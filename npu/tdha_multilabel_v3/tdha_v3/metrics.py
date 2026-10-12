import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.stats import spearmanr
from .common import *

def threshold(y,z):
    y=np.asarray(y,bool);z=np.asarray(z,np.float32)
    if not y.any() or y.all():return None
    # The highest feasible threshold is the ceil(.95*n_pos)-th largest positive score; ties use >=.
    k=int(np.ceil(.95*y.sum()));result=np.sort(z[y])[-k]
    candidates=np.r_[np.nextafter(z.min(),np.float32(-np.inf)),np.unique(z),np.nextafter(z.max(),np.float32(np.inf))]
    assert result in candidates and (z[y]>=result).mean()>=.95
    return float(result)

def conditional(y,z,p):
    bins=np.floor(p/50).astype(np.int64);out=np.full(len(y),np.nan);coverage=[]
    for b in np.unique(bins[y==1]):
        on=(bins==b)&(y==1);off=np.sort(z[(bins==b)&(y==0)])
        coverage.append({'bin':int(b),'on':int(on.sum()),'off':len(off)})
        if len(off):out[on]=(np.searchsorted(off,z[on],'left')+np.searchsorted(off,z[on],'right'))/(2*len(off))
    pos=out[y==1];return float(np.nanmean(pos)) if np.isfinite(pos).any() else None,float(np.isfinite(pos).sum()/max(len(pos),1)),coverage

def binary(y,z,t,p,high):
    y=np.asarray(y,int);z=np.asarray(z,float)
    out={'n':len(y),'positive':int(y.sum()),'AP':None,'ROC_AUC':None}
    if len(y) and len(np.unique(y))==2:
        out.update(AP=float(average_precision_score(y,z)),ROC_AUC=float(roc_auc_score(y,z)))
    if t is None:return out|{'threshold_available':False}
    pred=z>=t;tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum());tn=int((~pred&(y==0)).sum());fn=int((~pred&(y==1)).sum())
    ratio=lambda a,b:float(a/b) if b else None
    rec=ratio(tp,tp+fn);fpr=ratio(fp,fp+tn)
    cc,cov,bins=conditional(y,z,p)
    out.update(threshold_available=True,Recall=rec,Precision=ratio(tp,tp+fp),F1=ratio(2*tp,2*tp+fp+fn),OFF_FPR=fpr,
        Balanced_accuracy=(rec+1-fpr)/2 if rec is not None and fpr is not None else None,
        confusion=[[tn,fp],[fn,tp]],conditional=cc,coverage=cov,bins=bins,
        high_FPR=ratio((pred&(y==0)&(p>=high)).sum(),((y==0)&(p>=high)).sum()))
    out['power_FPR']=[ratio((pred&(y==0)&(p>=lo)&(p<hi)).sum(),((y==0)&(p>=lo)&(p<hi)).sum()) for lo,hi in [(float('-inf'),800),(800,1500),(1500,2500),(2500,float('inf'))]]
    off=y==0
    out['OFF_power_spearman']=float(spearmanr(z[off],p[off]).statistic) if off.sum()>2 and np.ptp(z[off])>0 and np.ptp(p[off])>0 else None
    return clean(out)

def summarize(rows):
    result={}
    for metric in ('AP','Recall','OFF_FPR','high_FPR','conditional','F1','Precision','ROC_AUC'):
        classes=[]
        for c in range(6):
            # Hierarchical equal weighting: record, class, fold and seed aggregation is completed by callers; absence does not contribute AP/Recall.
            vals=[x[metric] for x in rows if x['class']==c and x.get(metric) is not None and (metric not in ('AP','Recall') or x['present'])]
            classes.append(float(np.mean(vals)) if vals else None)
        result[metric]=float(np.mean(classes)) if all(v is not None for v in classes) else None
        result[metric+'_classes']=classes
    return result

def weighted_ap(y,z,w):
    order=np.argsort(-z,kind='stable');y=y[order];z=z[order];w=w[order]
    tp=np.cumsum(w*y);n=np.cumsum(w);ends=np.r_[z[1:]!=z[:-1],True]
    tp=tp[ends];n=n[ends]
    if not len(tp) or tp[-1]<=0 or n[-1]<=tp[-1]:return np.nan
    return float(np.sum(np.diff(np.r_[0,tp])*tp/np.maximum(n,1))/tp[-1])

def block_bootstrap_pair(records,iterations=2000,seed=42):
    "Each item contains all seeds for one fold/record; share block weights across heads, models and seeds.\n    records: y/mask [N,C], a/b [seed,N,C], ta/tb [seed,C], time, present.\n    Macro AP/Recall use present classes; OFF FPR includes absence. Preserve equal record and fold weights.\n    "
    rng=np.random.default_rng(seed);results=[];draws=[];invalid=0
    prepared=[]
    for r in records:
        block=(r['time']-r['time'][0])//60000000;_,inv=np.unique(block,return_inverse=True)
        r['_sorted']={}
        for c in range(6):
            ok=r['mask'][:,c].astype(bool)
            for tag in ('a','b'):
                for si in range(len(r[tag])):
                    z=r[tag][si,ok,c];order=np.argsort(-z,kind='stable');zz=z[order]
                    r['_sorted'][tag,si,c]=(order,np.r_[zz[1:]!=zz[:-1],True] if len(zz) else np.empty(0,bool))
        prepared.append((r,inv,int(inv.max()+1)))
    for attempt in range(iterations*20):
        if len(results)==iterations:break
        per_fold={};draw={};valid_repeat=True
        for ri,(r,inv,k) in enumerate(prepared):
            counts=np.bincount(rng.integers(k,size=k),minlength=k);w=counts[inv];draw[str(ri)]=counts.tolist()
            fold=per_fold.setdefault(r['fold'],{c:[] for c in range(6)})
            for c in range(6):
                if not r['trained'][c]:continue
                ok=r['mask'][:,c].astype(bool);y=r['y'][ok,c];ww=w[ok]
                if not len(y):continue
                if (y==0).any() and (ww*(1-y)).sum()==0:valid_repeat=False
                if (y==1).any() and (ww*y).sum()==0:valid_repeat=False
                values=[]
                for tag,scores,thresholds in (('a',r['a'],r['ta']),('b',r['b'],r['tb'])):
                    seeds=[]
                    for si in range(len(scores)):
                        z=scores[si,ok,c];t=thresholds[si,c];on=ww*y;off=ww*(1-y)
                        order,ends=r['_sorted'][tag,si,c]
                        tp=np.cumsum(ww[order]*y[order])[ends];total=np.cumsum(ww[order])[ends]
                        ap=float(np.sum(np.diff(np.r_[0,tp])*tp/np.maximum(total,1))/tp[-1]) if r['present'][c] and len(tp) and tp[-1]>0 and total[-1]>tp[-1] else np.nan
                        if r['present'][c] and len(np.unique(y))==2 and not np.isfinite(ap):valid_repeat=False
                        if not np.isfinite(t):valid_repeat=False
                        rec=np.sum(on*(z>=t))/on.sum() if on.sum() and r['present'][c] else np.nan
                        fpr=np.sum(off*(z>=t))/off.sum() if off.sum() else np.nan
                        seeds.append([ap,rec,fpr])
                    values.append(np.mean(seeds,axis=0))
                fold[c].append(np.array(values))
        folds=[]
        for f,classes in per_fold.items():
            cs=[]
            for c,vals in classes.items():
                if not vals:cs.append(np.full(3,np.nan));continue
                v=np.stack(vals)
                # Absent AP/Recall does not enter record means; failed positive-support resamples cannot be accepted silently.
                a=np.nanmean(v,axis=0);cs.append(a[1]-a[0])
            folds.append(np.mean(cs,axis=0))
        delta=np.mean(folds,axis=0)
        if valid_repeat and np.isfinite(delta).all():results.append(delta);draws.append(draw)
        else:invalid+=1
    a=np.array(results)
    return {'metrics':['AP','Recall','OFF_FPR'],'valid':len(a),'invalid':invalid,'complete':len(a)==iterations,
            'CI':np.quantile(a,[.025,.975],axis=0).tolist() if len(a) else None,'samples':a,'draws':draws}
