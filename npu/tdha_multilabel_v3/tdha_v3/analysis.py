"Exploratory analysis after freezing; results do not reselect a model."
from pathlib import Path
import numpy as np
from .common import *
from .runtime import task_id
from .training import Windows,pairs_for,metadata
from .metrics import block_bootstrap_pair

def comparison_records(run,candidate,part='cross'):
    run=Path(run);records=[]
    for fold in ('A','B'):
        ds=Windows(run,fold,'P',pairs_for(run,fold,part));meta=metadata(ds)
        arrays=[];thresholds=[];trained=None
        for net,rep in [('N0','W1'),(candidate['net'],candidate['rep'])]:
            zs=[];ts=[]
            for seed in (11,42,73):
                root=run/'tasks'/task_id(net,rep,fold,seed);zs.append(np.load(root/f'{part}.npy'))
                th=read(root/'thresholds.json');ts.append([np.nan if x is None else x for x in th['logit']]);trained=th['active']
            arrays.append(np.stack(zs));thresholds.append(np.asarray(ts))
        mapping=read(run/'audit/mapping.json')
        for r in np.unique(meta['record']):
            ix=meta['record']==r
            records.append({'fold':fold,'record':int(r),'y':meta['y'][ix],'mask':meta['mask'][ix],
                'time':meta['time'][ix],'near':meta['near'][ix],'a':arrays[0][:,ix],'b':arrays[1][:,ix],
                'ta':thresholds[0],'tb':thresholds[1],'trained':trained,
                'present':[str(c) in mapping[str(r)] for c in range(11)]})
    return records

def absent_bootstrap(records):
    rng=np.random.default_rng(42);samples=[];draws=[];attempts=0
    points=[];class_points={};class_samples={}
    for r in records:
        cc=[]
        for c in range(11):
            if r['present'][c] or not r['trained'][c] or not np.isfinite(r['ta'][:,c]).all() or not np.isfinite(r['tb'][:,c]).all():continue
            off=r['mask'][:,c].astype(bool)&(r['y'][:,c]==0)
            if off.any():
                point=float(((r['b'][:,off,c]>=r['tb'][:,c,None]).mean(1)-(r['a'][:,off,c]>=r['ta'][:,c,None]).mean(1)).mean())
                cc.append(point);key=f"{r['fold']}_{c}";class_points[key]=point;class_samples[key]=[]
        points.append(np.mean(cc) if cc else np.nan)
    while len(samples)<2000 and attempts<40000:
        attempts+=1;folds=[];draw=[];class_draw={}
        for r in records:
            _,inv=np.unique((r['time']-r['time'][0])//60000000,return_inverse=True);k=inv.max()+1
            count=np.bincount(rng.integers(k,size=k),minlength=k);w=count[inv];classes=[];draw.append(count.tolist())
            # Include all trained absent classes, not only the primary classes; both sides use their own calibration thresholds.
            for c in range(11):
                if r['present'][c] or not r['trained'][c] or not np.isfinite(r['ta'][:,c]).all() or not np.isfinite(r['tb'][:,c]).all():continue
                off=r['mask'][:,c].astype(bool)&(r['y'][:,c]==0);ww=w*off
                if not ww.sum():classes.append(np.nan);continue
                a=((r['a'][:,:,c]>=r['ta'][:,c,None])*ww).sum(1)/ww.sum()
                b=((r['b'][:,:,c]>=r['tb'][:,c,None])*ww).sum(1)/ww.sum();value=float((b-a).mean());classes.append(value);class_draw[f"{r['fold']}_{c}"]=value
            folds.append(np.mean(classes) if classes else np.nan)
        if np.isfinite(folds).all() and set(class_draw)==set(class_samples):
            samples.append(float(np.mean(folds)));draws.append(draw)
            for key,value in class_draw.items():class_samples[key].append(value)
    class_ci={k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in class_samples.items()}
    return {'point':float(np.mean(points)),'per_class_point':class_points,'per_class_CI':class_ci,'valid':len(samples),'invalid':attempts-len(samples),'CI':np.quantile(samples,[.025,.975]).tolist() if samples else None,'samples':samples,'draws':draws}

def analyze(run):
    run=Path(run);require((run/'execution_complete.json').exists(),'Execution is incomplete; a final success decision cannot be generated')
    candidate=read(run/'selection.json')['candidate'];records=comparison_records(run,candidate)
    boot=block_bootstrap_pair(records);save_array(run/'analysis/bootstrap.npy',boot.pop('samples'));dump(run/'analysis/bootstrap_indices.json',boot.pop('draws'));dump(run/'analysis/bootstrap.json',boot)
    absent=absent_bootstrap(comparison_records(run,candidate,'third'));save_array(run/'analysis/absent_bootstrap.npy',np.array(absent.pop('samples')));dump(run/'analysis/absent_indices.json',absent.pop('draws'));dump(run/'analysis/absent.json',absent)
    points=[];perclass=[];seedchecks={};steady=[]
    for fold in ('A','B'):
        seedchecks[fold]=0
        for seed in (11,42,73):
            all_metrics=[read(run/'tasks'/task_id(n,r,fold,seed)/'cross_metrics.json') for n,r in [('N0','W1'),(candidate['net'],candidate['rep'])]]
            a,b=[x['summary'] for x in all_metrics]
            require(all(a[k] is not None and b[k] is not None for k in ('AP','Recall','OFF_FPR')),'Cross-script primary metrics are undefined')
            points.append([b[k]-a[k] for k in ('AP','Recall','OFF_FPR')]);seedchecks[fold]+=b['AP']>a['AP']
            perclass.append(np.array(b['Recall_classes'])-np.array(a['Recall_classes']))
            from .metrics import summarize
            ss=[summarize([v for v in x['rows'] if v.get('region')=='steady']) for x in all_metrics]
            steady.append(ss[1]['AP']-ss[0]['AP'] if all(x['AP'] is not None for x in ss) else np.nan)
    delta=np.mean(points,0);classdelta=np.mean(perclass,0);ci=np.array(boot['CI']) if boot['CI'] else np.full((2,3),np.nan)
    # Keep the historical serialized criterion labels for existing report readers.
    criteria={
        '跨脚本AP提升及CI':bool(delta[0]>=.01 and ci[0,0]>0),
        'Recall非劣':bool(delta[1]>=-.05 and ci[0,1]>=-.05),
        'OFF误报保护':bool(delta[2]<=.02 and ci[1,2]<=.02),
        '逐类别Recall保护':bool(classdelta.min()>=-.10),
        '两折seed一致性':all(v>=2 for v in seedchecks.values()),
        '第三脚本缺席类别误报':bool(absent['valid']==2000 and absent['per_class_point'] and all(v<=.02 and absent['per_class_CI'][k][1]<=.02 for k,v in absent['per_class_point'].items())),
        '稳态方向':bool(np.isfinite(steady).all() and np.mean(steady)>=0),
        'Bootstrap有效数量':boot['complete']}
    conclusion='值得扩大复验（仅限TDHA受控记录上的探索性证据）' if all(criteria.values()) else '未达到原冻结扩大复验判据；完整保留诊断结果'
    out={'candidate':candidate,'delta':dict(zip(('AP','Recall','OFF_FPR'),delta)),'CI':boot['CI'],'class_recall_delta':classdelta,'seed_AP_improvements':seedchecks,'steady_AP_delta':float(np.mean(steady)),'criteria':criteria,'conclusion':conclusion}
    dump(run/'analysis/decision.json',out);return out
