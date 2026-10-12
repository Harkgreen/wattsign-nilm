from __future__ import annotations
import copy
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .common import *
from .data import load_record
from .models import Model,PretrainModel
from .metrics import threshold,binary,summarize
from .memory import cache,host_batch,to_device,BatchReader,device_encoding,encoding_windows,settings

def backend():
    import torch_npu  # Import only in an explicit server worker; failure must not trigger a CPU fallback.
    require(torch.npu.is_available() and torch.npu.device_count()==1,'Each worker must see exactly one available NPU')
    torch.npu.set_device(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.npu.config.allow_internal_format=False
    if hasattr(torch.npu,'conv'):torch.npu.conv.allow_hf32=False
    if hasattr(torch.npu,'matmul'):torch.npu.matmul.allow_hf32=False
    return torch.device('npu:0')

def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'accelerator':torch.npu.get_rng_state() if device.type=='npu' else None}

def restore_rng(s,device):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'])
    if device.type=='npu':torch.npu.set_rng_state(s['accelerator'])

def rng_hash(s):
    return digest({'python':str(s['python']),'numpy':[s['numpy'][0],array_hash(s['numpy'][1]),*s['numpy'][2:]],'cpu':array_hash(s['cpu'].numpy()),'npu':array_hash(s['accelerator'].cpu().numpy()) if s['accelerator'] is not None else None})

def state_hash(state):return digest({k:array_hash(v.detach().cpu().numpy()) for k,v in sorted(state.items())})

def checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);os.replace(tmp,path)

def sync(device):
    if device.type=='npu':torch.npu.synchronize()

class Windows(Dataset):
    def __init__(self,run,fold,rep,pairs,source='aggregate',tokens=256):
        self.run=Path(run);self.fold=fold;self.rep=rep;self.pairs=np.asarray(pairs,np.int64);self.tokens=tokens
        self.source=source;self.records={};self.features={};self.waves={}
        self.root=self.run/'cache'/f'fold{fold}' if source=='aggregate' else self.run/'pretrain'/f'fold{fold}'
        self.norm=read(self.run/'cache'/f'fold{fold}'/'normalizer.json')
    def record_path(self,r):
        return self.run/'cache'/f'record{r}' if self.source=='aggregate' else self.root/f'record{r}'
    def arrays(self,r):
        if r not in self.records:
            self.records[r]=load_record(self.record_path(r))
            self.features[r]={tag:cache().get(self.root/f'r{r}_{tag}.npy') for tag in (('power','F') if self.rep=='F' else ('power',))}
            if self.rep=='DVI':self.records[r]['dvi']=cache().get(self.record_path(r)/'dvi.npy')
        return self.records[r],self.features[r]
    def __len__(self):return len(self.pairs)
    def __getitem__(self,j):
        r,e=map(int,self.pairs[j]);arr,feat=self.arrays(r);end=e//5;lo=end-self.tokens+1
        require(lo>=0,'Window is outside the available record');x={'power':np.array(feat['power'][lo:end+1]),'y':np.array(arr['y'][e],np.float32),'mask':np.array(arr['known'][e],np.float32)}
        if self.rep=='F':x['extra']=np.array(feat['F'][lo:end+1])
        elif self.rep=='DVI':
            x['extra']=np.array(arr['dvi'][lo:end+1]);x['valid']=np.array(arr['dvi_valid'][lo:end+1],np.float32)
        elif self.rep=='W1':
            n=len(arr['time'])
            if r not in self.waves:self.waves[r]=cache().get(self.record_path(r)/'aligned.bin',dtype=np.float32,shape=(n,3,1024))
            wave=np.array(self.waves[r][lo*5:(end+1)*5]);sc=self.norm['wave_scale']
            wave[:,0]/=sc[0];wave[:,1:]/=sc[1]
            x['extra']=wave.reshape(self.tokens,5,3,1024)
            # V2 W1 validity requires voltage-fundamental and absolute-harmonic conditions, not only Urms > 0.
            x['valid']=np.array(arr['fm'][lo*5:(end+1)*5,0],np.float32).reshape(self.tokens,5)
        return x

def collate(ds,indices,device):
    return to_device(host_batch(ds,indices),device)

def class_weights(ds):
    pos=np.zeros(11);neg=np.zeros(11)
    for r in np.unique(ds.pairs[:,0]):
        arr,_=ds.arrays(int(r));ix=ds.pairs[ds.pairs[:,0]==r,1];y=arr['y'][ix];m=arr['known'][ix]
        pos+=(y*m).sum(0);neg+=((1-y)*m).sum(0)
    active=(pos>0)&(neg>0);n=pos+neg
    return {'positive':pos,'negative':neg,'active':active,'wp':n/(2*np.maximum(pos,1)),'wn':n/(2*np.maximum(neg,1))}

def balanced_loss(logits,y,mask,weights):
    w=y*weights['wp']+(1-y)*weights['wn'];m=mask*weights['active']
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*w*m
    # Use fixed training-distribution weights and equal active-head weighting; each head denominator is its known sample count in this batch.
    den=m.sum(0);head=den>0
    return (loss.sum(0)/den.clamp_min(1))[head].mean()

def calibration_loss(y,z,mask,active,primary=True):
    losses=np.logaddexp(0,z.astype(np.float64))-y*z
    values=[]
    for c in range(6 if primary else 11):
        on=mask[:,c].astype(bool)&(y[:,c]==1);off=mask[:,c].astype(bool)&(y[:,c]==0)
        if active[c] and on.any() and off.any():values.append((losses[on,c].mean()+losses[off,c].mean())/2)
    require(len(values)==6 if primary else len(values)>0,'Early-stopping data does not support the required primary classes')
    return float(np.mean(values))

def predict_reference(model,ds,device,micro):
    model.eval();before=rng_state(device);parts=[]
    # Infer each window independently as a safe reference; never carry GRU state or stale waveform embeddings across windows.
    with torch.inference_mode():
        for lo in range(0,len(ds),micro):parts.append(model(collate(ds,range(lo,min(lo+micro,len(ds))),device)).cpu().numpy().astype(np.float32))
    sync(device);require(rng_hash(before)==rng_hash(rng_state(device)),'Evaluation consumed the training RNG state')
    return np.concatenate(parts) if parts else np.empty((0,11),np.float32)

def predict(model,ds,device,micro):
    if isinstance(model,PretrainModel):return predict_reference(model,ds,device,micro)
    model.eval();before=rng_state(device);out=np.zeros((len(ds),11),np.float32)
    with torch.inference_mode():
        for r in np.unique(ds.pairs[:,0]):
            where=np.flatnonzero(ds.pairs[:,0]==r);ep=ds.pairs[where,1]//5
            first=int(ep.min()-255);last=int(ep.max());indices=np.arange(first,last+1)
            single=Windows(ds.run,ds.fold,ds.rep,np.stack((np.full(len(indices),r),indices*5+4),1),ds.source,tokens=1)
            encoded=np.empty((len(indices),128),np.float32)
            # Use temporary eval encodings from the current checkpoint; never persist them or reuse them for training.
            for lo in range(0,len(single),micro*8):
                batch=collate(single,range(lo,min(lo+micro*8,len(single))),device)
                encoded[lo:lo+len(batch['power'])]=model.front(batch)[:,0].cpu().numpy()
            resident=device_encoding(encoded,device)
            for lo in range(0,len(ep),micro):
                ends=ep[lo:lo+micro]-first
                windows=encoding_windows(encoded,ends,device,resident)
                if lo==0 and resident is not None:
                    reference=encoding_windows(encoded,ends,device)
                    require(torch.equal(windows,reference),'Device-cache window indexing changed the input')
                    del reference
                out[where[lo:lo+micro]]=model.backbone(windows).cpu().numpy()
            del resident
    sync(device);require(rng_hash(before)==rng_hash(rng_state(device)),'Optimized evaluation consumed RNG state')
    require(np.isfinite(out).all(),'Nonfinite inference output')
    # Check ordinary, first/last and middle endpoints on every validation pass instead of assuming the accelerated path is correct.
    probe=np.unique(np.linspace(0,len(ds)-1,min(5,len(ds)),dtype=int))
    if len(probe):
        ref=predict_reference(model,Windows(ds.run,ds.fold,ds.rep,ds.pairs[probe],ds.source),device,micro)
        require(np.allclose(out[probe],ref,rtol=2e-5,atol=2e-5),'Optimized inference is not equivalent; stop without changing the model')
    return out

def metadata(ds):
    rows={k:[] for k in ('y','mask','strict','time','power','near','record','endpoint')}
    for r,e in ds.pairs:
        a,_=ds.arrays(int(r));e=int(e)
        for k,tag in [('y','y'),('mask','known'),('strict','strict'),('time','time'),('near','near')]:rows[k].append(a[tag][e])
        rows['power'].append(a['basic'][e-4:e+1,0].mean());rows['record'].append(r);rows['endpoint'].append(e)
    return {k:np.asarray(v) for k,v in rows.items()}

def pairs_for(run,fold,part):
    ep=read(Path(run)/'cache'/f'fold{fold}'/'endpoints.json')
    return np.array([(r,e) for r in range(1,8) for e in ep[f'{r}_{part}'] if (r==1)==(part=='third')],np.int64).reshape(-1,2) if part!='third' else np.array([(1,e) for e in ep['1_cross']],np.int64)

def evaluate_prediction(run,fold,ds,z,thresholds,active,high):
    meta=metadata(ds);mapping=read(Path(run)/'audit/mapping.json');rows=[]
    for r in np.unique(meta['record']):
        rr=meta['record']==r
        for c in range(11):
            if not active[c]:rows.append({'record':int(r),'class':c,'status':'未训练','present':str(c) in mapping[str(r)]});continue
            for region in ('main','strict500','steady','overlap','main_05'):
                m=rr&meta['mask'][:,c].astype(bool)
                if region=='strict500':m&=meta['strict'][:,c].astype(bool)
                if region=='steady':m&=meta['near']>1000000
                if region=='overlap':m&=(meta['y']*meta['mask']).sum(1)>=2
                row=binary(meta['y'][m,c],z[m,c],0. if region=='main_05' else thresholds[c],meta['power'][m],high[c])
                row.update(record=int(r),class_name=CLASSES[c],**{'class':c,'present':str(c) in mapping[str(r)],'region':region})
                rows.append(row)
    return rows,summarize([r for r in rows if r.get('region')=='main'])

def fit(run,task,device):
    run=Path(run);fold=task['fold'];seed=task['seed'];rep=task['rep'];net=task['net'];dest=run/'tasks'/task['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'done.json').exists():return read(dest/'done.json')
    capacity=read(run/'benchmark.json');micro=capacity['microbatch']
    pre=task.get('kind')=='pretrain';source='single' if pre else 'aggregate'
    root=run/'pretrain'/f'fold{fold}' if pre else run/'cache'/f'fold{fold}'
    pairs=np.load(root/'pairs.npy');orders=np.load(root/f'order{seed}.npy')
    calpairs=np.load(root/'calibration.npy') if pre else pairs_for(run,fold,'calibration')
    train=Windows(run,fold,rep,pairs,source,tokens=1 if pre else 256);cal=Windows(run,fold,rep,calpairs,source,tokens=1 if pre else 256)
    weight=class_weights(train);active=weight['active'];require(active[:6].all() if not pre else active.any(),'Insufficient training class coverage')
    weights={k:torch.tensor(weight[k],dtype=torch.float32,device=device) for k in ('wp','wn','active')}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='npu':torch.npu.manual_seed_all(seed)
    model=(PretrainModel(rep,seed) if pre else Model(net,rep,seed)).to(device)
    if task.get('pretrained'):
        pretrained=torch.load(run/'tasks'/task['pretrained']/'best.pt',map_location='cpu',weights_only=False)['model']
        model.front.load_state_dict({k[6:]:v for k,v in pretrained.items() if k.startswith('front.')})
    initial=state_hash(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False)
    ep=0;position=0;best=float('inf');bad=0;history=[];start=time.monotonic();elapsed=0.
    latest=dest/'last.pt'
    if latest.exists():
        last=torch.load(latest,map_location='cpu',weights_only=False)
        require(last['code']==code_hash() and last['sample']==array_hash(pairs),'Resume identity mismatch')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);ep=last['epoch'];position=last['position'];best=last['best'];bad=last['bad'];history=last['history'];elapsed=last['elapsed'];restore_rng(last['rng'],device)
    calmeta=metadata(cal);trainmeta=metadata(train)
    def save(epoch,pos):
        checkpoint(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'position':pos,'best':best,'bad':bad,'history':history,'rng':rng_state(device),'sample':array_hash(pairs),'code':code_hash(),'elapsed':elapsed+time.monotonic()-start})
    maxepochs=20 if pre else 40;patience=3 if pre else 5
    for epoch in range(ep,maxepochs):
        if bad>=patience:break
        model.train();losses=[]
        with BatchReader() as reader:
            for lo in range(position,len(pairs),64):
                indices=orders[epoch,lo:lo+64];optimizer.zero_grad(set_to_none=True)
                # Preserve the effective-batch denominator, microbatch size and random streams.
                ys=trainmeta['mask'][indices];den=torch.tensor(ys.sum(0),device=device)*weights['active'];heads=den>0
                groups=(indices[q:q+micro] for q in range(0,len(indices),micro))
                for batch in reader.batches(train,groups,device):
                    z=model(batch)
                    raw=torch.nn.functional.binary_cross_entropy_with_logits(z,batch['y'],reduction='none')
                    w=batch['y']*weights['wp']+(1-batch['y'])*weights['wn'];m=batch['mask']*weights['active']
                    loss=((raw*w*m).sum(0)/den.clamp_min(1))[heads].mean()
                    require(torch.isfinite(loss).item(),'Nonfinite training loss');loss.backward();losses.append(float(loss.detach().cpu()))
                require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),'Gradient contains NaN or Inf')
                optimizer.step();save(epoch,lo+len(indices))
        position=0;z=predict(model,cal,device,micro);value=calibration_loss(calmeta['y'],z,calmeta['mask'],active,not pre)
        history.append({'epoch':epoch+1,'calibration_balanced_BCE':value,'train_loss_sum':sum(losses)})
        if value<best-1e-8:
            best=value;bad=0;checkpoint(dest/'best.pt',{'model':model.state_dict(),'epoch':epoch+1})
        else:bad+=1
        save(epoch+1,0);dump(dest/'progress.json',{'epoch':epoch+1,'best':best,'patience':bad,'task':task})
        print(task['id'],epoch+1,value,flush=True)
    model.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=False)['model'])
    hashes={'model':state_hash(model.state_dict()),'initial':initial,'sampling':array_hash(pairs),'batches':array_hash(orders)}
    dump(dest/'history.json',history)
    dump(dest/'memory.json',{'settings':settings(),'ram_cache':cache().stats(),'note':'缓存预算不是进程总RSS；显存缓存仅用于当前checkpoint的eval编码'})
    if pre:
        done={'task':task,'hashes':hashes,'elapsed_s':elapsed+time.monotonic()-start,'parameters':sum(p.numel() for p in model.parameters()),'status':'预训练完成'}
        dump(dest/'done.json',done);return done
    z=predict(model,cal,device,micro);thresholds=[]
    for c in range(11):
        m=calmeta['mask'][:,c].astype(bool)
        thresholds.append(threshold(calmeta['y'][m,c],z[m,c]) if active[c] else None)
    highmeta=metadata(Windows(run,fold,'P',pairs_for(run,fold,'train_natural')));high=[]
    for c in range(11):
        off=highmeta['mask'][:,c].astype(bool)&(highmeta['y'][:,c]==0)
        high.append(float(np.quantile(highmeta['power'][off],.8)) if off.any() else None)
    # The frozen high-power cutoff uses natural training endpoints, never calibration or test data.
    dump(dest/'thresholds.json',{'logit':thresholds,'active':active,'high_power':high,'shared_target':.95})
    save_array(dest/'calibration.npy',z);hashes['calibration']=array_hash(z);hashes['thresholds']=digest(thresholds)
    ds=Windows(run,fold,rep,pairs_for(run,fold,'time'));pred=predict(model,ds,device,micro)
    rows,summary=evaluate_prediction(run,fold,ds,pred,thresholds,active,high)
    save_array(dest/'time.npy',pred);dump(dest/'time_metrics.json',{'rows':rows,'summary':summary});hashes['time_prediction']=array_hash(pred);hashes['time_metrics']=digest(rows)
    done={'task':task,'hashes':hashes,'elapsed_s':elapsed+time.monotonic()-start,'parameters':sum(p.numel() for p in model.parameters()),'summary':summary,'device':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'status':'已完成'}
    dump(dest/'done.json',done);return done

def readout(run,task,device):
    run=Path(run);require((run/'freeze.json').exists(),'Frozen model selection is required before cross-script prediction');dest=run/'tasks'/task['id']
    require((dest/'done.json').exists(),'Model training has not completed');model=Model(task['net'],task['rep'],task['seed']).to(device)
    model.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=False)['model'])
    require(state_hash(model.state_dict())==read(run/'freeze.json')['models'][task['id']]['model'],'Model identity changed after freezing')
    th=read(dest/'thresholds.json');micro=read(run/'benchmark.json')['microbatch']
    result={}
    for part in ('cross','third'):
        ds=Windows(run,task['fold'],task['rep'],pairs_for(run,task['fold'],part));z=predict(model,ds,device,micro)
        rows,summary=evaluate_prediction(run,task['fold'],ds,z,th['logit'],th['active'],th['high_power'])
        save_array(dest/f'{part}.npy',z);dump(dest/f'{part}_metrics.json',{'rows':rows,'summary':summary})
        result[part]={'prediction':array_hash(z),'metrics':digest(rows)}
    dump(dest/'readout.json',result)
