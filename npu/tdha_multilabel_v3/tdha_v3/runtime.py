"Run at most three queued standalone workers; do not use DDP or kill other users' processes."
from __future__ import annotations
import importlib.metadata
import importlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .common import *

def dependency_imports(*, include_report=False):
    """Check queue imports; rendering packages are needed only for report export."""
    errors={}
    modules = ('numpy','scipy','pandas','sklearn','psutil')
    if include_report:
        modules += ('matplotlib','PIL','reportlab.pdfbase.pdfmetrics')
    for module in modules:
        try:importlib.import_module(module)
        except Exception as exc:errors[module]=f'{type(exc).__name__}: {exc}'
    return errors

def doctor(run, *, include_report=False):
    from .memory import settings
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    errors=dependency_imports(include_report=include_report)
    dump(run/'audit/dependencies.json',{'passed':not errors,'import_errors':errors})
    require(not errors,'Runtime dependencies are missing or cannot be imported: '+str(errors)+'; install requirements-server.txt in the verified environment without changing torch/CANN')
    import psutil
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
    out={'python':sys.version,'executable':sys.executable,'machine':platform.machine(),'packages':packages,
         'pip_check':check.stdout+check.stderr,'npu_smi':smi.stdout+smi.stderr,'source_code':code_hash(),
         'memory_policy':settings(),'host_memory':psutil.virtual_memory()._asdict()}
    dump(run/'audit/environment.json',out)
    require(sys.version_info[:2]==(3,11),'The server runtime requires Python 3.11')
    require(packages.get('torch','').split('+')[0]=='2.7.1' and packages.get('torch-npu',packages.get('torch_npu'))=='2.7.1.post8','PyTorch/TorchNPU versions do not match the verified runtime')
    require(smi.returncode==0,'npu-smi is unavailable')
    # A profiler missing from the mandatory benchmark blocks the run; do not treat training as successful or install it to hide an environment mismatch.
    print('Environment manifest saved. Optional profiler dependency warnings remain recorded; benchmark still checks actual operators.',flush=True)

def available(card):
    p=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
    require(p.returncode==0,'Cannot inspect device occupancy')
    return f'No running processes found in NPU {card}' in p.stdout

def task_id(net,rep,fold,seed,suffix=''):return f'{net}-{rep}-{fold}-s{seed}'+suffix

def task(net,rep,fold,seed,suffix='',**kw):return {'id':task_id(net,rep,fold,seed,suffix),'net':net,'rep':rep,'fold':fold,'seed':seed,**kw}

def stage_a():
    return [task(n,r,f,42) for n in ('N0','N1','N2') for r in ('F','W1','DVI') for f in ('A','B')]+[task('N0','P',f,42) for f in ('A','B')]

def schedule(run,tasks,cards,action='fit'):
    require(1<=len(cards)<=3 and len(set(cards))==len(cards),'Specify one to three distinct NPUs')
    run=Path(run);pending=list(tasks);running={}
    for t in tasks:dump(run/'queue'/f"{t['id']}.json",t)
    while pending or running:
        for card,(proc,handle,t) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del running[card]
            if code!=0:
                # Never kill an occupied worker; wait for its current optimization/sample step to finish before graceful exit.
                for p,h,_ in running.values():p.wait();h.close()
                raise Blocked(f"Task {t['id']} failed; inspect logs. Later stages were not started")
        for card in cards:
            if card in running:continue
            eligible=next((t for t in pending if t.get('required_card',card)==card),None)
            if eligible is None or not available(card):continue
            pending.remove(eligible);t=eligible
            env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONHASHSEED='0')
            log=run/'logs'/f"{t['id']}_{action}.log";log.parent.mkdir(parents=True,exist_ok=True);h=log.open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-m','tdha_v3','worker','--run',str(run),'--task',str(run/'queue'/f"{t['id']}.json"),'--action',action],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
            running[card]=(p,h,t)
        dump(run/'status.json',{'pending':[t['id'] for t in pending],'running':{str(c):t['id'] for c,(_,_,t) in running.items()},'action':action})
        if pending or running:time.sleep(10)

def select(run):
    run=Path(run);base=[read(run/'tasks'/task_id('N0','W1',f,42)/'done.json') for f in ('A','B')];ranking=[]
    def avg(items,key):
        v=[x['summary'][key] for x in items];require(all(x is not None for x in v),f'Selection metric {key} is not computable');return np.mean(v,axis=0)
    for n in ('N0','N1','N2'):
        for r in ('F','W1','DVI'):
            if (n,r)==('N0','W1'):continue
            a=[read(run/'tasks'/task_id(n,r,f,42)/'done.json') for f in ('A','B')]
            dr=avg(a,'Recall')-avg(base,'Recall');df=avg(a,'OFF_FPR')-avg(base,'OFF_FPR');dc=avg(a,'Recall_classes')-avg(base,'Recall_classes')
            passed=dr>=-.05 and df<=.02 and dc.min()>=-.10
            ranking.append({'net':n,'rep':r,'protected':bool(passed),'AP':float(avg(a,'AP')),'FPR':float(avg(a,'OFF_FPR')),'parameters':a[0]['parameters'],'Recall_delta':float(dr),'class_recall_delta':dc.tolist()})
    ranking.sort(key=lambda a:(not a['protected'],-a['AP'],a['FPR'],a['parameters'],a['net'],a['rep']))
    out={'candidate':ranking[0],'ranking':ranking,'selection_data':'训练家族后段的候选筛选集','code':code_hash()}
    dump(run/'selection.json',out);return out['candidate']

def benchmark(run,device):
    import torch
    from .training import Windows,collate,pairs_for,balanced_loss,class_weights,predict_reference,predict,rng_state,rng_hash,sync
    from .models import Model
    from .memory import cache,settings,BatchReader
    run=Path(run);require((run/'cache/ready.json').exists(),'Prepare the data first')
    require(read(run/'cache/ready.json')['details']['code']==code_hash(),'Input cache does not match the current code')
    records=[]
    for micro in (16,8):
        records=[];oom=False
        for n in ('N0','N1','N2'):
            for r in ('P','F','W1','DVI'):
                if r=='P' and n!='N0':continue
                try:
                    ds=Windows(run,'A',r,np.load(run/'cache/foldA/pairs.npy')[:micro])
                    t=time.monotonic();batch=collate(ds,range(len(ds)),device);sync(device);cold_load=time.monotonic()-t
                    t=time.monotonic();warm=collate(ds,range(len(ds)),device);sync(device);warm_load=time.monotonic()-t
                    require(all(torch.equal(batch[k],warm[k]) for k in batch),'RAM cache path changed the inputs')
                    before=rng_state(device)
                    with BatchReader() as reader:
                        for prefetched in reader.batches(ds,[range(len(ds))],device):
                            require(all(torch.equal(batch[k],prefetched[k]) for k in batch),'Prefetch path changed the inputs')
                    require(rng_hash(before)==rng_hash(rng_state(device)),'CPU prefetch changed the random stream')
                    del warm,prefetched
                    model=Model(n,r,42).to(device)
                    torch.npu.reset_peak_memory_stats();before=rng_state(device)
                    model.eval()
                    with torch.inference_mode():model(batch)
                    require(rng_hash(before)==rng_hash(rng_state(device)),'Evaluation changed the RNG state')
                    opt=torch.optim.AdamW(model.parameters(),lr=.0003,foreach=False);model.train();times=[]
                    for step in range(3):
                        sync(device);t=time.monotonic();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(batch),batch['y']);loss.backward()
                        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),'Network gradients are missing or non-finite')
                        opt.step();sync(device);times.append(time.monotonic()-t)
                    cal=Windows(run,'A',r,pairs_for(run,'A','calibration')[:64]);t=time.monotonic();predict(model,cal,device,micro);cal_time=time.monotonic()-t
                    from .cost import macs
                    count=macs(model,{k:v[:1] for k,v in batch.items()})
                    records.append({'network':n,'representation':r,'parameters':sum(p.numel() for p in model.parameters()),'MAC_per_endpoint':count,'microbatch':micro,'seconds_per_microbatch':float(np.mean(times[1:])),'calibration_64_seconds':cal_time,'peak_bytes':int(torch.npu.max_memory_allocated()),'initial_load_and_transfer_s':cold_load,'warm_load_and_transfer_s':warm_load,'ram_cache':cache().stats()})
                    del opt,model,batch,ds;torch.npu.empty_cache()
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and micro==16:
                        import gc
                        oom=True;model=batch=opt=ds=loss=None;gc.collect();torch.npu.empty_cache();break
                    raise
            if oom:break
        if not oom:break
    require(len(records)==10,'Capacity preflight did not cover all combinations')
    # The budget covers 40 epochs and full calibration; actual early stopping may shorten execution.
    ntrain=len(np.load(run/'cache/foldA/pairs.npy'));ncal=len(pairs_for(run,'A','calibration'))
    for r in records:r['max40_single_fit_hours']=(r['seconds_per_microbatch']*np.ceil(ntrain/micro)+r['calibration_64_seconds']*ncal/64)*40/3600
    dump(run/'benchmark.json',{'microbatch':micro,'accumulation':64//micro,'rows':records,'code':code_hash(),'memory_policy':settings(),'估计':'实测批次外推；预训练、缓存、报告与I/O另计，12–24小时不是截止时间'})

def run_queue(data,run,cards,reference):
    import fcntl
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with (run/'.controller.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Blocked('Another controller holds this run lock; refusing to start a second queue')
        try:return _run_queue_impl(data,run,cards,reference)
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def _run_queue_impl(data,run,cards,reference):
    from .pretrain import prepare_single
    run=Path(run);require((run/'benchmark.json').exists(),'Run benchmark first')
    require(read(run/'benchmark.json')['code']==code_hash(),'Benchmark does not match the current code identity')
    a=stage_a();schedule(run,a,cards)
    candidate=read(run/'selection.json')['candidate'] if (run/'selection.json').exists() else select(run)
    n,r=candidate['net'],candidate['rep']
    b=[task(nn,rr,f,s) for nn,rr in [('N0','W1'),(n,r)] for f in ('A','B') for s in (11,73)]
    schedule(run,b,cards)
    c=[];pretrain_tasks=[]
    for f in ('A','B'):
        prepare_single(data,run,f,r,reference)
        pt=task(n,r,f,42,'-pretrain',kind='pretrain');pretrain_tasks.append(pt)
        ft=task(n,r,f,42,'-finetune',pretrained=pt['id']);c.append(ft)
    schedule(run,pretrain_tasks,cards)
    schedule(run,c,cards)
    d=[]
    for nn,rr in [('N0','W1'),(n,r)]:
        base=read(run/'tasks'/task_id(nn,rr,'A',42)/'done.json');card=int(base['device'])
        require(card in cards,'Independent replay requires the original physical NPU')
        d.append(task(nn,rr,'A',42,'-replay',required_card=card))
    schedule(run,d,cards)
    checks=[]
    for t in d:
        old=read(run/'tasks'/task_id(t['net'],t['rep'],'A',42)/'done.json')['hashes'];new=read(run/'tasks'/t['id']/'done.json')['hashes']
        checks.append({'task':t['id'],'pass':old==new,'original':old,'replay':new})
    dump(run/'reproducibility.json',checks);require(all(x['pass'] for x in checks),'Independent replay hashes differ; cross-script readout is blocked')
    all_tasks=a+b+c+d
    freeze={'code':code_hash(),'protocol':PROTOCOL,'candidate':candidate,'reproduction':checks,'cache':file_hash(run/'cache/ready.json'),
            'models':{t['id']:read(run/'tasks'/t['id']/'done.json')['hashes'] for t in all_tasks}}
    if (run/'freeze.json').exists():require(read(run/'freeze.json')==clean(freeze),'Frozen contents changed')
    else:dump(run/'freeze.json',freeze)
    schedule(run,all_tasks,cards,action='readout')
    for t in d:
        old=read(run/'tasks'/task_id(t['net'],t['rep'],'A',42)/'readout.json');new=read(run/'tasks'/t['id']/'readout.json')
        require(old==new,'Cross-script replay inference hashes differ')
    dump(run/'execution_complete.json',{'main_fits':28,'pretrain_related':4,'independent_replays':2,'readout_reproduction':True,'code':code_hash()})
