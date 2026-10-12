from __future__ import annotations
import csv
import io
import os
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
from .common import *
from .physics import quantities, harmonics, aligned, changes, frequency_changes, pool5, raster5

def timestamp(s): return np.datetime64(s.replace('/','-'),'us').astype(np.int64)

def mappings(path):
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}; result={}
    with zipfile.ZipFile(path) as z:
        shared=[''.join(t.text or '' for t in e.findall('.//m:t',ns)) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        for rec in range(1,8):
            rows=[]
            for row in ET.fromstring(z.read(f'xl/worksheets/sheet{rec}.xml')).findall('.//m:sheetData/m:row',ns):
                d={}
                for cell in row.findall('m:c',ns):
                    v=cell.find('m:v',ns)
                    if v is not None: d[re.match('[A-Z]+',cell.attrib['r']).group()]=shared[int(v.text)] if cell.attrib.get('t')=='s' else v.text
                rows.append(d)
            mapping={}
            for row in rows:
                name=row.get('A','').strip()
                if name not in NAMES: continue
                marks=[j for j,c in enumerate('BCDEF') if row.get(c,'') not in ('','0')]
                require(len(marks)<=1,f'Multiple sockets mapped to the same class: {rec}/{name}')
                if marks: mapping[NAMES.index(name)]={'socket':marks[0],'mode':row.get('G','').strip()}
            require(len(mapping)==5 and len({x['socket'] for x in mapping.values()})==5,f'Invalid socket mapping: {rec}')
            result[rec]=mapping
    return result

def labels(z,member,mapping,single_class=None):
    with z.open(member) as f: rows=list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')))
    t=np.array([timestamp(r['Time']) for r in rows],np.int64)
    require(np.all(np.diff(t)>0),'Label timestamps must increase strictly')
    y=np.zeros((len(t),11),np.uint8)
    if single_class is None:
        states=[]
        for row in rows:
            pairs=re.findall(r'Socket([0-4]):\s*([01])',row['event'])
            require(len(pairs)==5 and len(dict(pairs))==5,'Could not parse all five socket states from the label')
            states.append([int(dict(pairs)[str(j)]) for j in range(5)])
        states=np.array(states,np.uint8)
        for c,entry in mapping.items(): y[:,int(c)]=states[:,entry['socket']]
        any_change=np.r_[False,np.any(states[1:]!=states[:-1],1)]
    else:
        valid=np.ones_like(y,bool);anomalies=[]
        for k,row in enumerate(rows):
            event=row['event']
            if event not in ('1-1','1-0'):
                # Handle only the reviewed malformed state; do not infer ON/OFF from the last character.
                require(event=='1)0',f'Invalid single-appliance state format: {member} CSV row {k+2} event={event!r}')
                valid[k,single_class]=False
                anomalies.append({'member':member,'csv_line':k+2,'id':row.get('id'),
                                  'time':row['Time'],'event':event,'class':single_class,
                                  'action':'本行至下一合法标签无监督；不推断OFF'})
            else:y[k,single_class]=int(event[-1])
        edge=np.r_[not valid[0,single_class],
                   (y[1:,single_class]!=y[:-1,single_class]) |
                   ~valid[1:,single_class] | ~valid[:-1,single_class]]
        if anomalies:
            for item in anomalies:
                k=item['csv_line']-2
                nxt=next((j for j in range(k+1,len(rows)) if valid[j,single_class]),None)
                item['unknown_until']=rows[nxt]['Time'] if nxt is not None else None
            return {'time':t,'state':y,'any_switch':t[edge],
                    'state_valid':valid,'label_anomalies':anomalies}
        any_change=edge
    return {'time':t,'state':y,'any_switch':t[any_change]}

def distance(t,events):
    if not len(events): return np.full(len(t),10**15,np.int64)
    p=np.searchsorted(events,t)
    return np.minimum(np.abs(t-events[np.maximum(p-1,0)]),np.abs(t-events[np.minimum(p,len(events)-1)]))

def project(lab,t):
    ix=np.searchsorted(lab['time'],t,side='right')-1
    span=(ix>=0)&(t<=lab['time'][-1]); y=lab['state'][np.maximum(ix,0)].copy();y[~span]=0
    dist=np.stack([distance(t,lab['time'][np.r_[False,np.diff(lab['state'][:,c].astype(int))!=0]]) for c in range(11)],1)
    if 'state_valid' in lab:
        valid=lab['state_valid'];state_known=valid[np.maximum(ix,0)]
        for c in range(11):
            edges=np.r_[not valid[0,c],
                        (np.diff(lab['state'][:,c].astype(int))!=0) |
                        ~valid[1:,c] | ~valid[:-1,c]]
            dist[:,c]=distance(t,lab['time'][edges])
        return {'y':y,'known':span[:,None]&state_known&(dist>150000),
                'strict':span[:,None]&state_known&(dist>500000),
                'near':distance(t,lab['any_switch']),'switch_distance':dist}
    return {'y':y,'known':span[:,None]&(dist>150000),'strict':span[:,None]&(dist>500000),
            'near':distance(t,lab['any_switch']),'switch_distance':dist}

def extract(archive,member,out):
    "Stream the complete member to trigger ZIP CRC checks; original files remain read-only."
    out=Path(out);out.mkdir(parents=True,exist_ok=True); stats=Counter(); pending={};seen=set()
    times=[];indices=[];basics=[];masks=[];fs=[];fms=[];raw_block=[];tb=[];ib=[]
    rawfile=out/'adc.bin.tmp'; wavefile=out/'aligned.bin.tmp'
    def flush(rh,wh):
        if not raw_block:return
        raw=np.stack(raw_block);raw.tofile(rh)
        q,m=quantities(raw);f,fm=harmonics(raw);w,g=aligned(raw);w.tofile(wh)
        basics.append(q);masks.append(m);fs.append(f);fms.append(fm);times.extend(tb);indices.extend(ib)
        raw_block.clear();tb.clear();ib.clear()
    with zipfile.ZipFile(archive) as z, z.open(member) as stream, rawfile.open('wb') as rh, wavefile.open('wb') as wh:
        for row in csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline='')):
            stats['rows']+=1;key=(row['HomeID'],row['RecvTime'],row['InternalIndex']);ch=int(row['Channel'])
            require(ch in (1,2),'Unknown waveform channel')
            pair=pending.setdefault(key,{})
            require(ch not in pair and key not in seen,'Duplicate key/channel: preserve the temporary audit and stop without overwriting the anomalous data')
            pair[ch]=row
            if len(pair)!=2: continue
            del pending[key];seen.add(key)
            a=[np.fromstring(pair[c]['Value'],sep='|',dtype=np.int32) for c in (1,2)]
            require(all(len(v)==1024 and np.all((v>=-32768)&(v<=32767)) for v in a),'Invalid waveform length or ADC range')
            require(all(pair[c]['Rate']=='1024' for c in (1,2)),'Invalid declared sample count per cycle')
            raw_block.append(np.stack(a).astype(np.int16));tb.append(timestamp(row['RecvTime']));ib.append(int(row['InternalIndex']))
            if len(raw_block)>=512:flush(rh,wh)
        flush(rh,wh)
    # Unexplained isolated rows block preparation rather than silently changing the existing timeline.
    require(not pending,f'Unpaired channels: {len(pending)} ; manual inspection is required')
    t=np.asarray(times,np.int64);ind=np.asarray(indices,np.uint8)
    dt=np.diff(t); step=np.diff(ind.astype(np.int16))%256
    cont=np.cumsum(np.r_[True,(dt<=0)|(dt>100000)|(step!=1)],dtype=np.int32)-1
    arr={'time':t,'index':ind,'basic':np.concatenate(basics),'bmask':np.concatenate(masks),'f':np.concatenate(fs),'fm':np.concatenate(fms),'continuity':cont}
    for k,v in arr.items(): save_array(out/f'{k}.npy',v)
    os.replace(rawfile,out/'adc.bin');os.replace(wavefile,out/'aligned.bin')
    stats.update({'cycles':len(t),'jitter':int(((dt<10000)|(dt>30000)).sum()),'gaps':int((dt>100000).sum()),'index_jump':int((step!=1).sum()),'time_reverse':int((dt<=0).sum())})
    dump(out/'quality.json',stats)
    return arr

def load_record(path):
    return {p.stem:np.load(p,mmap_mode='r') for p in Path(path).glob('*.npy')}

def segments(t,continuity,lab,fold,rec):
    span=(t>=lab['time'][0])&(t<=lab['time'][-1]);phase=np.full(len(t),-1,np.int8)
    if rec in FOLDS[fold]['train']: phase[span]=np.searchsorted(lab['time'][FOLDS[fold]['cut']],t[span],side='right')
    else: phase[span]=3
    change=np.r_[True,(phase[1:]!=phase[:-1])|(continuity[1:]!=continuity[:-1])]
    seg=np.cumsum(change);start=np.maximum.accumulate(np.where(change,np.arange(len(t)),0))
    return phase,seg,start

def moments_update(state,v,m):
    x=np.where(m,v,0).astype(np.float64)
    state[0]+=x.sum(0);state[1]+=(x*x).sum(0);state[2]+=m.sum(0)

def fit_moments(state):
    s,q,n=state;mu=s/np.maximum(n,1);sd=np.sqrt(np.maximum(q/np.maximum(n,1)-mu*mu,0))
    return {'mean':mu,'std':np.maximum(sd,1e-6),'count':n}

def normalize(v,m,n): return np.concatenate((np.where(m,(v-n['mean'])/n['std'],0),m),1).astype(np.float32)

def prepare(data,run,reference):
    data=Path(data);run=Path(run);ref=read(reference); cache=run/'cache'
    ident={'code':code_hash(),'protocol':digest(PROTOCOL),'sources':ref['sources']}
    if (cache/'ready.json').exists():
        require(read(cache/'ready.json')['identity']==digest(ident),'Cache configuration or source code changed; use a new run ID')
        for name in ('Aggregate.zip','SocketRecord.xlsx'):stable_source(data/name,ref['sources'][name])
        for name,expected in read(cache/'ready.json')['files'].items():require(file_hash(cache/name)==expected,f'Corrupt cache: {name}')
        return
    run.mkdir(parents=True,exist_ok=True)
    # Approximately 55 GB for materialized sequences/images/waveforms plus models/predictions; apply a 1.5 safety factor.
    require(shutil.disk_usage(run).free>=100*1024**3,'Less than 100 GiB of free storage; stop without deleting other files')
    for name in ('Aggregate.zip','SocketRecord.xlsx'): stable_source(data/name,ref['sources'][name])
    mapping=mappings(data/'SocketRecord.xlsx');dump(run/'audit/mapping.json',mapping)
    with zipfile.ZipFile(data/'Aggregate.zip') as z:
        labs={r:labels(z,f'Aggregate/Event/Aggregation_sign_{r}.csv',mapping[r]) for r in range(1,8)}
    for r in range(1,8):
        dest=cache/f'record{r}';print(f'Reading the complete aggregate record {r}, including ZIP CRC verification',flush=True)
        if (dest/'record_ready.json').exists():
            ready=read(dest/'record_ready.json')
            require(ready['identity']==digest(ident),'Record cache identity mismatch')
            for name,h in ready['files'].items():require(file_hash(dest/name)==h,f'Corrupt record cache: {r}/{name}')
            continue
        arr=extract(data/'Aggregate.zip',f'Aggregate/Aggregation_{r}.csv',dest)
        pr=project(labs[r],arr['time'])
        actual={k:array_hash(arr[k]) for k in ('time','index','basic','bmask','continuity')}
        actual.update({'microwave_y':array_hash(pr['y'][:,5]),'microwave_known':array_hash(pr['known'][:,5].astype(np.uint8))})
        expected=ref['regression'][str(r)]
        dump(run/f'audit/regression{r}.json',{'expected':expected,'actual':actual,'pass':actual==expected})
        require(actual==expected,f'Record {r} does not match the inherited V2 timeline; stopping')
        for k,v in pr.items():save_array(dest/f'{k}.npy',v)
        n=len(arr['time']);raw=np.memmap(dest/'adc.bin',dtype=np.int16,mode='r',shape=(n,2,1024))
        dv=np.lib.format.open_memmap(dest/'dvi.tmp.npy',mode='w+',dtype=np.float32,shape=(n//5,6,32,32))
        valid=np.zeros(n//5,bool);clips=np.zeros(n//5,np.float32)
        for b in range(n//5):
            if arr['continuity'][b*5]==arr['continuity'][b*5+4]:dv[b],valid[b],clips[b]=raster5(raw[b*5:b*5+5])
            else:dv[b]=0
        dv.flush();del dv;os.replace(dest/'dvi.tmp.npy',dest/'dvi.npy')
        save_array(dest/'dvi_valid.npy',valid);save_array(dest/'clipping.npy',clips)
        dump(dest/'labels.json',{k:v.tolist() for k,v in labs[r].items()})
        dump(dest/'record_ready.json',{'identity':digest(ident),'files':{p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.name!='record_ready.json'}})
    for fold in FOLDS: prepare_fold(cache,fold,labs)
    stable_source(data/'Aggregate.zip',ref['sources']['Aggregate.zip'])
    # Hash file contents, not mtime or NPZ container metadata, for numerical regression.
    files={str(p.relative_to(cache)):file_hash(p) for p in sorted(cache.rglob('*')) if p.is_file()}
    dump(cache/'ready.json',{'identity':digest(ident),'details':ident,'files':files})
    # Historical audit field names are retained for evidence compatibility.
    dump(run/'audit/exposure.json',{'数据':'全部七份 TDHA 聚合记录在此前实验中已暴露','读取':'聚合输入、标签与监督掩码构建','禁止':['UK-DALE','REFIT'],'任务':PROTOCOL['task']})

def prepare_fold(cache,fold,labs):
    root=cache/f'fold{fold}';root.mkdir(parents=True,exist_ok=True)
    accum={tag:[np.zeros(k),np.zeros(k),np.zeros(k)] for tag,k in [('power',54),('F',600)]}
    scale=np.zeros(3); endpoints={}; counts=[]
    for r in range(1,8):
        arr=load_record(cache/f'record{r}');phase,seg,start=segments(arr['time'],arr['continuity'],labs[r],fold,r)
        v,m=changes(arr['basic'],arr['bmask'],seg);pv,pm=pool5(v,m,seg)
        f,fm=frequency_changes(arr['f'],arr['fm'],seg);fv,fm=pool5(f,fm,seg)
        for name,a in [('power_v',pv),('power_m',pm),('F_v',fv),('F_m',fm),('phase',phase),('segment',seg),('start',start)]:save_array(root/f'r{r}_{name}.npy',a)
        token_end=np.arange(len(pv))*5+4;train=(phase[token_end]==0)&(start[token_end]<=token_end-4)
        for tag,x,mask in [('power',pv,pm),('F',fv,fm)]:moments_update(accum[tag],x[train],mask[train])
        ix=(phase==0)&arr['bmask'][:,3].astype(bool)&arr['bmask'][:,4].astype(bool)
        scale[:2]+=(arr['basic'][ix,3:5].astype(np.float64)**2).sum(0);scale[2]+=ix.sum()
        idx=np.arange(len(phase));qualified=(idx-start>=2609)&(idx%5==4)&(arr['bmask'][:,0]!=0)
        for name,part in [('train',0),('train_natural',0),('calibration',1),('time',2),('cross',3)]:
            ep=idx[qualified&(phase==part)]
            if name=='train':ep=ep[ep%50==49]
            endpoints[f'{r}_{name}']=ep.tolist()
            counts.append({'record':r,'part':name,'endpoints':len(ep),'positive':(arr['y'][ep]*arr['known'][ep]).sum(0).tolist()})
    norm={tag:fit_moments(st) for tag,st in accum.items()};norm['wave_scale']=np.sqrt(scale[:2]/scale[2]).tolist()
    dump(root/'normalizer.json',norm);dump(root/'endpoints.json',endpoints);dump(root/'coverage.json',counts)
    for r in range(1,8):
        for tag in ('power','F'):
            v=np.load(root/f'r{r}_{tag}_v.npy',mmap_mode='r');m=np.load(root/f'r{r}_{tag}_m.npy',mmap_mode='r')
            save_array(root/f'r{r}_{tag}.npy',normalize(v,m,norm[tag]))
    pairs=np.array([(r,e) for r in FOLDS[fold]['train'] for e in endpoints[f'{r}_train']],np.int64)
    require(len(pairs)>0,'No eligible training endpoints');save_array(root/'pairs.npy',pairs)
    for seed in (11,42,73):
        rng=np.random.default_rng(seed+1000*(fold=='B'));orders=np.stack([rng.permutation(len(pairs)) for _ in range(40)])
        save_array(root/f'order{seed}.npy',orders)
    # Compare V2 microwave natural ON/OFF under matching endpoint eligibility; the multi-label microwave mask stays independent of other classes.
    micro=sum(row['positive'][5] for row in counts if row['part']=='train')
    require(micro=={'A':2538,'B':2483}[fold],f'{fold} fold eligibility regression failed: microwave ON count {micro}')
