"Read only the first single-appliance unit with matching mode in the training partition, never units used by test data."
import os
import re
import shutil
import zipfile
from pathlib import Path
import numpy as np
from .common import *
from .data import labels,extract,project,load_record,normalize
from .physics import changes,frequency_changes,pool5,raster5

SOURCES={
 (0,''):'Displayer',(1,'1st Speed'):'Fan_1stSpeed',(1,'3rd Speed'):'Fan_3rdSpeed',
 (2,'1st Speed And Cool Wind'):'Hairdryer_1stSpeedCoolWind',
 (2,'1st Speed And Low Temperature Wind'):'Hairdryer_1stSpeedLowTemperatureWind',
 (3,'Quarter Power'):'LargeHumidifierQuarterPower',(3,'Half Power'):'LargeHumidifierHalfPower',
 (3,'Max Power'):'LargeHumidifierMaxPower',(3,'Three Quarters Power'):'LargeHumidifierThreeQuartersPower',
 (4,'Half Power'):'SmallHumidifierHalfPower',(4,'Max Power'):'SmallHumidifierMaxPower',
 (5,'Low Heat'):'MicrowaveLowHeat',(5,'Medium Heat'):'MicrowaveMediumHeat',
 (5,'Defrosting'):'MicrowaveDefrosting',(5,'High Heat'):'MicrowaveHighHeat',
 (6,''):'TableLamp',(7,''):'WaterHeater',(8,''):'Router',(9,''):'MobilePhone',(10,''):'EggBoiler'}

def prepare_single(data,run,fold,rep,reference):
    data=Path(data);run=Path(run);root=run/'pretrain'/f'fold{fold}';root.mkdir(parents=True,exist_ok=True)
    if (root/'ready.json').exists():
        ready=read(root/'ready.json');require(ready['code']==code_hash() and ready['rep']==rep,'Single-appliance cache configuration changed')
        return
    mapping=read(run/'audit/mapping.json');norm=read(run/'cache'/f'fold{fold}'/'normalizer.json');ref=read(reference)
    allowed=sorted({(int(c),v['mode']) for r in FOLDS[fold]['train'] for c,v in mapping[str(r)].items()})
    names=[SOURCES[key]+'.zip' for key in allowed]
    estimate=0
    for name in names:
        stable_source(data/name,ref['sources'][name])
        with zipfile.ZipFile(data/name) as z:estimate+=sum(x.file_size for x in z.infolist())*2
    require(shutil.disk_usage(root).free>=estimate*1.5,'Insufficient single-appliance cache space (twice the extracted size plus a 1.5x margin)')
    # Audit the small label files before scanning waveforms; write anomalies and propagation intervals separately.
    label_audit=[]
    for c,mode in allowed:
        archive=data/(SOURCES[c,mode]+'.zip')
        with zipfile.ZipFile(archive) as z:
            for number in range(1,5):
                events=[n for n in z.namelist() if n.lower().endswith(f'_{number}.csv') and 'sign' in n.lower()]
                require(len(events)==1,'Ambiguous single-appliance label filename')
                lab=labels(z,events[0],{},single_class=c)
                label_audit.extend(dict(item,archive=archive.name) for item in lab.get('label_anomalies',[]))
    # Preserve the historical label-policy text in audit metadata for compatibility.
    dump(run/'audit/pretrain_labels'/f'fold{fold}.json',{'fold':fold,'anomalies':label_audit,
         'policy':'仅已审查的1)0设未知，直到下一合法标签；两端±150ms/±500ms；其余格式停止'})
    pairs=[];cal=[];manifest=[]
    for i,(c,mode) in enumerate(allowed):
        archive=data/(SOURCES[c,mode]+'.zip')
        with zipfile.ZipFile(archive) as z:
            for number in range(1,5):
                waveform=[n for n in z.namelist() if n.lower().endswith(f'_{number}.csv') and 'sign' not in n.lower()]
                events=[n for n in z.namelist() if n.lower().endswith(f'_{number}.csv') and 'sign' in n.lower()]
                require(len(waveform)==len(events)==1,'Ambiguous single-appliance waveform or label filename')
                rid=100+i*4+number;dest=root/f'record{rid}'
                print(f'Preparing single-appliance fold {fold}: {archive.name}, recording {number} -> record{rid}',flush=True)
                lab=labels(z,events[0],{},single_class=c);a=extract(archive,waveform[0],dest);pr=project(lab,a['time'])
                for k,v in pr.items():save_array(dest/f'{k}.npy',v)
                seg=a['continuity'];v,m=changes(a['basic'],a['bmask'],seg);pv,pm=pool5(v,m,seg)
                f,fm=frequency_changes(a['f'],a['fm'],seg);fv,fm=pool5(f,fm,seg)
                for tag,v,m in [('power',pv,pm),('F',fv,fm)]:
                    nn={k:np.asarray(x) for k,x in norm[tag].items()}
                    save_array(root/f'r{rid}_{tag}.npy',normalize(v,m,nn))
                n=len(a['time']);raw=np.memmap(dest/'adc.bin',mode='r',dtype=np.int16,shape=(n,2,1024))
                if rep=='DVI':
                    dv=np.lib.format.open_memmap(dest/'dvi.npy',mode='w+',dtype=np.float32,shape=(n//5,6,32,32));valid=np.zeros(n//5,bool)
                    for b in range(n//5):
                        if seg[b*5]==seg[b*5+4]:dv[b],valid[b],_=raster5(raw[b*5:b*5+5])
                        else:dv[b]=0
                    dv.flush();del dv;save_array(dest/'dvi_valid.npy',valid)
                idx=np.arange(n);start=np.maximum.accumulate(np.where(np.r_[True,np.diff(seg)!=0],idx,0))
                # Representation pretraining uses only the current causal cycle token; keep the shared 50-cycle feature boundary and leave the temporal backbone untrained.
                ep=idx[(idx%50==49)&(idx-start>=54)&pr['known'][:,c]]
                (pairs if number<=3 else cal).extend((rid,int(e)) for e in ep)
                manifest.append({'record':rid,'class':c,'mode':mode,'source':archive.name,'member':waveform[0],'role':'train' if number<=3 else 'early_stop'})
    pairs=np.array(pairs,np.int64);cal=np.array(cal,np.int64);rng=np.random.default_rng(42+1000*(fold=='B'))
    save_array(root/'pairs.npy',pairs);save_array(root/'calibration.npy',cal);save_array(root/'order42.npy',np.stack([rng.permutation(len(pairs)) for _ in range(20)]))
    dump(root/'ready.json',{'code':code_hash(),'fold':fold,'rep':rep,'normalizer':digest(norm),'source_manifest':manifest,'sampling':array_hash(pairs),
         'label_policy':'unknown-invalid-v1','anomalies':label_audit})
