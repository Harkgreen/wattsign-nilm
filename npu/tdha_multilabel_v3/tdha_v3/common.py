from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

# Historical dataset labels and protocol text remain unchanged for artifact compatibility.
CLASSES = ['显示器','风扇','吹风机','大加湿器','小加湿器','微波炉','台灯','热水器','路由器','手机','煮蛋器']
NAMES = ['Displayer','Fan','Hairdryer','Large humidifier','Small humidifier','Microwave','Table Lamp','Water Heater','Router','Mobilephone','Egg Boiler']
FOLDS = {'A': {'train':[2,4,6], 'cross':[3,5,7], 'cut':[255,340]},
         'B': {'train':[3,5,7], 'cross':[2,4,6], 'cut':[236,315]}}
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {'version':3, 'classes':CLASSES,'primary':list(range(6)), 'folds':FOLDS,
 'history_cycles':1280,'eligibility':2609,'token_cycles':5,'seeds':[11,42,73],
 'train_stride':50,'eval_stride':5,'epochs':40,'patience':5,'lr':0.0003,
 'weight_decay':0.0001,'batch':64,'recall_target':0.95,'bootstrap':2000,
 'bootstrap_seed':42,'unknown_us':150000,'sensitivity_us':500000,
 'representation':['F','W1','DVI'],'networks':['N0','N1','N2'],
 'task':'插座供电状态；TDHA 已暴露受控聚合探索性证据',
 'normalization':'训练段固定统计；mask 不标准化',
 'not_deployment':True}

class Blocked(RuntimeError):
    "Fail closed when data, device or protocol constraints are not satisfied."

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,Path): return str(x)
    return x

def canonical(x):
    return json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def digest(x): return hashlib.sha256(canonical(x)).hexdigest()

def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def code_hash():
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'tdha_v3').glob('*.py'))})

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(json.dumps(clean(obj),ensure_ascii=False,indent=2,sort_keys=True).encode('utf-8'))
    os.replace(tmp,path)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def save_array(path,a):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.save(f,a,allow_pickle=False)
    os.replace(tmp,path)

def require(condition,message):
    if not condition: raise Blocked(message)

def stable_source(path, expected):
    path=Path(path); require(path.is_file(),f'Missing source file: {path.name}')
    before=path.stat(); actual=file_hash(path); after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source file is still changing; wait until the copy is complete')
    require(actual==expected['sha256'] and after.st_size==expected['size'],f'Source checksum verification failed: {path.name}')
    return actual
