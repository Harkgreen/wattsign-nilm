"Bounded read-only memory mapping and one-batch CPU prefetch; preserve order, random streams and model computation."
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import os
import threading
import time
import numpy as np
import psutil
import torch

GIB = 1024**3


def settings():
    def number(name, default):
        value = float(os.environ.get(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
        return value
    prefetch = os.environ.get('TDHA_PREFETCH', '1')
    if prefetch not in ('0', '1'):raise ValueError('TDHA_PREFETCH must be 0 or 1')
    return dict(ram_bytes=int(number('TDHA_RAM_CACHE_GB', 64)*GIB),
                reserve_bytes=int(number('TDHA_RAM_RESERVE_GB', 256)*GIB),
                eval_bytes=int(number('TDHA_EVAL_CACHE_GB', 4)*GIB),
                hbm_reserve_bytes=int(number('TDHA_HBM_RESERVE_GB', 16)*GIB),
                prefetch=prefetch == '1')


class ArrayCache:
    "One shared mapping per process; multiple Windows objects never duplicate materialized windows.\n\n    Warm the mapping once to avoid external page-cache races in timing or capacity estimates. Unprepared files still use mmap.\n    "
    def __init__(self, budget, reserve, available=None):
        self.budget=budget;self.reserve=reserve
        self.available=available or (lambda:psutil.virtual_memory().available)
        self.items={};self.bytes=0;self.hits=0;self.skips=0;self.seconds=0.
        self.lock=threading.Lock()

    def get(self, path, dtype=None, shape=None):
        path=Path(path).resolve();stat=path.stat()
        key=(str(path),stat.st_size,stat.st_mtime_ns,str(dtype),shape)
        with self.lock:
            if key in self.items:
                self.hits+=1
                return self.items[key]
            mapped=(np.load(path,mmap_mode='r',allow_pickle=False) if dtype is None
                    else np.memmap(path,mode='r',dtype=dtype,shape=shape))
            # Estimate the total address space in advance; avoid competing for resources when workers start concurrently.
            headroom=max(0,self.available()-self.reserve)//3
            if mapped.nbytes>min(self.budget-self.bytes,headroom) or self.budget==0:
                self.skips+=1
                return mapped
            start=time.monotonic();array=np.array(mapped,copy=True);array.flags.writeable=False
            self.items[key]=array;self.bytes+=array.nbytes;self.seconds+=time.monotonic()-start
            return array

    def stats(self):
        return dict(budget_bytes=self.budget,resident_bytes=self.bytes,files=len(self.items),
                    hits=self.hits,mmap_fallbacks=self.skips,copy_seconds=self.seconds)


_CACHE=None
_CACHE_LOCK=threading.Lock()


def cache():
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            cfg=settings();_CACHE=ArrayCache(cfg['ram_bytes'],cfg['reserve_bytes'])
    return _CACHE


def host_batch(ds, indices):
    rows=[ds[int(i)] for i in indices]
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]}


def to_device(batch,device):
    # Keep all NPU calls on the main thread; avoid unverified asynchronous or pinned-memory interfaces.
    return {k:torch.from_numpy(v).to(device) for k,v in batch.items()}


class BatchReader:
    "Prefetch at most one CPU batch, preserve delivery order and propagate the original exception."
    def __init__(self, enabled=None):
        self.enabled=settings()['prefetch'] if enabled is None else enabled
        self.pool=None

    def __enter__(self):
        if self.enabled:self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tdha-prefetch')
        return self

    def __exit__(self,*args):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)

    def batches(self,ds,groups,device):
        groups=iter(groups)
        if self.pool is None:
            for indices in groups:yield to_device(host_batch(ds,indices),device)
            return
        first=next(groups,None)
        if first is None:return
        future=self.pool.submit(host_batch,ds,first)
        try:
            while True:
                batch=future.result();indices=next(groups,None)
                future=self.pool.submit(host_batch,ds,indices) if indices is not None else None
                yield to_device(batch,device)
                if future is None:break
        finally:
            # Wait for the current read to finish even on early exit, so it cannot leak into a later phase.
            if future is not None:future.result()


def device_encoding(encoded,device):
    cfg=settings()
    if device.type!='npu' or encoded.nbytes>cfg['eval_bytes'] or cfg['eval_bytes']==0:return None
    if not hasattr(torch.npu,'mem_get_info'):
        raise RuntimeError('NPU free-memory query is unavailable; set TDHA_EVAL_CACHE_GB=0 to disable the evaluation device cache explicitly')
    free,_=torch.npu.mem_get_info(device)
    if encoded.nbytes>max(0,free-cfg['hbm_reserve_bytes']):return None
    return torch.from_numpy(encoded).to(device)


def encoding_windows(encoded,ends,device,resident=None):
    if resident is None:
        return torch.from_numpy(np.stack([encoded[e-255:e+1] for e in ends])).to(device)
    indices=np.asarray(ends)[:,None]+np.arange(-255,1)[None,:]
    return resident[torch.from_numpy(indices).to(device)]
