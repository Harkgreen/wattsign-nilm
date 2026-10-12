"""Train and evaluate the UK-DALE 10-second TCN with P/Q features.

run() accepts explicit mains, appliance-channel and time-interval inputs.
It returns an execution plan unless execute=True. Training writes five-seed
checkpoints, calibration thresholds and predictions to a fresh output folder."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
import copy, hashlib, json, random, time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from wattsign.low_frequency_workflow import *
SEEDS = (11,29,42,73,101)
_EXECUTION_APPROVED = False

def require_execution():
    if not _EXECUTION_APPROVED:raise RuntimeError('Use run(..., execute=True) after supplying explicit inputs and a new output')

from wattsign._ukdale_10s import RichCausalTCN

def model():return RichCausalTCN()
def sj(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def sh(x):
    return hashlib.sha256(x).hexdigest()

def ah(x):
    x = np.ascontiguousarray(x)
    return sh(f'{x.dtype}|{x.shape}|'.encode() + x.view(np.uint8).tobytes())

def sth(d):
    h = hashlib.sha256()
    for k in sorted(d):
        x = d[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().view(np.uint8).tobytes())
    return h.hexdigest()

def reset(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
def mode(z, m):
    z = z.copy()
    if m == 'A':
        z[:, 6:] = 0
    elif m == 'PS':
        z[:, 12:] = 0
    elif m == 'PQ':
        z[:, 6:12] = 0
    elif m == 'M':
        z[:, 6:9] = 0
        z[:, 12:15] = 0
    elif m in ('B_snapshot', 'B\u5feb\u7167'):
        z[:, :, :63] = 0
    elif m == 'B':
        pass
    else:
        raise ValueError(m)
    return z

class DS(Dataset):

    def __init__(self, e, m, feat=None, t=None):
        self.e = np.asarray(e)
        self.m = m
        self.feat = x if feat is None else feat
        self.t = h1 if t is None else t

    def __len__(self):
        return len(self.e)

    def __getitem__(self, i):
        e = self.e[i]
        q = mode(np.ascontiguousarray(self.feat[e - T + 1:e + 1].T[None]), self.m)[0]
        return (torch.from_numpy(q), torch.tensor(np.float32(self.t['target'][e])))

def edges(c, l, r):
    return np.r_[l, np.flatnonzero(c[l + 1:r] != c[l:r - 1]) + l + 1, r]

@torch.no_grad()
def seqinfer(m, feat, cont, e, l, r, kind):
    out = np.empty(len(e), np.float32)
    e = np.asarray(e)
    for a, b in zip(edges(cont, l, r)[:-1], edges(cont, l, r)[1:], strict=True):
        lo, hi = (np.searchsorted(e, a), np.searchsorted(e, b))
        if lo == hi:
            continue
        local = e[lo:hi]
        for cl in range(int(a), int(b), 65536):
            cr = min(int(b), cl + 65536)
            ilo, ihi = (np.searchsorted(local, cl), np.searchsorted(local, cr))
            if ilo == ihi:
                continue
            h = max(int(a), cl - m.rf + 1)
            q = mode(np.ascontiguousarray(feat[h:cr].T[None]), kind)
            v = m.seq(torch.from_numpy(q).to(DEVICE))[0]
            ii = torch.as_tensor(local[ilo:ihi] - h, device=DEVICE)
            out[lo + ilo:lo + ihi] = v[ii].cpu().numpy()
    return torch.sigmoid(torch.from_numpy(out)).numpy().astype(np.float32)

@torch.no_grad()
def wininfer(m, feat, e, kind, donor=None):
    ans = []
    e = np.asarray(e)
    for i in range(0, len(e), IBATCH):
        ee = e[i:i + IBATCH]
        ix = ee[:, None] - np.arange(T - 1, -1, -1)
        q = np.ascontiguousarray(feat[ix].transpose(0, 2, 1))
        if donor is not None:
            dd = np.asarray(donor[i:i + len(ee)])
            dx = dd[:, None] - np.arange(T - 1, -1, -1)
            q[:, 6:, :] = feat[dx][:, :, 6:].transpose(0, 2, 1)
        ans.append(torch.sigmoid(m(torch.from_numpy(mode(q, kind)).to(DEVICE))).cpu().numpy())
    return np.concatenate(ans).astype(np.float32)
def ckpath(n, s):
    return CP / f'{n}_seed{s}.pt'

def train(n, s):
    require_execution()
    reset(s)
    m = model().to(DEVICE)
    m.load_state_dict(ini[s])
    assert sth(m.state_dict()) == rep['initial_state_hashes'][str(s)]
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    ds = DS(samples[s], n, feat=x, t=h1)
    rng = np.random.default_rng(s + 10000)
    best = np.inf
    bst = None
    be = 0
    stale = 0
    hist = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if DEVICE.type == 'cuda' else None
    for ep in range(1, MAXE + 1):
        m.train()
        order = rng.permutation(len(ds))
        dl = DataLoader(Subset(ds, order.tolist()), batch_size=BATCH, shuffle=False, num_workers=0)
        ls = []
        for xb, yb in dl:
            xb, yb = (xb.to(DEVICE), yb.to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(m(xb), yb)
            loss.backward()
            opt.step()
            ls.append(float(loss.detach().cpu()))
        pr = wininfer(m, x, cal, n) if n in ('B_snapshot', 'B\u5feb\u7167') else seqinfer(m, x, h1['continuity'], cal, *CALIBRATION_BOUNDS, n)
        val = bce(caly, pr)
        hist.append({'model': n, 'seed': s, 'epoch': ep, 'training_bce': np.mean(ls), 'calibration_balanced_bce': val})
        if val < best - 1e-09:
            best = val
            bst = copy.deepcopy(m.state_dict())
            be = ep
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    m.load_state_dict(bst)
    meta = {'model': n, 'seed': s, 'best_epoch': be, 'best_balanced_bce': best, 'training_seconds': time.perf_counter() - start, 'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 ** 2 if DEVICE.type == 'cuda' else 0.0, 'initial_state_hash': sth(ini[s]), 'state_hash': sth(bst), 'sampling_hash': ah(samples[s]), 'normalizer_hash': sh(sj(normal).encode())}
    torch.save({'state_dict': bst, 'metadata': meta}, ckpath(n, s))
    return (meta, pd.DataFrame(hist))


def prepare_training(spec):
    # Original 60/20/20 chronological split, with a complete 640 s history per partition.
    global h1,x,normal,samples,ini,rep,cal,caly,CALIBRATION_BOUNDS
    h1 = read_timeline_10s(spec)
    count = len(h1['target']); training_end=int(.60*count); calibration_end=int(.80*count)
    splits = {'train':(0,training_end),'calibration':(training_end,calibration_end),'h1_final':(calibration_end,count)}
    partitions = {name:endpoints_10s(h1,*bounds) for name,bounds in splits.items()}
    for name,e in partitions.items():ensure_two_classes(h1['target'],e,name)
    normal = fit_masked_normalizer(h1['features_unscaled'],training_end)
    x = norm_apply(h1['features_unscaled'],normal)
    samples = {s:balanced_samples(partitions['train'],h1['target'],s) for s in SEEDS}
    ini={}
    for s in SEEDS:
        reset(s);ini[s]=copy.deepcopy(model().state_dict())
    rep={'initial_state_hashes':{str(s):sth(ini[s]) for s in SEEDS}}
    cal=partitions['calibration'];caly=h1['target'][cal].astype(np.uint8)
    CALIBRATION_BOUNDS=splits['calibration']
    return partitions,splits


def run(spec, output, *, evaluation_specs=(), protected_roots=(), device='cuda', execute=False, trainer=None):
    global DEVICE,OUT,CP,T,BATCH,IBATCH,MAXE,PATIENCE,LR,WD,_EXECUTION_APPROVED
    spec,evaluation,out = validate_run(spec,evaluation_specs,output,protected_roots)
    plan={'recipe':'V6 TCN/PQ','seeds':list(SEEDS),'input_seconds':10,'history_seconds':640,
          'output_seconds':10,'training_interval':[spec['start'],spec['end']],
          'evaluations':[s.get('name','evaluation') for s in evaluation],'output':str(out),
          'training':bool(execute),'device':device,'new_results_do_not_replace_saved_results':True}
    if not execute:return plan
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Select an available explicit cpu or cuda training device')
    DEVICE=torch.device(device);OUT=out;CP=out/'checkpoints'
    T=64;BATCH=64;IBATCH=2048;MAXE=40;PATIENCE=5;LR=1e-3;WD=1e-4
    if device=='cuda':
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    partitions,splits=prepare_training(spec)
    create_output(out);save_preparation(out,normal,samples)
    write_json(out/'protocol'/'run.json',{**plan,'split_bounds':splits,'normalizer_fit_end':normal['fit_end'],
        'preparation':'Original raw fallback readers, labels and causal features; new training context built without old caches',
        'initial_state_hashes':rep['initial_state_hashes'],'sampling_hashes':{str(s):ah(samples[s]) for s in SEEDS}})
    histories=[];training=[]
    train_selected = train if trainer is None else bind_visible_trainer(trainer,globals())
    _EXECUTION_APPROVED=True
    try:
        for seed in SEEDS:
            metadata,history=train_selected('PQ',seed);training.append(metadata);histories.append(history)
    finally:_EXECUTION_APPROVED=False
    pd.concat(histories,ignore_index=True).to_csv(out/'tables'/'training_history.csv',index=False)
    write_json(out/'tables'/'training_metadata.json',training)
    models=[];calibration=[]
    for seed in SEEDS:
        payload=torch.load(ckpath('PQ',seed),map_location='cpu',weights_only=True)
        m=model();m.load_state_dict(payload['state_dict'],strict=True);m.to(DEVICE).eval();models.append(m)
        calibration.append(seqinfer(m,x,h1['continuity'],cal,*splits['calibration'],'PQ'))
    calibration=np.stack(calibration)
    frozen_threshold=threshold_10s(caly,calibration)
    write_json(out/'protocol'/'frozen_threshold.json',{'space':'probability','threshold':frozen_threshold,'target_recall':.95,'calibration_only':True})
    rows=[]
    test_specs=[('h1_final',h1,x,partitions['h1_final'],splits['h1_final'])]
    for ev in evaluation:
        timeline=read_timeline_10s(ev);features=norm_apply(timeline['features_unscaled'],normal)
        e=endpoints_10s(timeline,0,len(timeline['target']))
        test_specs.append((ev.get('name','evaluation'),timeline,features,e,(0,len(timeline['target']))))
    for name,timeline,features,e,bounds in test_specs:
        stable=timeline['stable_valid'][e].astype(bool);kept=e[stable]
        ensure_two_classes(timeline['target'],kept,name)
        y=timeline['target'][kept];power=timeline['p'][kept]
        for seed,m in zip(SEEDS,models,strict=True):
            score=seqinfer(m,features,timeline['continuity'],e,*bounds,'PQ')[stable]
            row={'partition':name,'seed':seed,**metrics_10s(y,score,frozen_threshold),**off_fpr_10s(y,score,power,frozen_threshold)}
            rows.append(row)
            np.savez_compressed(out/'predictions'/f'{name}_seed{seed}.npz',endpoints=kept,target=y,power=power,score=score,threshold=frozen_threshold)
    write_json(out/'tables'/'metrics.json',rows)
    return {'output':str(out),'seeds':list(SEEDS),'threshold':frozen_threshold,'metrics':rows}


if __name__ == '__main__':
    args,spec,evaluations=parse_cli(__doc__)
    result=run(spec,args.output,evaluation_specs=evaluations,protected_roots=args.protected_root,device=args.device,execute=args.execute)
    print(json.dumps(result,ensure_ascii=False,indent=2))
