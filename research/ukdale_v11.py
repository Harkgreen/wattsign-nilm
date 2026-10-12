"""Train and evaluate the UK-DALE TCN with native 1-second inputs.

run() accepts explicit mains, appliance-channel and time-interval inputs.
It returns an execution plan unless execute=True. The selected workflow uses
five seeds, causal 640-second histories and 10-second output endpoints."""
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

from wattsign._ukdale_1s import M0TCN
def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256((str(x.dtype) + str(x.shape)).encode() + x.view(np.uint8).tobytes()).hexdigest()

def state_hash(s):
    h = hashlib.sha256()
    for k in sorted(s):
        h.update(k.encode())
        h.update(s[k].detach().cpu().contiguous().numpy().view(np.uint8).tobytes())
    return h.hexdigest()

def reset_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def capture_rng():
    return {'cpu': torch.get_rng_state().clone(), 'cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]}

def restore_rng(r):
    torch.set_rng_state(r['cpu'])
    torch.cuda.set_rng_state_all(r['cuda'])

def same_rng(a, b):
    return torch.equal(a['cpu'], b['cpu']) and all((torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'], strict=True)))
def provider_h1(endpoints, shape):
    return full_window(BASE18, EXTRA_VALUES, EXTRA_MASKS, endpoints, shape)
def stream_slice(source, left, right, shape):
    base, values, masks, _ = source
    out = np.zeros((1, 42, right - left), np.float32)
    out[0, :18] = base[left:right].T
    if shape:
        out[0, 18:30] = values[left:right].T
        out[0, 30:42] = masks[left:right].T
    return torch.from_numpy(np.ascontiguousarray(out)).to(DEVICE)

def infer(model, provider, endpoints, shape, source=None, bs=512):
    was = model.training
    rng = capture_rng()
    model.eval()
    endpoints = np.asarray(endpoints, np.int64)
    out = np.empty(len(endpoints), np.float32)
    try:
        with torch.inference_mode():
            if source is None:
                for i in range(0, len(endpoints), bs):
                    out[i:i + bs] = model(torch.from_numpy(provider(endpoints[i:i + bs], shape)).to(DEVICE)).float().cpu().numpy()
            else:
                _, _, _, continuity = source
                seconds = endpoints * 10 + 9
                cuts = np.r_[0, np.flatnonzero(continuity[seconds[1:]] != continuity[seconds[:-1]]) + 1, len(endpoints)]
                for lo0, hi0 in zip(cuts[:-1], cuts[1:], strict=True):
                    for lo in range(lo0, hi0, bs):
                        hi = min(hi0, lo + bs)
                        sec = seconds[lo:hi]
                        left = int(sec[0] - 639)
                        right = int(sec[-1] + 1)
                        x = stream_slice(source, left, right, shape)
                        if isinstance(model, M0TCN):
                            logits = model.forward_sequence(x)[0, torch.as_tensor(sec - left, device=DEVICE)]
                        else:
                            raise ValueError('Selected workflow supports M0_TCN only')
                        out[lo:hi] = logits.float().cpu().numpy()
        return out
    finally:
        model.train(was)
        assert same_rng(rng, capture_rng())
def train_neural(arch, shape, seed, repetition, phase):
    require_execution()
    reset_all(seed)
    model = MODELS[arch]().to(DEVICE)
    initial_hash = state_hash(model.state_dict())
    key = (arch, seed)
    assert key not in INITIAL_STATE_HASHES or INITIAL_STATE_HASHES[key] == initial_hash, f'Initial state differs: {key}'
    INITIAL_STATE_HASHES[key] = initial_hash
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = np.inf
    state = None
    wait = 5
    hist = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if DEVICE.type == 'cuda' else None
    for epoch in range(40):
        model.train()
        losses = []
        order = BATCH_ORDERS[seed][epoch]
        for st in range(0, len(order), 64):
            ep = SAMPLES[seed][order[st:st + 64]]
            x = torch.from_numpy(provider_h1(ep, shape)).to(DEVICE)
            yy = torch.from_numpy(h1['target'][ep].astype(np.float32)).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x), yy)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        z = infer(model, provider_h1, PARTITIONS['calibration'], shape, H1_SOURCE)
        val = bce_balanced(h1['target'][PARTITIONS['calibration']], z)
        hist.append({'epoch': epoch + 1, 'training_loss': np.mean(losses), 'calibration_balanced_bce': val})
        if val < best - 1e-08:
            best = val
            state = copy.deepcopy(model.state_dict())
            be = epoch + 1
            wait = 5
        else:
            wait -= 1
            if not wait:
                break
    model.load_state_dict(state)
    meta = {'architecture': arch, 'representation': 'F1' if shape else 'F0', 'seed': seed, 'repetition': repetition, 'initial_state_hash': initial_hash, 'best_epoch': be, 'training_seconds': time.perf_counter() - start, 'peak_memory_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == 'cuda' else 0.0, 'state_hash': state_hash(state), 'parameter_count': sum((p.numel() for p in model.parameters()))}
    torch.save({'state': state, 'metadata': meta}, OUT / 'checkpoints' / f'{phase}_cold{repetition}_{arch}_F{int(shape)}_seed{seed}.pt')
    return (model, meta, hist)


def prepare_training(spec):
    global h1,BASE18,EXTRA_VALUES,EXTRA_MASKS,NORMALIZER,SAMPLES,BATCH_ORDERS,PARTITIONS,H1_SOURCE
    h1,raw_q=read_timeline_1s(spec)
    count=len(h1['target']);train_end=int(.60*count);cal_end=int(.80*count)
    splits={'train':(0,train_end),'calibration':(train_end,cal_end),'h1_final':(cal_end,count)}
    PARTITIONS={name:endpoint_indices(h1,*bounds) for name,bounds in splits.items()}
    for name,e in PARTITIONS.items():ensure_two_classes(h1['target'],e,name)
    unnormalized,_,_=make_native_features(h1,raw_q,1.0)
    NORMALIZER=feature_normalizer(unnormalized,train_end*10)
    BASE18=apply_normalizer(unnormalized,NORMALIZER)
    # F0 has no shape channels. Broadcast immutable zero views avoid unnecessary full-length caches.
    EXTRA_VALUES=np.broadcast_to(np.zeros((1,12),np.float32),(len(BASE18),12))
    EXTRA_MASKS=np.broadcast_to(np.zeros((1,12),np.uint8),(len(BASE18),12))
    H1_SOURCE=(BASE18,EXTRA_VALUES,EXTRA_MASKS,h1['continuity1'])
    SAMPLES={s:balanced_samples(PARTITIONS['train'],h1['target'],s) for s in SEEDS}
    BATCH_ORDERS={s:[np.random.default_rng(s*100000+epoch).permutation(len(SAMPLES[s])).astype(np.int64)
                     for epoch in range(40)] for s in SEEDS}
    return splits,raw_q


def run(spec, output, *, evaluation_specs=(), protected_roots=(), device='cuda', execute=False, trainer=None):
    global DEVICE,OUT,MODELS,INITIAL_STATE_HASHES,_EXECUTION_APPROVED
    spec,evaluation,out=validate_run(spec,evaluation_specs,output,protected_roots)
    plan={'recipe':'V11 stage2_cold2 M0_TCN-F0','seeds':list(SEEDS),'input_seconds':1,
          'history_seconds':640,'output_seconds':10,'training_interval':[spec['start'],spec['end']],
          'evaluations':[s.get('name','evaluation') for s in evaluation],'output':str(out),
          'training':bool(execute),'device':device,'new_results_do_not_replace_saved_results':True}
    if not execute:return plan
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Select an available explicit cpu or cuda training device')
    DEVICE=torch.device(device);OUT=out;MODELS={'M0_TCN':M0TCN};INITIAL_STATE_HASHES={}
    if device=='cuda':
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    splits,raw_q=prepare_training(spec)
    create_output(out);save_preparation(out,NORMALIZER,SAMPLES,BATCH_ORDERS)
    write_json(out/'protocol'/'run.json',{**plan,'split_bounds':splits,'normalizer_fit_end_seconds':splits['train'][1]*10,
        'preparation':'Original native reader, label rules, float32 normalization and deterministic sampling/batch order rebuilt from raw files'})
    models=[];histories=[];training=[]
    train_selected = train_neural if trainer is None else bind_visible_trainer(trainer,globals())
    _EXECUTION_APPROVED=True
    try:
        for seed in SEEDS:
            m,metadata,history=train_selected('M0_TCN',False,seed,2,'stage2')
            models.append(m.eval());training.append(metadata);histories += [dict(r,seed=seed) for r in history]
    finally:_EXECUTION_APPROVED=False
    pd.DataFrame(histories).to_csv(out/'tables'/'training_history.csv',index=False)
    write_json(out/'tables'/'training_metadata.json',training)
    thresholds={};cal=PARTITIONS['calibration']
    for seed,m in zip(SEEDS,models,strict=True):
        z=infer(m,provider_h1,cal,False,H1_SOURCE)
        thresholds[seed]=threshold_1s(h1['target'][cal],z)
    write_json(out/'protocol'/'frozen_thresholds.json',{'space':'logit','thresholds':thresholds,'target_recall':.95,'calibration_only':True})
    rows=[]
    def evaluate(name,timeline,q,features,endpoints):
        e=endpoints[timeline['stable'][endpoints].astype(bool)]
        ensure_two_classes(timeline['target'],e,name)
        zeros=np.broadcast_to(np.zeros((1,12),np.float32),(len(features),12))
        masks=np.broadcast_to(np.zeros((1,12),np.uint8),(len(features),12))
        source=(features,zeros,masks,timeline['continuity1'])
        provider=lambda ep,shape:full_window(features,zeros,masks,ep,shape)
        for seed,m in zip(SEEDS,models,strict=True):
            score=infer(m,provider,e,False,source)
            metrics=metric_table(timeline['target'][e],timeline['p10'][e],q['q10'][1.0][e],timeline['event'][e],score,thresholds[seed])
            metrics.pop('_cond',None);rows.append({'partition':name,'seed':seed,**metrics})
            np.savez_compressed(out/'predictions'/f'{name}_seed{seed}.npz',endpoints=e,target=timeline['target'][e],power=timeline['p10'][e],score=score,threshold=thresholds[seed])
    evaluate('h1_final',h1,raw_q,BASE18,PARTITIONS['h1_final'])
    for ev in evaluation:
        timeline,q=read_timeline_1s(ev)
        features,_,_=make_native_features(timeline,q,1.0,normalizer=NORMALIZER)
        e=endpoint_indices(timeline,0,len(timeline['target']))
        evaluate(ev.get('name','evaluation'),timeline,q,features,e)
    write_json(out/'tables'/'metrics.json',rows)
    return {'output':str(out),'seeds':list(SEEDS),'thresholds':thresholds,'metrics':rows}


if __name__ == '__main__':
    args,spec,evaluations=parse_cli(__doc__)
    result=run(spec,args.output,evaluation_specs=evaluations,protected_roots=args.protected_root,device=args.device,execute=args.execute)
    print(json.dumps(result,ensure_ascii=False,indent=2))
