"""Train the phase-aligned waveform GRU on TDHA aggregate recordings.

The workflow pairs ADC cycles, constructs labels and causal features, and
trains two folds with three seeds. Checkpoints, calibration thresholds and
new evaluation outputs are written to a fresh directory."""
from __future__ import annotations
import argparse, csv, gc, hashlib, io, json, math, os, random, re, sys, time
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO/"src") not in sys.path:
    sys.path.insert(0,str(REPO/"src"))
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix)
from scipy.stats import spearmanr
from wattsign._tdha_waveform import AggregateModel

SEEDS=(11,42,73)
PAIR_FOLDS={"A":{"train":(2,4,6),"test":(3,5,7),"cut":(255,340)},
            "B":{"train":(3,5,7),"test":(2,4,6),"cut":(236,315)}}
MICROWAVE_SOCKET={1:None,2:4,3:None,4:3,5:4,6:None,7:4}
EPOCH=datetime(1970,1,1)
M,CHANNELS=242,520
DEVICE=None
ARCHIVE=None
BASE_OUT=None
OUT=None
LABELS={}
TIMELINES={}
SOURCE_AUDIT=[]
FOLD_FEATURES={}
FOLD_NORMS={}
ENDPOINTS={}
SAMPLE_MANIFEST={}
HF_RAW={}
RAW_MMAP={}
POWER5={}
HIGH_POWER={}


def us_of(s):
    d = datetime.fromisoformat(s.replace('/', '-')) - EPOCH
    return (d.days * 86400 + d.seconds) * 1000000 + d.microseconds

def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    return sha_bytes(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())

def dump_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rng_digest():
    h = hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    for q in torch.cuda.get_rng_state_all():
        h.update(q.cpu().numpy().tobytes())
    return h.hexdigest()

def socket_mapping(path):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as z:
        root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        shared = [''.join((t.text or '' for t in e.findall('.//m:t', ns))) for e in root.findall('m:si', ns)]
        result = {}
        for rec in range(1, 8):
            sheet = ET.fromstring(z.read(f'xl/worksheets/sheet{rec}.xml'))
            rows = []
            for row in sheet.findall('.//m:sheetData/m:row', ns):
                dd = {}
                for cell in row.findall('m:c', ns):
                    v = cell.find('m:v', ns)
                    if v is None:
                        continue
                    col = re.match('[A-Z]+', cell.attrib['r']).group()
                    dd[col] = shared[int(v.text)] if cell.attrib.get('t') == 's' else v.text
                rows.append(dd)
            microw = [r for r in rows if r.get('A', '').strip().lower() == 'microwave']
            if not microw:
                result[rec] = None
            else:
                assert len(microw) == 1
                marks = [i for i, c in enumerate('BCDEF') if c in microw[0] and microw[0][c] not in ('0', '')]
                assert len(marks) <= 1, (rec, microw[0])
                result[rec] = marks[0] if marks else None
    return result

def parse_state(s):
    pairs = re.findall('Socket([0-4]):\\s*([01])', s)
    assert len(pairs) == 5 and set((int(k) for k, _ in pairs)) == set(range(5)), s
    d = {int(k): int(v) for k, v in pairs}
    return tuple((d[i] for i in range(5)))

def load_labels(z, rec):
    with z.open(f'Aggregate/Event/Aggregation_sign_{rec}.csv') as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')))
    times = np.asarray([us_of(r['Time']) for r in rows], dtype=np.int64)
    states = np.asarray([parse_state(r['event']) for r in rows], dtype=np.uint8)
    assert np.all(np.diff(times) > 0) and len(times) > 300
    target = np.zeros(len(times), np.uint8) if MICROWAVE_SOCKET[rec] is None else states[:, MICROWAVE_SOCKET[rec]]
    if MICROWAVE_SOCKET[rec] is None:
        assert not target.any()
    changes = np.r_[False, target[1:] != target[:-1]]
    any_changes = np.r_[False, np.any(states[1:] != states[:-1], axis=1)]
    return {'time': times, 'states': states, 'target': target, 'change_time': times[changes], 'all_change_time': times[any_changes], 'n_labels': len(times), 'on_switches': int(np.sum((target[1:] == 1) & (target[:-1] == 0))), 'off_switches': int(np.sum((target[1:] == 0) & (target[:-1] == 1)))}

def cycle_quantities(raw):
    assert raw.ndim == 3 and raw.shape[1:] == (2, 1024)
    u = raw[:, 0].astype(np.float32) / 12.9
    i = raw[:, 1].astype(np.float32) * (28.47 / 32768.0)
    ur = np.sqrt(np.mean(u * u, axis=1, dtype=np.float64))
    ir = np.sqrt(np.mean(i * i, axis=1, dtype=np.float64))
    p = np.mean(u * i, axis=1, dtype=np.float64)
    s = ur * ir
    rad = s * s - p * p
    n_valid = rad >= -1e-05 * np.maximum(s * s, 1.0)
    nonactive = np.sqrt(np.maximum(rad, 0.0))
    pf_valid = s > 1e-05
    pf = np.divide(p, s, out=np.zeros_like(p), where=pf_valid)
    basic = np.stack((p, s, nonactive, ur, ir, pf), axis=1).astype(np.float32)
    bmask = np.stack((np.isfinite(p), np.isfinite(s), n_valid, np.isfinite(ur), np.isfinite(ir), pf_valid), axis=1)
    bmask &= np.isfinite(basic)
    basic[~bmask] = 0.0
    fu = np.fft.rfft(u, axis=1)
    fi = np.fft.rfft(i, axis=1)
    phase = np.angle(fu[:, 1])
    im = np.maximum(ir, 1e-12)
    harmonics = []
    for h in (3, 5, 7, 9, 11):
        z = np.sqrt(2.0) / 1024.0 * fi[:, h] * np.exp(-1j * h * phase) / im
        harmonics.extend((z.real, z.imag))

    def thd(spec):
        return np.sqrt(np.sum(np.abs(spec[:, 2:26]) ** 2, axis=1)) / np.maximum(np.abs(spec[:, 1]), 1e-12)
    central = i - i.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(central * central, axis=1))
    kurt = np.mean((central / np.maximum(std[:, None], 1e-06)) ** 4, axis=1)
    other = (thd(fi), thd(fu), np.max(np.abs(i), axis=1) / im, kurt, i.mean(axis=1) / im, np.cos(np.angle(fi[:, 1]) - phase))
    hf = np.stack((*harmonics, *other), axis=1).astype(np.float32)
    assert hf.shape[1] == 16
    hmask = np.isfinite(hf) & (ir[:, None] > 0.0001) & (ur[:, None] > 1.0)
    hmask[:, 11] &= np.abs(fi[:, 1]) > 0.001
    hmask[:, 10] &= np.abs(fi[:, 1]) > 0.001
    hmask[:, 11] &= np.abs(fu[:, 1]) > 0.001
    hmask[:, 15] &= (np.abs(fi[:, 1]) > 0.001) & (np.abs(fu[:, 1]) > 0.001)
    hmask[:, 13] &= std > 0.0001
    hf[~hmask] = 0.0
    return (basic, bmask.astype(np.uint8), hf, hmask.astype(np.uint8))

def make_labels(rec, times):
    lab = LABELS[rec]
    ix = np.searchsorted(lab['time'], times, side='right') - 1
    known = (ix >= 0) & (times <= lab['time'][-1])
    y = np.zeros(len(times), np.uint8)
    y[known] = lab['target'][ix[known]]
    changes = lab['change_time']
    if len(changes):
        pos = np.searchsorted(changes, times)
        left = changes[np.maximum(pos - 1, 0)]
        right = changes[np.minimum(pos, len(changes) - 1)]
        dist = np.minimum(np.abs(times - left), np.abs(times - right))
        known &= dist > 150000
    else:
        dist = np.full(len(times), np.iinfo(np.int64).max, dtype=np.int64)
    event = np.full(len(times), -1, np.int32)
    starts = np.flatnonzero((lab['target'] == 1) & np.r_[True, lab['target'][:-1] == 0])
    if len(starts):
        ords = np.searchsorted(lab['time'][starts], times, side='right') - 1
        active = (y == 1) & (ords >= 0)
        event[active] = rec * 1000 + ords[active].astype(np.int32)
    any_ch = lab['all_change_time']
    pos = np.searchsorted(any_ch, times)
    near = np.minimum(np.abs(times - any_ch[np.maximum(pos - 1, 0)]), np.abs(times - any_ch[np.minimum(pos, len(any_ch) - 1)])) if len(any_ch) else np.full(len(times), 10 ** 15, dtype=np.int64)
    return (y, known.astype(np.uint8), event, dist, near)

def extract_record(rec):
    cache = BASE_OUT / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    raw_path = cache / f'agg{rec}_adc_int16.bin'
    arrays_path = cache / f'agg{rec}_timeline.npz'
    meta_path = cache / f'agg{rec}_meta.json'
    if meta_path.exists() and arrays_path.exists() and raw_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if raw_path.stat().st_size == meta['cycles'] * 2 * 1024 * 2:
            with np.load(arrays_path, allow_pickle=False) as zz:
                arr = {k: zz[k] for k in zz.files}
            assert len(arr['time']) == meta['cycles']
            return (arr, meta)
    started = time.time()
    stats = Counter()
    pending = {}
    seen = set()
    lists = {k: [] for k in ('time', 'index', 'basic', 'bmask', 'hf', 'hmask')}
    block = []
    block_times = []
    block_idx = []

    def flush(handle):
        if not block:
            return
        raw = np.stack(block).astype(np.int16)
        raw.tofile(handle)
        basic, bmask, hf, hmask = cycle_quantities(raw)
        lists['time'].append(np.asarray(block_times, np.int64))
        lists['index'].append(np.asarray(block_idx, np.uint8))
        lists['basic'].append(basic)
        lists['bmask'].append(bmask)
        lists['hf'].append(hf)
        lists['hmask'].append(hmask)
        block.clear()
        block_times.clear()
        block_idx.clear()
    with ZipFile(ARCHIVE) as z, z.open(f'Aggregate/Aggregation_{rec}.csv') as f, raw_path.open('wb') as handle:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='', errors='strict'))
        assert reader.fieldnames == ['id', 'DeviceTime', 'HomeID', 'RecvTime', 'InternalIndex', 'Channel', 'Rate', 'Value'], reader.fieldnames
        for row in reader:
            stats['rows'] += 1
            key = (row['HomeID'], row['RecvTime'], row['InternalIndex'])
            ch = int(row['Channel'])
            if ch not in (1, 2):
                stats['invalid_channel'] += 1
                continue
            pair = pending.setdefault(key, {})
            if ch in pair:
                stats['duplicate_channel'] += 1
            pair[ch] = row
            if 1 not in pair or 2 not in pair:
                continue
            del pending[key]
            if key in seen:
                stats['duplicate_pair_key'] += 1
                continue
            seen.add(key)
            stats['paired'] += 1
            if pair[1]['Rate'] != '1024' or pair[2]['Rate'] != '1024':
                stats['invalid_rate'] += 1
                continue
            uv = np.fromstring(pair[1]['Value'], sep='|', dtype=np.int16)
            iv = np.fromstring(pair[2]['Value'], sep='|', dtype=np.int16)
            if uv.size != 1024 or iv.size != 1024:
                stats['invalid_length'] += 1
                continue
            t = us_of(row['RecvTime'])
            idx = int(row['InternalIndex'])
            block.append(np.stack((uv, iv)))
            block_times.append(t)
            block_idx.append(idx)
            if len(block) >= 2048:
                flush(handle)
            if stats['paired'] % 100000 == 0:
                print(f"Record {rec}: paired {stats['paired']:,} cycles in {time.time() - started:.0f}s", flush=True)
        flush(handle)
    stats['unpaired_keys_at_eof'] = len(pending)
    del pending, seen
    gc.collect()
    arr = {k: np.concatenate(v, axis=0) for k, v in lists.items()}
    n = len(arr['time'])
    assert n == stats['paired'] - stats['invalid_rate'] - stats['invalid_length'] - stats['duplicate_pair_key']
    tt = arr['time']
    ii = arr['index'].astype(np.int16)
    delta = np.diff(tt)
    step = np.diff(ii) % 256 == 1
    stats['jitter_outside_10_30ms'] = int(np.sum((delta < 10000) | (delta > 30000)))
    stats['time_nonincreasing'] = int(np.sum(delta <= 0))
    stats['time_gap_over_100ms'] = int(np.sum(delta > 100000))
    stats['index_jump'] = int(np.sum(~step))
    split = np.r_[True, (delta <= 0) | (delta > 100000) | ~step]
    arr['continuity'] = np.cumsum(split, dtype=np.int32) - 1
    y, known, event, dist, near = make_labels(rec, tt)
    arr.update({'y': y, 'known': known, 'event': event, 'distance_switch_us': dist, 'distance_any_us': near})
    np.savez_compressed(arrays_path, **arr)
    meta = {'record': rec, 'cycles': int(n), 'first_time_us': int(tt[0]), 'last_time_us': int(tt[-1]), 'stats': dict(stats), 'raw_size': raw_path.stat().st_size, 'elapsed_s': round(time.time() - started, 1), 'basic_hash': array_hash(arr['basic']), 'hf_hash': array_hash(arr['hf']), 'label_hash': array_hash(y)}
    dump_json(meta, meta_path)
    return (arr, meta)

def phase_of(rec, fold):
    lab = LABELS[rec]
    tt = TIMELINES[rec]['time']
    in_span = (tt >= lab['time'][0]) & (tt <= lab['time'][-1])
    phase = np.full(len(tt), -1, np.int8)
    if rec in PAIR_FOLDS[fold]['train']:
        ca, cb = PAIR_FOLDS[fold]['cut']
        cuts = lab['time'][[ca, cb]]
        phase[in_span] = np.searchsorted(cuts, tt[in_span], side='right').astype(np.int8)
    else:
        phase[in_span] = 3
    return phase

def segment_of(rec, fold):
    arr = TIMELINES[rec]
    phase = phase_of(rec, fold)
    n = len(phase)
    change = np.r_[True, (phase[1:] != phase[:-1]) | (arr['continuity'][1:] != arr['continuity'][:-1])]
    seg = np.cumsum(change, dtype=np.int32)
    start = np.maximum.accumulate(np.where(change, np.arange(n, dtype=np.int64), 0))
    return (phase, seg, start)

def causal_feature_block(vals, masks, seg):
    n, k = vals.shape
    assert masks.shape == (n, k)
    out = np.zeros((n, k * 6), np.float32)
    for j in range(k):
        base = vals[:, j].astype(np.float64)
        valid = masks[:, j].astype(bool)
        left = np.r_[False, (seg[1:] == seg[:-1]) & valid[1:] & valid[:-1]]
        delta = np.zeros(n, np.float64)
        delta[1:] = np.where(left[1:], base[1:] - base[:-1], 0.0)
        rolling = pd.Series(np.where(valid, base, np.nan)).groupby(seg, sort=False).transform(lambda x: x.shift(1).rolling(50, min_periods=42).median()).to_numpy()
        bvalid = valid & np.isfinite(rolling)
        out[:, 6 * j] = np.where(valid, base, 0.0)
        out[:, 6 * j + 1] = delta
        out[:, 6 * j + 2] = np.where(bvalid, base - rolling, 0.0)
        out[:, 6 * j + 3] = valid
        out[:, 6 * j + 4] = left
        out[:, 6 * j + 5] = bvalid
    return out

def endpoint_index(rec, fold, part, train=False):
    arr = TIMELINES[rec]
    phase, seg, start = segment_of(rec, fold)
    idx = np.arange(len(phase), dtype=np.int64)
    cadence = idx % 50 == 49 if train else idx % 5 == 4
    ok = (phase == part) & (arr['known'] == 1) & cadence & (idx - start >= 2609) & (arr['bmask'][:, 0] == 1)
    ep = idx[ok]
    assert not len(ep) or np.all(seg[ep] == seg[ep - 2559])
    assert not len(ep) or np.all(seg[ep] == seg[ep - 2609])
    return ep

def fit_norm(all_values, all_masks):
    mat = np.concatenate(all_values, axis=0).astype(np.float64)
    mask = np.concatenate(all_masks, axis=0).astype(bool)
    assert mat.shape == mask.shape
    mu = np.zeros(mat.shape[1], np.float64)
    sd = np.ones(mat.shape[1], np.float64)
    for j in range(mat.shape[1]):
        v = mat[mask[:, j], j]
        assert len(v) > 0, j
        mu[j] = float(v.mean())
        sd[j] = max(float(v.std()), 1e-06)
    return {'mean': mu, 'std': sd}

def build_fold_features(fold):
    feature_dir = BASE_OUT / 'cache' / f'fold_{fold}'
    feature_dir.mkdir(parents=True, exist_ok=True)
    norm_path = feature_dir / 'normalizer.npz'
    filepaths = {rec: feature_dir / f'agg{rec}_features.npz' for rec in range(1, 8)}
    if norm_path.exists() and all((p.exists() for p in filepaths.values())):
        with np.load(norm_path) as z:
            norm = {k: z[k] for k in z.files}
        feat = {}
        for rec, p in filepaths.items():
            with np.load(p, allow_pickle=False) as z:
                feat[rec] = {k: z[k] for k in z.files}
        return (feat, norm)
    train = PAIR_FOLDS[fold]['train']
    raw = {}
    segments = {}
    for rec in range(1, 8):
        phase, seg, start = segment_of(rec, fold)
        arr = TIMELINES[rec]
        base = causal_feature_block(arr['basic'], arr['bmask'], seg)
        raw[rec] = base
        segments[rec] = (phase, seg, start)
    value_list = []
    mask_list = []
    hval = []
    hmask = []
    for rec in train:
        phase, _, _ = segments[rec]
        ix = phase == 0
        vals = raw[rec][ix]
        value_list.append(np.stack([vals[:, 6 * j + k] for j in range(6) for k in range(3)], axis=1))
        mask_list.append(np.stack([vals[:, 6 * j + 3 + k] for j in range(6) for k in range(3)], axis=1).astype(bool))
        hval.append(TIMELINES[rec]['hf'][ix])
        hmask.append(TIMELINES[rec]['hmask'][ix])
    bn = fit_norm(value_list, mask_list)
    hn = fit_norm(hval, hmask)
    v_rms = math.sqrt(float(np.mean(np.concatenate([TIMELINES[r]['basic'][segments[r][0] == 0, 3].astype(np.float64) ** 2 for r in train]))))
    i_rms = math.sqrt(float(np.mean(np.concatenate([TIMELINES[r]['basic'][segments[r][0] == 0, 4].astype(np.float64) ** 2 for r in train]))))
    assert v_rms > 0 and i_rms > 0
    norm = {'base_mean': bn['mean'], 'base_std': bn['std'], 'hf_mean': hn['mean'], 'hf_std': hn['std'], 'voltage_scale': np.float64(v_rms), 'current_scale': np.float64(i_rms)}
    np.savez(norm_path, **norm)
    feat = {}
    for rec in range(1, 8):
        base = raw[rec]
        for j in range(6):
            for k in range(3):
                v = base[:, 6 * j + k]
                ok = base[:, 6 * j + 3 + k].astype(bool)
                q = j * 3 + k
                v[ok] = (v[ok] - norm['base_mean'][q]) / norm['base_std'][q]
                v[~ok] = 0.0
        hf = TIMELINES[rec]['hf'].copy()
        mask = TIMELINES[rec]['hmask'].astype(bool)
        hf[mask] = (hf[mask] - np.broadcast_to(norm['hf_mean'], hf.shape)[mask]) / np.broadcast_to(norm['hf_std'], hf.shape)[mask]
        hf[~mask] = 0.0
        phase, seg, start = segments[rec]
        feat[rec] = {'base': base, 'hf': hf, 'hmask': mask.astype(np.uint8), 'phase': phase, 'segment': seg, 'start': start}
        np.savez_compressed(filepaths[rec], **feat[rec])
    return (feat, norm)

def physical_wave(raw):
    u = raw[:, 0].astype(np.float32) / 12.9
    i = raw[:, 1].astype(np.float32) * (28.47 / 32768.0)
    p = np.mean(u * i, axis=1, dtype=np.float64)
    u2 = np.mean(u * u, axis=1, dtype=np.float64)
    good = np.isfinite(u2) & (u2 > 1e-12)
    f = i - (np.divide(p, u2, out=np.zeros_like(p), where=good)[:, None] * u).astype(np.float32)
    return (u, i, f, good)

def f_current(raw):
    u, i, _, good = physical_wave(raw)
    ur = np.sqrt(np.mean(u * u, axis=1, dtype=np.float64))
    ir = np.sqrt(np.mean(i * i, axis=1, dtype=np.float64))
    uf = np.fft.rfft(u, axis=1)[:, 1]
    ff = np.fft.rfft(i, axis=1)[:, 1:26]
    h = np.arange(1, 26)
    rot = np.exp(-1j * np.angle(uf)[:, None] * h[None, :])
    amp = np.sqrt(2.0) / 1024 * ff * rot
    ab = np.stack((amp.real, amp.imag), axis=-1).reshape(-1, 50)
    rel = ab / np.maximum(ir[:, None], 1e-12)
    valid = np.isfinite(ab).all(1) & np.isfinite(uf) & (ur > 1.0) & (np.abs(uf) > 0.001) & good
    relvalid = valid & (ir > 1e-06)
    vals = np.concatenate((ab, rel), axis=1).astype(np.float32)
    vals[~valid, :50] = 0
    vals[~relvalid, 50:] = 0
    mask = np.stack((valid, relvalid), axis=1).astype(np.uint8)
    return (vals, mask)

def raw_memmap(rec):
    if rec not in RAW_MMAP:
        n=SOURCE_AUDIT[rec-1]["cycles"]
        RAW_MMAP[rec]=np.memmap(BASE_OUT/"cache"/f"agg{rec}_adc_int16.bin",dtype=np.int16,mode="r",shape=(n,2,1024))
    return RAW_MMAP[rec]


def window_arrays(fold,rec,eps,rep,length=2560):
    if rep!="W1":raise ValueError("Only selected W1 is supported")
    eps=np.asarray(eps,dtype=np.int64)
    x=np.zeros((len(eps),CHANNELS,length),np.float32)
    feat=FOLD_FEATURES[fold][rec];source=HF_RAW[rec]
    for k,ep in enumerate(eps):
        ix=np.arange(ep-length+1,ep+1,dtype=np.int64)
        assert ix[0]>=feat["start"][ep] and feat["segment"][ix[0]]==feat["segment"][ep]
        x[k,:36]=feat["base"][ix].T
        x[k,36+M:36+M+16]=source["fm"][ix,0].astype(np.float32)[None]
    return x


def feature_timeline(fold,rec,rep):
    if rep!="W1":raise ValueError("Only selected W1 is supported")
    src=FOLD_FEATURES[fold][rec];n=len(src["base"])
    x=np.zeros((CHANNELS,n),np.float32);x[:36]=src["base"].T
    x[36+M:36+M+16]=HF_RAW[rec]["fm"][:,0][None]
    return x


def collect_endpoints(fold, kind):
    records = PAIR_FOLDS[fold]['train'] if kind in ('train', 'calibration', 'time') else PAIR_FOLDS[fold]['test']
    pairs = np.concatenate([np.stack((np.full(len(ENDPOINTS[fold][r, kind]), r, dtype=np.int32), ENDPOINTS[fold][r, kind]), axis=1) for r in records], axis=0)
    return pairs

def pairs_y(pairs):
    return np.asarray([TIMELINES[int(r)]['y'][int(e)] for r, e in pairs], np.uint8)

def chunk_features(fold, pairs, rep, length):
    x = np.empty((len(pairs), CHANNELS, length), np.float32)
    for rec in np.unique(pairs[:, 0]):
        sel = np.flatnonzero(pairs[:, 0] == rec)
        x[sel] = window_arrays(fold, int(rec), pairs[sel, 1], rep, length)
    return x

def chunk_raw(pairs, length):
    return np.stack([raw_memmap(int(r))[int(e) - length + 1:int(e) + 1] for r, e in pairs])

def train_threshold(y, score, target=0.95):
    y = np.asarray(y)
    score = np.asarray(score)
    assert np.any(y == 1) and np.all(np.isfinite(score))
    pos = np.sort(score[y == 1])
    need = math.ceil(target * len(pos))
    candidate = np.unique(np.r_[np.nextafter(score.min(), -np.inf), score, np.nextafter(score.max(), np.inf)])
    recall = (len(pos) - np.searchsorted(pos, candidate, side='left')) / len(pos)
    feasible = candidate[recall >= target - 1e-12]
    assert len(feasible) > 0 and recall[candidate == feasible[-1]][0] >= target - 1e-12
    return float(feasible[-1])

def model_state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()

def binary_bce(y, logits):
    y = np.asarray(y)
    z = np.asarray(logits, dtype=np.float64)
    loss = np.logaddexp(0.0, z) - y * z
    assert (y == 0).any() and (y == 1).any()
    return float(0.5 * (loss[y == 0].mean() + loss[y == 1].mean()))

def encode_full_record(model, rec, batch=512):
    assert model.encoder is not None
    mm = raw_memmap(rec)
    out = np.empty((len(mm), 16), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(mm), batch):
            raw = torch.from_numpy(np.asarray(mm[lo:lo + batch]).copy()).to(DEVICE)
            out[lo:lo + len(raw)] = model.encode_cycles(raw[:, None]).reshape(-1, 16).float().cpu().numpy()
    return out

def score_neural(model, fold, pairs, batch=64):
    if (model.family, model.rep, model.length) != ('GRU', 'W1', 2560):
        raise ValueError('Only selected GRU/W1/L2560 readout is supported')
    prior = rng_digest()
    old = model.training
    model.eval()
    out = np.empty(len(pairs), np.float32)
    try:
        with torch.inference_mode():
            for rec in np.unique(pairs[:, 0]):
                sel = np.flatnonzero(pairs[:, 0] == rec)
                ep = pairs[sel, 1].astype(np.int64)
                xx = feature_timeline(fold, int(rec), model.rep)
                emb = encode_full_record(model, int(rec)) if model.encoder is not None else None
                if emb is not None:
                    xx[36:52] = emb.T * xx[36 + M:36 + M + 16]
                tensor = torch.from_numpy(np.ascontiguousarray(xx[None])).to(DEVICE)
                stem = model.temporal.stem
                z = F.relu(stem.conv(tensor))
                usable = z.shape[-1] // 5 * 5
                z = z[:, :, :usable].reshape(1, 32, usable // 5, 5)
                tokens = stem.norm(stem.proj(torch.cat((z.mean(-1), z.amax(-1)), dim=1).transpose(1, 2)))[0]
                for lo in range(0, len(ep), batch):
                    ids = ep[lo:lo + batch]
                    start = ids - model.length + 1
                    assert np.all(start % 5 == 0)
                    positions = torch.as_tensor(start[:, None] // 5 + np.arange(model.length // 5)[None, :], device=DEVICE)
                    group = tokens[positions].clone()
                    first = np.stack([xx[:, v:v + 5] for v in start])
                    first = torch.from_numpy(first).to(DEVICE)
                    group[:, 0] = stem(first)[:, 0]
                    h = model.temporal.memory(group)[0][:, -1]
                    out[sel[lo:lo + len(ids)]] = model.temporal.head(h).squeeze(-1).float().cpu().numpy()
                del tensor, z, tokens
                del xx, emb
        return out
    finally:
        model.train(old)
        assert rng_digest() == prior, 'Evaluation altered training RNG state'

def fit_neural(fold, seed, family, rep, length, calpairs, cal_y):
    fix_seed(seed)
    norm = FOLD_NORMS[fold]
    model = AggregateModel(family, rep, length, norm['voltage_scale'], norm['current_scale']).to(DEVICE)
    init_hash = model_state_hash(model.temporal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    sample = SAMPLE_MANIFEST[fold][seed]
    pairs = sample['pairs']
    y = pairs_y(pairs).astype(np.float32)
    best = None
    best_loss = np.inf
    best_epoch = 0
    bad = 0
    history = []
    t0 = time.time()
    for epoch in range(40):
        model.train()
        order = sample['order'][epoch]
        total = 0.0
        for left in range(0, len(order), 64):
            group = order[left:left + 64]
            den = len(group)
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, den, 8):
                ids = group[j:j + 8]
                batchpairs = pairs[ids]
                x = torch.from_numpy(chunk_features(fold, batchpairs, rep, length)).to(DEVICE)
                raw = torch.from_numpy(chunk_raw(batchpairs, length).copy()).to(DEVICE) if rep in ('W0', 'W1') else None
                logit = model(x, raw=raw)
                target = torch.from_numpy(y[ids]).to(DEVICE)
                loss = F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / den
                loss.backward()
                total += float(loss.detach().cpu()) * den
                del x, raw, logit, target, loss
            optimizer.step()
        scores = score_neural(model, fold, calpairs)
        cal_loss = binary_bce(cal_y, scores)
        history.append({'epoch': epoch + 1, 'sample_BCE': total / len(order), 'balanced_cal_BCE': cal_loss})
        print(f'{fold} seed{seed} {family}-{rep} L{length} epoch{epoch + 1}: {cal_loss:.5f}', flush=True)
        if cal_loss < best_loss - 1e-08:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss = cal_loss
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
        if bad >= 5:
            break
    model.load_state_dict(best)
    model.eval()
    return (model, history, best_epoch, init_hash, round(time.time() - t0, 2))

def checkpoint_name(fold, seed, fam, rep, length):
    return f'{fold}_{seed}_{fam}_{rep}_L{length}'

def power5(rec):
    p = TIMELINES[rec]['basic'][:, 0].astype(np.float64)
    cs = np.r_[0.0, np.cumsum(p)]
    z = np.full(len(p), np.nan, np.float32)
    z[4:] = ((cs[5:] - cs[:-5]) / 5.0).astype(np.float32)
    return z

def sigmoid(x):
    x = np.asarray(x, np.float64)
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

def condition_50w(y, p, event, score):
    y = np.asarray(y)
    p = np.asarray(p)
    event = np.asarray(event)
    score = np.asarray(score)
    pos = np.flatnonzero((y == 1) & (event >= 0))
    all_events = np.unique(event[pos])
    if not len(pos):
        return (np.nan, 0.0, {})
    off = np.flatnonzero(y == 0)
    bins = np.floor(p / 50).astype(np.int64)
    ev_value = defaultdict(list)
    for bin_ in np.unique(bins[pos]):
        ref = off[bins[off] == bin_]
        if not len(ref):
            continue
        ordered = np.sort(score[ref])
        ids = pos[bins[pos] == bin_]
        left = np.searchsorted(ordered, score[ids], side='left')
        right = np.searchsorted(ordered, score[ids], side='right')
        for j, val in zip(ids, (left + right) / (2 * len(ref))):
            ev_value[int(event[j])].append(float(val))
    event_mean = {k: float(np.mean(v)) for k, v in ev_value.items()}
    return (float(np.mean(list(event_mean.values()))) if event_mean else np.nan, len(event_mean) / len(all_events), event_mean)

def record_metrics(rec, fold, logit, threshold, endpoint, stage):
    a = TIMELINES[rec]
    y = a['y'][endpoint]
    p = POWER5[rec][endpoint]
    prob = sigmoid(logit)
    pred = np.asarray(logit) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_two = len(np.unique(y)) == 2
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    cond, coverage, ev = condition_50w(y, p, a['event'][endpoint], logit)
    off = y == 0
    high = off & (p >= HIGH_POWER[fold])
    steady = a['distance_any_us'][endpoint] > 1000000
    safe500 = a['distance_switch_us'][endpoint] > 500000
    ys = y[steady]
    ps = pred[steady]
    return {'record': rec, 'fold': fold, 'stage': stage, 'n': len(endpoint), 'positive': int(y.sum()), 'prevalence': float(y.mean()), 'AP': float(average_precision_score(y, logit)) if has_two else np.nan, 'ROC_AUC': float(roc_auc_score(y, logit)) if has_two else np.nan, 'precision': float(precision), 'recall': float(recall) if y.any() else np.nan, 'F1': float(f1) if y.any() else np.nan, 'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if has_two else np.nan, 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp), 'off_FPR': float(fp / off.sum()) if off.any() else np.nan, 'high_off_FPR': float(pred[high].mean()) if high.any() else np.nan, 'conditional_50w': cond, 'condition_coverage': coverage, 'off_score_P_spearman': float(spearmanr(logit[off], p[off]).statistic) if off.sum() > 2 else np.nan, 'steady_AP': float(average_precision_score(ys, logit[steady])) if len(np.unique(ys)) == 2 else np.nan, 'steady_ON_recall': float(ps[ys == 1].mean()) if np.any(ys == 1) else np.nan, 'steady_OFF_FPR': float(ps[ys == 0].mean()) if np.any(ys == 0) else np.nan, 'mask_500ms_n': int(safe500.sum()), 'mask_500ms_AP': float(average_precision_score(y[safe500], logit[safe500])) if len(np.unique(y[safe500])) == 2 else np.nan, 'mask_500ms_recall': float(pred[safe500 & (y == 1)].mean()) if np.any(safe500 & (y == 1)) else np.nan, 'mask_500ms_off_FPR': float(pred[safe500 & (y == 0)].mean()) if np.any(safe500 & (y == 0)) else np.nan}

def per_record_metrics(fold, pairs, y, logit, threshold, stage):
    out = []
    for rec in np.unique(pairs[:, 0]):
        select = np.flatnonzero(pairs[:, 0] == rec)
        eps = pairs[select, 1].astype(np.int64)
        assert np.array_equal(TIMELINES[int(rec)]['y'][eps], y[select])
        out.append(record_metrics(int(rec), fold, logit[select], threshold, eps, stage))
    return out

def validate_paths(archive,mapping,output):
    archive=Path(archive).resolve();mapping=Path(mapping).resolve();output=Path(output).resolve()
    if not archive.is_file() or not mapping.is_file():
        raise FileNotFoundError("Provide local Aggregate.zip and SocketRecord.xlsx")
    if output.exists():
        raise FileExistsError("Output must be a new directory; existing experiments are never overwritten")
    if output.is_relative_to(archive.parent) or output.is_relative_to(mapping.parent) or archive.is_relative_to(output) or mapping.is_relative_to(output):
        raise ValueError("Output must be separate from the input files")
    for ancestor in (output,*output.parents):
        if (ancestor/".git").exists() or ((ancestor/"pyproject.toml").is_file() and (ancestor/"src").is_dir()):
            raise ValueError("Training output must be outside source checkouts; choose an external artifact directory")
    return archive,mapping,output


def prepare(archive,mapping,output,device="cuda"):
    """Build selected causal inputs in a fresh explicitly supplied output folder."""
    global ARCHIVE,BASE_OUT,OUT,DEVICE
    archive,mapping,output=validate_paths(archive,mapping,output)
    if device not in ("cpu","cuda"):
        raise ValueError("Choose the CPU or CUDA training environment")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable; no device fallback")
    ARCHIVE=archive;BASE_OUT=output;OUT=output;DEVICE=torch.device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(1)
    for d in ("audit","cache","models","predictions","tables","protocol"):(OUT/d).mkdir(parents=True,exist_ok=False)
    for store in (LABELS,TIMELINES,FOLD_FEATURES,FOLD_NORMS,ENDPOINTS,SAMPLE_MANIFEST,HF_RAW,RAW_MMAP,POWER5,HIGH_POWER):store.clear()
    SOURCE_AUDIT.clear()
    assert socket_mapping(mapping)==MICROWAVE_SOCKET
    with ZipFile(ARCHIVE) as z:
        LABELS.update({rec:load_labels(z,rec) for rec in range(1,8)})
    for rec in range(1,8):
        TIMELINES[rec],meta=extract_record(rec);SOURCE_AUDIT.append(meta)
        raw=raw_memmap(rec);fm=np.zeros((len(raw),2),np.uint8)
        for lo in range(0,len(raw),4096):
            _,mask=f_current(np.asarray(raw[lo:lo+4096]))
            fm[lo:lo+len(mask)]=mask
        HF_RAW[rec]={"fm":fm}
    dump_json(SOURCE_AUDIT,OUT/"audit"/"source_quality.json")
    for fold in PAIR_FOLDS:
        FOLD_FEATURES[fold],FOLD_NORMS[fold]=build_fold_features(fold)
        ENDPOINTS[fold]={}
        for rec in range(1,8):
            part=3 if rec not in PAIR_FOLDS[fold]["train"] else 0
            ENDPOINTS[fold][(rec,"train")]=endpoint_index(rec,fold,part,train=True) if part==0 else np.empty(0,np.int64)
            for name,p in (("calibration",1),("time",2),("cross",3)):
                ENDPOINTS[fold][(rec,name)]=endpoint_index(rec,fold,p)
        train_pos=np.concatenate([np.stack((np.full(len(ENDPOINTS[fold][(r,"train")]),r,dtype=np.int32),ENDPOINTS[fold][(r,"train")]),axis=1) for r in PAIR_FOLDS[fold]["train"]],axis=0)
        y=np.asarray([TIMELINES[int(r)]["y"][int(i)] for r,i in train_pos],np.uint8)
        on=train_pos[y==1];off=train_pos[y==0]
        assert len(on)>0 and len(off)>=len(on)
        SAMPLE_MANIFEST[fold]={}
        for seed in SEEDS:
            rng=np.random.default_rng(seed+1000*(fold=="B"))
            chosen=off[rng.choice(len(off),size=len(on),replace=False)]
            pairs=np.concatenate((on,chosen),axis=0)
            order=np.stack([rng.permutation(len(pairs)) for _ in range(40)])
            SAMPLE_MANIFEST[fold][seed]={"pairs":pairs.astype(np.int32),"order":order.astype(np.int32),"hash":array_hash(pairs)}
            np.savez(OUT/"audit"/f"sampling_{fold}_{seed}.npz",pairs=pairs,order=order)
    POWER5.update({r:power5(r) for r in range(1,8)})
    for fold in PAIR_FOLDS:
        train=collect_endpoints(fold,"train")
        neg=np.asarray([POWER5[int(r)][int(e)] for r,e in train if TIMELINES[int(r)]["y"][int(e)]==0],np.float32)
        HIGH_POWER[fold]=float(np.quantile(neg,.8))
    dump_json({"family":"GRU","representation":"W1","history_cycles":2560,"history_seconds":51.2,"input_hz":51200,"output_seconds":.1,
               "folds":PAIR_FOLDS,"seeds":SEEDS,"high_power":HIGH_POWER,"scope":"TDHA GRU/W1 training with fold-specific preprocessing and calibration"},OUT/"protocol"/"method.json")


def train_case(fold,seed):
    """Train one selected run and freeze its calibration threshold before cross readout."""
    family,rep,length="GRU","W1",2560
    calpairs=collect_endpoints(fold,"calibration");cal_y=pairs_y(calpairs)
    key=checkpoint_name(fold,seed,family,rep,length)
    model,history,best_epoch,initial_hash,fit_seconds=fit_neural(fold,seed,family,rep,length,calpairs,cal_y)
    cal=score_neural(model,fold,calpairs);threshold=train_threshold(cal_y,cal)
    state_hash=model_state_hash(model)
    checkpoint={"family":family,"rep":rep,"length":length,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    target=OUT/"models"/f"{key}.pt";temporary=target.with_suffix(".pt.tmp")
    torch.save(checkpoint,temporary);os.replace(temporary,target)
    pd.DataFrame(history).to_csv(OUT/"tables"/f"history_{key}.csv",index=False)
    np.savez_compressed(OUT/"predictions"/f"{key}_cal.npz",pairs=calpairs,y=cal_y,logit=cal)
    row={"key":key,"fold":fold,"seed":seed,"family":family,"rep":rep,"length":length,
         "best_epoch":best_epoch,"threshold":threshold,"state_hash":state_hash,"init_hash":initial_hash,
         "sample_hash":SAMPLE_MANIFEST[fold][seed]["hash"],"batch_hash":array_hash(SAMPLE_MANIFEST[fold][seed]["order"]),"fit_seconds":fit_seconds}
    dump_json(row,OUT/"protocol"/f"{key}_frozen_before_cross.json")
    for part in ("time","cross","pressure"):
        if part=="pressure":
            eps=ENDPOINTS[fold][(1,"cross")]
            pairs=np.stack((np.full(len(eps),1,dtype=np.int32),eps),axis=1)
        else:
            pairs=collect_endpoints(fold,part)
        y=pairs_y(pairs);logit=score_neural(model,fold,pairs)
        np.savez_compressed(OUT/"predictions"/f"{key}_{part}.npz",pairs=pairs,y=y,logit=logit)
        metrics=per_record_metrics(fold,pairs,y,logit,threshold,part)
        pd.DataFrame(metrics).to_csv(OUT/"tables"/f"metrics_{key}_{part}.csv",index=False)
    del model;gc.collect()
    if DEVICE.type=="cuda":torch.cuda.empty_cache()
    return row


def run_selected(archive,mapping,output,device="cuda",folds=("A","B"),seeds=SEEDS):
    """Explicit full workflow; no call is made on import or by plan-only mode."""
    if len(set(folds))!=len(folds) or any(f not in PAIR_FOLDS for f in folds):raise ValueError("Choose unique folds A/B")
    if len(set(seeds))!=len(seeds) or any(s not in SEEDS for s in seeds):raise ValueError("Choose the retained seeds 11/42/73")
    prepare(archive,mapping,output,device)
    rows=[train_case(fold,seed) for fold in folds for seed in seeds]
    pd.DataFrame(rows).to_csv(OUT/"tables"/"selected_runs.csv",index=False)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--mapping",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--folds",nargs="+",choices=("A","B"),default=("A","B"))
    parser.add_argument("--seeds",nargs="+",type=int,choices=SEEDS,default=SEEDS)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--execute",action="store_true",help="Explicitly prepare, train and evaluate a new run")
    mode.add_argument("--plan-only",action="store_true",help="Read-only; also the default")
    args=parser.parse_args(argv)
    if not args.execute:
        validate_paths(args.archive,args.mapping,args.output)
        print(json.dumps({"family":"GRU","representation":"W1","length":2560,"folds":args.folds,"seeds":args.seeds,"training":False,"writes":False}))
        return
    run_selected(args.archive,args.mapping,args.output,args.device,tuple(args.folds),tuple(args.seeds))


if __name__=="__main__":main()

