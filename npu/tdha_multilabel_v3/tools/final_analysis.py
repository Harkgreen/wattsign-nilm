"""Paired TDHA V3 statistics for CPU/CUDA evaluations with fixed checkpoints.

The 60-second paired bootstrap uses exact integer sufficient statistics.
CUDA/CPU supplemental evaluation and NPU execution records have separate scope.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path
import warnings
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from paths import output_separate
WORK = INPUTS = ARTIFACT = None
from tdha_v3.common import dump, read, save_array, file_hash, require, digest, array_hash
from tdha_v3.metrics import summarize

ITERATIONS = 2000
BOOTSTRAP_SEED = 42
BLOCK_US = 60_000_000
PRIMARY = 6
PREFIX_MEMORY_BUDGET = 384 * 1024**2
SEEDS = (11, 42, 73)
ARCHIVE = None


def task_id(net, rep, fold, seed):
    return f'{net}-{rep}-{fold}-s{seed}'


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(text.encode('utf-8'))
    temp.replace(path)


def _task_root(run, tid):
    return (ARTIFACT / 'predictions' if ARTIFACT else Path(run) / 'tasks') / tid


def _threshold_root(run, tid):
    return (ARTIFACT / 'models' if ARTIFACT else Path(run) / 'tasks') / tid


def _compact_metadata(fold, part):
    path=ARTIFACT/'evaluation-metadata'/f'fold{fold}_{part}.npz'
    with np.load(path,allow_pickle=False) as saved:
        keys={'y','mask','strict','time','near','record','endpoint'}
        require(keys <= set(saved.files), 'Compact metadata fields are missing')
        result={key:np.array(saved[key]) for key in keys}
    n=len(result['record'])
    require(n>0 and all(len(x)==n for x in result.values()), 'Compact metadata row counts differ')
    require(result['y'].shape==(n,11) and result['mask'].shape==(n,11)
            and result['strict'].shape==(n,11), 'Compact label/mask shape mismatch')
    require(all(result[k].shape==(n,) for k in ('record','endpoint','time','near')), 'Compact endpoint metadata shape mismatch')
    require(all(np.issubdtype(result[k].dtype,np.integer) for k in ('record','endpoint','time','near')), 'Endpoint metadata must retain integer precision')
    require(np.isin(result['mask'],[0,1]).all() and np.isin(result['strict'],[0,1]).all(), 'Invalid compact masks')
    require(np.isin(result['y'][result['mask'].astype(bool)],[0,1]).all(), 'Invalid known labels')
    pairs=np.column_stack((result['record'],result['endpoint']))
    require(len(np.unique(pairs,axis=0))==n, 'Duplicate compact endpoints')
    return result


def validate_compact_artifacts():
    """Verify the selected evidence package using its manifests and prediction metadata."""
    manifest=read(ARTIFACT/'manifest.json')
    rows=manifest.get('files',[]);entries={r['path']:r for r in rows}
    require(len(entries)==len(rows) and bool(rows), 'Invalid compact file manifest')
    def verified(relative):
        require(relative in entries, f'Missing integrity binding: {relative}')
        path=(ARTIFACT/relative).resolve()
        require(path.is_relative_to(ARTIFACT.resolve()), 'Artifact path escapes case directory')
        require(path.is_file(), f'Missing evidence attachment: {relative}')
        expected=entries[relative]
        require(path.stat().st_size==expected['bytes'] and file_hash(path)==expected['sha256'], f'Artifact checksum mismatch: {relative}')
        return path
    for name in ('config.json','mapping.json','acceptance.json','freeze.json','selection.json'):
        verified(name)
    config=read(ARTIFACT/'config.json');tasks=config['tasks']
    expected={task_id(n,r,f,seed) for n,r in [('N0','W1'),('N1','F')] for f in ('A','B') for seed in SEEDS}
    require(len(tasks)==12 and set(tasks)==expected, 'Compact statistics require all 12 frozen paired runs, with no seed selection')
    frozen=read(ARTIFACT/'freeze.json');candidate=read(ARTIFACT/'selection.json')['candidate']
    require(candidate['net']=='N1' and candidate['rep']=='F' and digest(candidate)==digest(frozen['candidate']), 'Frozen candidate changed')
    acceptance=read(ARTIFACT/'acceptance.json')
    require(acceptance.get('status')=='passed' and acceptance.get('npu_verification') is False, 'Invalid local acceptance evidence')
    require(set(acceptance['selected_tasks'])==expected, 'Acceptance task selection mismatch')
    evidence={(x['task'],x['part']):x for x in acceptance['selected_readouts']['parts_evidence']}
    for fold in ('A','B'):
        for part in ('cross','third'):
            verified(f'evaluation-metadata/fold{fold}_{part}.npz');_compact_metadata(fold,part)
    conversion=read(verified('weight_conversion.json'))
    require(conversion.get('status')=='verified' and conversion.get('arbitrary_pickle_loading') is False, 'Invalid conversion evidence')
    converted={row['task']:row for row in conversion['tasks']}
    for tid in tasks:
        weight=verified(f'models/{tid}/best.cpu.pt')
        done=read(verified(f'npu-original/{tid}/done.json'))
        receipt=converted[tid]
        require(receipt['state_hash']==frozen['models'][tid]['model']==done['hashes']['model'], 'Frozen model tensor identity mismatch')
        require(receipt['destination_sha256']==file_hash(weight) and receipt['strict_load'] is True
                and receipt['roundtrip_hash_match'] is True, 'Converted checkpoint identity mismatch')
        fold='A' if '-A-' in tid else 'B'
        threshold=read(verified(f'models/{tid}/thresholds.json'))
        require(digest(threshold['logit'])==frozen['models'][tid]['thresholds'], f'Frozen thresholds changed: {tid}')
        for part in ('cross','third'):
            pred=np.load(verified(f'predictions/{tid}/{part}.npy'),mmap_mode='r',allow_pickle=False)
            metric=read(verified(f'predictions/{tid}/{part}_metrics.json'));meta=_compact_metadata(fold,part)
            require(pred.dtype==np.float32 and pred.shape==(len(meta['record']),11) and np.isfinite(pred).all(), f'Invalid compact predictions: {tid}/{part}')
            receipt=evidence[(tid,part)]
            require(receipt['samples']==len(pred) and receipt['prediction_hash']==array_hash(pred)
                    and receipt['metrics_rows_hash']==digest(metric['rows']), 'Prediction/metric acceptance identity mismatch')
            require(digest(summarize([r for r in metric['rows'] if r.get('region')=='main']))==digest(metric['summary']), 'Compact metric summary mismatch')
    return dict(completed=True,completed_tasks=12,task_ids=tasks,trained=False,thresholds_changed=False,
                backend='cuda',original_npu_status='partial',parts=['cross','third']),candidate


def _pairs(run, fold, part):
    if ARTIFACT:
        meta=_compact_metadata(fold,part)
        return np.column_stack((meta['record'],meta['endpoint'])).astype(np.int64)
    endpoints = read((INPUTS or run) / 'cache' / f'fold{fold}' / 'endpoints.json')
    if part == 'third':
        pairs = [(1, e) for e in endpoints['1_cross']]
    else:
        pairs = [(r, e) for r in range(1, 8)
                 for e in endpoints[f'{r}_{part}'] if r != 1]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _metadata(run, fold, part):
    """Same endpoint order and array selection as training.metadata, without models."""
    if ARTIFACT:return _compact_metadata(fold,part)
    pairs = _pairs(run, fold, part)
    out = {'record': pairs[:, 0], 'endpoint': pairs[:, 1]}
    specifications = {'y': 'y', 'mask': 'known', 'time': 'time', 'near': 'near'}
    for key, filename in specifications.items():
        data = []
        for record in np.unique(pairs[:, 0]):
            points = pairs[pairs[:, 0] == record, 1]
            array = np.load((INPUTS or run) / 'cache' / f'record{record}' / f'{filename}.npy',
                            mmap_mode='r', allow_pickle=False)
            require(len(points) and points.max() < len(array), 'Endpoint outside restored metadata')
            data.append(np.asarray(array[points]))
        require(bool(data), f'No restored metadata for fold {fold}, {part}')
        out[key] = np.concatenate(data)
    return out


def comparison_records(run, candidate, part='cross'):
    run = Path(run)
    mapping = read(ARTIFACT/'mapping.json' if ARTIFACT else (INPUTS or run)/'audit/mapping.json')
    records = []
    for fold in ('A', 'B'):
        meta = _metadata(run, fold, part)
        scores, thresholds, active_flags = [], [], []
        for net, rep in [('N0', 'W1'), (candidate['net'], candidate['rep'])]:
            zs, ts = [], []
            for seed in SEEDS:
                tid=task_id(net,rep,fold,seed)
                root = _task_root(run,tid)
                z = np.load(root / f'{part}.npy', allow_pickle=False)
                require(z.shape == (len(meta['record']), 11) and np.isfinite(z).all(),
                        f'Invalid frozen scores: {root.name}/{part}')
                th = read(_threshold_root(run,tid) / 'thresholds.json')
                require(len(th['active']) == 11 and len(th['logit']) == 11,
                        f'Invalid thresholds: {root.name}')
                zs.append(z)
                ts.append([np.nan if x is None else x for x in th['logit']])
                active_flags.append(np.asarray(th['active'], bool))
            scores.append(np.stack(zs))
            thresholds.append(np.asarray(ts))
        require(all(np.array_equal(active_flags[0], a) for a in active_flags),
                f'Trained-head masks differ across paired methods/seeds in fold {fold}')
        for record in np.unique(meta['record']):
            ix = meta['record'] == record
            records.append({'fold': fold, 'record': int(record),
                            'y': meta['y'][ix], 'mask': meta['mask'][ix],
                            'time': meta['time'][ix], 'near': meta['near'][ix],
                            'a': scores[0][:, ix], 'b': scores[1][:, ix],
                            'ta': thresholds[0], 'tb': thresholds[1],
                            'trained': active_flags[0],
                            'present': [str(c) in mapping[str(record)] for c in range(11)]})
    return records


def _blocks(record):
    t = np.asarray(record['time'])
    require(len(t) > 0 and np.all(np.diff(t) > 0), 'Non-increasing endpoint time')
    _, inverse = np.unique((t - t[0]) // BLOCK_US, return_inverse=True)
    return inverse, int(inverse.max() + 1)


def _prepare_class(record, c, inverse, k):
    ok = np.asarray(record['mask'][:, c], bool)
    y = np.asarray(record['y'][ok, c], np.int64)
    inv = inverse[ok]
    require(np.isin(y, (0, 1)).all(), 'Nonbinary known labels')
    on = np.bincount(inv[y == 1], minlength=k).astype(np.int64)
    off = np.bincount(inv[y == 0], minlength=k).astype(np.int64)
    heads = {}
    for tag in ('a', 'b'):
        heads[tag] = []
        for si in range(len(record[tag])):
            z = record[tag][si, ok, c]
            threshold = record['t' + tag][si, c]
            needs_ap = bool(record['present'][c] and on.sum() and off.sum())
            order = np.argsort(-z, kind='stable') if needs_ap else np.empty(0, np.int64)
            zz = z[order]
            ends = np.r_[zz[1:] != zz[:-1], True] if len(zz) else np.empty(0, bool)
            # A fixed score order, including every tie. Integer prefix counts per
            # temporal block give the exact same weighted cumulative counts.
            prefix = None
            if needs_ap and len(z) * k <= 2_000_000:
                prefix = np.empty((2, int(ends.sum()), k), dtype=np.float64)
                ordered_inv, ordered_y = inv[order], y[order]
                for block in range(k):
                    in_block = ordered_inv == block
                    prefix[0, :, block] = np.cumsum(in_block, dtype=np.int64)[ends]
                    prefix[1, :, block] = np.cumsum(in_block * ordered_y, dtype=np.int64)[ends]
            hits = z >= threshold
            hit_on = np.bincount(inv[hits & (y == 1)], minlength=k).astype(np.int64)
            hit_off = np.bincount(inv[hits & (y == 0)], minlength=k).astype(np.int64)
            heads[tag].append({'prefix': prefix, 'inverse': inv[order], 'y': y[order],
                               'ends': ends, 'hit_on': hit_on, 'hit_off': hit_off,
                               'threshold_finite': bool(np.isfinite(threshold))})
    require(len(y) * max(k, 1) < 2**53,
            'Bootstrap integer prefix totals would exceed float64 exactness')
    return {'on': on, 'off': off, 'heads': heads,
            'present': bool(record['present'][c]), 'n': len(y),
            'has_on': bool(on.sum()), 'has_off': bool(off.sum())}


def _class_batch(specification, counts):
    n = len(counts)
    on = counts @ specification['on']
    off = counts @ specification['off']
    valid = np.ones(n, dtype=bool)
    if specification['has_on']:
        valid &= on > 0
    if specification['has_off']:
        valid &= off > 0
    values = []
    for tag in ('a', 'b'):
        seeds = []
        for head in specification['heads'][tag]:
            ap = np.full(n, np.nan)
            if specification['present'] and specification['has_on'] and specification['has_off']:
                if head['prefix'] is not None:
                    total = counts @ head['prefix'][0].T
                    tp = counts @ head['prefix'][1].T
                else:
                    weights = counts[:, head['inverse']]
                    total = np.cumsum(weights, axis=1, dtype=np.int64)[:, head['ends']]
                    tp = np.cumsum(weights * head['y'], axis=1, dtype=np.int64)[:, head['ends']]
                allowed = (tp[:, -1] > 0) & (total[:, -1] > tp[:, -1])
                numerator = np.diff(np.c_[np.zeros(n), tp], axis=1) * tp / np.maximum(total, 1)
                ap[allowed] = np.sum(numerator[allowed], axis=1) / tp[allowed, -1]
                if specification['has_on'] and specification['has_off']:
                    valid &= np.isfinite(ap)
            if not head['threshold_finite']:
                valid[:] = False
            hit_on, hit_off = counts @ head['hit_on'], counts @ head['hit_off']
            rec = np.divide(hit_on, on, out=np.full(n, np.nan), where=on > 0)
            if not specification['present']:
                rec[:] = np.nan
            fpr = np.divide(hit_off, off, out=np.full(n, np.nan), where=off > 0)
            seeds.append(np.stack((ap, rec, fpr), axis=1))
        values.append(np.mean(np.stack(seeds, axis=1), axis=1))
    return np.stack(values, axis=1), valid


def block_bootstrap_pair_fast(records, iterations=ITERATIONS, seed=BOOTSTRAP_SEED, batch_size=32, progress=None):
    """Original paired bootstrap; all draw identities and weighting are retained.

    Vectorized prefixes replace repeated sorting and weighted cumsums. Exact
    integer counts are represented in float64 only below its exact-integer bound.
    Quantile method remains NumPy's default linear interpolation.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    prefix_bytes = 0
    for r in records:
        inv, k = _blocks(r)
        specs = {}
        for c in range(PRIMARY):
            if not r['trained'][c] or not r['mask'][:, c].any():
                continue
            spec = _prepare_class(r, c, inv, k)
            for heads in spec['heads'].values():
                for head in heads:
                    if head['prefix'] is None:
                        continue
                    size = head['prefix'].nbytes
                    if prefix_bytes + size <= PREFIX_MEMORY_BUDGET:
                        prefix_bytes += size
                    else:
                        head['prefix'] = None
            specs[c] = spec
        prepared.append((r, k, specs))
    accepted, draws, invalid, attempts = [], [], 0, 0
    next_progress = 256
    while len(accepted) < iterations and attempts < iterations * 20:
        batch = min(batch_size, iterations * 20 - attempts)
        all_counts = [np.empty((batch, k), dtype=np.int64) for _, k, _ in prepared]
        batch_draws = []
        # Exact original RNG order: attempt -> record, sharing each draw across
        # methods, classes, and all three seeds.
        for bi in range(batch):
            draw = {}
            for ri, (_, k, _) in enumerate(prepared):
                counts = np.bincount(rng.integers(k, size=k), minlength=k)
                all_counts[ri][bi] = counts
                draw[str(ri)] = counts.tolist()
            batch_draws.append(draw)
        per_fold = {}
        valid = np.ones(batch, dtype=bool)
        for ri, (r, _, specs) in enumerate(prepared):
            fold = per_fold.setdefault(r['fold'], {c: [] for c in range(PRIMARY)})
            for c, spec in specs.items():
                value, okay = _class_batch(spec, all_counts[ri])
                fold[c].append(value)
                valid &= okay
        folds = []
        for classes in per_fold.values():
            cs = []
            for vals in classes.values():
                if not vals:
                    cs.append(np.full((batch, 3), np.nan))
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    v = np.nanmean(np.stack(vals, axis=1), axis=1)
                cs.append(v[:, 1] - v[:, 0])
            folds.append(np.mean(np.stack(cs, axis=1), axis=1))
        delta = np.mean(np.stack(folds, axis=1), axis=1)
        valid &= np.isfinite(delta).all(axis=1)
        for bi in range(batch):
            attempts += 1
            if valid[bi]:
                accepted.append(delta[bi])
                draws.append(batch_draws[bi])
                if len(accepted) == iterations:
                    break
            else:
                invalid += 1
        if progress is not None and len(accepted) >= next_progress:
            progress(len(accepted), invalid)
            next_progress += 256
    values = np.asarray(accepted)
    return {'metrics': ['AP', 'Recall', 'OFF_FPR'], 'valid': len(values),
            'invalid': invalid, 'complete': len(values) == iterations,
            'CI': np.quantile(values, [.025, .975], axis=0).tolist() if len(values) else None,
            'samples': values, 'draws': draws}


def absent_bootstrap_fast(records, iterations=ITERATIONS, seed=BOOTSTRAP_SEED, batch_size=128):
    rng = np.random.default_rng(seed)
    prepared, points, class_points, class_samples = [], [], {}, {}
    for r in records:
        inv, k = _blocks(r)
        specs, values = {}, []
        for c in range(11):
            if (r['present'][c] or not r['trained'][c] or
                    not np.isfinite(r['ta'][:, c]).all() or not np.isfinite(r['tb'][:, c]).all()):
                continue
            off = r['mask'][:, c].astype(bool) & (r['y'][:, c] == 0)
            key = f"{r['fold']}_{c}"
            on = []
            for tag in ('a', 'b'):
                hits = r[tag][:, :, c] >= r['t' + tag][:, c, None]
                on.append(np.stack([np.bincount(inv[off & hit], minlength=k) for hit in hits]))
            specs[key] = {'off': np.bincount(inv[off], minlength=k), 'a': on[0], 'b': on[1]}
            # The original draw becomes invalid when any eligible absent head
            # has no known OFF support. Never silently drop that head.
            if not off.any():
                continue
            point = float(((r['b'][:, off, c] >= r['tb'][:, c, None]).mean(1) -
                           (r['a'][:, off, c] >= r['ta'][:, c, None]).mean(1)).mean())
            values.append(point)
            class_points[key] = point
            class_samples[key] = []
        prepared.append((r, k, specs))
        points.append(float(np.mean(values)) if values else np.nan)
    samples, draws, attempts = [], [], 0
    while len(samples) < iterations and attempts < iterations * 20:
        batch = min(batch_size, iterations * 20 - attempts)
        counts = [np.empty((batch, k), dtype=np.int64) for _, k, _ in prepared]
        batch_draws = []
        for bi in range(batch):
            draw = []
            for ri, (_, k, _) in enumerate(prepared):
                value = np.bincount(rng.integers(k, size=k), minlength=k)
                counts[ri][bi] = value
                draw.append(value.tolist())
            batch_draws.append(draw)
        folds, class_draw = [], {}
        for ri, (_, _, specs) in enumerate(prepared):
            classes = []
            for key, spec in specs.items():
                denominator = counts[ri] @ spec['off']
                a = np.divide(counts[ri] @ spec['a'].T, denominator[:, None],
                              out=np.full((batch, len(spec['a'])), np.nan), where=denominator[:, None] > 0)
                b = np.divide(counts[ri] @ spec['b'].T, denominator[:, None],
                              out=np.full((batch, len(spec['b'])), np.nan), where=denominator[:, None] > 0)
                value = np.mean(b - a, axis=1)
                classes.append(value)
                class_draw[key] = value
            folds.append(np.mean(np.stack(classes, axis=1), axis=1) if classes else np.full(batch, np.nan))
        fold_values = np.stack(folds, axis=1)
        valid = np.isfinite(fold_values).all(axis=1) & (set(class_draw) == set(class_samples))
        values = np.mean(fold_values, axis=1)
        for bi in range(batch):
            attempts += 1
            if valid[bi]:
                samples.append(float(values[bi]))
                draws.append(batch_draws[bi])
                for key, value in class_draw.items():
                    class_samples[key].append(float(value[bi]))
                if len(samples) == iterations:
                    break
    return {'point': float(np.mean(points)), 'per_class_point': class_points,
            'per_class_CI': {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in class_samples.items()},
            'valid': len(samples), 'invalid': attempts - len(samples),
            'CI': np.quantile(samples, [.025, .975]).tolist() if samples else None,
            'samples': samples, 'draws': draws}


def validate_local_completion(run):
    if ARTIFACT:return validate_compact_artifacts()
    run = Path(run)
    marker = read(run / 'local_readout_complete.json')
    require(marker.get('completed') is True and marker.get('completed_tasks', 0) >= 12,
            'All 12 frozen paired-comparison tasks must be complete before final analysis')
    require(marker.get('trained') is False and marker.get('thresholds_changed') is False,
            'Training/threshold changes invalidate frozen local analysis')
    require(marker.get('backend') in ('cpu', 'cuda') and marker.get('original_npu_status') == 'partial',
            'Local analysis cannot certify completion of original NPU execution')
    require(set(marker.get('parts', [])) == {'cross', 'third'}, 'Missing readout part')
    task_ids = marker.get('task_ids', [])
    require(len(task_ids) == marker['completed_tasks'] and len(set(task_ids)) == len(task_ids) and
            all('-pretrain' not in t for t in task_ids), 'Invalid supervised task manifest')
    require(file_hash(run / 'selection.json') == file_hash(ARCHIVE / 'selection.json'),
            'Frozen original candidate selection changed')
    candidate = read(run / 'selection.json')['candidate']
    require(candidate['net'] == 'N1' and candidate['rep'] == 'F', 'Original frozen candidate is not N1-F')
    manifest_path = marker.get('task_manifest_path')
    require(bool(manifest_path) and bool(marker.get('task_manifest_sha256')), 'Task manifest binding is required')
    if manifest_path:
        path = (run / manifest_path).resolve()
        require(path.is_relative_to(run.resolve()), 'Task manifest must stay inside local run')
        require(file_hash(path) == marker.get('task_manifest_sha256'), 'Task manifest SHA-256 mismatch')
        manifest = read(path)
        manifest_rows = {row['task']['id']: row for row in manifest['tasks']}
        require(len(manifest['tasks']) == len(task_ids) and set(manifest_rows) == set(task_ids),
                'Marker task IDs do not match frozen supervised task manifest')
        require(manifest['selection_sha256'] == file_hash(run / 'selection.json'),
                'Manifest selection binding changed')
        require(manifest['freeze_sha256'] == file_hash(run / 'freeze.json') and
                file_hash(run / 'freeze.json') == file_hash(ARCHIVE / 'freeze.json'),
                'Original frozen model identity binding changed')
        frozen = read(run / 'freeze.json')
        require(set(task_ids) <= set(frozen['models']), 'Local task IDs differ from original frozen models')
        require(digest(frozen['candidate']) == digest(candidate), 'Frozen candidate identity changed')
        require(all(row['model'] == frozen['models'][tid]['model'] for tid, row in manifest_rows.items()),
                'Manifest model identity differs from original frozen model')
        require(all(row['task'] == read(ARCHIVE / 'tasks' / tid / 'done.json')['task']
                    for tid, row in manifest_rows.items()),
                'Task architecture/fold/seed differs from original completed training record')

    require(all(task_id(n, rep, f, seed) in task_ids
                for n, rep in [('N0', 'W1'), ('N1', 'F')]
                for f in ('A', 'B') for seed in SEEDS), 'Paired comparison task missing')
    for tid in task_ids:
        local = run / 'tasks' / tid
        require(file_hash(local / 'thresholds.json') == file_hash(ARCHIVE / 'tasks' / tid / 'thresholds.json'),
                f'Frozen threshold bytes changed: {tid}')
        require(digest(read(local / 'thresholds.json')['logit']) == frozen['models'][tid]['thresholds'],
                f'Threshold vector differs from original frozen manifest: {tid}')
        fold = '-A-' if '-A-' in tid else '-B-'
        fold = fold[1]
        for part in ('cross', 'third'):
            prediction = np.load(local / f'{part}.npy', mmap_mode='r', allow_pickle=False)
            require(prediction.shape == (len(_pairs(run, fold, part)), 11) and np.isfinite(prediction).all(),
                    f'Incomplete or nonfinite local prediction: {tid}/{part}')
            metrics = read(local / f'{part}_metrics.json')
            require(isinstance(metrics.get('rows'), list) and isinstance(metrics.get('summary'), dict),
                    f'Missing task metric evidence: {tid}/{part}')
            row = manifest_rows[tid]
            audit = read(local / f'{part}_local_audit.json')
            require(audit.get('completed') is True and audit.get('backend') == marker['backend'] and
                    audit.get('trained') is False and audit.get('thresholds_changed') is False and
                    audit.get('original_npu_status') == 'partial',
                    f'Not a completed local frozen-weight receipt: {tid}/{part}')
            require(audit.get('task') == tid and audit.get('part') == part and
                    audit.get('samples') == len(prediction) and prediction.dtype == np.float32,
                    f'Local task/part/dtype mismatch: {tid}/{part}')
            recovery_sha = audit['input_recovery_sha256']
            require(isinstance(recovery_sha, str) and len(recovery_sha) == 64 and
                    all(c in '0123456789abcdef' for c in recovery_sha),
                    f'Invalid input recovery receipt identity: {tid}/{part}')
            snapshot_path = run / 'audit' / 'input_receipts' / f'{recovery_sha}.json'
            require(file_hash(snapshot_path) == recovery_sha, f'Input recovery snapshot changed: {tid}/{part}')
            snapshot = read(snapshot_path)
            require(snapshot.get('frozen_cache_identity_preserved') is True and
                    row['task']['rep'] in snapshot.get('verified_representations', []),
                    f'Input representation not accepted: {tid}/{part}')
            checks = audit['reference_checks']
            require({c['record'] for c in checks} == set(_pairs(run, fold, part)[:, 0].tolist()) and
                    all(c.get('passed') is True and c.get('full_window_reference') is True and
                        c.get('independent_initial_state') is True and c.get('rtol') == 2e-5 and
                        c.get('atol') == 2e-5 for c in checks),
                    f'Local full-window inference equivalence checks incomplete: {tid}/{part}')
            require(audit['model'] == row['model'] and audit['cpu_weight_sha256'] == row['cpu_weight_sha256'] and
                    audit['thresholds_sha256'] == file_hash(local / 'thresholds.json') == row['thresholds_sha256'],
                    f'Frozen model/threshold provenance mismatch: {tid}/{part}')
            require(audit['prediction'] == array_hash(prediction) and audit['metrics'] == digest(metrics['rows']) and
                    audit['pairs'] == array_hash(_pairs(run, fold, part)),
                    f'Prediction/metric/endpoint receipt mismatch: {tid}/{part}')
            summary = summarize([v for v in metrics['rows'] if v.get('region') == 'main'])
            require(digest(summary) == digest(metrics['summary']),
                    f'Task summary no longer matches per-class metric rows: {tid}/{part}')

    return marker, candidate


def _decision(run, candidate, boot, absent):
    points, perclass, steady, seedchecks = [], [], [], {}
    absolute = {'N0-W1': [], 'N1-F': []}
    for fold in ('A', 'B'):
        seedchecks[fold] = 0
        for seed in SEEDS:
            metrics = [read(_task_root(run,task_id(n,r,fold,seed)) / 'cross_metrics.json')
                       for n, r in [('N0', 'W1'), (candidate['net'], candidate['rep'])]]
            a, b = [x['summary'] for x in metrics]
            absolute['N0-W1'].append(a)
            absolute['N1-F'].append(b)
            require(all(a[k] is not None and b[k] is not None for k in ('AP', 'Recall', 'OFF_FPR')),
                    'Cross-script primary metric is undefined')
            points.append([b[k] - a[k] for k in ('AP', 'Recall', 'OFF_FPR')])
            seedchecks[fold] += b['AP'] > a['AP']
            perclass.append(np.array(b['Recall_classes']) - np.array(a['Recall_classes']))
            ss = [summarize([v for v in x['rows'] if v.get('region') == 'steady']) for x in metrics]
            steady.append(ss[1]['AP'] - ss[0]['AP'] if all(x['AP'] is not None for x in ss) else np.nan)
    delta, classdelta = np.mean(points, 0), np.mean(perclass, 0)
    ci = np.array(boot['CI']) if boot['CI'] else np.full((2, 3), np.nan)
    criteria = {
        '跨脚本AP提升及CI': bool(delta[0] >= .01 and ci[0, 0] > 0),
        'Recall非劣': bool(delta[1] >= -.05 and ci[0, 1] >= -.05),
        'OFF误报保护': bool(delta[2] <= .02 and ci[1, 2] <= .02),
        '逐类别Recall保护': bool(classdelta.min() >= -.10),
        '两折seed一致性': all(v >= 2 for v in seedchecks.values()),
        '第三脚本缺席类别误报': bool(absent['valid'] == ITERATIONS and absent['per_class_point'] and
                                  all(v <= .02 and absent['per_class_CI'][k][1] <= .02
                                      for k, v in absent['per_class_point'].items())),
        '稳态方向': bool(np.isfinite(steady).all() and np.mean(steady) >= 0),
        'Bootstrap有效数量': boot['complete']}
    conclusion = ('值得扩大复验（仅限TDHA受控记录上的探索性证据）' if all(criteria.values()) else
                  '未达到预先冻结的扩大复验判据')
    method_summary = {name: {key: float(np.mean([row[key] for row in values]))
                              if all(row.get(key) is not None for row in values) else None
                              for key in ('AP', 'Recall', 'OFF_FPR', 'F1')}
                      for name, values in absolute.items()}
    return {'candidate': candidate, 'paired_method_summary': method_summary,
            'delta': dict(zip(('AP', 'Recall', 'OFF_FPR'), delta)),
            'CI': boot['CI'], 'class_recall_delta': classdelta,
            'seed_AP_improvements': seedchecks, 'steady_AP_delta': float(np.mean(steady)),
            'criteria': criteria, 'conclusion': conclusion,
            'evaluation_scope': 'Local frozen-weight CPU/CUDA supplemental evaluation; not NPU validation',
            'candidate_reselected': False, 'thresholds_changed': False}


def _write_reports(run, marker, candidate, out):
    fields = ['task_id', 'fold', 'seed', 'part', 'backend', 'AP', 'Recall', 'OFF_FPR', 'F1', 'Precision']
    destination = WORK / 'reports' / 'task_summary_local.csv'
    rows = []
    for tid in marker['task_ids']:
        for part in ('cross', 'third'):
            summary = read(_task_root(run,tid) / f'{part}_metrics.json')['summary']
            seed = tid.split('-s')[-1].split('-')[0]
            rows.append({'task_id': tid, 'fold': 'A' if '-A-' in tid else 'B', 'seed': seed,
                         'part': part, 'backend': marker['backend'],
                         **{k: summary.get(k) for k in fields[5:]}})
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(destination, stream.getvalue())
    lines = ['# TDHA V3 本地冻结模型补充评测', '',
             f"推理后端：**{marker['backend'].upper()}**。NPU训练与时间评测已有结果；原NPU执行记录为partial。跨脚本和第三记录的结果来自独立CUDA/CPU补充评测。", '',
             '## 冻结比较', '',
             '**N1-F 对照 N0-W1**：种子11/42/73、A/B两折，使用各任务的固定校准阈值。', '',
             '| 指标 | 候选减基线 | 95% 配对块 Bootstrap CI |',
             '|---|---:|---:|']
    for i, name in enumerate(('AP', 'Recall', 'OFF_FPR')):
        lines.append(f"| {name} | {float(out['delta'][name]):.6f} | [{out['CI'][0][i]:.6f}, {out['CI'][1][i]:.6f}] |")
    lines += ['', '两折三种子等权平均的绝对指标：', '',
              '| 冻结方案 | AP | Recall | OFF FPR | F1 |', '|---|---:|---:|---:|---:|']
    for name, values in out['paired_method_summary'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{values[k]:.6f}" if values[k] is not None else 'null' for k in ('AP', 'Recall', 'OFF_FPR', 'F1')) + ' |')

    lines += ['', f"结论：**{out['conclusion']}**", '',
              '## 原冻结统计判据', '', '| 判据 | 结果 |', '|---|---|']
    lines += [f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in out['criteria'].items()]
    lines += ['', '## 解释范围', '',
              '主 AP/Recall 限于共同训练支持的前六类；缺席类别不纳入 AP/Recall，OFF FPR 可纳入缺席。',
              '第三记录采用全部已训练、有冻结阈值的缺席类别逐类误报差值及区间；总体缺少类别支持的指标保持 null。',
              'CUDA/CPU与NPU的浮点环境不同，补充评测与NPU预测分别报告。',
              '完整逐任务汇总见 [task_summary_local.csv](task_summary_local.csv)，统计工件见 `../analysis/`。', '']
    _atomic_text(WORK / 'reports' / 'final_statistics.md', '\n'.join(lines))
    names = ['Cross-script AP improvement and CI', 'Recall noninferiority',
             'OFF false-positive protection', 'Per-class recall protection',
             'AP seed consistency in both folds', 'Third-record absent-class false positives',
             'Steady-state AP direction', 'Valid bootstrap draw count']
    english = ['# TDHA V3 local frozen-weight supplemental evaluation', '',
               f"Inference backend: **{marker['backend'].upper()}**. NPU training and within-script time results are available; the NPU execution record is partial. Cross-script and third-record results use separate CUDA/CPU supplemental evaluation.", '',
               '**Frozen comparison: N1-F versus N0-W1**, seeds 11/42/73, folds A/B, with the original per-task calibration thresholds.', '',
               '| Metric | Candidate minus baseline | 95% paired-block bootstrap CI |',
               '|---|---:|---:|']
    for i, name in enumerate(('AP', 'Recall', 'OFF_FPR')):
        english.append(f"| {name} | {float(out['delta'][name]):.6f} | [{out['CI'][0][i]:.6f}, {out['CI'][1][i]:.6f}] |")
    english += ['', 'Absolute metrics, equally averaged over both folds and all three seeds:', '',
                '| Frozen method | AP | Recall | OFF FPR | F1 |', '|---|---:|---:|---:|---:|']
    for name, values in out['paired_method_summary'].items():
        english.append('| ' + name + ' | ' + ' | '.join(f"{values[k]:.6f}" if values[k] is not None else 'null' for k in ('AP', 'Recall', 'OFF_FPR', 'F1')) + ' |')

    english += ['', '**Conclusion:** ' + ('Supports a larger exploratory replication on controlled TDHA recordings.' if all(out['criteria'].values()) else 'Does not meet the predefined criteria for a larger replication.'), '',
                '| Frozen statistical criterion | Result |', '|---|---|']
    english += [f"| {name} | {'Pass' if passed else 'Fail'} |" for name, passed in zip(names, out['criteria'].values())]
    english += ['', 'Primary AP/Recall require the six shared supported classes. Absent classes do not enter AP/Recall, but can enter OFF FPR.',
                'Third-record analysis includes every eligible trained absent class with frozen thresholds. Unsupported aggregate AP/Recall values remain null.',
                'CUDA/CPU evaluation uses the fixed checkpoints and per-task thresholds; its outputs are reported separately from NPU predictions.',
                'See [task_summary_local.csv](task_summary_local.csv), and `../analysis/` for complete evidence.', '']
    _atomic_text(WORK / 'reports' / 'final_statistics.en.md', '\n'.join(english))



def analyze(run):
    run = Path(run).resolve()
    require(WORK is not None, 'Explicit separate output directory required')
    started = time.monotonic()
    marker, candidate = validate_local_completion(run)
    records = comparison_records(run, candidate)
    print('Paired cross-script bootstrap started: 2,000 valid draws', flush=True)
    boot = block_bootstrap_pair_fast(records, progress=lambda valid, invalid: print(f'Bootstrap valid={valid}, invalid={invalid}', flush=True))
    require(boot['complete'] and boot['valid'] == ITERATIONS, 'Insufficient valid paired bootstrap draws')
    save_array(WORK / 'analysis' / 'bootstrap.npy', boot.pop('samples'))
    dump(WORK / 'analysis' / 'bootstrap_indices.json', boot.pop('draws'))
    dump(WORK / 'analysis' / 'bootstrap.json', boot)
    third_records = comparison_records(run, candidate, 'third')
    print('Third-record absent-class bootstrap started: 2,000 valid draws', flush=True)
    absent = absent_bootstrap_fast(third_records)
    require(absent['valid'] == ITERATIONS, 'Insufficient absent-class bootstrap draws')
    save_array(WORK / 'analysis' / 'absent_bootstrap.npy', np.asarray(absent.pop('samples')))
    dump(WORK / 'analysis' / 'absent_indices.json', absent.pop('draws'))
    dump(WORK / 'analysis' / 'absent.json', absent)
    out = _decision(run, candidate, boot, absent)
    dump(WORK / 'analysis' / 'decision.json', out)
    _write_reports(run, marker, candidate, out)
    evidence = {'schema_version': 1, 'completed': True, 'backend': marker['backend'],
                'scope': 'local supplemental readout and frozen statistical analysis',
                'original_npu_status': 'partial', 'trained': False, 'thresholds_changed': False,
                'candidate_reselected': False, 'bootstrap_iterations': ITERATIONS,
                'bootstrap_block_us': BLOCK_US, 'bootstrap_seed': BOOTSTRAP_SEED,
                'bootstrap_valid': boot['valid'], 'absent_valid': absent['valid'],
                'elapsed_seconds': time.monotonic() - started,
                'adapter_sha256': file_hash(Path(__file__)),
                'source_evidence_sha256': file_hash(ARTIFACT/'manifest.json' if ARTIFACT else run/'local_readout_complete.json'),
                'decision_sha256': file_hash(WORK / 'analysis' / 'decision.json')}
    dump(WORK / 'audit' / 'analysis_completion.json', evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return out


def main(argv=None):
    global WORK, ARCHIVE, INPUTS, ARTIFACT
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--artifact-root',type=Path,help='Unpacked results/tdha-npu case with manifest, predictions and compact metadata')
    mode.add_argument('--run',type=Path,help='Alternatively, a read-only original completed local inference run')
    parser.add_argument('--archive',type=Path,help='Original mode only: read-only frozen NPU archive')
    parser.add_argument('--inputs',type=Path,help='Original mode only: read-only prepared input run')
    parser.add_argument('--output',type=Path,required=True,help='Separate statistics output directory')
    execution=parser.add_mutually_exclusive_group()
    execution.add_argument('--execute',action='store_true',help='Explicitly calculate frozen statistics; default validates only')
    execution.add_argument('--plan-only',action='store_true',help='Read-only validation; also the default')
    args=parser.parse_args(argv)
    if args.artifact_root:
        if args.archive or args.inputs:parser.error('--artifact-root does not use --archive or --inputs')
        ARTIFACT=args.artifact_root.resolve();ARCHIVE=INPUTS=None;run=ARTIFACT
        WORK=output_separate(args.output,ARTIFACT,PACKAGE)
    else:
        if not args.archive or not args.inputs:parser.error('--run requires --archive and --inputs')
        ARTIFACT=None;ARCHIVE=args.archive.resolve();INPUTS=args.inputs.resolve();run=args.run.resolve()
        WORK=output_separate(args.output,run,ARCHIVE,INPUTS,PACKAGE)
    if not args.execute:
        marker,_=validate_local_completion(run)
        print(json.dumps({'valid':True,'tasks':marker['completed_tasks'],'writes':False}))
        return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):analyze(run)


if __name__=='__main__':
    main()
