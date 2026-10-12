"""Frozen local inference only. No fit, calibration, CLI worker or NPU calls."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TDHA_RAM_CACHE_GB', '0')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
import hashlib
import json
import shutil
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from paths import output_separate, task_path
W = A = RUN = INPUTS = WEIGHTS = None
from tdha_v3.common import read, dump, require, digest, array_hash, save_array
from tdha_v3.models import Model
from tdha_v3.training import Windows as OriginalWindows, state_hash, pairs_for, evaluate_prediction
from tdha_v3.memory import host_batch, to_device, cache
from tdha_v3.metrics import summarize


class Windows(OriginalWindows):
    """Original windows with explicit read-only inputs; never map recovery temp files."""
    def __init__(self, run, *args, **kwargs):
        super().__init__(INPUTS, *args, **kwargs)

    def arrays(self, r):
        if r not in self.records:
            names = ['time', 'y', 'known', 'strict', 'near', 'basic']
            if self.rep == 'W1':
                names.append('fm')
            if self.rep == 'DVI':
                names += ['dvi', 'dvi_valid']
            self.records[r] = {name: np.load(self.record_path(r) / f'{name}.npy',
                                           mmap_mode='r', allow_pickle=False) for name in names}
            self.features[r] = {tag: cache().get(self.root / f'r{r}_{tag}.npy')
                                for tag in (('power', 'F') if self.rep == 'F' else ('power',))}
        return self.records[r], self.features[r]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def log(event, **fields):
    row = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    (W / 'logs').mkdir(parents=True, exist_ok=True)
    with (W / 'logs' / 'local_readout.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def setup(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == 'cuda':
        require(torch.cuda.is_available(), 'Local CUDA unavailable; no automatic backend substitution')
    return torch.device(device)


def plan(task_ids, write=False):
    freeze = read(A / 'freeze.json')
    require(bool(task_ids) and len(task_ids)==len(set(task_ids)), 'Explicit unique task IDs required')
    require(set(task_ids) <= set(freeze['models']), 'Unknown frozen task ID')
    rows = []
    for tid in sorted(task_ids):
        task_path(A / 'tasks', tid)
        task_path(RUN / 'tasks', tid)
        done = read(A / 'tasks' / tid / 'done.json')
        task = done['task']
        require(task['id'] == tid, 'Task identity mismatch')
        require(done['hashes']['model'] == freeze['models'][tid]['model'], 'Model not frozen')
        thpath = A / 'tasks' / tid / 'thresholds.json'
        th = read(thpath)
        require(digest(th['logit']) == freeze['models'][tid]['thresholds'], 'Thresholds not frozen')
        dest = RUN / 'tasks' / tid
        if write:
            dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'thresholds.json').exists():
            require(sha(dest / 'thresholds.json') == sha(thpath), 'Existing local thresholds changed')
        elif write:
            shutil.copy2(thpath, dest / 'thresholds.json')
        rows.append(dict(task=task, model=freeze['models'][tid]['model'],
                         thresholds_sha256=sha(thpath),
                         source_best_sha256=sha(A / 'tasks' / tid / 'best.pt'),
                         cpu_weight_sha256=sha(task_path(WEIGHTS, tid) / 'best.cpu.pt'),
                         source_readout_parts=[p for p in ('cross', 'third')
                                               if (A / 'tasks' / tid / f'{p}.npy').exists()]))
    manifest = dict(schema=1, tasks=rows, freeze_sha256=sha(A / 'freeze.json'),
                    selection_sha256=sha(A / 'selection.json'),
                    trained=False, thresholds_changed=False, original_npu_status='partial')
    path = RUN / 'task_manifest.json'
    if path.exists():
        require(read(path) == manifest, 'Local task manifest identity changed')
    elif write:
        dump(path, manifest)
    for name in ('freeze.json', 'selection.json'):
        if (RUN / name).exists():
            require(sha(RUN / name) == sha(A / name), f'{name} changed')
        elif write:
            shutil.copy2(A / name, RUN / name)
    return manifest


def cache_ready(rep):
    status = INPUTS / 'audit' / 'input_recovery.json'
    # Numerical acceptance is provided by the input recovery tool, never by this runner.
    require(status.exists(), 'Missing verified input recovery receipt')
    receipt = read(status)
    require(receipt.get('verified_representations') and rep in receipt['verified_representations'],
            f'Representation {rep} not yet accepted by input recovery')
    require(receipt.get('frozen_cache_identity_preserved') is True,
            'Original frozen input identity has not been verified')
    value = sha(status)
    snapshot = RUN / 'audit' / 'input_receipts' / f'{value}.json'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copy2(status, snapshot)
    require(sha(snapshot) == value, 'Input receipt snapshot changed')
    return value


def verify_part(row, part, backend):
    """Bind every completed artifact to the exact frozen task and endpoint order."""
    task = row['task']
    dest = RUN / 'tasks' / task['id']
    audit = read(dest / f'{part}_local_audit.json')
    pairs = pairs_for(INPUTS, task['fold'], part)
    require(audit.get('completed') is True and audit.get('trained') is False
            and audit.get('thresholds_changed') is False, 'Readout not complete and frozen')
    require(audit['task'] == task['id'] and audit['part'] == part
            and audit['backend'] == backend, 'Readout task/part/backend mismatch')
    require(audit['model'] == row['model']
            and audit['thresholds_sha256'] == row['thresholds_sha256']
            and audit['cpu_weight_sha256'] == row['cpu_weight_sha256'], 'Frozen identity mismatch')
    require(sha(dest / 'thresholds.json') == row['thresholds_sha256']
            and sha(A / 'tasks' / task['id'] / 'thresholds.json') == row['thresholds_sha256'],
            'Frozen threshold file changed')
    require(audit['pairs'] == array_hash(pairs) and audit['samples'] == len(pairs),
            'Eligible endpoints changed')
    receipt_path = RUN / 'audit' / 'input_receipts' / f"{audit['input_recovery_sha256']}.json"
    require(receipt_path.exists() and sha(receipt_path) == audit['input_recovery_sha256'],
            'Input recovery receipt missing/changed')
    receipt = read(receipt_path)
    require(receipt.get('frozen_cache_identity_preserved') is True
            and task['rep'] in receipt.get('verified_representations', []),
            'Required evaluation inputs not verified')
    checks = audit['reference_checks']
    require({c['record'] for c in checks} == set(pairs[:, 0].tolist())
            and all(c.get('passed') is True and c.get('full_window_reference') is True
                    and c.get('independent_initial_state') is True
                    and c.get('rtol') == 2e-5 and c.get('atol') == 2e-5 for c in checks),
            'Missing independent full-window verification')
    pred = np.load(dest / f'{part}.npy', mmap_mode='r', allow_pickle=False)
    require(pred.dtype == np.float32 and pred.shape == (len(pairs), 11)
            and np.isfinite(pred).all() and array_hash(pred) == audit['prediction'],
            'Prediction shape/dtype/content changed')
    metrics = read(dest / f'{part}_metrics.json')
    require(digest(metrics['rows']) == audit['metrics'], 'Metric content changed')
    summary = summarize([r for r in metrics['rows'] if r.get('region') == 'main'])
    require(digest(summary) == digest(metrics['summary']), 'Metric summary inconsistent')
    return dict(prediction=audit['prediction'], metrics=audit['metrics'])


def front_batch(ds, lo, hi, device):
    return to_device(host_batch(ds, range(lo, hi)), device)


def reference(model, ds, indices, device, micro, original_rnn=None):
    saved = None
    if original_rnn is not None:
        saved = model.backbone.rnn
        model.backbone.rnn = original_rnn
    try:
        parts = []
        with torch.inference_mode():
            for lo in range(0, len(indices), min(micro, 8)):
                ix = indices[lo:lo + min(micro, 8)]
                batch = to_device(host_batch(ds, ix), device)
                parts.append(model(batch).cpu().numpy().astype(np.float32))
        return np.concatenate(parts) if parts else np.empty((0, 11), np.float32)
    finally:
        if saved is not None:
            model.backbone.rnn = saved


def predict_local(model, ds, device, micro, front_micro, original_rnn=None):
    require(len(ds) > 0, 'No eligible held-out endpoints')
    model.eval()
    out = np.empty((len(ds), 11), np.float32)
    checks = []
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device.type == 'cuda' else None
    started = time.monotonic()
    with torch.inference_mode():
        for r in np.unique(ds.pairs[:, 0]):
            where = np.flatnonzero(ds.pairs[:, 0] == r)
            ep = ds.pairs[where, 1] // 5
            first, last = int(ep.min() - 255), int(ep.max())
            single_pairs = np.column_stack((np.full(last - first + 1, r),
                                            np.arange(first, last + 1) * 5 + 4))
            single = Windows(ds.run, ds.fold, ds.rep, single_pairs, tokens=1)
            encoded = torch.empty((len(single), 128), dtype=torch.float32, device=device)
            for lo in range(0, len(single), front_micro):
                hi = min(lo + front_micro, len(single))
                encoded[lo:hi] = model.front(front_batch(single, lo, hi, device))[:, 0]
            require(bool(torch.isfinite(encoded).all()), 'Non-finite frontend embedding')
            offset = torch.arange(-255, 1, device=device, dtype=torch.long)
            for lo in range(0, len(ep), micro):
                end = torch.as_tensor(ep[lo:lo + micro] - first, device=device, dtype=torch.long)
                windows = encoded[end[:, None] + offset[None, :]]
                if model.backbone.name == 'N0' and device.type == 'cuda':
                    from gru_eval_acceleration import eval_backbone
                    z = eval_backbone(model, windows)
                else:
                    z = model.backbone(windows)
                out[where[lo:lo + len(end)]] = z.cpu().numpy()
                if lo == 0 or (lo // micro) % 100 == 0:
                    log('prediction_progress', fold=ds.fold, rep=ds.rep,
                        record=int(r), endpoints_completed=min(lo + micro, len(ep)),
                        endpoints_total=len(ep), elapsed_s=round(time.monotonic() - started, 2))
            # Independent full-window reference for first/last/quartiles in every record.
            probe = where[np.unique(np.linspace(0, len(where) - 1, min(9, len(where)), dtype=int))]
            ref = reference(model, ds, probe.tolist(), device, micro, original_rnn)
            error = np.abs(out[probe].astype(np.float64) - ref.astype(np.float64))
            passed = bool(np.allclose(out[probe], ref, rtol=2e-5, atol=2e-5))
            check = dict(record=int(r), indices=probe.tolist(),
                         full_window_reference=True, independent_initial_state=True,
                         rtol=2e-5, atol=2e-5, max_absolute_error=float(error.max()), passed=passed)
            checks.append(check)
            log('reference_check', **check)
            require(passed, 'Local accelerated inference failed original full-window reference tolerance')
            del encoded, single
    require(torch.equal(rng, torch.get_rng_state()), 'Inference changed CPU RNG')
    if cuda_rng is not None:
        require(torch.equal(cuda_rng, torch.cuda.get_rng_state()), 'Inference changed CUDA RNG')
    require(np.isfinite(out).all(), 'Non-finite predictions')
    return out, checks


def evaluate(model, ds, device, args, original_rnn):
    return predict_local(model, ds, device, args.micro, args.front_micro, original_rnn)


def reuse_identical_readout(row, part, backend, receipt_hash):
    """A replay with identical frozen model, thresholds and inputs needs no duplicate inference."""
    tid = row['task']['id']
    for other in read(RUN / 'task_manifest.json')['tasks']:
        if (other['task']['id'] == tid or other['model'] != row['model']
                or other['thresholds_sha256'] != row['thresholds_sha256']
                or any(other['task'][key] != row['task'][key] for key in ('net', 'rep', 'fold'))):
            continue
        source = RUN / 'tasks' / other['task']['id']
        if not (source / 'readout.json').exists():
            continue
        result = verify_part(other, part, backend)
        audit = read(source / f'{part}_local_audit.json')
        dest = RUN / 'tasks' / tid
        for name in (f'{part}.npy', f'{part}_metrics.json'):
            target = dest / name
            temporary = target.with_suffix(target.suffix + '.tmp')
            shutil.copy2(source / name, temporary)
            os.replace(temporary, target)
        audit.update(task=tid, cpu_weight_sha256=row['cpu_weight_sha256'],
                     input_recovery_sha256=receipt_hash, reused_from=other['task']['id'],
                     reuse_reason='Identical frozen parameter hash, thresholds, fold and representation',
                     elapsed_s=0.0)
        dump(dest / f'{part}_local_audit.json', audit)
        require(verify_part(row, part, backend) == result, 'Reused frozen readout failed validation')
        log('part_reused_identical', task=tid, part=part, source_task=other['task']['id'])
        return result
    return None


def run_task(row, device, args):
    task = row['task']
    tid, rep, fold = task['id'], task['rep'], task['fold']
    receipt_hash = cache_ready(rep)
    dest = RUN / 'tasks' / tid
    th = read(dest / 'thresholds.json')
    require(sha(dest / 'thresholds.json') == row['thresholds_sha256'], 'Threshold file changed')
    path = task_path(WEIGHTS, tid) / 'best.cpu.pt'
    require(sha(path) == row['cpu_weight_sha256'], 'Converted weight file changed')
    model = Model(task['net'], rep, task['seed'])
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'], strict=True)
    require(state_hash(model.state_dict()) == row['model'], 'Model differs from frozen checkpoint')
    model.to(device).eval()
    original_rnn = None
    log('task_start', task=tid, backend=device.type, trained=False, thresholds_changed=False)
    result = {}
    for part in ('cross', 'third'):
        ds = Windows(RUN, fold, rep, pairs_for(INPUTS, fold, part))
        audit_path = dest / f'{part}_local_audit.json'
        if audit_path.exists() and not args.validate_only:
            audit = read(audit_path)
            if (audit.get('completed') and audit.get('model') == row['model']
                    and audit.get('thresholds_sha256') == row['thresholds_sha256']
                    and audit.get('backend') == device.type
                    and audit.get('pairs') == array_hash(ds.pairs)):
                result[part] = verify_part(row, part, device.type)
                log('part_resume_verified', task=tid, part=part)
                continue
        if not args.validate_only:
            reused = reuse_identical_readout(row, part, device.type, receipt_hash)
            if reused is not None:
                result[part] = reused
                continue
        started = time.monotonic()
        if args.validate_only:
            idx = np.unique(np.concatenate([np.flatnonzero(ds.pairs[:, 0] == r)[
                np.unique(np.linspace(0, sum(ds.pairs[:, 0] == r)-1, 4, dtype=int))]
                for r in np.unique(ds.pairs[:, 0])]))
            ds = Windows(RUN, fold, rep, ds.pairs[idx])
        pred, checks = evaluate(model, ds, device, args, original_rnn)
        if args.validate_only:
            log('task_validation', task=tid, part=part, samples=len(ds), checks=checks)
            continue
        rows, summary = evaluate_prediction(INPUTS, fold, ds, pred,
                                            th['logit'], th['active'], th['high_power'])
        save_array(dest / f'{part}.npy', pred)
        dump(dest / f'{part}_metrics.json', dict(rows=rows, summary=summary))
        audit = dict(schema=1, completed=True, task=tid, part=part, backend=device.type,
                     device=torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
                     original_npu_status='partial', trained=False, thresholds_changed=False,
                     model=row['model'], thresholds_sha256=row['thresholds_sha256'],
                     cpu_weight_sha256=row['cpu_weight_sha256'], pairs=array_hash(ds.pairs),
                     input_recovery_sha256=receipt_hash, prediction=array_hash(pred),
                     metrics=digest(rows), samples=len(ds), reference_checks=checks,
                     gru_acceleration='cuda_graph_original_ops' if task['net']=='N0' and device.type=='cuda' else None,
                     microbatch=args.micro, frontend_batch=args.front_micro,
                     float_precision='float32',
                     tf32=False, deterministic=True, elapsed_s=time.monotonic() - started)
        dump(audit_path, audit)
        result[part] = dict(prediction=audit['prediction'], metrics=audit['metrics'])
        log('part_complete', task=tid, part=part, samples=len(ds),
            summary=summary, elapsed_s=round(audit['elapsed_s'], 2))
    if not args.validate_only:
        require(state_hash(model.state_dict()) == row['model'], 'Inference changed model parameters')
        dump(dest / 'readout.json', result)
    if task['net'] == 'N0' and device.type == 'cuda':
        from gru_eval_acceleration import clear_graph_cache
        clear_graph_cache(model)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def completion(manifest, device):
    completed = []
    for row in manifest['tasks']:
        tid = row['task']['id']
        dest = RUN / 'tasks' / tid
        if not (dest / 'readout.json').exists():
            return False
        for part in ('cross', 'third'):
            p = dest / f'{part}_local_audit.json'
            if not p.exists():
                return False
            result = verify_part(row, part, device.type)
            require(read(dest / 'readout.json')[part] == result, 'Readout index changed')
        completed.append(tid)
    marker = dict(schema=1, completed=True, task_ids=completed, completed_tasks=len(completed),
                  parts=['cross', 'third'], backend=device.type, original_npu_status='partial',
                  trained=False, thresholds_changed=False,
                  task_manifest_path='task_manifest.json',
                  task_manifest_sha256=sha(RUN / 'task_manifest.json'),
                  completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    dump(RUN / 'local_readout_complete.json', marker)
    log('all_readouts_complete', tasks=len(completed), backend=device.type)
    return True


def main(argv=None):
    global W, A, RUN, INPUTS, WEIGHTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Read-only frozen NPU task archive')
    parser.add_argument('--inputs', type=Path, required=True, help='Read-only prepared run containing cache/ and audit/input_recovery.json')
    parser.add_argument('--weights', type=Path, required=True, help='Read-only directory containing <task>/best.cpu.pt')
    parser.add_argument('--output', type=Path, required=True, help='Separate local output directory; creates run/, audit/, logs/')
    parser.add_argument('--tasks', nargs='+', required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true', help='Explicitly run frozen inference')
    modes.add_argument('--plan-only', action='store_true', help='Read-only plan (also the default)')
    parser.add_argument('--validate-only', action='store_true', help='With --execute, check a small sample instead of saving full predictions')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Required for --execute; never inferred or substituted')
    parser.add_argument('--micro', type=int, default=64)
    parser.add_argument('--front-micro', type=int, default=256)
    args=parser.parse_args(argv)
    require(args.micro>0 and args.front_micro>0, 'Batch size must be positive')
    if args.execute and args.device is None: parser.error('--execute requires --device cpu or cuda')
    if args.validate_only and not args.execute: parser.error('--validate-only requires --execute')
    A=args.archive.resolve();INPUTS=args.inputs.resolve();WEIGHTS=args.weights.resolve()
    W=output_separate(args.output,A,INPUTS,WEIGHTS,PACKAGE);RUN=W/'run'
    manifest=plan(args.tasks,write=False)
    if not args.execute:
        print(json.dumps(dict(event='plan_complete',tasks=len(manifest['tasks']),writes=False,trained=False)))
        return
    # Only explicit execution may create the selected-task manifest and copy immutable thresholds.
    manifest=plan(args.tasks,write=True)
    device=setup(args.device)
    dump(W/'audit/local_environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),backend=device.type,
        tf32=False,deterministic=True,trained=False,npu_verification=False))
    for row in manifest['tasks']:run_task(row,device,args)
    if not args.validate_only:completion(manifest,device)


if __name__=='__main__':
    main()
