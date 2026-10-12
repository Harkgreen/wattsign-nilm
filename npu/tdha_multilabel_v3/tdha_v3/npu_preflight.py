"""Data-free NPU acceptance. Each network gets a fresh NPU process.

Synthetic tensors test operators only; they never count as research results.
Full real-data capacity/performance checking still belongs to benchmark().
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import platform
from .common import ROOT, code_hash, dump, read, require


CASES = [(n, r, 'main') for n in ('N0', 'N1', 'N2') for r in ('P', 'F', 'W1', 'DVI')
         if r != 'P' or n == 'N0'] + [('front', r, 'pretrain') for r in ('F', 'W1', 'DVI')]


def record_skip(run,cards):
    "Record an explicit request to skip this check; retain the last actual result and never fabricate a passed status."
    require(1<=len(cards)<=3 and len(set(cards))==len(cards),'Specify one to three distinct NPU cards')
    root=Path(run);previous=root/'operator_preflight.json'
    # Historical audit reason/scope values are retained; console messages use English.
    receipt={'skipped':True,'passed':None,'complete':False,'code':code_hash(),
             'cards':cards,'time_ns':time.time_ns(),'python':sys.version,'machine':platform.machine(),
             'reason':'启动命令显式使用--skip-preflight；未在本次执行NPU算子测试',
             'previous':read(previous) if previous.exists() else None}
    dump(root/'audit/operator_preflight_skips'/f'{receipt["time_ns"]}.json',receipt)
    print('NPU operator checks skipped with --skip-preflight; dependency, data, capacity and reproducibility checks remain enabled.',flush=True)


def batch_for(rep, batch_size=2, tokens=256, missing=False):
    import torch
    g = torch.Generator().manual_seed(91023)
    b, t = batch_size, tokens
    batch = {'power': torch.randn(b, t, 108, generator=g),
             'y': torch.randint(0, 2, (b, 11), generator=g).float(),
             'mask': torch.ones(b, 11)}
    if rep == 'F':
        batch['extra'] = torch.randn(b, t, 1200, generator=g)
        batch['extra'][..., 600:] = 0 if missing else 1
    elif rep == 'W1':
        batch['extra'] = torch.randn(b, t, 5, 3, 1024, generator=g)
        batch['valid'] = torch.zeros(b, t, 5) if missing else torch.ones(b, t, 5)
    elif rep == 'DVI':
        batch['extra'] = torch.randn(b, t, 6, 32, 32, generator=g)
        batch['valid'] = torch.zeros(b, t) if missing else torch.ones(b, t)
    return batch


def exercise(net, rep, kind, device, tokens=256):
    import torch
    from .models import Model, PretrainModel
    from .training import balanced_loss, rng_state, rng_hash, sync
    torch.manual_seed(42)
    if device.type == 'npu':
        torch.npu.manual_seed_all(42)
    model = (PretrainModel(rep, 42) if kind == 'pretrain' else Model(net, rep, 42)).to(device)
    # AdamW settings match fit(); no CPU fallback or autocast.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, foreach=False)
    results = []
    for size, missing in ((2, False), (1, True)):
        batch = {k: v.to(device) for k, v in batch_for(rep, size, 1 if kind == 'pretrain' else tokens, missing).items()}
        model.eval()
        before = rng_hash(rng_state(device))
        with torch.no_grad():
            a = model(batch); sync(device)
            b = model(batch); sync(device)
        require(a.shape == (size, 11) and torch.isfinite(a).all().item(), 'invalid eval output')
        require(torch.equal(a, b), 'eval not repeatable')
        require(before == rng_hash(rng_state(device)), 'eval consumed RNG')
        model.train(); optimizer.zero_grad(set_to_none=True)
        weights = {k: torch.ones(11, device=device) for k in ('wp', 'wn', 'active')}
        z = model(batch); sync(device)
        loss = balanced_loss(z, batch['y'], batch['mask'], weights); sync(device)
        require(torch.isfinite(loss).item(), 'nonfinite loss')
        loss.backward(); sync(device)
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), 'invalid gradients')
        optimizer.step(); sync(device)
        require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'invalid parameters after AdamW')
        results.append({'batch': size, 'tokens': batch['power'].shape[1], 'missing_high_frequency': missing,
                        'forward_backward_adamw': True, 'eval_repeatable': True})
    return {'checks': results, 'parameters': sum(p.numel() for p in model.parameters()), 'precision': 'FP32'}


def preflight(run, cards, timeout=1200):
    from .runtime import available
    require(1 <= len(cards) <= 3 and len(set(cards)) == len(cards) and all(c >= 0 for c in cards), 'Specify one to three distinct non-negative NPU card IDs')
    # Sequential on the first requested physical card; no fourth card/process.
    card = cards[0]
    folder = Path(run) / 'audit/operator_preflight'
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for net, rep, kind in CASES:
        require(available(card), 'The preflight card is occupied; other processes will not be terminated')
        name = f'{kind}-{net}-{rep}'
        dest = folder / f'{name}.json'
        # A crashed child must never inherit a previous successful receipt.
        if dest.exists():
            dest.unlink()
        env = os.environ.copy()
        env['ASCEND_RT_VISIBLE_DEVICES'] = str(card)
        env['ASCEND_LAUNCH_BLOCKING'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        print(f'NPU operator preflight {name} (isolated process; no dataset required)', flush=True)
        command = [sys.executable, '-m', 'tdha_v3.npu_preflight', '--net', net, '--rep', rep,
                   '--kind', kind, '--output', str(dest)]
        error = ''; returncode = None
        with (folder / f'{name}.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                error = f'Preflight subprocess exceeded {timeout} seconds and was terminated'
        try:
            result = read(dest) if dest.exists() else {}
        except (ValueError, OSError):
            result = {}
        if returncode != 0 or not result:
            result.update(passed=False, error=error or result.get('error') or f'Process exited with code {returncode}; see {name}.log')
        result.update(case=name, physical_device=card, returncode=returncode)
        results.append(result)
        dump(Path(run) / 'operator_preflight.json', {'passed': False, 'complete': False, 'code': code_hash(), 'rows': results})
    passed = all(r.get('passed') and r.get('code') == code_hash() for r in results)
    dump(Path(run) / 'operator_preflight.json', {'passed': passed, 'complete': True, 'code': code_hash(), 'rows': results,
         'scope': '仅合成算子验收；实际microbatch16/8、容量、速度和真实数据等价检查仍由benchmark完成；未验证其他物理卡'})
    require(passed, 'NPU operator preflight failed; results for 13 configurations are in operator_preflight.json and audit/operator_preflight/*.log')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net', required=True); p.add_argument('--rep', required=True)
    p.add_argument('--kind', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        from .session import require_tmux
        require_tmux()
        require((args.net, args.rep, args.kind) in CASES, 'unknown operator case')
        from .training import backend
        started = time.monotonic()
        result = exercise(args.net, args.rep, args.kind, backend())
        dump(args.output, {**result, 'passed': True, 'code': code_hash(), 'elapsed_s_debug_sync': time.monotonic() - started})
        return 0
    except Exception as exc:
        traceback.print_exc()
        dump(args.output, {'passed': False, 'code': code_hash(), 'error': str(exc)})
        return 1


if __name__ == '__main__':
    sys.exit(main())
