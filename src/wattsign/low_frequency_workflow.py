"""UK-DALE readers, causal features, appliance labels and evaluation metrics.

The 10-second and 1-second training recipes share these preprocessing routines.
Input specifications name the mains file, appliance-channel file and interval."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
RULE = {'on_threshold_w':800., 'off_threshold_w':100., 'minimum_activity_seconds':20., 'short_off_merge_seconds':15., 'maximum_sample_gap_seconds':30.}
MICROWAVE_RULE = RULE
VALUE_MASK_PAIRS = [(g+j,g+3+j) for g in (0,6,12) for j in range(3)]
def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def array_sha256(array: np.ndarray) -> str:
    x = np.ascontiguousarray(array)
    return sha256_bytes(f'{x.dtype}|{x.shape}|'.encode() + x.view(np.uint8).tobytes())
def _mark(array: np.ndarray, left: int, right: int, value=True) -> None:
    left, right = (max(0, int(left)), min(array.size, int(right)))
    if left < right:
        array[left:right] = value

def _ceil_index(value: float) -> int:
    return int(math.ceil(value - 1e-10))

def _floor_index(value: float) -> int:
    return int(math.floor(value + 1e-10))

def scan_psq_10s(path: Path, start_s: int, end_s: int) -> dict:
    n1, n10 = (end_s - start_s, (end_s - start_s) // 10)
    p = np.zeros(n10, np.float32)
    s = np.zeros(n10, np.float32)
    q = np.zeros(n10, np.float32)
    pv = np.zeros(n10, bool)
    sv = np.zeros(n10, bool)
    qv = np.zeros(n10, bool)
    p1_valid = np.zeros(n1, bool)
    break1 = np.zeros(n1, bool)
    break10 = np.zeros(n10, bool)
    audit = {'raw_records': 0, 'records_in_interval': 0, 'nonmonotonic_records': 0, 'recovery_breaks': 0, 'gaps_over_30s': 0, 'invalid_raw_q_records': 0}
    carry = pd.DataFrame()
    previous_kept = None
    recovery_pending = False
    reached_end = False

    def consume(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        grouped = frame.groupby('bin10', sort=False).agg(p=('p_clean', 'median'), p_bad=('p_bad', 'max'), s=('s_clean', 'median'), q=('q_clean', 'median'))
        idx = grouped.index.to_numpy(np.int64)
        ok = (idx >= 0) & (idx < n10)
        idx = idx[ok]
        grouped = grouped.iloc[np.flatnonzero(ok)]
        pp = grouped['p'].to_numpy(float)
        ss = grouped['s'].to_numpy(float)
        qq = grouped['q'].to_numpy(float)
        pgood = np.isfinite(pp) & ~grouped['p_bad'].to_numpy(bool)
        sgood = np.isfinite(ss)
        qgood = np.isfinite(qq)
        p[idx[pgood]] = pp[pgood].astype(np.float32)
        pv[idx[pgood]] = True
        s[idx[sgood]] = ss[sgood].astype(np.float32)
        sv[idx[sgood]] = True
        q[idx[qgood]] = qq[qgood].astype(np.float32)
        qv[idx[qgood]] = True
        grouped1 = frame.groupby('bin1', sort=False).agg(p=('p_clean', 'median'), p_bad=('p_bad', 'max'))
        idx1 = grouped1.index.to_numpy(np.int64)
        ok1 = (idx1 >= 0) & (idx1 < n1)
        idx1 = idx1[ok1]
        grouped1 = grouped1.iloc[np.flatnonzero(ok1)]
        p1good = np.isfinite(grouped1['p'].to_numpy(float)) & ~grouped1['p_bad'].to_numpy(bool)
        p1_valid[idx1[p1good]] = True
    for chunk in pd.read_csv(path, sep='\\s+', header=None, usecols=[0, 1, 2], names=['ts', 'p', 's'], chunksize=500000):
        ts_raw = pd.to_numeric(chunk['ts'], errors='coerce').to_numpy(float)
        pp_raw = pd.to_numeric(chunk['p'], errors='coerce').to_numpy(float)
        ss_raw = pd.to_numeric(chunk['s'], errors='coerce').to_numpy(float)
        finite_ts = np.isfinite(ts_raw)
        ts_raw, pp_raw, ss_raw = (ts_raw[finite_ts], pp_raw[finite_ts], ss_raw[finite_ts])
        audit['raw_records'] += int(ts_raw.size)
        if ts_raw.size == 0:
            continue
        prior = np.maximum.accumulate(np.r_[-np.inf if previous_kept is None else previous_kept, ts_raw[:-1]])
        accepted_mask = ts_raw > prior
        accepted_index = np.flatnonzero(accepted_mask)
        audit['nonmonotonic_records'] += int((~accepted_mask).sum())
        if accepted_index.size == 0:
            recovery_pending = True
            continue
        recovery = np.zeros(accepted_index.size, bool)
        if recovery_pending or (~accepted_mask).any():
            pending = recovery_pending
            previous_raw_index = -1
            for accepted_position, raw_index in enumerate(accepted_index):
                if raw_index > previous_raw_index + 1:
                    pending = True
                if pending and start_s <= ts_raw[raw_index] < end_s:
                    recovery[accepted_position] = True
                    pending = False
                previous_raw_index = int(raw_index)
            if accepted_index[-1] < ts_raw.size - 1:
                pending = True
            recovery_pending = pending
        else:
            recovery_pending = False
        ts, pp, ss = (ts_raw[accepted_index], pp_raw[accepted_index], ss_raw[accepted_index])
        diffs = np.diff(np.r_[previous_kept if previous_kept is not None else ts[0], ts])
        gaps = diffs > MICROWAVE_RULE['maximum_sample_gap_seconds']
        in_span = (ts >= start_s) & (ts < end_s)
        boundary = (gaps | recovery) & in_span
        if boundary.any():
            boundary_ts = ts[boundary]
            i1 = np.floor(boundary_ts).astype(np.int64) - start_s
            i10 = (np.floor(boundary_ts / 10).astype(np.int64) * 10 - start_s) // 10
            break1[i1] = True
            break10[i10] = True
        audit['gaps_over_30s'] += int((gaps & in_span).sum())
        audit['recovery_breaks'] += int((recovery & in_span).sum())
        previous_kept = float(ts[-1])
        reached_end = bool((ts >= end_s).any())
        ts, pp, ss = (ts[in_span], pp[in_span], ss[in_span])
        audit['records_in_interval'] += int(ts.size)
        if ts.size:
            p_bad = ~np.isfinite(pp) | np.isfinite(pp) & np.isfinite(ss) & (pp == 0.0) & (ss > 50.0)
            s_bad = ~np.isfinite(ss) | (ss < 0.0)
            q_ok = ~p_bad & ~s_bad & (ss >= np.abs(pp))
            q_raw = np.where(q_ok, np.sqrt(np.maximum(ss * ss - pp * pp, 0.0)), np.nan)
            audit['invalid_raw_q_records'] += int((~q_ok).sum())
            frame = pd.DataFrame({'bin10': (np.floor(ts / 10).astype(np.int64) * 10 - start_s) // 10, 'bin1': np.floor(ts).astype(np.int64) - start_s, 'p_clean': np.where(p_bad, np.nan, pp), 'p_bad': p_bad, 's_clean': np.where(s_bad, np.nan, ss), 'q_clean': q_raw})
            if not carry.empty:
                frame = pd.concat([carry, frame], ignore_index=True)
            last_bin = int(frame['bin10'].iloc[-1])
            complete = frame['bin10'].to_numpy() != last_bin
            consume(frame.loc[complete])
            carry = frame.loc[~complete].copy()
        if reached_end:
            break
    consume(carry)
    break1[0] = False
    break10[0] = False
    continuity1 = np.cumsum(break1, dtype=np.int32)
    continuity10 = np.cumsum(break10, dtype=np.int32)
    audit.update({'valid_p_bins_1s': int(p1_valid.sum()), 'valid_p_bins_10s': int(pv.sum()), 'valid_s_bins_10s': int(sv.sum()), 'valid_q_bins_10s': int(qv.sum()), 'segments_1s': int(continuity1.max() + 1), 'segments_10s': int(continuity10.max() + 1), 'q_coverage_fraction': float(qv.mean())})
    return {'p': p, 's': s, 'q': q, 'p_valid': pv, 's_valid': sv, 'q_valid': qv, 'p1_valid': p1_valid, 'break1': break1, 'break10': break10, 'continuity1': continuity1, 'continuity10': continuity10, 'breaks': break10, 'audit': audit}

def build_signal_features(value: np.ndarray, valid: np.ndarray, continuity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = value.size
    values = np.zeros((n, 3), np.float32)
    masks = np.zeros((n, 3), np.uint8)
    values[:, 0] = value
    masks[:, 0] = valid
    for i in range(1, n):
        if valid[i] and valid[i - 1] and (continuity[i] == continuity[i - 1]):
            values[i, 1] = np.float32(float(value[i]) - float(value[i - 1]))
            masks[i, 1] = 1
        history = np.arange(max(0, i - 6), i, dtype=np.int64)
        history = history[valid[history] & (continuity[history] == continuity[i])]
        if valid[i] and i >= 6 and (history.size >= 5):
            values[i, 2] = np.float32(float(value[i]) - float(np.median(value[history])))
            masks[i, 2] = 1
    return (values, masks)

def pack_18_channels(signals: list[tuple[np.ndarray, np.ndarray]], continuity: np.ndarray) -> np.ndarray:
    groups = []
    for value, valid in signals:
        values, masks = build_signal_features(value, valid, continuity)
        groups.append(np.concatenate([values, masks.astype(np.float32)], axis=1))
    result = np.concatenate(groups, axis=1).astype(np.float32)
    assert result.shape[1] == 18
    return result

def scan_microwave_native(path: Path, start_s: int, end_s: int) -> dict:
    stop_s = float(end_s + 2 * MICROWAVE_RULE['maximum_sample_gap_seconds'])
    previous_timestamp = None
    state = 'unknown'
    active_start = active_on_prev = trusted_start = None
    coverage_segments, raw_intervals = ([], [])
    recovery_break_pending = False
    audit = {'native_records': 0, 'native_gaps': 0, 'native_nonmonotonic_records': 0}

    def close_active(end_timestamp, reason, off_prev):
        nonlocal active_start, active_on_prev
        if active_start is not None and end_timestamp > active_start:
            raw_intervals.append({'start_s': float(active_start), 'end_s': float(end_timestamp), 'end_reason': reason, 'on_prev_s': float(active_on_prev if active_on_prev is not None else active_start), 'off_prev_s': float(off_prev if off_prev is not None else end_timestamp)})
        active_start = active_on_prev = None
    reached_stop = False
    for chunk in pd.read_csv(path, sep='\\s+', header=None, usecols=[0, 1], names=['ts', 'power'], chunksize=250000):
        for ts, power in zip(pd.to_numeric(chunk['ts'], errors='coerce').to_numpy(float), pd.to_numeric(chunk['power'], errors='coerce').to_numpy(float), strict=True):
            if not (math.isfinite(ts) and math.isfinite(power)):
                continue
            if ts > stop_s:
                reached_stop = True
                break
            audit['native_records'] += 1
            if previous_timestamp is not None and ts <= previous_timestamp:
                audit['native_nonmonotonic_records'] += 1
                recovery_break_pending = True
                continue
            is_gap = previous_timestamp is not None and ts - previous_timestamp > MICROWAVE_RULE['maximum_sample_gap_seconds']
            if is_gap or recovery_break_pending:
                audit['native_gaps'] += int(is_gap)
                if active_start is not None and previous_timestamp is not None:
                    close_active(previous_timestamp, 'data_gap', previous_timestamp)
                if trusted_start is not None and previous_timestamp is not None:
                    coverage_segments.append((trusted_start, previous_timestamp))
                state = 'unknown'
                trusted_start = None
                recovery_break_pending = False
            if state == 'unknown' and power <= MICROWAVE_RULE['off_threshold_w']:
                state = 'off'
                trusted_start = float(ts)
            elif state == 'off' and power >= MICROWAVE_RULE['on_threshold_w']:
                active_start = float(ts)
                active_on_prev = float(previous_timestamp if previous_timestamp is not None else ts)
                state = 'on'
            elif state == 'on' and power <= MICROWAVE_RULE['off_threshold_w']:
                close_active(ts, 'off_threshold', previous_timestamp if previous_timestamp is not None else ts)
                state = 'off'
            previous_timestamp = float(ts)
        if reached_stop:
            break
    if active_start is not None and previous_timestamp is not None:
        close_active(previous_timestamp, 'right_censored', previous_timestamp)
    if trusted_start is not None and previous_timestamp is not None:
        coverage_segments.append((trusted_start, previous_timestamp))
    raw_intervals.sort(key=lambda x: (x['start_s'], x['end_s']))
    merged = []
    for interval in raw_intervals:
        if merged and merged[-1]['end_reason'] == 'off_threshold' and (0 <= interval['start_s'] - merged[-1]['end_s'] <= MICROWAVE_RULE['short_off_merge_seconds']):
            merged[-1].update({'end_s': interval['end_s'], 'end_reason': interval['end_reason'], 'off_prev_s': interval['off_prev_s']})
        else:
            merged.append(dict(interval))
    accepted = []
    for interval in merged:
        interval['span_seconds'] = interval['end_s'] - interval['start_s']
        if interval['span_seconds'] >= MICROWAVE_RULE['minimum_activity_seconds']:
            accepted.append(interval)
    audit.update({'accepted_events': len(accepted), 'merged_intervals': len(merged), 'trusted_coverage_segments': len(coverage_segments)})
    return {'coverage_segments': coverage_segments, 'merged_intervals': merged, 'accepted_intervals': accepted, 'audit': audit}

def project_labels_to_10s(scan: dict, start_s: int, end_s: int) -> dict:
    n1 = end_s - start_s
    covered = np.zeros(n1, bool)
    overlaps = np.zeros(n1, bool)
    accepted = np.zeros(n1, bool)
    event = np.full(n1, -1, np.int32)
    uncertain = np.zeros(n1, bool)
    guard = np.zeros(n1, bool)
    for left, right in scan['coverage_segments']:
        _mark(covered, _ceil_index(left - start_s), _floor_index(right - start_s))
    for interval in scan['merged_intervals']:
        _mark(overlaps, _floor_index(interval['start_s'] - start_s), _ceil_index(interval['end_s'] - start_s))
    for eid, interval in enumerate(scan['accepted_intervals']):
        left, right = (_ceil_index(interval['start_s'] - start_s), _floor_index(interval['end_s'] - start_s))
        _mark(accepted, left, right)
        _mark(event, left, right, eid)
        _mark(uncertain, math.floor(interval['on_prev_s'] - start_s), math.ceil(interval['start_s'] - start_s))
        if interval['end_reason'] == 'off_threshold':
            _mark(uncertain, math.floor(interval['off_prev_s'] - start_s), math.ceil(interval['end_s'] - start_s))
        for boundary in (interval['start_s'], interval['end_s']):
            _mark(guard, math.floor(boundary - 10 - start_s), math.ceil(boundary + 10 - start_s))
    target = (covered & accepted).astype(np.uint8)
    valid = covered & (~overlaps | accepted)
    supervised = valid & ~uncertain
    stable = supervised & ~guard
    endpoint = np.arange(9, n1, 10)
    return {'target': target[endpoint], 'valid': supervised[endpoint].astype(np.uint8), 'stable_valid': stable[endpoint].astype(np.uint8), 'event_id': event[endpoint]}
def fit_masked_normalizer(features: np.ndarray, fit_end: int) -> dict:
    means, stds = (np.zeros(18, np.float64), np.ones(18, np.float64))
    for value_col, mask_col in VALUE_MASK_PAIRS:
        ok = features[:fit_end, mask_col] > 0.5
        values = features[:fit_end, value_col][ok].astype(np.float64)
        assert values.size > 100
        means[value_col], stds[value_col] = (values.mean(), max(values.std(), 1e-06))
    return {'means': means.tolist(), 'stds': stds.tolist(), 'fit_end': int(fit_end), 'fit_hash': array_sha256(features[:fit_end])}
def norm_apply(raw, n):
    x = raw.astype(np.float32, copy=True)
    mu = np.array(n['means'])
    sd = np.array(n['stds'])
    for g in (0, 6, 12):
        for j in range(3):
            v, m = (g + j, g + 3 + j)
            ok = x[:, m] > 0.5
            x[:, v] = 0
            x[ok, v] = ((raw[ok, v] - mu[v]) / sd[v]).astype(np.float32)
    return x

def endpoints_10s(t, left, right):
    e = np.arange(left + 64 - 1, right, dtype=np.int64)
    ok = t['valid'][e].astype(bool) & t['endpoint_valid'][e].astype(bool)
    ok &= t['continuity'][e] == t['continuity'][e - 64 + 1]
    return e[ok]
def scan_psq_seconds(path, start_s, end_s, gains):
    """Transform raw P/S records before aggregation, retaining frozen quality rules."""
    n1, n10 = (end_s - start_s, (end_s - start_s) // 10)
    p1 = np.zeros(n1, np.float32)
    p10 = np.zeros(n10, np.float32)
    p1_valid = np.zeros(n1, bool)
    p10_valid = np.zeros(n10, bool)
    p1_bad = np.zeros(n1, bool)
    p10_bad = np.zeros(n10, bool)
    q1 = {g: np.zeros(n1, np.float32) for g in gains}
    q10 = {g: np.zeros(n10, np.float32) for g in gains}
    q1_valid = {g: np.zeros(n1, bool) for g in gains}
    q10_valid = {g: np.zeros(n10, bool) for g in gains}
    break1 = np.zeros(n1, bool)
    break10 = np.zeros(n10, bool)
    current1 = current10 = None
    p_values1, p_values10 = ([], [])
    q_values1, q_values10 = ({g: [] for g in gains}, {g: [] for g in gains})
    previous, recovery = (None, False)
    audit = {'accepted_records': 0, 'nonmonotonic': 0, 'gap_over_30': 0, 'first_excluded_timestamp': None}

    def flush1():
        nonlocal current1, p_values1, q_values1
        if current1 is not None:
            if p_values1 and (not p1_bad[current1]):
                p1[current1] = np.float32(np.median(np.asarray(p_values1, np.float64)))
                p1_valid[current1] = True
            for gain in gains:
                if q_values1[gain]:
                    q1[gain][current1] = np.float32(np.median(np.asarray(q_values1[gain], np.float64)))
                    q1_valid[gain][current1] = True
        current1 = None
        p_values1 = []
        q_values1 = {g: [] for g in gains}

    def flush10():
        nonlocal current10, p_values10, q_values10
        if current10 is not None:
            if p_values10 and (not p10_bad[current10]):
                p10[current10] = np.float32(np.median(np.asarray(p_values10, np.float64)))
                p10_valid[current10] = True
            for gain in gains:
                if q_values10[gain]:
                    q10[gain][current10] = np.float32(np.median(np.asarray(q_values10[gain], np.float64)))
                    q10_valid[gain][current10] = True
        current10 = None
        p_values10 = []
        q_values10 = {g: [] for g in gains}
    with Path(path).open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                timestamp = float(fields[0])
            except ValueError:
                continue
            if not math.isfinite(timestamp):
                continue
            if timestamp >= end_s:
                audit['first_excluded_timestamp'] = timestamp
                break
            try:
                p, s = (float(fields[1]), float(fields[2]))
            except ValueError:
                continue
            if previous is not None and timestamp <= previous:
                audit['nonmonotonic'] += 1
                recovery = True
                continue
            gap = previous is not None and timestamp - previous > RULE['maximum_sample_gap_seconds']
            if timestamp < start_s:
                previous = timestamp
                continue
            i1 = int(math.floor(timestamp)) - start_s
            i10 = (int(math.floor(timestamp / 10)) * 10 - start_s) // 10
            if not (0 <= i1 < n1 and 0 <= i10 < n10):
                continue
            if gap or recovery:
                break1[i1] = True
                break10[i10] = True
                audit['gap_over_30'] += int(gap)
                recovery = False
            if current1 is None:
                current1 = i1
            elif current1 != i1:
                flush1()
                current1 = i1
            if current10 is None:
                current10 = i10
            elif current10 != i10:
                flush10()
                current10 = i10
            p_bad = not math.isfinite(p) or (math.isfinite(s) and p == 0.0 and (s > 50.0))
            if p_bad:
                p1_bad[i1] = True
                p10_bad[i10] = True
            else:
                p_values1.append(p)
                p_values10.append(p)
            for gain in gains:
                scaled_s = gain * s
                if not p_bad and math.isfinite(scaled_s) and (scaled_s >= abs(p)):
                    q = math.sqrt(max(scaled_s * scaled_s - p * p, 0.0))
                    q_values1[gain].append(q)
                    q_values10[gain].append(q)
            audit['accepted_records'] += 1
            previous = timestamp
    flush1()
    flush10()
    break1[0] = False
    break10[0] = False
    assert audit['first_excluded_timestamp'] is not None and audit['first_excluded_timestamp'] >= end_s
    return {'p1': p1, 'p1_valid': p1_valid, 'p10': p10, 'p10_valid': p10_valid, 'q1': q1, 'q10': q10, 'q1_valid': q1_valid, 'q10_valid': q10_valid, 'continuity1': np.cumsum(break1, dtype=np.int32), 'continuity10': np.cumsum(break10, dtype=np.int32), 'audit': audit}

def scan_microwave_labels(path, start_s, end_s):
    """Dedicated microwave labels with the 800/100/20/15/30 hysteresis and duration rules."""
    stop_s = end_s + 2 * RULE['maximum_sample_gap_seconds']
    previous = None
    state = 'unknown'
    active = on_previous = trusted = None
    coverage, intervals, recovery = ([], [], False)

    def close(end, reason, off_previous):
        nonlocal active, on_previous
        if active is not None and end > active:
            intervals.append({'start': active, 'end': end, 'reason': reason, 'on_previous': on_previous if on_previous is not None else active, 'off_previous': off_previous if off_previous is not None else end})
        active = on_previous = None
    with Path(path).open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                timestamp, power = (float(fields[0]), float(fields[1]))
            except ValueError:
                continue
            if not (math.isfinite(timestamp) and math.isfinite(power)):
                continue
            if timestamp > stop_s:
                break
            if previous is not None and timestamp <= previous:
                recovery = True
                continue
            gap = previous is not None and timestamp - previous > RULE['maximum_sample_gap_seconds']
            if gap or recovery:
                if active is not None and previous is not None:
                    close(previous, 'gap', previous)
                if trusted is not None and previous is not None:
                    coverage.append((trusted, previous))
                state = 'unknown'
                trusted = None
                recovery = False
            if state == 'unknown' and power <= RULE['off_threshold_w']:
                state = 'off'
                trusted = timestamp
            elif state == 'off' and power >= RULE['on_threshold_w']:
                active = timestamp
                on_previous = previous if previous is not None else timestamp
                state = 'on'
            elif state == 'on' and power <= RULE['off_threshold_w']:
                close(timestamp, 'off', previous if previous is not None else timestamp)
                state = 'off'
            previous = timestamp
    if active is not None and previous is not None:
        close(previous, 'censored', previous)
    if trusted is not None and previous is not None:
        coverage.append((trusted, previous))
    intervals.sort(key=lambda row: (row['start'], row['end']))
    merged = []
    for row in intervals:
        if merged and merged[-1]['reason'] == 'off' and (0 <= row['start'] - merged[-1]['end'] <= RULE['short_off_merge_seconds']):
            merged[-1].update(end=row['end'], reason=row['reason'], off_previous=row['off_previous'])
        else:
            merged.append(dict(row))
    accepted = [row for row in merged if row['end'] - row['start'] >= RULE['minimum_activity_seconds']]
    n1 = end_s - start_s
    covered = np.zeros(n1, bool)
    overlaps = np.zeros(n1, bool)
    active_mask = np.zeros(n1, bool)
    event = np.full(n1, -1, np.int32)
    uncertain = np.zeros(n1, bool)
    stable_guard = np.zeros(n1, bool)

    def mark(array, left, right, value=True):
        left, right = (max(0, int(left)), min(n1, int(right)))
        if left < right:
            array[left:right] = value
    for left, right in coverage:
        mark(covered, math.ceil(left - start_s - 1e-10), math.floor(right - start_s + 1e-10))
    for row in merged:
        mark(overlaps, math.floor(row['start'] - start_s + 1e-10), math.ceil(row['end'] - start_s - 1e-10))
    for event_id, row in enumerate(accepted):
        left = math.ceil(row['start'] - start_s - 1e-10)
        right = math.floor(row['end'] - start_s + 1e-10)
        mark(active_mask, left, right)
        event[max(0, left):min(n1, right)] = event_id
        mark(uncertain, math.floor(row['on_previous'] - start_s), math.ceil(row['start'] - start_s))
        if row['reason'] == 'off':
            mark(uncertain, math.floor(row['off_previous'] - start_s), math.ceil(row['end'] - start_s))
        for boundary in (row['start'], row['end']):
            mark(stable_guard, math.floor(boundary - 10 - start_s), math.ceil(boundary + 10 - start_s))
    valid = covered & (~overlaps | active_mask)
    supervised = valid & ~uncertain
    stable = supervised & ~stable_guard
    return {'target': (active_mask & covered)[9::10].astype(np.uint8), 'valid': supervised[9::10], 'stable': stable[9::10], 'event': event[9::10], 'accepted': accepted}
def causal_signal_features(value, valid, continuity, lookback=60, minimum=50):
    """Use only prior valid observations and reset each continuous segment."""
    n = len(value)
    values = np.zeros((n, 3), np.float32)
    masks = np.zeros((n, 3), np.uint8)
    values[:, 0] = value
    masks[:, 0] = valid
    same = np.zeros(n, bool)
    same[1:] = valid[1:] & valid[:-1] & (continuity[1:] == continuity[:-1])
    positions = np.flatnonzero(same)
    values[positions, 1] = value[positions] - value[positions - 1]
    masks[positions, 1] = 1
    borders = np.flatnonzero(np.r_[True, continuity[1:] != continuity[:-1], True])
    for left, right in zip(borders[:-1], borders[1:], strict=True):
        series = np.where(valid[left:right], value[left:right], np.nan)
        baseline = pd.Series(series).rolling(lookback, min_periods=minimum).median().shift(1).to_numpy(copy=True)
        baseline[:min(lookback, len(baseline))] = np.nan
        good = valid[left:right] & np.isfinite(baseline)
        local = np.flatnonzero(good)
        values[left + local, 2] = value[left + local] - baseline[local]
        masks[left + local, 2] = 1
    return np.concatenate([values, masks.astype(np.float32)], axis=1)

def feature_normalizer(features, end_second):
    result = {}
    for column in (0, 1, 2, 12, 13, 14):
        good = features[:end_second, column + 3].astype(bool)
        values = features[:end_second, column][good]
        assert values.size
        result[str(column)] = {'mean': float(values.mean()), 'std': max(float(values.std()), 1e-06)}
    return result

def apply_normalizer(features, normalizer):
    out = features.copy()
    for key, info in normalizer.items():
        column = int(key)
        good = out[:, column + 3].astype(bool)
        out[good, column] = (out[good, column] - info['mean']) / info['std']
        out[~good, column] = 0.0
    return out

def endpoint_indices(timeline, left, right):
    endpoint = np.arange(left + 63, right, dtype=np.int64)
    second = endpoint * 10 + 9
    valid = timeline['valid'][endpoint] & timeline['p10_valid'][endpoint] & timeline['p1_valid'][second] & (timeline['continuity10'][endpoint] == timeline['continuity10'][endpoint - 63]) & (timeline['continuity1'][second] == timeline['continuity1'][second - 639])
    return endpoint[valid]

def make_native_features(timeline, raw_q, gain, p_group=None, normalizer=None):
    p_valid_common = timeline['p1_valid'] & np.repeat(timeline['p10_valid'], 10)
    q_valid_common = raw_q['q1_valid'][gain] & np.repeat(raw_q['q10_valid'][gain], 10)
    if p_group is None:
        p_group = causal_signal_features(timeline['p1'], p_valid_common, timeline['continuity1'])
    q_group = causal_signal_features(raw_q['q1'][gain], q_valid_common, timeline['continuity1'])
    features = np.zeros((len(timeline['p1']), 18), np.float32)
    features[:, :6] = p_group
    features[:, 12:] = q_group
    assert not np.count_nonzero(features[:, 6:12]), 'All six S channels must remain zero.'
    if normalizer is not None:
        features = apply_normalizer(features, normalizer)
    return (features, p_group, q_group)
def full_window(features18, values12, masks12, endpoints, with_shape):
    seconds = np.asarray(endpoints, dtype=np.int64) * 10 + 9
    b = len(seconds)
    out = np.zeros((b, 42, 640), np.float32)
    for i, s in enumerate(seconds):
        out[i, :18] = features18[s - 639:s + 1].T
        if with_shape:
            out[i, 18:30] = values12[s - 639:s + 1].T
            out[i, 30:42] = masks12[s - 639:s + 1].T
    return out
def bce(y, prob):
    log = np.log(np.clip(prob, 1e-07, 1 - 1e-07) / (1 - np.clip(prob, 1e-07, 1 - 1e-07)))
    l = np.logaddexp(0, log) - y * log
    return float(0.5 * l[y == 1].mean() + 0.5 * l[y == 0].mean())
def recall(y, s, t):
    yy = np.tile(y, s.shape[0])
    ss = s.ravel()
    return float((ss[yy == 1] >= t).mean())

def threshold_10s(y, s):
    yy = np.tile(y, s.shape[0])
    ss = s.ravel()
    pos = np.sort(ss[yy == 1])
    can = np.r_[np.nextafter(ss.min(), -np.inf), np.unique(ss), np.nextafter(ss.max(), np.inf)]
    assert can[0] < ss.min() and can[-1] > ss.max()
    need = min(max(math.ceil(0.95 * len(pos) - 1e-12), 1), len(pos))
    t = float(pos[-need])
    assert recall(y, s, t) >= 0.95
    return t

def metrics_10s(y, s, t):
    q = (s >= t).astype(np.uint8)
    a, b, c, _ = precision_recall_fscore_support(y, q, average='binary', zero_division=0)
    z = confusion_matrix(y, q, labels=[0, 1])
    return {'PR-AUC': float(average_precision_score(y, s)), 'ROC-AUC': float(roc_auc_score(y, s)), 'Precision': float(a), 'Recall': float(b), 'F1': float(c), 'balanced_accuracy': float(balanced_accuracy_score(y, q)), 'TN': int(z[0, 0]), 'FP': int(z[0, 1]), 'FN': int(z[1, 0]), 'TP': int(z[1, 1])}

def off_fpr_10s(y, s, p, t):
    r = {}
    for lab, lo, hi in [('<800 W', -np.inf, 800.0), ('800–1500 W', 800.0, 1500.0), ('1500–2500 W', 1500.0, 2500.0), ('≥2500 W', 2500.0, np.inf)]:
        k = (y == 0) & (p >= lo) & (p < hi)
        r['OFF_FPR ' + lab] = float((s[k] >= t).mean()) if k.any() else np.nan
    k = (y == 0) & (p >= 800)
    r['high_power_OFF_FPR'] = float((s[k] >= t).mean()) if k.any() else np.nan
    return r
def bce_balanced(y, z):
    l = np.logaddexp(0, z) - np.asarray(y) * z
    return float(0.5 * (l[np.asarray(y) == 1].mean() + l[np.asarray(y) == 0].mean()))

def threshold_1s(y, s):
    p = np.sort(s[y == 1])
    need = math.ceil(0.95 * len(p))
    c = np.unique(np.r_[np.nextafter(s.min(), -np.inf), s, np.nextafter(s.max(), np.inf)])
    return float(c[len(p) - np.searchsorted(p, c, side='left') >= need][-1])

def condition(y, p, event, s):
    yy = np.asarray(y)
    pp = np.asarray(p)
    ee = np.asarray(event)
    ss = np.asarray(s)
    pos = np.flatnonzero((yy == 1) & (ee >= 0))
    all_events = np.unique(ee[pos])
    if not len(pos):
        return ({}, 0.0)
    off = np.flatnonzero(yy == 0)
    bins = np.floor(pp / 50.0).astype(np.int64)
    value_parts = []
    event_parts = []
    for b in np.unique(bins[pos]):
        refs = off[bins[off] == b]
        if not len(refs):
            continue
        ix = pos[bins[pos] == b]
        ranked = np.sort(ss[refs])
        left = np.searchsorted(ranked, ss[ix], side='left')
        right = np.searchsorted(ranked, ss[ix], side='right')
        value_parts.append((left + right).astype(np.float64) / (2.0 * len(ranked)))
        event_parts.append(ee[ix])
    if not value_parts:
        return ({}, 0.0)
    ev = np.concatenate(event_parts)
    value = np.concatenate(value_parts)
    ids, inv = np.unique(ev, return_inverse=True)
    total = np.bincount(inv, weights=value, minlength=len(ids))
    count = np.bincount(inv, minlength=len(ids))
    result = {int(e): float(a / b) for e, a, b in zip(ids, total, count, strict=True)}
    return (result, len(result) / max(1, len(all_events)))

def metric_table(y, p, q, event, s, thr):
    pred = s >= thr
    pr, re, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    off = y == 0
    high = off & (p >= 800)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cond, cover = condition(y, p, event, s)
    result = {'AP': float(average_precision_score(y, s)), 'ROC-AUC': float(roc_auc_score(y, s)), 'Precision': float(pr), 'Recall': float(re), 'F1': float(f), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'high_power_OFF_FPR': float(pred[high].mean()) if high.any() else np.nan, 'power_conditioned_concordance': float(np.mean(list(cond.values()))) if cond else np.nan, 'matched_event_coverage': cover, 'OFF-P Spearman': float(spearmanr(s[off], p[off]).statistic), 'OFF-Q Spearman': float(spearmanr(s[off], q[off]).statistic), 'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]), 'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]), '_cond': cond}
    for name, left, right in [('＜800 W', -np.inf, 800), ('800–1500 W', 800, 1500), ('1500–2500 W', 1500, 2500), ('≥2500 W', 2500, np.inf)]:
        band = off & (p >= left) & (p < right)
        result[f'OFF_FPR {name}'] = float(pred[band].mean()) if band.any() else np.nan
    return result


def read_timeline_10s(spec):
    raw = scan_psq_10s(spec['mains'], spec['start'], spec['end'])
    scan = scan_microwave_native(spec['microwave'], spec['start'], spec['end'])
    labels = project_labels_to_10s(scan, spec['start'], spec['end'])
    last_second = np.arange(len(raw['p']), dtype=np.int64)*10+9
    eligible = raw['p_valid'] & raw['p1_valid'][last_second]
    mature = last_second >= 639
    eligible[mature] &= raw['continuity1'][last_second[mature]] == raw['continuity1'][last_second[mature]-639]
    features = pack_18_channels([(raw['p'],raw['p_valid']),(raw['s'],raw['s_valid']),(raw['q'],raw['q_valid'])], raw['continuity10'])
    return {'features_unscaled':features,'continuity':raw['continuity10'],
        'endpoint_valid':eligible.astype(np.uint8), 'p':raw['p'],'s':raw['s'],'q':raw['q'],**labels}


def read_timeline_1s(spec):
    raw = scan_psq_seconds(spec['mains'],spec['start'],spec['end'],gains=(1.0,))
    labels = scan_microwave_labels(spec['microwave'],spec['start'],spec['end'])
    timeline = {k:raw[k] for k in ('p1','p1_valid','p10','p10_valid','continuity1','continuity10')}
    timeline.update({k:labels[k] for k in ('target','valid','stable','event')})
    timeline['start_s'] = spec['start']
    return timeline, {k:raw[k] for k in ('q1','q10','q1_valid','q10_valid')}


def balanced_samples(endpoints, labels, seed):
    # Exact original balanced sampling, before the unused ranking-pair draw.
    positive = endpoints[labels[endpoints] == 1]
    negative = endpoints[labels[endpoints] == 0]
    if len(positive) == 0 or len(negative) < len(positive):
        raise ValueError('Balanced original sampling requires positive examples and enough negative examples')
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(negative)[:len(positive)]
    sample = np.concatenate([positive, chosen])
    return sample[rng.permutation(len(sample))]


def validate_spec(spec):
    required = {'mains','microwave','start','end'}
    if required - spec.keys():
        raise ValueError(f'Missing data specification: {sorted(required-spec.keys())}')
    start,end = int(spec['start']),int(spec['end'])
    if start % 10 or end % 10 or end-start < 2000:
        raise ValueError('Specify a 10-second aligned interval of at least 2000 seconds')
    checked = dict(spec,start=start,end=end)
    for name in ('mains','microwave'):
        if not str(spec[name]).strip():raise ValueError(f'Set the existing {name} data file explicitly')
        checked[name] = Path(spec[name]).expanduser().resolve()
        if not checked[name].is_file():raise FileNotFoundError(checked[name])
    return checked


def validate_run(spec, evaluation_specs, output, protected_roots=()):
    checked = validate_spec(spec)
    evaluation = [validate_spec(s) for s in evaluation_specs]
    names = [s.get('name','evaluation') for s in evaluation]
    if len(names) != len(set(names)) or 'h1_final' in names or any(not n or '/' in n or '\\' in n or n in ('.','..') for n in names):
        raise ValueError('Evaluation names must be unique safe path components')
    if not str(output).strip():raise ValueError('Set a new independent output directory')
    out = Path(output).expanduser().resolve()
    if any((parent/'.git').exists() or ((parent/'pyproject.toml').is_file() and (parent/'src').is_dir()) for parent in (out,*out.parents)):
        raise ValueError('Output must be outside source code checkouts')
    roots = [Path(p).expanduser().resolve() for p in protected_roots]
    roots += [Path(__file__).resolve().parents[2]]
    roots += [s[key].parent for s in [checked,*evaluation] for key in ('mains','microwave')]
    if out.exists() or any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in roots):
        raise ValueError('Output must not exist and must be disjoint from the repository, data and protected roots')
    return checked,evaluation,out


def ensure_two_classes(labels, endpoints, name):
    if len(endpoints)==0 or set(np.unique(labels[endpoints]).tolist()) != {0,1}:
        raise ValueError(f'{name} needs nonempty positive and negative eligible endpoints')


def write_json(path, value):
    def encode(obj):
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,Path):return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=encode)+'\n',encoding='utf-8')


def create_output(out):
    # Only called after explicit execution approval and input/path validation.
    out.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoints','predictions','tables','preparation','protocol'):
        (out/name).mkdir()


def save_preparation(out, normalizer, samples, batches=None):
    write_json(out/'preparation'/'normalizer.json',normalizer)
    arrays = {f'sample_{s}':v for s,v in samples.items()}
    if batches is not None:
        arrays.update({f'order_{s}_{e}':order for s,orders in batches.items() for e,order in enumerate(orders)})
    np.savez_compressed(out/'preparation'/'sampling_and_batches.npz',**arrays)


def parse_cli(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mains',required=True)
    parser.add_argument('--microwave',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--start',type=int,default=1363564800)
    parser.add_argument('--end',type=int,default=1371340800)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--protected-root',action='append',default=[])
    parser.add_argument('--evaluations-json',help='Optional JSON list of explicit mains/microwave/start/end/name specifications')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--plan-only',action='store_true')
    args = parser.parse_args()
    evaluation=[]
    if args.evaluations_json:
        evaluation=json.loads(Path(args.evaluations_json).read_text(encoding='utf-8'))
        if not isinstance(evaluation,list):raise ValueError('Evaluation specification must be a JSON list')
    return args,dict(mains=args.mains,microwave=args.microwave,start=args.start,end=args.end),evaluation


def bind_visible_trainer(trainer, namespace):
    """Bind a visible notebook training definition to its prepared recipe context."""
    from types import FunctionType
    if not isinstance(trainer,FunctionType) or trainer.__closure__ is not None:
        raise TypeError('Provide a plain training function without hidden closure dependencies')
    return FunctionType(trainer.__code__,namespace,trainer.__name__,trainer.__defaults__)
