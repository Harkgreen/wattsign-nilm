"""Selected model and preprocessing definitions reused without fitting.

Parameter names and forward computations retain checkpoint compatibility.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Module):

    def __init__(self, a, b, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.c = nn.Conv1d(a, b, k, dilation=d)

    def forward(self, x):
        return self.c(F.pad(x, (self.pad, 0)))

class RB(nn.Module):

    def __init__(self, a, b, d):
        super().__init__()
        self.a = CausalConv(a, b, 3, d)
        self.b = CausalConv(b, b, 3, d)
        self.s = nn.Identity() if a == b else nn.Conv1d(a, b, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return F.relu(self.s(x) + self.drop(F.relu(self.b(self.drop(F.relu(self.a(x)))))))

class M0TCN(nn.Module):

    def __init__(self):
        super().__init__()
        ch = 42
        self.ds = (1, 2, 4, 8, 16, 32, 32, 64)
        self.blocks = nn.Sequential(*[RB(42 if i == 0 else 22, 22, d) for i, d in enumerate(self.ds)])
        self.out = nn.Conv1d(22, 1, 1)

    def forward_sequence(self, x):
        return self.out(self.blocks(x))[:, 0]

    def forward(self, x):
        return self.forward_sequence(x)[:, -1]

def causal_signal_features(value, valid, continuity, lookback=60, minimum=50):
    """Use only valid observations before the current bin; compute each continuous segment independently."""
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

def apply_normalizer(features, normalizer):
    out = features.copy()
    for key, info in normalizer.items():
        column = int(key)
        good = out[:, column + 3].astype(bool)
        out[good, column] = (out[good, column] - info['mean']) / info['std']
        out[~good, column] = 0.0
    return out

def make_native_features(timeline, raw_q, gain, p_group=None, normalizer=None):
    p_valid_common = timeline['p1_valid'] & np.repeat(timeline['p10_valid'], 10)
    q_valid_common = raw_q['q1_valid'][gain] & np.repeat(raw_q['q10_valid'][gain], 10)
    if p_group is None:
        p_group = causal_signal_features(timeline['p1'], p_valid_common, timeline['continuity1'])
    q_group = causal_signal_features(raw_q['q1'][gain], q_valid_common, timeline['continuity1'])
    features = np.zeros((len(timeline['p1']), 18), np.float32)
    features[:, :6] = p_group
    features[:, 12:] = q_group
    assert not np.count_nonzero(features[:, 6:12]), 'S 六通道必须恒为零'
    if normalizer is not None:
        features = apply_normalizer(features, normalizer)
    return (features, p_group, q_group)
