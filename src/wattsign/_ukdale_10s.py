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


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))

class CausalResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, 3, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, 3, dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        hidden = self.dropout(F.relu(self.conv1(x)))
        hidden = self.dropout(F.relu(self.conv2(hidden)))
        return F.relu(hidden + self.residual(x))

class RichCausalTCN(nn.Module):

    def __init__(self):
        super().__init__()
        blocks = []
        channels = 18
        for dilation in (1, 2, 4, 8):
            blocks.append(CausalResidualBlock(channels, 32, dilation, 0.1))
            channels = 32
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(32, 1, 1)

    def forward_sequence(self, x):
        return self.head(self.blocks(x))[:, 0, :]

    def seq(self, x):
        return self.forward_sequence(x)

    def forward(self, x):
        return self.forward_sequence(x)[:, -1]

    @property
    def receptive_field(self):
        return 61

    @property
    def rf(self):
        return self.receptive_field

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
