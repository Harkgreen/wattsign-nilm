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

M = 242
CHANNELS = 520


class CausalConv(nn.Module):

    def __init__(self, a, b, k=3, d=1):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(a, b, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

class SegmentStem(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = CausalConv(CHANNELS, 32, 5)
        self.proj = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        z = F.relu(self.conv(x))
        n = z.shape[-1]
        assert n % 5 == 0
        z = z.reshape(len(z), 32, n // 5, 5)
        return self.norm(self.proj(torch.cat((z.mean(-1), z.amax(-1)), dim=1).transpose(1, 2)))

class RecurrentTime(nn.Module):

    def __init__(self, family, length):
        super().__init__()
        self.stem = SegmentStem()
        layer = nn.GRU if family == 'GRU' else nn.LSTM
        self.memory = layer(32, 48, 2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.memory(self.stem(x))
        return self.head(out[:, -1]).squeeze(-1)

class W1Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(3, 8, 15, stride=4, padding=7), nn.ReLU(), nn.Conv1d(8, 16, 9, stride=4, padding=4), nn.ReLU(), nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU())
        self.out = nn.Linear(256, 16)

    def forward(self, w):
        z = self.conv(w)
        assert z.shape[-1] == 32
        z = z.reshape(len(z), 16, 8, 4)
        pooled = torch.cat((z.mean(-1), z.amax(-1)), dim=1).flatten(1)
        return self.out(pooled)

class AggregateModel(nn.Module):

    def __init__(self, family, rep, length, vscale, iscale):
        super().__init__()
        self.family = family
        self.rep = rep
        self.length = length
        if (family, rep, length) != ('GRU', 'W1', 2560):
            raise ValueError('The public adapter supports the selected GRU/W1/2560 model only')
        self.temporal = RecurrentTime('GRU', length)
        self.encoder = W1Encoder()
        self.register_buffer('vscale', torch.tensor(float(vscale)))
        self.register_buffer('iscale', torch.tensor(float(iscale)))

    def encode_cycles(self, raw):
        n = raw.shape[0]
        a = raw.reshape(-1, 2, 1024)
        u = a[:, 0].float() / 12.9
        i = a[:, 1].float() * (28.47 / 32768.0)
        if self.rep == 'W0':
            inp = torch.stack((u / self.vscale, i / self.iscale), dim=1)
        else:
            p = (u * i).mean(-1)
            u2 = (u * u).mean(-1).clamp_min(1e-12)
            f = i - p[:, None] / u2[:, None] * u
            phase = torch.angle(torch.fft.rfft(u, dim=-1)[:, 1])
            shifts = torch.round(phase * 1024 / (2 * math.pi)).long()
            positions = (torch.arange(1024, device=u.device)[None] - shifts[:, None]) % 1024
            inp = torch.stack((u / self.vscale, i / self.iscale, f / self.iscale), dim=1)
            inp = torch.gather(inp, 2, positions[:, None].expand(-1, 3, -1))
        return self.encoder(inp).reshape(n, -1, 16)

    def forward(self, x, raw=None, embedding=None):
        if self.encoder is not None:
            if embedding is None:
                assert raw is not None
                embedding = self.encode_cycles(raw)
            x = x.clone()
            mask = x[:, 36 + M:36 + M + 16].clone()
            x[:, 36:52] = embedding.transpose(1, 2) * mask
        return self.temporal(x)
