"""Explicit PyTorch-semantics GRU for the FP32 TDHA baseline.

Keeps nn.GRU parameter names, shapes and initialization, but never dispatches
the fused GRU operator. Only the configuration used by this experiment is
supported. Numerical equivalence is tolerance-based, not fused-kernel bitwise
identity; dropout has the same semantics, not necessarily the same RNG stream.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FP32GRU(nn.GRU):
    def flatten_parameters(self):
        # No cuDNN/NPU packed-weight storage is needed by the explicit path.
        pass

    def forward(self, input, hx=None):
        if not self.batch_first or self.bidirectional:
            raise ValueError('FP32GRU requires batch_first=True and unidirectional')
        if not isinstance(input, torch.Tensor) or input.ndim != 3:
            raise ValueError('FP32GRU requires a dense [batch,time,features] tensor')
        b, t, width = input.shape
        if not b or not t or width != self.input_size:
            raise ValueError('Invalid FP32GRU input shape')
        if input.dtype != torch.float32 or any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('TDHA GRU must use FP32 inputs and parameters; AMP is not enabled')
        if hx is None:
            hx = input.new_zeros(self.num_layers, b, self.hidden_size)
        if hx.shape != (self.num_layers, b, self.hidden_size) or hx.dtype != input.dtype or hx.device != input.device:
            raise ValueError('Invalid FP32GRU initial hidden state')
        output = input
        final = []
        for layer in range(self.num_layers):
            wi = getattr(self, f'weight_ih_l{layer}')
            wh = getattr(self, f'weight_hh_l{layer}')
            bi = getattr(self, f'bias_ih_l{layer}') if self.bias else None
            bh = getattr(self, f'bias_hh_l{layer}') if self.bias else None
            # Batch the input projection; only the recurrent projection is serial.
            projected = F.linear(output, wi, bi)
            h = hx[layer]
            sequence = []
            for x in projected.unbind(1):
                ir, iz, inn = x.chunk(3, dim=-1)
                hr, hz, hn = F.linear(h, wh, bh).chunk(3, dim=-1)
                reset = torch.sigmoid(ir + hr)
                update = torch.sigmoid(iz + hz)
                candidate = torch.tanh(inn + reset * hn)
                h = candidate + update * (h - candidate)
                sequence.append(h)
            output = torch.stack(sequence, dim=1)
            final.append(h)
            if layer < self.num_layers - 1 and self.dropout:
                output = F.dropout(output, p=self.dropout, training=self.training)
        return output, torch.stack(final, dim=0)
