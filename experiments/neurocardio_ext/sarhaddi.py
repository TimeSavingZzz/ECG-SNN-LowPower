#!/usr/bin/env python3
"""Reimplementation of Sarhaddi et al. (BSN 2026) SNN-SRNN for PTB-XL.

Paper: "Spiking Recurrent ECG Classification for Resource-Constrained Wearable
Devices" (Sarhaddi et al., 22nd Body Sensor Networks, 2026).

SPECIFIED in the paper (implemented exactly):
  * Continuous dual-polarity encoding: per-lead per-sample z-score, then ReLU
    split into positive/negative, concat -> 12-lead becomes 24 channels.
  * Two spiking conv blocks: Conv1d(24->32, k=7) and Conv1d(32->64, k=5), each
    Conv1d -> BatchNorm1d -> LIF (leaky, arctangent surrogate) -> temporal max-pool.
  * Recurrent spiking layer: I_t = W_in x_t + W_rec s_{t-1}, then LIF.
  * Global spike-rate pooling (mean spikes over time) -> FC -> logits.
  * Multi-label: sigmoid + weighted BCE (pos_weight = neg/pos ratio).

INFERRED (NOT in the paper — documented for reproducibility, R7):
  * LIF unrolled along the ECG time axis (same convention as the base S-TCN),
    so "time steps" = pooled signal length.
  * beta=0.9, threshold=1.0, arctangent surrogate alpha=2.0, max-pool kernel=2.
  * Recurrent hidden dim = 128 — recovered from the published param count.

Param counts match Table I exactly: plain SNN = 16,229, SNN-SRNN = 41,381.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


def _lif(beta: float = 0.9):
    return snn.Leaky(beta=beta,
                     spike_grad=surrogate.atan(alpha=2.0),
                     init_hidden=False,
                     reset_mechanism="subtract")


def _run_lif_along_time(lif, current: torch.Tensor) -> torch.Tensor:
    """Unroll a LIF along the ECG time axis: current (B, C, T) -> spikes (B, C, T)."""
    mem = lif.init_leaky()
    out = torch.zeros_like(current)
    for t in range(current.shape[-1]):
        sp, mem = lif(current[:, :, t], mem)
        out[:, :, t] = sp
    return out


def dual_polarity_encode(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-lead per-sample z-score, then ReLU split pos/neg, concat -> 2C channels."""
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    z = (x - mu) / sd
    return torch.cat([F.relu(z), F.relu(-z)], dim=1)


class SpikingConvBlock(nn.Module):
    """Conv1d -> BatchNorm1d -> LIF -> temporal MaxPool1d(2)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, beta: float = 0.9):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(c_out)
        self.lif = _lif(beta)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.conv(x))
        s = _run_lif_along_time(self.lif, h)
        return self.pool(s)


class SarhaddiSNN(nn.Module):
    """Feedforward plain SNN (dual-polarity + 2 spiking conv blocks + rate-pool + FC)."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__()
        self.block1 = SpikingConvBlock(24, 32, 7, beta)
        self.block2 = SpikingConvBlock(32, 64, 5, beta)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = dual_polarity_encode(x)          # (B, 24, T)
        s = self.block2(self.block1(enc))      # (B, 64, T/4)
        rate = s.mean(dim=-1)                  # (B, 64) global spike-rate pooling
        return self.fc(rate)


class SarhaddiSNNSRNN(nn.Module):
    """SNN-SRNN: adds a recurrent spiking layer I_t = W_in x_t + W_rec s_{t-1}."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9, hidden: int = 128):
        super().__init__()
        self.block1 = SpikingConvBlock(24, 32, 7, beta)
        self.block2 = SpikingConvBlock(32, 64, 5, beta)
        self.r_in = nn.Linear(64, hidden)        # W_in
        self.r_rec = nn.Linear(hidden, hidden)   # W_rec
        self.lif_rec = _lif(beta)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = dual_polarity_encode(x)          # (B, 24, T)
        s = self.block2(self.block1(enc))      # (B, 64, L), L = T/4
        B, C, L = s.shape
        mem = self.lif_rec.init_leaky()
        s_prev = torch.zeros(B, self.r_in.out_features, device=s.device)
        spikes = []
        for t in range(L):
            cur = self.r_in(s[:, :, t]) + self.r_rec(s_prev)
            s_t, mem = self.lif_rec(cur, mem)
            spikes.append(s_t)
            s_prev = s_t
        rate = torch.stack(spikes, dim=-1).mean(dim=-1)   # (B, hidden)
        return self.fc(rate)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 12, 1000)
    for cls in (SarhaddiSNN, SarhaddiSNNSRNN):
        m = cls()
        y = m(x)
        print(cls.__name__, "params =", count_parameters(m), "logits =", tuple(y.shape))
