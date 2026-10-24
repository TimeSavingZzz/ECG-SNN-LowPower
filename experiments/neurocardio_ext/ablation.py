#!/usr/bin/env python3
"""Ablation variants for NeuroCardio (Task E/F).

Two groups, all trained under the SAME 60-epoch config so they are comparable:

  Encoder ablation — how much of the 4.5pp gap is caused by the encoder?
    * ab_polarity : replace the delta encoder with Sarhaddi's dual-polarity
                    encoder (per-lead z-score then ReLU pos/neg split).
    * ab_multiq   : delta encoder WITHOUT the slew-rate limit — the reference
                    may move several quanta per step (still quantized, still
                    fires on |diff| >= theta).

  Structural ablation — do the two non-standard blocks earn their keep?
    * ab_noattn   : spiking attention pool -> global mean spike-rate pool
                    (this is exactly what Sarhaddi uses)
    * ab_nogate   : membrane-gated residual -> plain block (no skip path)
    * ab_nodilate : dilation plan (1,2,4,8,16,32) -> all 1
    * ab_shallow  : 6 blocks -> 3 blocks

ab_full is the same-config control (delta encoder, 6 blocks, gate + attn pool).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from neurocardio.model import (NeuroCardio, STCNBlock, _lif,
                               _run_lif_along_time)
from neurocardio.sarhaddi import dual_polarity_encode


# --------------------------------------------------------------------------
# Encoder variants
# --------------------------------------------------------------------------
def delta_encode_multiquantum(x: torch.Tensor, theta: float = 0.15,
                              maxq: int = 8) -> torch.Tensor:
    """Delta encoder with the slew-rate limit removed.

    Upstream snaps the reference one quantum per step (`ref += ±theta`), so a
    fast-rising lead can never be tracked. Here the reference jumps to the
    nearest quantized level (up to ±maxq quanta per step) while spikes still
    fire on |diff| >= theta. Keeps the quantized-reference / spike semantics.
    """
    B, C, T = x.shape
    assert C == 12, f"expected 12 leads, got {C}"
    out = torch.zeros(B, 2 * C, T, device=x.device, dtype=x.dtype)
    ref = x[:, :, 0].clone()
    for t in range(T):
        diff = x[:, :, t] - ref
        on = diff >= theta
        off = diff <= -theta
        out[:, :C, t] = on.float()
        out[:, C:, t] = off.float()
        ref = ref + torch.clamp(torch.round(diff / theta), -maxq, maxq) * theta
    return out


class NeuroCardioPolarity(NeuroCardio):
    """Delta encoder -> Sarhaddi dual-polarity encoder (still 24 channels)."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta)
        # shadow the parent's staticmethod with an instance-level encoder
        self.delta_encode = lambda x, theta=0.15: dual_polarity_encode(x)


class NeuroCardioMultiQuantum(NeuroCardio):
    """Delta encoder without the slew-rate limit."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta)
        self.delta_encode = lambda x, theta=0.15: delta_encode_multiquantum(x, theta=theta)


# --------------------------------------------------------------------------
# Structural ablation
# --------------------------------------------------------------------------
class _MeanSpikeRatePool(nn.Module):
    """Global mean spike-rate pooling (Sarhaddi-style), no gate."""

    def forward(self, s: torch.Tensor):
        return s.mean(dim=-1), None


class NeuroCardioNoAttn(NeuroCardio):
    """Spiking attention pool -> global mean spike-rate pool."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta)
        self.pool = _MeanSpikeRatePool()


class STCNBlockNoGate(nn.Module):
    """STCNBlock with the membrane-gated residual removed (main path only)."""

    def __init__(self, c_in: int, c_out: int, dilation: int, pool: int = 1,
                 beta: float = 0.9):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.lif1 = _lif(beta=beta)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.lif2 = _lif(beta=beta)
        self.tpool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, s_in: torch.Tensor) -> torch.Tensor:
        h1 = self.bn1(self.conv1(s_in))
        s1 = _run_lif_along_time(self.lif1, h1)
        h2 = self.bn2(self.conv2(s1))
        s2 = _run_lif_along_time(self.lif2, h2)
        return self.tpool(s2)


class NeuroCardioNoGate(NeuroCardio):
    """Membrane-gated residual -> plain blocks (no skip path)."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta)
        in_ch = self.stem_conv.out_channels
        blocks = []
        for c, d, p in zip((64, 128, 128, 256, 256, 384),
                           (1, 2, 4, 8, 16, 32), (1, 2, 1, 2, 1, 2)):
            blocks.append(STCNBlockNoGate(in_ch, c, dilation=d, pool=p, beta=beta))
            in_ch = c
        self.blocks = nn.ModuleList(blocks)


class NeuroCardioNoDilation(NeuroCardio):
    """Dilation plan (1,2,4,8,16,32) -> all 1."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta,
                         dilation_plan=(1, 1, 1, 1, 1, 1))


class NeuroCardioShallow(NeuroCardio):
    """6 blocks -> 3 blocks."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta,
                         channel_plan=(64, 128, 256),
                         dilation_plan=(1, 2, 4),
                         pool_plan=(1, 2, 2))


# --------------------------------------------------------------------------
# Combined models (P1): stack the winning single changes one at a time so the
# effect of each addition stays attributable to exactly one change.
# --------------------------------------------------------------------------
class NeuroCardioShallowPolarity(NeuroCardio):
    """P1a = ab_shallow + dual-polarity encoder (one change vs ab_shallow)."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta,
                         channel_plan=(64, 128, 256),
                         dilation_plan=(1, 2, 4),
                         pool_plan=(1, 2, 2))
        self.delta_encode = lambda x, theta=0.15: dual_polarity_encode(x)


class NeuroCardioImproved(NeuroCardio):
    """P1b = P1a + mean spike-rate pool (one change vs P1a)."""

    def __init__(self, num_classes: int = 5, beta: float = 0.9):
        super().__init__(num_classes=num_classes, beta=beta,
                         channel_plan=(64, 128, 256),
                         dilation_plan=(1, 2, 4),
                         pool_plan=(1, 2, 2))
        self.delta_encode = lambda x, theta=0.15: dual_polarity_encode(x)
        self.pool = _MeanSpikeRatePool()


if __name__ == "__main__":
    from neurocardio.model import count_parameters
    x = torch.randn(2, 12, 1000)
    for cls in (NeuroCardioPolarity, NeuroCardioMultiQuantum, NeuroCardioNoAttn,
                NeuroCardioNoGate, NeuroCardioNoDilation, NeuroCardioShallow,
                NeuroCardioShallowPolarity, NeuroCardioImproved):
        m = cls()
        y = m(x)
        print(f"{cls.__name__:28s} params={count_parameters(m):>9,} logits={tuple(y.shape)}")
