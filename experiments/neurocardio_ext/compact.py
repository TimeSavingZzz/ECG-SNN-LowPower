#!/usr/bin/env python3
"""Compact NeuroCardio (~54K params) for fair comparison with Sarhaddi's
compact models (SNN 16K / SNN-SRNN 41K / CNN 55K / LSTM 53K).

Same architecture as NeuroCardio (delta encoder + dilated S-TCN stack +
spiking attention pool + LIF classifier), just shrunk: 3 blocks, small
channels, and a small dense head. Keeps every design choice (membrane-gated
residuals, event-driven attention pooling) so the comparison isolates *scale*
from *architecture*.
"""
from __future__ import annotations

import torch.nn as nn

from neurocardio.model import NeuroCardio


class CompactNeuroCardio(NeuroCardio):
    def __init__(self, num_classes: int = 5, beta: float = 0.9,
                 head_channels: int = 32):
        super().__init__(
            num_classes=num_classes,
            beta=beta,
            T_dense=8,
            stem_channels=16,
            channel_plan=(32, 48, 64),
            dilation_plan=(1, 2, 4),
            pool_plan=(1, 2, 2),
        )
        # Shrink the dense head from the default 256-wide one.
        # The LIF layers (self.lif_fc / self.lif_out) are dimension-agnostic,
        # so only the linear/bn projections need replacing.
        in_ch = 64  # == channel_plan[-1]
        self.fc1 = nn.Linear(in_ch, head_channels)
        self.fc1_bn = nn.BatchNorm1d(head_channels)
        self.fc2 = nn.Linear(head_channels, num_classes)
