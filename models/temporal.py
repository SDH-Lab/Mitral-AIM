"""Continuous cardiac-time encoding and variable-length temporal encoders."""
import math

import torch
from torch import nn


class ContinuousTimeEncoding(nn.Module):
    """Fixed Fourier features of normalized cardiac time in ``[0, 1]``.

    This is not a learned table of length 84. Frequencies run from one cycle
    over the cardiac cycle up to ``max_cycles`` cycles, so both coarse phase
    and finer within-cycle timing are represented for any sequence length.
    """

    def __init__(self, d_model=256, max_cycles=32.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")
        half = d_model // 2
        cycles = torch.exp(torch.linspace(0, math.log(float(max_cycles)), half))
        self.register_buffer("omega", (2.0 * math.pi) * cycles, persistent=False)

    def forward(self, tau):
        angles = tau.float().unsqueeze(-1) * self.omega
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return encoding.to(dtype=tau.dtype)


class TemporalEncoder(nn.Module):
    """Two-layer pre-norm Transformer. Padding keys are masked."""

    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=512,
                 dropout=0.10):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.d_model = d_model

    def forward(self, tokens, key_padding_mask):
        """tokens [B, T, D], key_padding_mask True on padded positions."""
        if tokens.shape[1] == 0:
            return tokens
        # A row of only padding makes attention softmax undefined. Unmask one
        # dummy key, then zero that row with the original mask.
        padding = key_padding_mask
        all_pad = padding.all(dim=1)
        if bool(all_pad.any()):
            padding = padding.clone()
            padding[all_pad, 0] = False
        hidden = self.encoder(tokens, src_key_padding_mask=padding)
        return hidden * (~key_padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)


def normalized_cardiac_time(lengths, steps, device):
    """``tau_t = t / (T - 1)`` on valid frames. Padded steps stay 0."""
    tau = torch.zeros(lengths.shape[0], steps, device=device, dtype=torch.float32)
    index = torch.arange(steps, device=device, dtype=torch.float32)
    for batch_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 0:
            continue
        if length == 1:
            tau[batch_index, 0] = 0.0
        else:
            tau[batch_index, :length] = index[:length] / float(length - 1)
    return tau


def map_motion_to_state(h_motion, lengths):
    """Broadcast [B, T-1, D] motion states onto [B, T, D] anatomical times.

    Endpoints copy the nearest transition. Interior times average the
    transition that ends there and the transition that starts there.
    """
    batch, transitions, dim = h_motion.shape
    steps = transitions + 1
    h_left = torch.cat([h_motion.new_zeros(batch, 1, dim), h_motion], dim=1)
    h_right = torch.cat([h_motion, h_motion.new_zeros(batch, 1, dim)], dim=1)
    weight_left = h_motion.new_zeros(batch, steps)
    weight_right = h_motion.new_zeros(batch, steps)
    for batch_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length < 2:
            continue
        weight_right[batch_index, 0] = 1
        weight_left[batch_index, length - 1] = 1
        if length > 2:
            weight_left[batch_index, 1:length - 1] = 0.5
            weight_right[batch_index, 1:length - 1] = 0.5
    return weight_left.unsqueeze(-1) * h_left + weight_right.unsqueeze(-1) * h_right
