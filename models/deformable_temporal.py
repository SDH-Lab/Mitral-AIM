"""Optional 1D temporal deformable attention (TadTR / FTC style).

Default MotionR_S keeps dense ``TemporalEncoder`` + linear ``map_motion_to_state``.
These modules are only constructed when config selects:

* ``temporal_mixer=deform_attn`` — vanilla self-DeformAttn over state tokens (#2)
* ``motion_align=cross_deform`` — CrossDeform from state queries into motion memory (#4)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


def deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Deformable attention with grid_sample (sequence viewed as H=1, W=T)."""
    n_, _s, m_, d_ = value.shape
    _, lq_, _m, levels_, points_, _ = sampling_locations.shape
    value_list = value.split([int(h) * int(w) for h, w in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (h_, w_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(n_ * m_, d_, int(h_), int(w_))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        n_ * m_, 1, lq_, levels_ * points_,
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(n_, m_ * d_, lq_)
    return output.transpose(1, 2).contiguous()


class DeformAttn1D(nn.Module):
    """Sparse 1D temporal deformable attention (single feature level)."""

    def __init__(self, d_model=256, n_heads=4, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        d_per_head = d_model // n_heads
        if not _is_power_of_2(d_per_head):
            warnings.warn(
                "DeformAttn1D works best when d_model/n_heads is a power of 2 "
                f"(got {d_per_head})."
            )
        self.d_model = d_model
        self.n_levels = 1
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * self.n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (4.0 * math.pi / self.n_heads)
        grid_init = thetas.cos()[:, None]
        grid_init = grid_init.view(self.n_heads, 1, 1, 1).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, temporal_len, input_padding_mask=None):
        """
        query / input_flatten: [N, L, C]
        reference_points: [N, Lq, 1, 1] in [0, 1]
        temporal_len: scalar int length of the value sequence (usually L)
        input_padding_mask: [N, L] True on padding
        """
        n, len_q, _ = query.shape
        n, len_in, _ = input_flatten.shape
        if int(temporal_len) != len_in:
            raise ValueError(f"temporal_len {temporal_len} != value length {len_in}")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(n, len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points, 1,
        )
        attention_weights = self.attention_weights(query).view(
            n, len_q, self.n_heads, self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points,
        )

        if reference_points.shape[-1] != 1:
            raise ValueError(
                f"reference_points last dim must be 1, got {reference_points.shape[-1]}"
            )
        offset_normalizer = reference_points.new_tensor([float(temporal_len)])
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, None]
        )
        # Fake 2D coords for grid_sample: (t, y=0.5)
        sampling_locations = torch.cat(
            (sampling_locations, torch.ones_like(sampling_locations) * 0.5),
            dim=-1,
        )
        spatial_shapes = torch.stack(
            (
                torch.ones(1, dtype=torch.long, device=query.device),
                torch.as_tensor([temporal_len], dtype=torch.long, device=query.device),
            ),
            dim=-1,
        )
        output = deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights,
        )
        return self.output_proj(output)


class _DeformEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, n_heads, n_points):
        super().__init__()
        self.self_attn = DeformAttn1D(d_model, n_heads=n_heads, n_points=n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, reference_points, temporal_len, padding_mask):
        src2 = self.self_attn(src, reference_points, src, temporal_len, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src


class TemporalDeformEncoder(nn.Module):
    """Drop-in temporal mixer: same call signature as ``TemporalEncoder``."""

    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=512,
                 dropout=0.10, n_points=4):
        super().__init__()
        layer = _DeformEncoderLayer(d_model, dim_feedforward, dropout, nhead, n_points)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.d_model = d_model
        self.n_points = int(n_points)

    @staticmethod
    def _reference_points(batch, steps, device, dtype):
        # Centers of padded slots; padded queries are zeroed after the stack.
        ref = (torch.arange(steps, device=device, dtype=dtype) + 0.5) / float(max(steps, 1))
        return ref.view(1, steps, 1, 1).expand(batch, -1, -1, -1)

    def forward(self, tokens, key_padding_mask):
        """tokens [B, T, D], key_padding_mask True on padded positions."""
        if tokens.shape[1] == 0:
            return tokens
        batch, steps, _ = tokens.shape
        padding = key_padding_mask
        reference = self._reference_points(batch, steps, tokens.device, tokens.dtype)
        hidden = tokens
        for layer in self.layers:
            hidden = layer(hidden, reference, steps, padding)
        return hidden * (~key_padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)


class CrossTemporalDeformAlign(nn.Module):
    """State queries sample motion memory with 1D deformable attention.

    Replaces linear ``map_motion_to_state`` when ``motion_align=cross_deform``.
    """

    def __init__(self, d_model=256, nhead=4, n_points=4, dropout=0.10):
        super().__init__()
        self.attn = DeformAttn1D(d_model, n_heads=nhead, n_points=n_points)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    @staticmethod
    def _motion_reference_for_state(batch, state_steps, motion_steps, device, dtype):
        """Map each state index onto a reference in the motion timeline [0, 1]."""
        if motion_steps <= 0:
            return torch.zeros(batch, state_steps, 1, 1, device=device, dtype=dtype)
        if state_steps <= 1:
            ref = torch.full((state_steps,), 0.5, device=device, dtype=dtype)
        else:
            # t=0 -> first motion center; t=T-1 -> last motion center.
            start = 0.5 / float(motion_steps)
            end = (motion_steps - 0.5) / float(motion_steps)
            ref = torch.linspace(start, end, state_steps, device=device, dtype=dtype)
        return ref.view(1, state_steps, 1, 1).expand(batch, -1, -1, -1)

    def forward(self, h_state, h_motion, frame_mask, motion_mask):
        """
        h_state:  [B, T, C]
        h_motion: [B, T-1, C]
        frame_mask / motion_mask: True on valid positions
        returns:  [B, T, C]
        """
        batch, state_steps, _ = h_state.shape
        motion_steps = h_motion.shape[1]
        if motion_steps == 0:
            return h_state.new_zeros(batch, state_steps, self.d_model)

        reference = self._motion_reference_for_state(
            batch, state_steps, motion_steps, h_state.device, h_state.dtype,
        )
        padding = ~motion_mask
        sampled = self.attn(
            h_state, reference, h_motion, motion_steps, input_padding_mask=padding,
        )
        aligned = self.norm(self.dropout(sampled))
        return aligned * frame_mask.unsqueeze(-1).to(dtype=aligned.dtype)
