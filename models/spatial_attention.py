"""Spatial pooling that can keep a small anatomical region before collapse."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def sector_at_resolution(volumes, spatial):
    """Fraction of each pooled cell inside the ultrasound FOV (volume > 0).

    ``volumes``: [N, 1, D0, H0, W0]. Returns [N, 1, *spatial] in [0, 1].
    """
    mask = (volumes > 0).to(dtype=torch.float32)
    return F.adaptive_avg_pool3d(mask, spatial)


class SpatialAttentionPool3d(nn.Module):
    """Softmax over spatial locations, then a weighted sum of channel vectors."""

    def __init__(self, in_channels, hidden_channels, out_dim=None):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )
        self.proj = None
        if out_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_channels, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, features, sector=None, return_map=False):
        """features: [N, C, D, H, W]. Attention is normalized over D*H*W.

        Optional ``sector`` [N, 1, D, H, W] is a field-of-view prior:
        logits += log(sector); sector==0 locations are masked out.
        """
        if features.numel() == 0:
            channels = self.proj[0].out_features if self.proj is not None else features.shape[1]
            empty = features.new_zeros(0, channels)
            attn = features.new_zeros(0, 1, *features.shape[-3:]) if return_map else None
            return empty, attn
        logits = self.score(features).float()
        if sector is not None:
            if sector.shape[-3:] != features.shape[-3:] or sector.shape[0] != features.shape[0]:
                raise ValueError(
                    f"sector shape {tuple(sector.shape)} incompatible with features {tuple(features.shape)}"
                )
            sector = sector.to(device=logits.device, dtype=logits.dtype)
            logits = logits + sector.clamp_min(1e-6).log()
            logits = logits.masked_fill(sector <= 0, float("-inf"))
        flat = logits.flatten(2)
        # All-outside FOV → fall back to uniform over the volume.
        all_masked = ~torch.isfinite(flat).any(dim=-1)
        if all_masked.any():
            flat = flat.clone()
            flat[all_masked] = 0
        alpha = torch.softmax(flat, dim=-1).view_as(logits)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = (features.float() * alpha).sum(dim=(-3, -2, -1)).to(dtype=features.dtype)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return pooled, (alpha if return_map else None)


class SpatialGapPool3d(nn.Module):
    """Ablation: global average pooling in place of spatial attention."""

    def __init__(self, in_channels, out_dim=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = None
        if out_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_channels, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, features, sector=None, return_map=False):
        if features.numel() == 0:
            channels = self.proj[0].out_features if self.proj is not None else features.shape[1]
            empty = features.new_zeros(0, channels)
            attn = features.new_zeros(0, 1, *features.shape[-3:]) if return_map else None
            return empty, attn
        if sector is None:
            pooled = self.pool(features).flatten(1)
            attn = None
            if return_map:
                spatial = features.shape[-3:]
                attn = features.new_full(
                    (features.shape[0], 1, *spatial), 1.0 / float(math.prod(spatial)),
                )
        else:
            if sector.shape[-3:] != features.shape[-3:] or sector.shape[0] != features.shape[0]:
                raise ValueError(
                    f"sector shape {tuple(sector.shape)} incompatible with features {tuple(features.shape)}"
                )
            sector = sector.to(device=features.device, dtype=torch.float32)
            weight = sector / sector.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
            pooled = (features.float() * weight).sum(dim=(-3, -2, -1)).to(dtype=features.dtype)
            attn = weight if return_map else None
        if self.proj is not None:
            pooled = self.proj(pooled)
        return pooled, attn
