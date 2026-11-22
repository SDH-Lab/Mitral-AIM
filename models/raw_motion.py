"""Low-level raw inter-frame motion, fused with learned anatomical transitions."""
import torch
from torch import nn


def _kaiming_conv(module):
    for child in module.modules():
        if isinstance(child, nn.Conv3d):
            nn.init.kaiming_normal_(child.weight, mode="fan_out", nonlinearity="relu")


class RawDifferenceEncoder(nn.Module):
    """Map a signed difference and its magnitude from 64³ to 128×8³."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )
        _kaiming_conv(self)

    def forward(self, raw_motion):
        """raw_motion: [N, 2, 64, 64, 64] -> [N, 128, 8, 8, 8]."""
        if raw_motion.numel() == 0:
            return raw_motion.new_zeros(0, 128, 8, 8, 8)
        features = self.net(raw_motion)
        if tuple(features.shape[-3:]) != (8, 8, 8) or features.shape[1] != 128:
            raise RuntimeError(f"raw difference encoder produced {tuple(features.shape)}")
        return features


class RawMotionProjection(nn.Module):
    """1×1 projection of the raw-motion feature map. Channels stay 128."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(128, 128, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(16, 128)
        self.act = nn.GELU()
        _kaiming_conv(self)

    def forward(self, features):
        if features.numel() == 0:
            return features
        return self.act(self.norm(self.proj(features)))


class RawToMotionChannels(nn.Module):
    """Lift raw-only features from 128 to 256 before spatial attention."""

    def __init__(self, out_channels=256):
        super().__init__()
        self.proj = nn.Conv3d(128, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(16, out_channels)
        self.act = nn.GELU()
        _kaiming_conv(self)

    def forward(self, features):
        if features.numel() == 0:
            return features.new_zeros(0, self.proj.out_channels, *features.shape[-3:])
        return self.act(self.norm(self.proj(features)))


class RawFeatureMotionFusion(nn.Module):
    """Concatenate raw and feature motion maps, then refine them jointly."""

    def __init__(self, in_channels, out_channels=256, dropout=0.10):
        super().__init__()
        if out_channels % 16 != 0:
            raise ValueError("fusion output channels must be divisible by 16")
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(16, out_channels)
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(16, out_channels)
        self.drop = nn.Dropout3d(dropout)
        self.act = nn.GELU()
        _kaiming_conv(self)

    def forward(self, features):
        if features.numel() == 0:
            return features.new_zeros(0, self.proj.out_channels, *features.shape[-3:])
        projected = self.act(self.norm1(self.proj(features)))
        refined = self.norm2(self.conv(projected))
        return self.drop(self.act(projected + refined))


def valid_raw_motion_pairs(volumes, pair_index, batch, transitions):
    """Signed difference and its absolute value for valid frame pairs only.

    ``volumes`` is the tensor passed into the network, with no extra normalization.
    ``pair_index`` selects transitions where both neighbors are real frames.
    """
    left = volumes[:, :-1].reshape(batch * transitions, *volumes.shape[2:]).index_select(0, pair_index)
    right = volumes[:, 1:].reshape(batch * transitions, *volumes.shape[2:]).index_select(0, pair_index)
    difference = right - left
    return torch.cat([difference, difference.abs()], dim=1)
