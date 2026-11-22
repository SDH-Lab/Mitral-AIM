"""Spatially resolved anatomical transition between consecutive feature maps."""
import torch
from torch import nn


class SpatialTransitionEncoder(nn.Module):
    """1×1 projection, two 3×3 convolutions, residual from the projected map.

    GroupNorm is used because a physical batch may contain few transitions.
    """

    def __init__(self, in_channels, hidden_channels=256, output_channels=None, dropout=0.10, groups=16):
        super().__init__()
        if output_channels is None:
            output_channels = hidden_channels
        if hidden_channels % groups != 0:
            raise ValueError(f"hidden_channels {hidden_channels} must divide into {groups} groups")
        self.output_channels = int(output_channels)
        self.proj = nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, hidden_channels)
        self.conv_a = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm_a = nn.GroupNorm(groups, hidden_channels)
        self.drop = nn.Dropout3d(dropout)
        self.conv_b = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm_b = nn.GroupNorm(groups, hidden_channels)
        self.act = nn.GELU()
        self.out_proj = None
        if self.output_channels != hidden_channels:
            self.out_proj = nn.Conv3d(hidden_channels, self.output_channels, kernel_size=1, bias=False)

    def forward(self, transition):
        """transition: [N, C_in, D, H, W] -> [N, output_channels, D, H, W]."""
        if transition.numel() == 0:
            return transition.new_zeros(0, self.output_channels, *transition.shape[-3:])
        projected = self.act(self.norm1(self.proj(transition)))
        hidden = self.drop(self.act(self.norm_a(self.conv_a(projected))))
        hidden = self.norm_b(self.conv_b(hidden))
        output = self.act(projected + hidden)
        if self.out_proj is not None:
            output = self.out_proj(output)
        return output


def build_transition_input(left, right, representation):
    """Channel-wise transition tensor from a consecutive feature pair.

    ``left`` and ``right`` are [N, C, D, H, W]. The difference is in feature
    space and is not divided by the time gap.
    """
    if representation == "full":
        return torch.cat([left, right, right - left], dim=1)
    if representation == "pair":
        return torch.cat([left, right], dim=1)
    if representation == "difference":
        return right - left
    raise ValueError(f"Unknown transition representation: {representation}")


def transition_in_channels(local_channels, representation):
    if representation == "full":
        return local_channels * 3
    if representation == "pair":
        return local_channels * 2
    if representation == "difference":
        return local_channels
    raise ValueError(f"Unknown transition representation: {representation}")
