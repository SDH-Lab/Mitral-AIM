"""Upsample learned spatial importance maps onto the 64³ volume.

These maps are not segmentations. They are the softmax weights used to pool
feature grids.
"""
import torch
from torch.nn import functional as F


def upsample_attention(attention, size=(64, 64, 64)):
    """Trilinear upsample of a spatial importance map to ``size``.

    Accepts [D, H, W], [N, D, H, W], or [N, 1, D, H, W].
    """
    tensor = torch.as_tensor(attention, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor[None, None]
    elif tensor.ndim == 4:
        tensor = tensor[:, None]
    elif tensor.ndim != 5 or tensor.shape[1] != 1:
        raise ValueError(f"unsupported attention shape {tuple(tensor.shape)}")
    if tuple(tensor.shape[-3:]) == tuple(size):
        return tensor
    return F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
