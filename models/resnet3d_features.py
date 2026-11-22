"""Classification 3D ResNet-18 that exposes intermediate feature maps.

Standard stem and four stride-2 stages with BatchNorm3d. On a 64³ input:

    layer1: 64 × 16 × 16 × 16
    layer2: 128 × 8 × 8 × 8
    layer3: 256 × 4 × 4 × 4
    layer4: 512 × 2 × 2 × 2

The network is created from scratch. Nothing in this module loads ImageNet,
MedicalNet, self-supervised, or previous-project weights. During ``train()``,
BatchNorm updates ``running_mean`` and ``running_var`` and the affine
parameters stay trainable.
"""
import contextlib

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet3D18Features(nn.Module):
    """3D ResNet-18 backbone. ``forward_stages`` returns layer2, layer3, layer4."""

    LAYER_CHANNELS = {"layer2": 128, "layer3": 256, "layer4": 512}

    def __init__(self, in_channels=1):
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError("in_channels must be 1 (intensity) or 2 (intensity + frame diff)")
        self.in_channels = int(in_channels)
        self.in_planes = 64
        self.conv1 = nn.Conv3d(self.in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(3, 2, 1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def normalize(self, volumes):
        """Scale intensity and optional frame-diff with the same mean/std.

        Intensity: ``(I - 0.15) / 0.25``. Frame-diff (signed or ``|Δ|``) is only
        divided by ``0.25`` so both stem channels share one amplitude scale.
        """
        mean, std = 0.15, 0.25
        if volumes.shape[1] == 1:
            return (volumes - mean) / std
        if volumes.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {volumes.shape[1]}"
            )
        intensity = (volumes[:, :1] - mean) / std
        motion = volumes[:, 1:] / std
        return torch.cat([intensity, motion], dim=1)

    def forward_stages(self, volumes):
        """volumes: [N, C, 64, 64, 64] with C matching ``in_channels`` -> (layer2, layer3, layer4)."""
        if volumes.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {volumes.shape[1]}"
            )
        x = self.normalize(volumes)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        if tuple(volumes.shape[-3:]) == (64, 64, 64):
            sizes = {
                "layer1": tuple(layer1.shape[-3:]),
                "layer2": tuple(layer2.shape[-3:]),
                "layer3": tuple(layer3.shape[-3:]),
                "layer4": tuple(layer4.shape[-3:]),
            }
            expected = {
                "layer1": (16, 16, 16),
                "layer2": (8, 8, 8),
                "layer3": (4, 4, 4),
                "layer4": (2, 2, 2),
            }
            if sizes != expected:
                raise RuntimeError(f"unexpected ResNet spatial sizes {sizes}")
        return layer2, layer3, layer4

    def forward_stages_checkpointed(self, volumes):
        """Same maps as ``forward_stages``, with one BatchNorm update per call.

        The backward pass recomputes the stages. Recomputation still uses
        batch statistics, and it does not write ``running_mean`` or
        ``running_var`` a second time.
        """
        if not (self.training and torch.is_grad_enabled()):
            return self.forward_stages(volumes)
        return _checkpoint_stages(self, volumes)


def _autocast_state(device_type):
    if torch.amp.is_autocast_available(device_type):
        device_state = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_state = None
    cpu_state = {
        "enabled": torch.is_autocast_enabled("cpu"),
        "dtype": torch.get_autocast_dtype("cpu"),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    return device_state, cpu_state


def _checkpoint_stages(encoder, volumes):
    """Checkpoint that keeps parameter gradients when the input does not require them."""
    device_type = volumes.device.type
    device_state, cpu_state = _autocast_state(device_type)
    # A custom Function is left out of the graph unless one input requires grad.
    # Volumes usually do not; parameters still must receive gradients.
    dummy = volumes.new_empty(()).requires_grad_()

    class _StageCheckpoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, _dummy):
            ctx.save_for_backward(inputs)
            ctx.device_type = device_type
            ctx.device_state = device_state
            ctx.cpu_state = cpu_state
            with torch.no_grad():
                return encoder.forward_stages(inputs)

        @staticmethod
        def backward(ctx, grad_layer2, grad_layer3, grad_layer4):
            (inputs,) = ctx.saved_tensors
            batch_norms = [module for module in encoder.modules() if isinstance(module, nn.BatchNorm3d)]
            flags = [module.track_running_stats for module in batch_norms]
            for module in batch_norms:
                module.track_running_stats = False
            input_grad = None
            try:
                device_ctx = (
                    torch.amp.autocast(device_type=ctx.device_type, **ctx.device_state)
                    if ctx.device_state is not None
                    else contextlib.nullcontext()
                )
                with torch.enable_grad(), device_ctx, torch.amp.autocast("cpu", **ctx.cpu_state):
                    inputs_in = inputs.detach().requires_grad_(True)
                    layer2, layer3, layer4 = encoder.forward_stages(inputs_in)
                    outputs, grads = [], []
                    for output, grad in (
                        (layer2, grad_layer2),
                        (layer3, grad_layer3),
                        (layer4, grad_layer4),
                    ):
                        if grad is not None:
                            outputs.append(output)
                            grads.append(grad)
                    if outputs:
                        torch.autograd.backward(outputs, grads)
                    input_grad = inputs_in.grad
            finally:
                for module, flag in zip(batch_norms, flags):
                    module.track_running_stats = flag
            return input_grad, None

    return _StageCheckpoint.apply(volumes, dummy)
