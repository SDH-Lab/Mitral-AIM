"""Scratch 3D ResNet-18: live BatchNorm, fixed normalization, standard pyramid."""
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.resnet3d_features import ResNet3D18Features
from models.spatial_transition import SpatialTransitionEncoder


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pair():
    torch.manual_seed(0)
    device = _device()
    reference = ResNet3D18Features().to(device).train()
    copied = ResNet3D18Features().to(device).train()
    copied.load_state_dict(reference.state_dict())
    return reference, copied


class ResNetBatchNormTests(unittest.TestCase):
    def test_train_mode_updates_running_stats_once_under_checkpoint(self):
        reference, checkpointed = _pair()
        volumes = torch.rand(2, 1, 64, 64, 64, device=_device())
        reference.train()
        checkpointed.train()
        self.assertTrue(all(module.training for module in reference.modules() if isinstance(module, nn.BatchNorm3d)))

        layer2, layer3, layer4 = reference.forward_stages(volumes)
        (layer2.float().sum() + layer3.float().sum() + layer4.float().sum()).backward()
        layer2c, layer3c, layer4c = checkpointed.forward_stages_checkpointed(volumes)
        (layer2c.float().sum() + layer3c.float().sum() + layer4c.float().sum()).backward()

        for left, right in zip(reference.modules(), checkpointed.modules()):
            if not isinstance(left, nn.BatchNorm3d):
                continue
            self.assertEqual(int(left.num_batches_tracked), 1)
            self.assertEqual(int(right.num_batches_tracked), 1)
            self.assertFalse(torch.equal(left.running_mean, torch.zeros_like(left.running_mean)))
            self.assertTrue(torch.allclose(left.running_mean, right.running_mean))
            self.assertTrue(torch.allclose(left.running_var, right.running_var))
            self.assertTrue(left.weight.requires_grad and left.bias.requires_grad)
            self.assertTrue(right.weight.grad is not None and torch.isfinite(right.weight.grad).all())
            self.assertTrue(right.track_running_stats)

        grad_gap = (reference.conv1.weight.grad - checkpointed.conv1.weight.grad).abs().max()
        grad_scale = reference.conv1.weight.grad.abs().mean().clamp_min(1e-6)
        self.assertLess(float(grad_gap / grad_scale), 0.02)
        self.assertTrue(torch.allclose(
            reference.layer4[1].conv2.weight.grad,
            checkpointed.layer4[1].conv2.weight.grad,
            atol=1e-4,
            rtol=1e-4,
        ))
        with torch.no_grad():
            checkpointed.forward_stages(volumes)
        self.assertEqual(int(checkpointed.bn1.num_batches_tracked), 2)
        self.assertEqual(tuple(layer2c.shape), (2, 128, 8, 8, 8))
        self.assertEqual(tuple(layer3c.shape), (2, 256, 4, 4, 4))
        self.assertEqual(tuple(layer4c.shape), (2, 512, 2, 2, 2))

    def test_normalization_init_and_strides(self):
        encoder = ResNet3D18Features()
        volumes = torch.tensor([0.0, 0.15, 0.40], dtype=torch.float32).view(3, 1, 1, 1, 1)
        expected = torch.tensor([-0.6, 0.0, 1.0], dtype=torch.float32).view(3, 1, 1, 1, 1)
        self.assertTrue(torch.equal(encoder.normalize(volumes), expected))
        for module in encoder.modules():
            if isinstance(module, nn.BatchNorm3d):
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                self.assertNotIsInstance(module, nn.GroupNorm)
            if isinstance(module, nn.Conv3d):
                self.assertTrue(module.weight.requires_grad)
        self.assertEqual(encoder.layer4[0].conv1.stride, (2, 2, 2))
        self.assertEqual(encoder.layer2[0].conv1.stride, (2, 2, 2))
        encoder.eval()
        self.assertTrue(all(not module.training for module in encoder.modules() if isinstance(module, nn.BatchNorm3d)))
        transition = SpatialTransitionEncoder(384)
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in transition.modules()))


if __name__ == "__main__":
    unittest.main()
