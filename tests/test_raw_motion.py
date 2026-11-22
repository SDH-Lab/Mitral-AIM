"""Raw frame difference, feature transition, and their fusion."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.anatomy_motion_4d import AnatomyMotion4DNet
from models.raw_motion import RawDifferenceEncoder, valid_raw_motion_pairs


def _device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 5:
        free = []
        for index in range(torch.cuda.device_count()):
            free_bytes = torch.cuda.mem_get_info(index)[0]
            free.append((free_bytes, index))
        free.sort(reverse=True)
        return torch.device(f"cuda:{free[0][1]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(length=3, steps=4):
    generator = torch.Generator().manual_seed(0)
    volumes = torch.rand(1, steps, 1, 64, 64, 64, generator=generator)
    mask = torch.zeros(1, steps, dtype=torch.bool)
    mask[0, :length] = True
    volumes[0, length:] = -7
    return volumes, mask


class RawMotionTests(unittest.TestCase):
    def test_valid_pairs_ignore_padding(self):
        volumes, mask = _batch()
        poisoned = volumes.clone()
        poisoned[0, 3] = 9
        pair_index = (mask[:, :-1] & mask[:, 1:]).reshape(-1).nonzero(as_tuple=False).flatten()
        clean = valid_raw_motion_pairs(volumes, pair_index, 1, 3)
        dirty = valid_raw_motion_pairs(poisoned, pair_index, 1, 3)
        self.assertEqual(tuple(clean.shape), (2, 2, 64, 64, 64))
        self.assertTrue(torch.equal(clean, dirty))
        self.assertGreaterEqual(float(clean[:, 1].min()), 0)
        reference = volumes[0, 1] - volumes[0, 0]
        self.assertTrue(torch.allclose(clean[0, 0], reference))

    def test_encoder_spatial_shape(self):
        encoder = RawDifferenceEncoder()
        output = encoder(torch.rand(1, 2, 64, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 128, 8, 8, 8))

    def test_motion_modes_and_gradients(self):
        device = _device()
        volumes, mask = _batch(length=3, steps=3)
        volumes, mask = volumes.to(device), mask.to(device)
        lengths = torch.tensor([3], device=device)
        modes = {
            "R": (True, False),
            "F": (False, True),
            "RF": (True, True),
            "0": (False, False),
        }
        for name, (raw, feature) in modes.items():
            model = AnatomyMotion4DNet(
                variant="M2",
                use_raw_difference=raw,
                use_feature_transition=feature,
                checkpoint_encoder=False,
            ).to(device).train()
            output = model(volumes, mask, lengths, return_visualization=True)
            self.assertEqual(tuple(output["logits"].shape), (1,))
            self.assertTrue(torch.isfinite(output["logits"]).all())
            self.assertTrue(torch.isfinite(output["H_motion"]).all())
            if raw and feature:
                self.assertEqual(tuple(output["raw_motion_feature"].shape), (1, 2, 128, 8, 8, 8))
                self.assertEqual(tuple(output["feature_transition_feature"].shape), (1, 2, 256, 8, 8, 8))
                self.assertEqual(tuple(output["fused_motion_feature"].shape), (1, 2, 256, 8, 8, 8))
            if feature and not raw:
                self.assertEqual(tuple(output["feature_transition_feature"].shape), (1, 2, 256, 8, 8, 8))
            if raw and not feature:
                self.assertEqual(tuple(output["raw_motion_feature"].shape), (1, 2, 128, 8, 8, 8))
                self.assertEqual(tuple(output["fused_motion_feature"].shape), (1, 2, 256, 8, 8, 8))
            if name == "0":
                self.assertIsNone(model.raw_difference_encoder)
                self.assertIsNone(model.transition_encoder)
                self.assertEqual(float(output["H_motion"].abs().sum()), 0)
            output["logits"].sum().backward()
            if raw:
                grad = model.raw_difference_encoder.net[0].weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all() and float(grad.abs().sum()) > 0)
            if feature:
                grad = model.transition_encoder.proj.weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all() and float(grad.abs().sum()) > 0)
            if raw and feature:
                grad = model.motion_fusion.proj.weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all() and float(grad.abs().sum()) > 0)
            if name != "0":
                grad = model.transition_pool.score[0].weight.grad
                motion_grad = model.motion_transformer.encoder.layers[0].linear1.weight.grad
                self.assertIsNotNone(grad)
                self.assertIsNotNone(motion_grad)
                self.assertTrue(torch.isfinite(grad).all() and float(grad.abs().sum()) > 0)
                self.assertTrue(torch.isfinite(motion_grad).all() and float(motion_grad.abs().sum()) > 0)


if __name__ == "__main__":
    unittest.main()
