"""Optional temporal_mixer / motion_align plugins must default to MotionR_S."""
import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.anatomy_motion_4d import build_model
from models.deformable_temporal import CrossTemporalDeformAlign, TemporalDeformEncoder
from models.temporal import TemporalEncoder


def _config():
    with (ROOT / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


class TemporalPluginTests(unittest.TestCase):
    def test_default_keeps_dense_linear_modules(self):
        model = build_model(_config())
        self.assertEqual(model.temporal_mixer, "dense")
        self.assertEqual(model.motion_align, "linear")
        self.assertIsInstance(model.state_transformer, TemporalEncoder)
        self.assertIsNone(model.state_deform_encoder)
        self.assertIsNone(model.cross_motion_align)

    def test_deform_attn_replaces_state_transformer(self):
        cfg = _config()
        cfg["temporal_options"]["temporal_mixer"] = "deform_attn"
        model = build_model(cfg)
        self.assertIsNone(model.state_transformer)
        self.assertIsInstance(model.state_deform_encoder, TemporalDeformEncoder)
        self.assertIsNone(model.cross_motion_align)

    def test_cross_deform_adds_align_module(self):
        cfg = _config()
        cfg["temporal_options"]["motion_align"] = "cross_deform"
        model = build_model(cfg)
        self.assertIsInstance(model.state_transformer, TemporalEncoder)
        self.assertIsInstance(model.cross_motion_align, CrossTemporalDeformAlign)

    def test_plugins_forward_shapes(self):
        cfg = _config()
        cfg["motion_branch"]["use_raw_difference"] = True
        cfg["motion_branch"]["use_feature_transition"] = False
        cfg["spatial_prior"]["use_sector_mask"] = True
        for mixer, align in (
            ("dense", "linear"),
            ("deform_attn", "linear"),
            ("dense", "cross_deform"),
        ):
            with self.subTest(mixer=mixer, align=align):
                local = copy.deepcopy(cfg)
                local["temporal_options"]["temporal_mixer"] = mixer
                local["temporal_options"]["motion_align"] = align
                model = build_model(local).eval()
                volumes = torch.randn(2, 5, 1, 64, 64, 64)
                frame_mask = torch.tensor(
                    [[True, True, True, True, False], [True, True, True, False, False]],
                )
                with torch.no_grad():
                    out = model(volumes, frame_mask)
                self.assertEqual(tuple(out["logits"].shape), (2,))
                self.assertEqual(out["temporal_mixer"], mixer)
                self.assertEqual(out["motion_align"], align)


if __name__ == "__main__":
    unittest.main()
