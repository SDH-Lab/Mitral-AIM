"""Config fields must change the modules they name, and spatial export must loop."""
import copy
import shutil
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.anatomy_motion_4d import build_model
from scripts.evaluate import _save_spatial


def _config():
    with (ROOT / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


class ConfigWiringTests(unittest.TestCase):
    def test_named_widths_and_dropout_change_the_network(self):
        cfg = _config()
        # The published MotionR_S config leaves this branch off. Turn it on
        # here so the width fields below still have a module to change.
        cfg["motion_branch"]["use_feature_transition"] = True
        changed = copy.deepcopy(cfg)
        changed["spatial_transition"]["hidden_channels"] = 128
        changed["transition_attention"]["attention_hidden"] = 32
        changed["selector_fusion"]["dropout"] = 0.35
        default = build_model(cfg)
        model = build_model(changed)
        self.assertEqual(default.transition_encoder.proj.out_channels, 256)
        self.assertIsNone(default.transition_encoder.out_proj)
        self.assertEqual(model.transition_encoder.proj.out_channels, 128)
        self.assertEqual(model.transition_encoder.out_proj.out_channels, 256)
        self.assertEqual(model.transition_pool.score[0].out_channels, 32)
        self.assertEqual(model.local_pool.score[0].out_channels, 64)
        self.assertEqual(model.selector_fusion[2].p, 0.35)
        self.assertEqual(model.start_head.net[3].p, 0.10)

    def test_spatial_export_survives_a_second_clip(self):
        class Dummy(torch.nn.Module):
            def forward(self, volumes, frame_mask, lengths, return_visualization=False):
                del lengths, return_visualization
                batch, steps = frame_mask.shape
                return {
                    "local_spatial_attention": torch.zeros(batch, steps, 1, 2, 2, 2),
                    "transition_spatial_attention": torch.zeros(batch, steps - 1, 1, 2, 2, 2),
                }

        items = []
        for name in ("clip_a", "clip_b"):
            items.append({
                "volume": torch.zeros(2, 1, 64, 64, 64),
                "label": 0.0,
                "patient_id": "p",
                "file_name": name,
                "length": 2,
            })
        out_dir = ROOT / "tests" / "_tmp_spatial_export"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        try:
            _save_spatial(Dummy(), items, torch.device("cpu"), out_dir, limit=2)
            self.assertTrue((out_dir / "clip_a.pt").is_file())
            self.assertTrue((out_dir / "clip_b.pt").is_file())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
