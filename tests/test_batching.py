"""length_bucket and bucket_width must change which clips share a batch."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.batch_sampler import VolumeBudgetBatchSampler
from training.trainer import make_loader


class _Lengths:
    def __init__(self, lengths):
        self._lengths = list(lengths)

    def lengths(self):
        return list(self._lengths)


class BatchingConfigTests(unittest.TestCase):
    def test_bucket_width_and_length_bucket_change_packing(self):
        lengths = [30, 5, 30]
        narrow = VolumeBudgetBatchSampler(
            lengths, max_valid_volumes=40, shuffle=False, length_bucket=True, bucket_width=10,
        )
        wide = VolumeBudgetBatchSampler(
            lengths, max_valid_volumes=40, shuffle=False, length_bucket=True, bucket_width=40,
        )
        flat = VolumeBudgetBatchSampler(
            lengths, max_valid_volumes=40, shuffle=False, length_bucket=False, bucket_width=10,
        )
        self.assertEqual(list(narrow), [[1], [0], [2]])
        self.assertEqual(list(wide), [[0, 1], [2]])
        self.assertEqual(list(flat), [[0, 1], [2]])
        self.assertEqual(sorted(index for batch in narrow for index in batch), [0, 1, 2])

    def test_loader_receives_both_settings(self):
        _loader, sampler = make_loader(
            _Lengths([30, 5, 30]), 40, False, 0, 0, torch.device("cpu"),
            length_bucket=True, bucket_width=10,
        )
        self.assertTrue(sampler.length_bucket)
        self.assertEqual(sampler.bucket_width, 10)
        self.assertEqual(list(sampler), [[1], [0], [2]])


if __name__ == "__main__":
    unittest.main()
