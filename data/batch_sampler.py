"""Length buckets packed under a valid-volume budget.

A single cycle longer than the budget is its own batch. Sequences are never
truncated. Each index appears once per epoch.
"""
import random
from collections import defaultdict

from torch.utils.data import Sampler


class VolumeBudgetBatchSampler(Sampler):
    def __init__(self, lengths, max_valid_volumes=80, shuffle=True, seed=0,
                 length_bucket=True, bucket_width=8):
        if max_valid_volumes < 1:
            raise ValueError("max_valid_volumes must be positive")
        self.length_bucket = bool(length_bucket)
        if self.length_bucket and int(bucket_width) < 1:
            raise ValueError("bucket_width must be positive when length bucketing is on")
        self.lengths = [int(length) for length in lengths]
        self.max_valid_volumes = int(max_valid_volumes)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_width = int(bucket_width)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _groups(self):
        if not self.length_bucket:
            return [list(range(len(self.lengths)))]
        buckets = defaultdict(list)
        for index, length in enumerate(self.lengths):
            buckets[length // self.bucket_width].append(index)
        return [buckets[key] for key in sorted(buckets)]

    def _batches(self, rng):
        packed = []
        for indices in self._groups():
            if self.shuffle:
                rng.shuffle(indices)
            current, volume_count = [], 0
            for index in indices:
                length = self.lengths[index]
                if current and volume_count + length > self.max_valid_volumes:
                    packed.append(current)
                    current, volume_count = [], 0
                current.append(index)
                volume_count += length
            if current:
                packed.append(current)
        if self.shuffle:
            rng.shuffle(packed)
        return packed

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        yield from self._batches(rng)

    def __len__(self):
        return len(self._batches(random.Random(self.seed + self.epoch)))


def group_optimizer_steps(clip_counts, target_clips):
    """Group physical batches until about ``target_clips`` clips are included.

    The final remainder is its own step so no clip is dropped. Each group
    stores the clip count that normalizes that step's loss.
    """
    if target_clips < 1:
        raise ValueError("target_clips must be positive")
    groups, current, count = [], [], 0
    for batch_index, size in enumerate(clip_counts):
        size = int(size)
        if size < 1:
            raise ValueError("physical batch is empty")
        if current and count >= target_clips:
            groups.append({"batch_indices": current, "patients": count})
            current, count = [], 0
        current.append(batch_index)
        count += size
    if current:
        groups.append({"batch_indices": current, "patients": count})
    return groups
