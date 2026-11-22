"""Read the audited 64³ clip cache. One item is one complete cardiac cycle."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def audit_manifest(manifest, cache):
    frame = pd.read_csv(manifest, dtype={"patient_id": str, "file_name": str})
    required = {"patient_id", "file_name", "fold", "binary"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if not frame.file_name.is_unique or frame.isna().any().any():
        raise ValueError("manifest file names must be unique and complete")
    if frame.groupby("patient_id").fold.nunique().max() != 1:
        raise ValueError("a patient appears in more than one fold")
    if frame.groupby("patient_id").binary.nunique().max() != 1:
        raise ValueError("a patient has inconsistent binary labels")
    if set(frame.fold.tolist()) != {0, 1, 2} or set(frame.binary.tolist()) != {0, 1}:
        raise ValueError("expected folds {0, 1, 2} and binary labels {0, 1}")
    metadata = json.loads((Path(cache) / "manifest.json").read_text())
    if metadata["version"] != "64_all_frames_clip_consistent_v1":
        raise ValueError(f"unexpected cache version {metadata['version']}")
    counts = {row["file_name"]: int(row["frames"]) for row in metadata["clips"]}
    if set(counts) != set(frame.file_name):
        raise ValueError("cache clips and manifest file names differ")
    if len(frame) != 623 or frame.patient_id.nunique() != 264:
        raise ValueError("expected 623 clips from 264 patients")
    if sum(counts.values()) != 14412 or min(counts.values()) != 3 or max(counts.values()) != 84:
        raise ValueError("unexpected cached frame counts")
    return frame[["patient_id", "file_name", "fold", "binary"]].copy(), counts


def split_folds(frame, fold):
    """Patient-disjoint split. ``fold`` is the validation partition."""
    fold = int(fold)
    train = frame[frame.fold != fold].copy()
    val = frame[frame.fold == fold].copy()
    if set(train.patient_id).intersection(set(val.patient_id)):
        raise ValueError("train and validation patients overlap")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def class_counts(frame):
    positive = int((frame.binary == 1).sum())
    negative = int((frame.binary == 0).sum())
    return {"negative": negative, "positive": positive, "pos_weight": negative / max(positive, 1)}


class ClipDataset(Dataset):
    def __init__(self, frame, cache, counts, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.cache = Path(cache)
        self.counts = counts
        self.augment = bool(augment)

    def __len__(self):
        return len(self.frame)

    def lengths(self):
        return [int(self.counts[name]) for name in self.frame.file_name.tolist()]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        array = np.load(self.cache / f"{row.file_name}.npy", allow_pickle=False)
        expected = (self.counts[row.file_name], 1, 64, 64, 64)
        if array.shape != expected:
            raise ValueError(f"{row.file_name} shape {array.shape} != {expected}")
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError(f"{row.file_name} is outside finite [0, 1]")
        volume = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        if self.augment:
            for axis in (-2, -1):
                if bool(torch.rand(()) < 0.5):
                    volume = volume.flip(axis)
            gain = 0.9 + 0.2 * float(torch.rand(()))
            volume = (gain * volume).clamp(0, 1)
        return {
            "volume": volume,
            "label": float(row.binary),
            "patient_id": str(row.patient_id),
            "file_name": str(row.file_name),
            "length": int(volume.shape[0]),
        }
