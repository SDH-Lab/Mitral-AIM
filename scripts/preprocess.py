"""Resize every original frame to 64^3 with one intensity scale per clip."""
import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.nn import functional as F

VERSION = "64_all_frames_clip_consistent_v1"
FRAME_NAME = re.compile(r"frame_(\d+)\.mha")


def spatial_resize(array):
    tensor = torch.from_numpy(array.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(64, 64, 64), mode="trilinear", align_corners=False)
    return resized[0, 0].numpy()


def normalize_clip(volume):
    positive = volume[volume > 0]
    scale = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    scale = max(scale, 1e-6)
    return np.clip(volume / scale, 0, 1), scale


def _as_list(value):
    if isinstance(value, dict):
        return {key: _as_list(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_list(item) for item in value]
    if hasattr(value, "tolist"):
        return _as_list(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def prepare(task):
    root, name, out, fallback = task
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    paths = sorted(
        (Path(root) / name).glob("frame_*.mha"),
        key=lambda path: int(FRAME_NAME.fullmatch(path.name).group(1)),
    )
    if len(paths) < 2:
        raise ValueError(f"{name} needs at least two frames")
    indices = [int(FRAME_NAME.fullmatch(path.name).group(1)) for path in paths]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} has duplicate frame indices")

    volumes = []
    geometry = None
    for path in paths:
        source = Path(fallback.get(str(path.relative_to(root)), str(path)))
        image = sitk.ReadImage(str(source))
        geo = (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())
        if geometry is None:
            geometry = geo
        elif not all(np.allclose(left, right, rtol=0, atol=1e-6) for left, right in zip(geometry, geo)):
            raise ValueError("Frame geometry differs: require explicit common physical resampling")
        array = sitk.GetArrayFromImage(image)
        if not np.isfinite(array).all():
            raise ValueError(f"{source} contains non-finite values")
        volumes.append(spatial_resize(array))

    stacked = np.stack(volumes)
    raw_mean = stacked.mean((1, 2, 3))
    raw_delta = np.abs(np.diff(stacked, axis=0)).mean((1, 2, 3))
    normalized, scale = normalize_clip(stacked)
    result = dict(
        file_name=name,
        frames=len(paths),
        original_frame_indices=indices,
        geometry=_as_list(geometry),
        clip_positive_q99=scale,
        frame_mean_before=raw_mean.tolist(),
        frame_mean_after=normalized.mean((1, 2, 3)).tolist(),
        adjacent_abs_change_before=raw_delta.tolist(),
        adjacent_abs_change_after=np.abs(np.diff(normalized, axis=0)).mean((1, 2, 3)).tolist(),
        temporal_smoothing=False,
        clipped_fraction=float(np.mean(normalized >= 1)),
    )
    np.save(Path(out) / f"{name}.npy", normalized[:, None].astype(np.float16))
    return result


def load_fallback(path):
    fallback = {}
    if not path:
        return fallback
    for record in json.loads(Path(path).read_text()):
        if not record.get("verified"):
            raise ValueError("fallback entry is not marked verified")
        backup = Path(record["backup"])
        digest = hashlib.sha256(backup.read_bytes()).hexdigest()
        if digest != record["backup_sha256"]:
            raise ValueError(f"fallback hash mismatch for {record['relative']}")
        fallback[record["relative"]] = str(backup)
    return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with patient_id, file_name, fold, binary")
    parser.add_argument("--root", required=True, help="Directory of per-clip folders containing frame_*.mha")
    parser.add_argument("--fallback", default="", help="Optional JSON of verified replacement frames")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=False)
    fallback = load_fallback(args.fallback)
    frame = pd.read_csv(args.manifest)
    records = []
    tasks = [(args.root, name, args.out, fallback) for name in frame.file_name]
    with ProcessPoolExecutor(args.workers) as pool:
        for index, record in enumerate(pool.map(prepare, tasks)):
            records.append(record)
            if index % 25 == 0:
                print("completed", index + 1, "frames", sum(item["frames"] for item in records), flush=True)

    fallback_sha256 = hashlib.sha256(Path(args.fallback).read_bytes()).hexdigest() if args.fallback else ""
    manifest = dict(
        version=VERSION,
        config=vars(args),
        clips=records,
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        fallback_sha256=fallback_sha256,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        spatial_mapping="Shared index-center mapping trilinear align_corners=False; all frame geometry asserted identical",
        intensity="One positive q99 over all resized voxels of entire clip; shared clipping [0,1]",
        frame_policy="ALL original frames, numeric order, no temporal interpolation or smoothing",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", len(records), sum(item["frames"] for item in records), flush=True)


if __name__ == "__main__":
    main()
