"""Evaluate a saved checkpoint. The decision threshold is fit on validation only."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import ClipDataset, audit_manifest, split_folds
from models.anatomy_motion_4d import build_model
from training.metrics import classification_metrics, select_threshold
from training.trainer import (
    assert_inside_target,
    load_training_checkpoint,
    make_loader,
    move_batch,
    predict_split,
)
from visualization.phase_plot import save_phase_plot
from visualization.spatial_attention_vis import upsample_attention
from visualization.systolic_metrics import (
    normalized_to_inclusive_frames,
    start_end_error,
    systolic_phase_mass,
    temporal_iou,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-plots", type=int, default=-1)
    parser.add_argument("--save-spatial", type=int, default=0)
    parser.add_argument("--systolic-csv", type=Path, default=None)
    args = parser.parse_args()
    out_dir = assert_inside_target(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint["config"]
    frame, counts = audit_manifest(cfg["data"]["manifest"], cfg["data"]["cache"])
    train_frame, val_frame = split_folds(frame, cfg["data"]["fold"])
    # Threshold is always chosen on the validation split of this checkpoint.
    val_data = ClipDataset(val_frame, cfg["data"]["cache"], counts, augment=False)
    batching = cfg["batching"]
    val_loader, _sampler = make_loader(
        val_data,
        int(batching["max_valid_volumes_per_physical_batch"]),
        False,
        int(cfg["training"].get("seed", 42)),
        args.workers,
        device,
        length_bucket=bool(batching["length_bucket"]),
        bucket_width=int(batching["bucket_width"]),
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    val_rows = predict_split(model, val_loader, device)
    threshold = select_threshold(
        [row["label"] for row in val_rows],
        [row["predicted_probability"] for row in val_rows],
    )
    if args.split == "val":
        rows = val_rows
    else:
        train_data = ClipDataset(train_frame, cfg["data"]["cache"], counts, augment=False)
        train_loader, _sampler = make_loader(
            train_data,
            int(batching["max_valid_volumes_per_physical_batch"]),
            False,
            int(cfg["training"].get("seed", 42)) + 1,
            args.workers,
            device,
            length_bucket=bool(batching["length_bucket"]),
            bucket_width=int(batching["bucket_width"]),
        )
        rows = predict_split(model, train_loader, device)
    metrics = classification_metrics(
        [row["label"] for row in rows],
        [row["predicted_probability"] for row in rows],
        threshold,
    )
    metrics["threshold_split"] = "val"
    metrics["reported_split"] = args.split
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plot_rows = [row for row in rows if "phase_weights" in row]
    if args.max_plots >= 0:
        plot_rows = plot_rows[:args.max_plots]
    for row in plot_rows:
        save_phase_plot(
            out_dir / "phase_plots" / f"{row['file_name']}.png",
            row["phase_weights"],
            row["expected_start"],
            row["expected_end"],
            title=f"{row['file_name']} label={int(row['label'])} p={row['predicted_probability']:.3f}",
        )
    if args.save_spatial > 0 and cfg["model"]["variant"] in {"M1", "M2", "M3", "M4"}:
        _save_spatial(model, val_data, device, out_dir / "spatial", args.save_spatial)
    if args.systolic_csv is not None:
        _score_systolic(rows, args.systolic_csv, out_dir / "systolic_metrics.json")
    print(json.dumps(metrics), flush=True)


def _save_spatial(model, dataset, device, out_dir, limit):
    out_dir = assert_inside_target(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    from data.collate import collate_clips
    for index in range(len(dataset)):
        if saved >= limit:
            break
        batch = move_batch(collate_clips([dataset[index]]), device)
        with torch.no_grad():
            outputs = model(batch["volumes"], batch["frame_mask"], batch["lengths"], return_visualization=True)
        length = int(batch["lengths"][0])
        payload = {}
        if outputs["local_spatial_attention"] is not None:
            local = outputs["local_spatial_attention"][0, :length].detach().float().cpu()
            payload["local_feature_grid"] = local
            payload["local_upsampled"] = upsample_attention(local[:, 0] if local.ndim == 4 else local)
        if outputs["transition_spatial_attention"] is not None:
            transition = outputs["transition_spatial_attention"][0, :max(length - 1, 0)].detach().float().cpu()
            payload["transition_feature_grid"] = transition
        torch.save(payload, out_dir / f"{batch['file_name'][0]}.pt")
        saved += 1


def _score_systolic(rows, csv_path, out_path):
    import pandas as pd
    table = pd.read_csv(csv_path)
    by_name = {row["file_name"]: row for row in rows}
    scored = []
    for record in table.to_dict("records"):
        row = by_name.get(str(record["file_name"]))
        if row is None or "phase_weights" not in row:
            continue
        length = row["sequence_length"]
        ref_start, ref_end = int(record["start_frame"]), int(record["end_frame"])
        pred_start, pred_end = normalized_to_inclusive_frames(row["expected_start"], row["expected_end"], length)
        mask = [ref_start <= frame <= ref_end for frame in range(length)]
        start_error, end_error = start_end_error(
            row["expected_start"],
            row["expected_end"],
            ref_start / max(length - 1, 1),
            ref_end / max(length - 1, 1),
        )
        scored.append({
            "file_name": row["file_name"],
            "systolic_phase_mass": systolic_phase_mass(row["phase_weights"], mask),
            "temporal_iou": temporal_iou(pred_start, pred_end, ref_start, ref_end),
            "start_error": start_error,
            "end_error": end_error,
        })
    out_path.write_text(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()
