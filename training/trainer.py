"""Epoch loop with volume-budget batches and clip-normalized accumulation."""
import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.batch_sampler import VolumeBudgetBatchSampler
from data.collate import collate_clips
from training.loss import AnatomyMotionLoss
from training.metrics import classification_metrics, select_threshold


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def output_root():
    """Directory that training and evaluation are allowed to write into.

    Defaults to this package. Set MOTIONR_OUTPUT_ROOT to a different directory
    when checkpoints should live outside the source tree.
    """
    override = os.environ.get("MOTIONR_OUTPUT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return PACKAGE_ROOT


def assert_inside_target(path):
    root = output_root()
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside {root}: {resolved}")
    return resolved


def load_training_checkpoint(path, map_location):
    """Load a checkpoint produced by this trainer.

    ``weights_only=True`` rejects pickled executable payloads. Checkpoints
    written here are a tensor state_dict plus plain Python config values.
    """
    return torch.load(path, map_location=map_location, weights_only=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_for(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), "bf16"
    return nullcontext(), "fp32"


def move_batch(batch, device):
    return {
        "volumes": batch["volumes"].to(device, non_blocking=True),
        "frame_mask": batch["frame_mask"].to(device, non_blocking=True),
        "lengths": batch["lengths"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "patient_id": batch["patient_id"],
        "file_name": batch["file_name"],
    }


def lr_factor(epoch, warmup_epochs, total_epochs):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    span = max(total_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / float(span)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(dataset, max_volumes, shuffle, seed, workers, device,
                 length_bucket=True, bucket_width=8):
    sampler = VolumeBudgetBatchSampler(
        dataset.lengths(),
        max_valid_volumes=max_volumes,
        shuffle=shuffle,
        seed=seed,
        length_bucket=length_bucket,
        bucket_width=bucket_width,
    )
    options = {
        "batch_sampler": sampler,
        "collate_fn": collate_clips,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        options["persistent_workers"] = False
        options["prefetch_factor"] = 1
    return DataLoader(dataset, **options), sampler


def snapshot_source(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "data", "training", "visualization", "scripts", "tests"):
        source_root = PACKAGE_ROOT / folder
        for path in source_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(PACKAGE_ROOT)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def build_optimizer(model, training_cfg):
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    other_params = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(training_cfg["encoder_lr"])},
            {"params": other_params, "lr": float(training_cfg["new_module_lr"])},
        ],
        weight_decay=float(training_cfg["weight_decay"]),
    )


def _optimizer_step(model, optimizer, patients, grad_clip):
    if patients < 1:
        return 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(patients)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


@torch.no_grad()
def predict_split(model, loader, device):
    model.eval()
    autocast, _precision = autocast_for(device)
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast:
            outputs = model(batch["volumes"], batch["frame_mask"], batch["lengths"], return_visualization=False)
        probabilities = torch.sigmoid(outputs["logits"].float()).detach().cpu()
        for index in range(probabilities.shape[0]):
            length = int(batch["lengths"][index])
            row = {
                "patient_id": batch["patient_id"][index],
                "file_name": batch["file_name"][index],
                "label": float(batch["labels"][index].detach().cpu()),
                "predicted_probability": float(probabilities[index]),
                "sequence_length": length,
            }
            if outputs["expected_start"] is not None:
                row["expected_start"] = float(outputs["expected_start"][index].detach().cpu())
                row["expected_end"] = float(outputs["expected_end"][index].detach().cpu())
                row["expected_phase_ratio"] = float(outputs["expected_phase_ratio"][index].detach().cpu())
                row["interval_entropy"] = float(outputs["interval_entropy"][index].detach().cpu())
                row["phase_weights"] = outputs["phase_weights"][index, :length].detach().float().cpu().tolist()
                row["phase_weights_raw"] = outputs["phase_weights_raw"][index, :length].detach().float().cpu().tolist()
            rows.append(row)
    return rows


def run_training(model, train_data, val_data, cfg, out_dir, device):
    out_dir = assert_inside_target(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = cfg["training"]
    batch_cfg = cfg["batching"]
    loss_cfg = cfg["loss"]
    epochs = int(training_cfg["epochs"])
    warmup = int(training_cfg["warmup_epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    # This counts clips. Folds are still split by patient.
    target_patients = int(batch_cfg["target_effective_patient_batch_size"])
    max_volumes = int(batch_cfg["max_valid_volumes_per_physical_batch"])
    grad_clip = float(training_cfg["gradient_clip_norm"])
    workers = int(cfg["data"].get("num_workers", 0))
    seed = int(training_cfg.get("seed", 42))

    train_loader, train_sampler = make_loader(
        train_data, max_volumes, True, seed + 1000, workers, device,
        length_bucket=bool(batch_cfg["length_bucket"]),
        bucket_width=int(batch_cfg["bucket_width"]),
    )
    val_loader, _val_sampler = make_loader(
        val_data, max_volumes, False, seed + 2000, workers, device,
        length_bucket=bool(batch_cfg["length_bucket"]),
        bucket_width=int(batch_cfg["bucket_width"]),
    )
    optimizer = build_optimizer(model, training_cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    pos_weight = loss_cfg.get("pos_weight")
    criterion = AnatomyMotionLoss(
        pos_weight=None if pos_weight in (None, "none") else float(pos_weight),
        lambda_interval_entropy=float(loss_cfg.get("lambda_interval_entropy", 0.0)),
        lambda_phase_length=float(loss_cfg.get("lambda_phase_length", 0.0)),
        min_phase_ratio=float(loss_cfg.get("min_phase_ratio", 0.05)),
        max_phase_ratio=float(loss_cfg.get("max_phase_ratio", 0.80)),
        aux_start_epoch=int(loss_cfg.get("aux_start_epoch", 10)),
    ).to(device)
    autocast, precision = autocast_for(device)
    log_path = out_dir / "train_log.jsonl"
    best_auroc = -1.0
    best_epoch = -1
    waits = 0
    history = []

    for epoch in range(epochs):
        factor = lr_factor(epoch, warmup, epochs)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        seen = 0
        loss_total = 0.0
        patient_total = 0
        grad_norms = []

        def step_if_needed(force=False):
            nonlocal seen, grad_norms
            if seen == 0:
                return
            if seen >= target_patients or force:
                grad_norms.append(_optimizer_step(model, optimizer, seen, grad_clip))
                seen = 0

        for batch in train_loader:
            step_if_needed(force=False)
            batch = move_batch(batch, device)
            with autocast:
                outputs = model(batch["volumes"], batch["frame_mask"], batch["lengths"])
            per_sample, extra = criterion(outputs, batch["labels"], epoch=epoch)
            loss = per_sample.sum() + extra.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss_total += float(loss.detach().cpu())
            patient_total += int(per_sample.shape[0])
            loss.backward()
            seen += int(per_sample.shape[0])
        step_if_needed(force=True)

        rows = predict_split(model, val_loader, device)
        labels = [row["label"] for row in rows]
        probabilities = [row["predicted_probability"] for row in rows]
        # Ranking metrics do not need a threshold. 0.5 is only a placeholder here.
        metrics = classification_metrics(labels, probabilities, threshold=0.5)
        record = {
            "epoch": epoch,
            "lr_factor": factor,
            "train_loss": loss_total / max(patient_total, 1),
            "precision": precision,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "val_auroc": metrics["auroc"],
            "val_auprc": metrics["auprc"],
        }
        entropies = [row["interval_entropy"] for row in rows if "interval_entropy" in row]
        if entropies:
            record["val_interval_entropy"] = float(np.mean(entropies))
            record["val_phase_ratio"] = float(np.mean([row["expected_phase_ratio"] for row in rows]))
        history.append(record)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_epoch = epoch
            waits = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_auroc": best_auroc, "config": cfg},
                out_dir / "best.pt",
            )
        else:
            waits += 1
            if waits >= patience:
                break

    checkpoint = load_training_checkpoint(out_dir / "best.pt", device)
    model.load_state_dict(checkpoint["model"])
    rows = predict_split(model, val_loader, device)
    labels = [row["label"] for row in rows]
    probabilities = [row["predicted_probability"] for row in rows]
    threshold = select_threshold(labels, probabilities)
    final_metrics = classification_metrics(labels, probabilities, threshold)
    final_metrics["best_epoch"] = best_epoch
    final_metrics["selection_metric"] = "val_AUROC"
    (out_dir / "best_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    with (out_dir / "val_predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"history": history, "metrics": final_metrics, "rows": rows}
