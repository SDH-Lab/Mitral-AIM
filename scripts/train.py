"""Train AnatomyMotion4DNet on one patient-disjoint validation fold."""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import ClipDataset, audit_manifest, class_counts, split_folds
from models.anatomy_motion_4d import build_model
from training.trainer import assert_inside_target, run_training, seed_all, snapshot_source


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def take_tiny(frame, count):
    if count < 2 or count % 2:
        raise ValueError("--tiny must be an even integer >= 2")
    parts = [frame[frame.binary == label].head(count // 2) for label in (0, 1)]
    return __import__("pandas").concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2])
    parser.add_argument("--variant", choices=["M0", "M1", "M2", "M3", "M4"])
    parser.add_argument("--local-layer", choices=["layer2", "layer3"])
    parser.add_argument("--local-pooling", choices=["attention", "gap"])
    parser.add_argument("--transition-representation", choices=["full", "pair", "difference"])
    parser.add_argument("--transition-pooling", choices=["attention", "gap"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--encoder-lr", type=float)
    parser.add_argument("--new-module-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--classifier-dropout", type=float)
    parser.add_argument(
        "--module-dropout",
        type=float,
        help="Sets state/motion transformer, transition, fusion, and selector dropout",
    )
    parser.add_argument("--tiny", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pos-weight", choices=["none", "auto"])
    parser.add_argument("--raw-difference", choices=["true", "false"])
    parser.add_argument("--feature-transition", choices=["true", "false"])
    parser.add_argument(
        "--head-features",
        choices=[
            "phase_anatomy_phase_motion_global",
            "phase_anatomy_phase_motion",
            "phase_motion_global",
        ],
    )
    parser.add_argument("--temporal-pool", choices=["phase", "uniform"])
    parser.add_argument("--sector-mask", choices=["true", "false"])
    parser.add_argument("--sector-mask-global", choices=["true", "false"])
    parser.add_argument("--stem-frame-diff", choices=["true", "false"])
    parser.add_argument("--stem-diff-mode", choices=["abs", "signed"])
    parser.add_argument("--temporal-mixer", choices=["dense", "deform_attn"])
    parser.add_argument("--motion-align", choices=["linear", "cross_deform"])
    parser.add_argument("--deform-n-points", type=int)
    parser.add_argument("--lambda-interval-entropy", type=float)
    parser.add_argument("--lambda-phase-length", type=float)
    parser.add_argument("--min-phase-ratio", type=float)
    parser.add_argument("--max-phase-ratio", type=float)
    parser.add_argument("--aux-start-epoch", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    if args.variant is not None:
        cfg["model"]["variant"] = args.variant
    if args.local_layer is not None:
        cfg["spatial_encoder"]["local_layer"] = args.local_layer
    if args.local_pooling is not None:
        cfg["local_anatomy"]["pooling"] = args.local_pooling
    if args.transition_representation is not None:
        cfg["spatial_transition"]["representation"] = args.transition_representation
    if args.transition_pooling is not None:
        cfg["transition_attention"]["pooling"] = args.transition_pooling
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.encoder_lr is not None:
        cfg["training"]["encoder_lr"] = float(args.encoder_lr)
    if args.new_module_lr is not None:
        cfg["training"]["new_module_lr"] = float(args.new_module_lr)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    if args.warmup_epochs is not None:
        cfg["training"]["warmup_epochs"] = int(args.warmup_epochs)
    if args.classifier_dropout is not None:
        cfg.setdefault("classifier", {})["dropout"] = float(args.classifier_dropout)
    if args.module_dropout is not None:
        md = float(args.module_dropout)
        cfg.setdefault("state_transformer", {})["dropout"] = md
        cfg.setdefault("motion_transformer", {})["dropout"] = md
        cfg.setdefault("spatial_transition", {})["dropout"] = md
        cfg.setdefault("selector_fusion", {})["dropout"] = md
        cfg.setdefault("interval_selector", {})["dropout"] = md
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.pos_weight is not None:
        cfg["loss"]["pos_weight"] = None if args.pos_weight == "none" else "auto"
    cfg.setdefault("motion_branch", {})
    if args.raw_difference is not None:
        cfg["motion_branch"]["use_raw_difference"] = args.raw_difference == "true"
    if args.feature_transition is not None:
        cfg["motion_branch"]["use_feature_transition"] = args.feature_transition == "true"
    if args.head_features is not None:
        cfg.setdefault("classifier", {})["head_features"] = args.head_features
        widths = {
            "phase_anatomy_phase_motion_global": 768,
            "phase_anatomy_phase_motion": 512,
            "phase_motion_global": 512,
        }
        cfg["classifier"]["input_dim"] = widths[args.head_features]
    if args.temporal_pool is not None:
        cfg.setdefault("classifier", {})["temporal_pool"] = args.temporal_pool
    cfg.setdefault("spatial_prior", {})
    if args.sector_mask is not None:
        cfg["spatial_prior"]["use_sector_mask"] = args.sector_mask == "true"
    if args.sector_mask_global is not None:
        cfg["spatial_prior"]["use_sector_mask_global"] = args.sector_mask_global == "true"
    cfg.setdefault("appearance_motion", {})
    if args.stem_frame_diff is not None:
        cfg["appearance_motion"]["use_stem_frame_diff"] = args.stem_frame_diff == "true"
    if args.stem_diff_mode is not None:
        cfg["appearance_motion"]["stem_diff_mode"] = args.stem_diff_mode
    cfg.setdefault("temporal_options", {})
    if args.temporal_mixer is not None:
        cfg["temporal_options"]["temporal_mixer"] = args.temporal_mixer
    if args.motion_align is not None:
        cfg["temporal_options"]["motion_align"] = args.motion_align
    if args.deform_n_points is not None:
        cfg["temporal_options"]["deform_n_points"] = int(args.deform_n_points)
    cfg.setdefault("loss", {})
    if args.lambda_interval_entropy is not None:
        cfg["loss"]["lambda_interval_entropy"] = float(args.lambda_interval_entropy)
    if args.lambda_phase_length is not None:
        cfg["loss"]["lambda_phase_length"] = float(args.lambda_phase_length)
    if args.min_phase_ratio is not None:
        cfg["loss"]["min_phase_ratio"] = float(args.min_phase_ratio)
    if args.max_phase_ratio is not None:
        cfg["loss"]["max_phase_ratio"] = float(args.max_phase_ratio)
    if args.aux_start_epoch is not None:
        cfg["loss"]["aux_start_epoch"] = int(args.aux_start_epoch)
    if int(cfg["training"]["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if int(cfg["training"]["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be positive")

    out_dir = assert_inside_target(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_all(int(cfg["training"]["seed"]))
    frame, counts = audit_manifest(cfg["data"]["manifest"], cfg["data"]["cache"])
    train_frame, val_frame = split_folds(frame, cfg["data"]["fold"])
    if args.tiny:
        train_frame = take_tiny(train_frame, args.tiny)
        val_frame = take_tiny(val_frame, args.tiny)
    balance = class_counts(train_frame)
    if cfg["loss"].get("pos_weight") == "auto":
        cfg["loss"]["pos_weight"] = balance["pos_weight"]
    print(json.dumps({
        "event": "data",
        "fold": int(cfg["data"]["fold"]),
        "train_clips": len(train_frame),
        "val_clips": len(val_frame),
        "train_class_counts": balance,
        "pos_weight": cfg["loss"].get("pos_weight"),
        "variant": cfg["model"]["variant"],
        "use_raw_difference": bool(cfg.get("motion_branch", {}).get("use_raw_difference", True)),
        "use_feature_transition": bool(cfg.get("motion_branch", {}).get("use_feature_transition", True)),
        "head_features": cfg.get("classifier", {}).get("head_features"),
        "temporal_pool": cfg.get("classifier", {}).get("temporal_pool", "phase"),
        "use_sector_mask": bool(cfg.get("spatial_prior", {}).get("use_sector_mask", False)),
        "use_sector_mask_global": bool(cfg.get("spatial_prior", {}).get("use_sector_mask_global", False)),
        "use_stem_frame_diff": bool(cfg.get("appearance_motion", {}).get("use_stem_frame_diff", False)),
        "stem_diff_mode": cfg.get("appearance_motion", {}).get("stem_diff_mode", "abs"),
        "temporal_mixer": cfg.get("temporal_options", {}).get("temporal_mixer", "dense"),
        "motion_align": cfg.get("temporal_options", {}).get("motion_align", "linear"),
        "deform_n_points": int(cfg.get("temporal_options", {}).get("deform_n_points", 4)),
        "lambda_interval_entropy": float(cfg.get("loss", {}).get("lambda_interval_entropy", 0.0)),
        "lambda_phase_length": float(cfg.get("loss", {}).get("lambda_phase_length", 0.0)),
        "min_phase_ratio": float(cfg.get("loss", {}).get("min_phase_ratio", 0.05)),
        "max_phase_ratio": float(cfg.get("loss", {}).get("max_phase_ratio", 0.80)),
        "aux_start_epoch": int(cfg.get("loss", {}).get("aux_start_epoch", 10)),
        "encoder_lr": float(cfg.get("training", {}).get("encoder_lr", 3.0e-5)),
        "new_module_lr": float(cfg.get("training", {}).get("new_module_lr", 9.0e-5)),
        "weight_decay": float(cfg.get("training", {}).get("weight_decay", 1.0e-4)),
        "warmup_epochs": int(cfg.get("training", {}).get("warmup_epochs", 5)),
        "seed": int(cfg.get("training", {}).get("seed", 42)),
        "classifier_dropout": float(cfg.get("classifier", {}).get("dropout", 0.20)),
        "module_dropout": float(cfg.get("state_transformer", {}).get("dropout", 0.10)),
    }), flush=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    snapshot_source(out_dir / "source")
    train_data = ClipDataset(train_frame, cfg["data"]["cache"], counts, augment=bool(cfg["data"]["augment"]) and not args.tiny)
    val_data = ClipDataset(val_frame, cfg["data"]["cache"], counts, augment=False)
    model = build_model(cfg).to(device)
    result = run_training(model, train_data, val_data, cfg, out_dir, device)
    print(json.dumps({"event": "complete", **result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
