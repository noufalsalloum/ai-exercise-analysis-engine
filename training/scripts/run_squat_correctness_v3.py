"""Run development-only Squat Correctness V3 LOSO cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from training.squat_correctness import FeatureCache, sha256
from training.squat_correctness_v3 import (
    DEVELOPMENT_SUBJECTS,
    development_only_manifest,
    run_v3_loso,
)


ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/squat_ai/data/repetition_manifest.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("datasets/window_cache/rehab24_squat_v1"),
    )
    parser.add_argument(
        "--motionbert-checkpoint",
        type=Path,
        default=Path("models/latest_epoch.bin"),
    )
    parser.add_argument(
        "--representation-checkpoint",
        type=Path,
        default=Path("checkpoints/exercise_representation/pilot/best.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/squat_ai_v3/correctness")
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/squat_ai_v3/correctness"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    manifest_path = _resolve(args.manifest)
    cache_dir = _resolve(args.cache_dir)
    raw = pd.read_csv(manifest_path, dtype={"subject_id": str})
    # Filter before any repetition cache path is resolved or opened.
    development = raw[raw["subject_id"].isin(DEVELOPMENT_SUBJECTS)].copy()
    development["repetition_cache_path"] = development["sample_id"].map(
        lambda value: str(
            (cache_dir / "repetitions" / f"{value}.npz").resolve()
        )
    )
    development = development_only_manifest(development)
    feature_dir = cache_dir / "motionbert_features"
    metadata = json.loads(
        (feature_dir / "metadata.json").read_text(encoding="utf-8")
    )
    feature_cache = FeatureCache(
        feature_dir / "features.npy",
        feature_dir / "metadata.json",
        tuple(metadata["sample_ids"]),
    )
    motionbert = _resolve(args.motionbert_checkpoint)
    representation = _resolve(args.representation_checkpoint)
    config = {
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "minimum_incorrect_recall": 0.60,
        "seed": int(args.seed),
        "device": str(device),
        "source_dataset": "REHAB24-6",
        "preprocessing_version": "physical_mp33_h36m_root_body_scale_xy_conf_pad_v4",
        "cache_version": "rehab24_squat_mp33_h36m_v4_rep60_v1",
        "motionbert_checkpoint_sha256": sha256(motionbert),
        "representation_checkpoint_sha256": sha256(representation),
        "trainable_scope": (
            "squat_adapter+temporal_attention+global_pooling+correctness_head"
        ),
        "source_manifest_sha256": sha256(manifest_path),
        "motionbert_frozen": True,
        "test_locked": True,
    }
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    development.drop(columns=["repetition_cache_path"]).to_csv(
        output_dir / "development_manifest.csv", index=False
    )
    result = run_v3_loso(
        development,
        feature_cache,
        motionbert,
        representation,
        output_dir,
        _resolve(args.checkpoint_dir),
        device,
        config,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
