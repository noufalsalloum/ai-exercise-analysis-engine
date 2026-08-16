"""Run Test-locked Squat repetition-boundary Experiment 2 development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from training.squat_rep_boundary import load_full_video_records
from training.squat_rep_boundary_v2 import run_boundary_experiments


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/squat_ai/data/repetition_manifest.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/window_cache/rehab24_squat_v1"))
    parser.add_argument("--v1-checkpoint", type=Path, default=Path("archive/checkpoints/squat_ai_v1/rep_boundary/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/squat_ai_v2/rep_boundary"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/squat_ai_v2/rep_boundary"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); resolve = lambda path: path if path.is_absolute() else ROOT / path
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    manifest = pd.read_csv(resolve(args.manifest), dtype={"subject_id": str})
    all_records = load_full_video_records(manifest, resolve(args.cache_dir) / "full_videos")
    development_records = [record for record in all_records if record.split in {"train", "validation"}]
    if len(development_records) != 14 or any(record.split == "test" for record in development_records):
        raise RuntimeError("Test lock or expected development video count violated.")
    config = {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "learning_rate": 3e-4, "weight_decay": 1e-4, "gradient_clip": 1.0, "window_size": 256, "stride": 128, "seed": args.seed, "test_locked": True}
    result = run_boundary_experiments(development_records, resolve(args.v1_checkpoint), resolve(args.output_dir), resolve(args.checkpoint_dir), device, config)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
