"""Train and evaluate the independent static Squat Error V1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.squat_error_v1 import run_experiment


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/squat_error_v1/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/squat_error_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/squat_error_v1/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args(); resolve = lambda path: path if path.is_absolute() else ROOT / path
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    result = run_experiment(resolve(args.data_dir), resolve(args.output_dir), resolve(args.checkpoint), device, seed=args.seed, development_only=args.development_only)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
