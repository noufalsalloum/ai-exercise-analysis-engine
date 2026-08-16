from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets.tools.audit_external_datasets import discover_dataset_roots
from datasets.adapters.squat_dataset_adapter import SquatDatasetAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the static squat-error task.")
    parser.add_argument("--external-root", type=Path, default=Path("datasets/external"))
    return parser.parse_args()


def main() -> None:
    """Validate Task 3 labels without inventing temporal sequences or training."""

    args = parse_args()
    root = args.external_root
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    adapter = SquatDatasetAdapter(discover_dataset_roots(root)["squat_dataset"])
    samples = list(adapter.iter_samples())
    labels = Counter(
        "good" if not sample.error_labels else sample.error_labels[0] for sample in samples
    )
    print(json.dumps({
        "task": "squat_static_error_classification",
        "images": len(samples),
        "classes": dict(labels),
        "input_contract": "one static RGB image per sample",
        "loss_contract": "CrossEntropyLoss (mutually-exclusive folder labels)",
        "motionbert_used": False,
        "status": "not_started",
        "blocker_before_reliable_validation": "No subject/video provenance for leakage-safe validation split.",
    }, indent=2))


if __name__ == "__main__":
    main()
