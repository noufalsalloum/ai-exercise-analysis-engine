from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.tools.audit_external_datasets import discover_dataset_roots
from datasets.adapters.intellirehabds_adapter import (
    INTELLIREHAB_TO_H36M,
    IntelliRehabDSAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the IntelliRehab correctness task.")
    parser.add_argument("--external-root", type=Path, default=Path("datasets/external"))
    return parser.parse_args()


def main() -> None:
    """Validate Task 2 inputs without starting unapproved training."""

    args = parse_args()
    root = args.external_root
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    dataset_root = discover_dataset_roots(root)["intellirehabds"]
    adapter = IntelliRehabDSAdapter(dataset_root)
    samples = list(adapter.iter_samples())
    missing = [row["h36m_joint"] for row in INTELLIREHAB_TO_H36M if row["mapping"] == "missing"]
    print(json.dumps({
        "task": "general_correctness_representation",
        "sequences": len(samples),
        "binary_labelled_sequences": sum(sample.correctness_label is not None for sample in samples),
        "input_contract": "(T,25,3) Kinect-v2 skeleton",
        "motionbert_safe": not missing,
        "missing_h36m_joints": missing,
        "status": "not_started",
        "next_required_component": "dataset-specific 25-joint temporal encoder",
    }, indent=2))


if __name__ == "__main__":
    main()
