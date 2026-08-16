from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.squat_correctness import SquatCorrectnessModel
from models.squat_rep_boundary import SquatRepBoundaryModel
from training.squat_correctness import load_checkpoint_strict


ROOT = Path(__file__).resolve().parents[1]


class Rehab24SquatPipelineTests(unittest.TestCase):
    def test_complete_cache_and_contract(self) -> None:
        cache = ROOT / "datasets/window_cache/rehab24_squat_v1"
        metadata = json.loads((cache / "cache_metadata.json").read_text(encoding="utf-8"))
        repetition_paths = list((cache / "repetitions").glob("*.npz"))
        full_paths = list((cache / "full_videos").glob("*.npz"))
        self.assertEqual(len(repetition_paths), 390)
        self.assertEqual(len(full_paths), 18)
        self.assertEqual(metadata["cached_camera_samples"], 390)
        with np.load(repetition_paths[0], allow_pickle=False) as archive:
            values = np.asarray(archive["motionbert_input"])
            mask = np.asarray(archive["temporal_mask"])
        self.assertEqual(values.shape, (60, 17, 3))
        self.assertEqual(mask.shape, (60,))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreaterEqual(float(values[..., 2].min()), 0.0)
        self.assertLessEqual(float(values[..., 2].max()), 1.0)

    def test_no_subject_or_camera_pair_leakage(self) -> None:
        manifest = pd.read_csv(ROOT / "results/squat_ai/data/repetition_manifest.csv", dtype={"subject_id": str})
        groups = {split: set(group["subject_id"]) for split, group in manifest.groupby("split")}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertTrue((manifest.groupby("pair_id")["split"].nunique() == 1).all())
        self.assertTrue((manifest.groupby("pair_id")["camera_id"].nunique() == 2).all())

    def test_correctness_and_boundary_checkpoints_load_strictly(self) -> None:
        device = torch.device("cpu")
        correctness = SquatCorrectnessModel(ROOT / "models/latest_epoch.bin")
        checkpoint = load_checkpoint_strict(
            ROOT / "archive/checkpoints/squat_ai_v1/correctness/best.pt", correctness, device
        )
        self.assertTrue(checkpoint["motionbert_frozen"])
        self.assertTrue(all(not parameter.requires_grad for parameter in correctness.backbone.parameters()))
        boundary_checkpoint = torch.load(
            ROOT / "archive/checkpoints/squat_ai_v1/rep_boundary/best.pt",
            map_location="cpu",
            weights_only=True,
        )
        boundary = SquatRepBoundaryModel()
        boundary.load_state_dict(boundary_checkpoint["model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
