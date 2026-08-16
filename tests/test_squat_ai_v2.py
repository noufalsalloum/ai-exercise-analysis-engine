from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.squat_correctness import SquatCorrectnessModel
from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from training.squat_correctness import load_checkpoint_strict, sha256
from training.squat_correctness_v2 import calibrate_threshold
from training.squat_rep_boundary_v2 import make_model, v2_segments


ROOT = Path(__file__).resolve().parents[1]


class SquatAIV2Tests(unittest.TestCase):
    def test_boundary_aux_shape_and_gradient(self) -> None:
        model = SquatRepBoundaryV2Model(channels=16, dropout=0.0)
        output = model(torch.randn(2, 80, 17, 3))
        self.assertEqual(tuple(output["active_logits"].shape), (2, 80))
        self.assertEqual(tuple(output["boundary_logits"].shape), (2, 80))
        (output["active_logits"].mean() + output["boundary_logits"].mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_boundary_head_splits_a_long_active_plateau(self) -> None:
        active = np.full(300, 0.95, np.float32)
        boundary = np.zeros(300, np.float32)
        boundary[96:105] = 0.95; boundary[196:205] = 0.95
        config = {"enter_threshold": 0.35, "exit_threshold": 0.2, "smoothing_kernel": 1, "min_duration": 30, "max_duration": 155, "merge_gap": 0, "boundary_threshold": 0.35, "boundary_smoothing_kernel": 1, "boundary_cluster_gap": 10}
        segments = v2_segments(active, boundary, config)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], 299)

    def test_threshold_calibration_enforces_incorrect_recall_constraint(self) -> None:
        frame = pd.DataFrame({"correctness": [0, 0, 0, 1, 1, 1, 1], "correct_probability": [0.1, 0.4, 0.55, 0.45, 0.7, 0.8, 0.9]})
        threshold, metrics, table = calibrate_threshold(frame, 0.60)
        self.assertGreaterEqual(metrics["classes"]["incorrect"]["recall"], 0.60)
        self.assertTrue(np.isfinite(threshold))
        self.assertTrue(table["constraint_satisfied"].any())

    def test_frozen_v2_checkpoints_load_strictly_and_test_opened_once(self) -> None:
        boundary_path = ROOT / "checkpoints/squat_ai_v2/rep_boundary/best.pt"
        boundary_checkpoint = torch.load(boundary_path, map_location="cpu", weights_only=True)
        boundary = make_model(boundary_checkpoint["experiment"]["architecture"])
        boundary.load_state_dict(boundary_checkpoint["model_state_dict"], strict=True)
        correctness = SquatCorrectnessModel(ROOT / "models/latest_epoch.bin")
        correctness_checkpoint = load_checkpoint_strict(ROOT / "archive/checkpoints/squat_ai_v2/correctness/best.pt", correctness, torch.device("cpu"))
        self.assertTrue(correctness_checkpoint["test_locked"])
        marker = json.loads((ROOT / "results/squat_ai_v2/test_opened_once.json").read_text(encoding="utf-8"))
        self.assertTrue(marker["opened_once"])
        self.assertEqual(marker["test_subjects"], ["4", "7"])

    def test_v1_and_motionbert_assets_are_unchanged(self) -> None:
        expected = {
            ROOT / "models/latest_epoch.bin": "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461",
            ROOT / "archive/checkpoints/squat_ai_v1/correctness/best.pt": "038fcb58ba75b7b4d59f1c9eb07ed2f0c4c6612e4d1a8348df19bbbaba927c99",
            ROOT / "archive/checkpoints/squat_ai_v1/rep_boundary/best.pt": "20eb442e800e5d1ca3821fe73aa3f3df3a84b1a87d959f0ffd4bea04bacd1563",
        }
        for path, digest in expected.items():
            self.assertEqual(sha256(path), digest)


if __name__ == "__main__":
    unittest.main()
