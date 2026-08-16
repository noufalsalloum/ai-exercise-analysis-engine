"""Safety and arithmetic tests for comparison-only averaged inference."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

from inference.squat_averaged_correctness import (
    AveragedCorrectnessConfig,
    AveragedCorrectnessExperiment,
    aggregate_probabilities,
    provisional_score,
)
from inference.squat_robustness import SquatRobustnessPolicy
from ui.squat_ai_presentation import present_ai_rep


ROOT = Path(__file__).resolve().parents[1]


class _ForbiddenModel:
    def __call__(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("Feature flag OFF must not invoke Correctness V3.")


class SquatAveragedCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AveragedCorrectnessConfig.load()
        cls.policy = SquatRobustnessPolicy.load()

    def test_three_window_mean_and_raw_threshold(self) -> None:
        result = aggregate_probabilities([0.50, 0.70, 0.66], 0.61)
        self.assertAlmostEqual(result["mean_probability"], 0.62)
        self.assertEqual(result["mean_decision"], "PASS")
        self.assertEqual(result["threshold"], 0.61)

    def test_five_window_median_std_min_and_max(self) -> None:
        values = [0.19, 0.37, 0.64, 0.55, 0.31]
        result = aggregate_probabilities(values, 0.61)
        self.assertAlmostEqual(result["mean_probability"], float(np.mean(values)))
        self.assertAlmostEqual(result["median_probability"], 0.37)
        self.assertAlmostEqual(result["std_probability"], float(np.std(values)))
        self.assertEqual(result["min_probability"], 0.19)
        self.assertEqual(result["max_probability"], 0.64)
        self.assertEqual(result["median_decision"], "FAIL")

    def test_calibrated_threshold_is_used_only_for_calibrated_values(self) -> None:
        raw = [0.62, 0.64, 0.66]
        calibrated = [self.policy.calibrate(value) for value in raw]
        raw_result = aggregate_probabilities(raw, self.config.raw_threshold)
        calibrated_result = aggregate_probabilities(
            calibrated, self.config.calibrated_threshold
        )
        self.assertEqual(raw_result["mean_decision"], "PASS")
        self.assertEqual(calibrated_result["mean_decision"], "PASS")
        self.assertAlmostEqual(
            self.policy.calibrate(self.config.raw_threshold),
            self.config.calibrated_threshold,
            places=9,
        )
        # This deliberately mismatched comparison demonstrates why the two
        # probability spaces cannot share a threshold.
        self.assertEqual(
            aggregate_probabilities(raw, self.config.calibrated_threshold)[
                "mean_decision"
            ],
            "FAIL",
        )

    def test_feature_flag_off_preserves_active_behavior(self) -> None:
        self.assertFalse(self.config.enabled)
        experiment = AveragedCorrectnessExperiment(self.config, self.policy)
        sequence = np.zeros((50, 17, 3), dtype=np.float32)
        sequence[..., 2] = 1.0
        self.assertIsNone(experiment.compare(_ForbiddenModel(), sequence, "cpu"))

    def test_temporal_views_keep_trained_shape_for_3_5_and_7(self) -> None:
        experiment = AveragedCorrectnessExperiment(
            self.config.with_enabled(True), self.policy
        )
        sequence = np.zeros((51, 17, 3), dtype=np.float32)
        sequence[..., 0] = np.linspace(-1.0, 1.0, 51)[:, None]
        sequence[..., 2] = 1.0
        for count in (3, 5, 7):
            views, metadata = experiment._temporal_views(sequence, count)
            self.assertEqual(views.shape, (count, 60, 17, 3))
            self.assertEqual(len(metadata), count)
            self.assertTrue(np.isfinite(views).all())

    def test_provisional_score_uses_mean_decision_and_no_error_penalty(self) -> None:
        rows = [
            {"raw": {"mean_decision": "PASS"}, "error_class": "bad_heel"},
            {"raw": {"mean_decision": "PASS"}, "error_class": "bad_back"},
            {"raw": {"mean_decision": "FAIL"}, "error_class": None},
        ]
        self.assertEqual(provisional_score(rows), 66.7)
        changed_errors = [dict(row, error_class="form_issue") for row in rows]
        self.assertEqual(provisional_score(changed_errors), 66.7)
        self.assertIsNone(provisional_score([]))

    def test_hidden_comparison_data_does_not_change_presentation(self) -> None:
        rep = {
            "rep_index": 1,
            "assessment": "NEEDS_REVIEW",
            "assessment_confidence": 0.45,
            "needs_review_reason": "Unsupported view",
        }
        baseline = present_ai_rep(rep)
        comparison = present_ai_rep(
            {
                **rep,
                "averaged_correctness_experiment": {
                    "raw": {"mean_decision": "PASS"}
                },
            }
        )
        self.assertEqual(baseline, comparison)

    def test_active_checkpoint_hashes_are_unchanged(self) -> None:
        expected = {
            "models/latest_epoch.bin": "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461",
            "checkpoints/squat_ai_v2/rep_boundary/best.pt": "284cb3adacf61568c4cf4cd2610e0bfe63b5df3bb620cc8a6b29e2f5a91c6d79",
            "checkpoints/squat_ai_v3/correctness/final_dev.pt": "49fe21f01d4f5aab5076b0e02728f1ecdbc1d37e90531516aa936d7e49e8d0bf",
            "checkpoints/squat_error_v1/best.pt": "9eabeec61746cda5e82f9e369e0e937b4ce2d91c5a8dbd3892edc171cb3c30cf",
            "checkpoints/exercise_representation/pilot/best.pt": "019884776d5a15925bffe61358fd8bcac627004b5a3ddb4165af969da46667ff",
        }
        for relative, digest in expected.items():
            with self.subTest(path=relative):
                actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
