"""Safety tests for Squat calibration, abstention, and performance score."""

from __future__ import annotations

import unittest

from inference.squat_robustness import SquatRobustnessPolicy


class SquatRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = SquatRobustnessPolicy.load()

    @staticmethod
    def rep(assessment: str) -> dict:
        return {"assessment": assessment, "pass_fail": assessment}

    def test_platt_calibration_is_monotonic_and_boundary_preserving(self) -> None:
        below = self.policy.calibrate(0.60)
        boundary = self.policy.calibrate(0.61)
        above = self.policy.calibrate(0.62)
        self.assertLess(below, boundary)
        self.assertAlmostEqual(boundary, self.policy.calibrated_threshold, places=12)
        self.assertGreater(above, boundary)

    def test_full_side_view_abstains_without_changing_raw_model_decision(self) -> None:
        result = self.policy.assess_rep(
            {
                "correct_probability": 0.20,
                "pass_fail": "FAIL",
                "error_class": "bad_heel",
                "error_confidence": 0.999,
            },
            "side",
        )
        self.assertEqual(result["raw_model_decision"], "FAIL")
        self.assertEqual(result["assessment"], "NEEDS_REVIEW")
        self.assertEqual(result["pass_fail"], "NEEDS_REVIEW")
        self.assertIsNone(result["error_class"])
        self.assertIsNone(result["error_confidence"])

    def test_supported_front_view_preserves_pass_and_fail(self) -> None:
        passed = self.policy.assess_rep(
            {"correct_probability": 0.80, "pass_fail": "PASS", "error_class": None},
            "front",
        )
        failed = self.policy.assess_rep(
            {
                "correct_probability": 0.20,
                "pass_fail": "FAIL",
                "error_class": "bad_back",
                "error_confidence": 1.0,
            },
            "front",
        )
        self.assertEqual(passed["assessment"], "PASS")
        self.assertEqual(failed["assessment"], "FAIL")
        self.assertEqual(failed["error_class"], "bad_back")

    def test_low_confidence_error_falls_back_to_form_issue(self) -> None:
        result = self.policy.assess_rep(
            {
                "correct_probability": 0.20,
                "pass_fail": "FAIL",
                "error_class": "bad_heel",
                "error_confidence": 0.80,
            },
            "front",
        )
        self.assertEqual(result["assessment"], "FAIL")
        self.assertEqual(result["error_class"], "form_issue")
        self.assertIsNone(result["error_confidence"])

    def test_all_pass_score_is_100(self) -> None:
        result = self.policy.performance_score([self.rep("PASS")] * 3)
        self.assertEqual(result.score, 100.0)
        self.assertEqual(result.coverage, 1.0)

    def test_all_fail_score_is_zero(self) -> None:
        result = self.policy.performance_score([self.rep("FAIL")] * 3)
        self.assertEqual(result.score, 0.0)

    def test_mixed_score_is_deterministic(self) -> None:
        values = [self.rep("PASS"), self.rep("PASS"), self.rep("FAIL")]
        first = self.policy.performance_score(values)
        second = self.policy.performance_score(values)
        self.assertEqual(first.score, 66.7)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_low_coverage_and_no_assessed_reps_are_unavailable(self) -> None:
        low = self.policy.performance_score(
            [self.rep("PASS"), self.rep("FAIL"), self.rep("NEEDS_REVIEW"), self.rep("NEEDS_REVIEW")]
        )
        none = self.policy.performance_score([self.rep("NEEDS_REVIEW")] * 4)
        self.assertIsNone(low.score)
        self.assertEqual(low.coverage, 0.5)
        self.assertIsNone(none.score)
        self.assertEqual(none.assessed_reps, 0)

    def test_score_is_always_bounded(self) -> None:
        for passes in range(4):
            values = [self.rep("PASS")] * passes + [self.rep("FAIL")] * (3 - passes)
            score = self.policy.performance_score(values).score
            self.assertIsNotNone(score)
            assert score is not None
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)


if __name__ == "__main__":
    unittest.main()
