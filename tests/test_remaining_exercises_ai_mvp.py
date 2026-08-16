from __future__ import annotations

import unittest

from application.exercise_registry import ExerciseRegistry
from inference.family_ai_status import deterministic_rep_score, unavailable_family_ai


class RemainingExerciseAIMVPTests(unittest.TestCase):
    def test_pushup_scope_is_not_silently_generalized(self) -> None:
        result = unavailable_family_ai("pushup", 4)
        assert result is not None
        self.assertIn("table/incline", result["scope"])
        self.assertFalse(result["available"])
        self.assertIsNone(result["ai_detected_count"])
        self.assertIsNone(result["performance_score"])

    def test_pullup_does_not_fabricate_learned_outputs(self) -> None:
        result = unavailable_family_ai("pullup", 3)
        assert result is not None
        self.assertEqual(result["official_count"], 3)
        self.assertIsNone(result["pass_count"])
        self.assertIsNone(result["fail_count"])
        self.assertEqual(result["per_rep_results"], [])

    def test_plank_contract_matches_marching_plank_not_static_hold(self) -> None:
        registry = ExerciseRegistry()
        plank = registry.get("plank")
        self.assertTrue(plank.can_analyze)
        self.assertTrue(plank.supports_repetitions)
        self.assertFalse(plank.supports_hold_time)
        result = unavailable_family_ai("plank", 2)
        assert result is not None
        self.assertIn("Marching Plank", result["scope"])

    def test_score_uses_assessed_reps_only(self) -> None:
        self.assertEqual(deterministic_rep_score(3, 1), 75.0)
        self.assertEqual(deterministic_rep_score(0, 2), 0.0)
        self.assertIsNone(deterministic_rep_score(0, 0))

    def test_other_families_unchanged(self) -> None:
        self.assertIsNone(unavailable_family_ai("lunge", 0))


if __name__ == "__main__":
    unittest.main()
