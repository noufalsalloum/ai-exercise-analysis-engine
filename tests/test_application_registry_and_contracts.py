from __future__ import annotations

import unittest
from dataclasses import replace

from application.exercise_registry import (
    ExerciseRegistry,
    PREPROCESSING_VERSION,
)
from application.runtime_router import FamilyRuntimeRouter
from application.session import SessionResultAggregator


class ExerciseApplicationRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ExerciseRegistry()

    def test_product_exercises_map_to_expected_families(self) -> None:
        expected = {
            "pushup": "pushup",
            "pullup": "pullup",
            "squat": "squat",
            "lunge": "lunge",
            "plank": "plank",
            "table_incline_pushup": "pushup",
            "wall_pushup": "pushup",
            "seal_pushup": "pushup",
            "air_squat": "squat",
            "back_squat": "squat",
            "bodyweight_lunges": "lunge",
            "reverse_dumbbell_lunges": "lunge",
            "cross_knee_plank": "plank",
            "marching_plank": "plank",
            "crunch": "crunch",
            "inchworm": "inchworm",
        }
        self.assertEqual(
            {item.exercise_id: item.family_id for item in self.registry.all()},
            expected,
        )

    def test_pushup_variations_reuse_the_same_family_expert_type(self) -> None:
        classes = {
            self.registry.get(exercise_id).family_expert_class
            for exercise_id in ("pushup", "wall_pushup", "seal_pushup")
        }
        self.assertEqual(len(classes), 1)
        self.assertIsNotNone(classes.pop())

    def test_validated_families_are_runnable_and_partial(self) -> None:
        runnable = [item.exercise_id for item in self.registry.all() if item.can_analyze]
        self.assertEqual(runnable, ["pushup", "pullup", "squat", "lunge", "plank", "table_incline_pushup"])
        self.assertEqual(self.registry.get("pushup").status, "partial")
        self.assertEqual(self.registry.get("pullup").status, "partial")
        self.assertEqual(self.registry.get("squat").status, "partial")
        self.assertEqual(self.registry.get("plank").status, "partial")
        self.assertEqual(self.registry.get("lunge").status, "partial")
        self.assertEqual(self.registry.get("table_incline_pushup").variation_id, "table_incline")
        for exercise_id in ("wall_pushup", "seal_pushup", "air_squat"):
            self.assertEqual(self.registry.get(exercise_id).status, "development")

    def test_main_screen_exposes_exactly_five_family_cards(self) -> None:
        self.assertEqual(
            [item.exercise_id for item in self.registry.main_families()],
            ["pushup", "pullup", "squat", "lunge", "plank"],
        )
        hidden_variations = {
            "wall_pushup",
            "seal_pushup",
            "air_squat",
            "back_squat",
            "bodyweight_lunges",
            "reverse_dumbbell_lunges",
            "cross_knee_plank",
            "marching_plank",
            "crunch",
            "inchworm",
        }
        self.assertTrue(hidden_variations.isdisjoint(
            {item.exercise_id for item in self.registry.main_families()}
        ))

    def test_development_exercise_cannot_run_fake_analysis(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "In Development"):
            self.registry.require_runnable("wall_pushup", "video")
        with self.assertRaisesRegex(RuntimeError, "In Development"):
            FamilyRuntimeRouter(self.registry).create("air_squat", "video")

    def test_untrained_capabilities_are_false_and_values_remain_none(self) -> None:
        exercise = self.registry.get("pushup")
        self.assertFalse(exercise.capabilities.pass_fail)
        self.assertFalse(exercise.capabilities.errors)
        self.assertFalse(exercise.capabilities.score)
        aggregator = SessionResultAggregator(exercise, "video", "side", "sample.mp4")
        result = aggregator.finalize(3.0)
        summary = result.summary
        self.assertEqual(summary.total_repetitions, 0)
        self.assertIsNone(summary.correct_repetitions)
        self.assertIsNone(summary.incorrect_repetitions)
        self.assertIsNone(summary.average_score)
        self.assertIsNone(summary.most_frequent_error)
        self.assertIsNone(summary.posture_breaks)
        self.assertEqual(result.sources["score"], "unavailable")

    def test_preprocessing_v4_contract_mismatch_is_rejected(self) -> None:
        original = self.registry.get("pushup")
        incompatible = replace(original, preprocessing_version="legacy_v3")
        with self.assertRaisesRegex(ValueError, "preprocessing version is incompatible"):
            ExerciseRegistry((incompatible,))
        self.assertEqual(original.preprocessing_version, PREPROCESSING_VERSION)


if __name__ == "__main__":
    unittest.main()
