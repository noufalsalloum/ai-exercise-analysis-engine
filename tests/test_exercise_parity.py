from __future__ import annotations

import unittest

from application.exercise_registry import ExerciseRegistry
from application.runtime_router import FamilyRuntimeRouter
from inference.family_ai_status import deterministic_rep_score, rule_based_parity_status, unavailable_family_ai
from ui.family_parity_presentation import present_family_dashboard, present_live_family_status


class ExerciseParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ExerciseRegistry()

    def test_table_incline_is_the_only_learned_pushup_variation(self) -> None:
        floor = self.registry.get("pushup")
        incline = self.registry.get("table_incline_pushup")
        self.assertEqual(floor.variation_id, "standard")
        self.assertIsNone(floor.assessment_checkpoint)
        self.assertEqual(incline.variation_id, "table_incline")
        self.assertIsNotNone(incline.assessment_checkpoint)
        self.assertIn("table/incline", unavailable_family_ai("pushup")["scope"])
        self.assertEqual(type(FamilyRuntimeRouter(self.registry).create("table_incline_pushup", "video")).__name__, "PushupRepetitionRuntime")

    def test_pushup_pass_fail_confidence_form_issue_score_and_coverage(self) -> None:
        ai = {
            "available": True, "scope": "Table/Incline Push-up", "ai_detected_reps": 2,
            "pass_count": 1, "fail_count": 1, "performance_score": deterministic_rep_score(1, 1),
            "assessment_coverage": 1.0,
            "per_rep_results": [
                {"rep_index": 1, "assessment": "PASS", "confidence": .81},
                {"rep_index": 2, "assessment": "FAIL", "confidence": .74, "error_class": "form_issue"},
            ],
            "model_status": {},
        }
        view = present_family_dashboard("pushup", ai)
        self.assertEqual(view["score"], 50.0)
        self.assertEqual(view["coverage"], 1.0)
        self.assertNotIn("Form Issue", view["rows"][0])
        self.assertIn("Form Issue", view["rows"][1])
        self.assertIn("74%", view["rows"][1])

    def test_pullup_rule_events_have_timing_without_fake_assessment_or_score(self) -> None:
        status = rule_based_parity_status("pullup", 1, [{"repetition_index": 1, "duration_seconds": 1.4, "confidence": .9}])
        self.assertIsNone(status["pass_count"])
        self.assertIsNone(status["performance_score"])
        view = present_family_dashboard("pullup", status)
        self.assertIn("1.40 s", view["rows"][0])
        self.assertFalse(view["learned"])

    def test_marching_plank_identity_and_cross_knee_status_are_truthful(self) -> None:
        plank = self.registry.get("plank")
        self.assertEqual(plank.variation_id, "marching")
        status = rule_based_parity_status("plank", 1, [{"repetition_index": 1, "duration_seconds": 1.0, "confidence": .8}])
        self.assertEqual(status["scope"], "Marching Plank")
        self.assertEqual(status["model_status"]["cross_knee_plank"], "In Development")
        self.assertIsNone(status["performance_score"])

    def test_live_and_new_contracts_do_not_leak_cross_family_results(self) -> None:
        pull = rule_based_parity_status("pullup", 1, [{"repetition_index": 1, "duration_seconds": 1.0}])
        plank = rule_based_parity_status("plank", 0, [])
        lunge = {
            "available": True,
            "ai_detected_reps": 0,
            "last_ai_rep": None,
            "per_rep_results": [],
            "model_status": {},
            "scope": "REHAB24 Ex5 Lunge",
            "performance_score": None,
        }
        self.assertIn("Rep 1", present_live_family_status("pullup", pull)["last"])
        self.assertIn("Waiting", present_live_family_status("plank", plank)["last"])
        self.assertIn("Waiting", present_live_family_status("lunge", lunge)["last"])
        self.assertEqual(plank["unit_count"], 0)
        self.assertIsNone(plank["performance_score"])
        self.assertEqual(present_family_dashboard("lunge", lunge)["rows"], [])
        self.assertIsNone(present_family_dashboard("lunge", lunge)["score"])


if __name__ == "__main__":
    unittest.main()
