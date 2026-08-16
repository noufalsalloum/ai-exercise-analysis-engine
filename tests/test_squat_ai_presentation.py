"""Focused tests for the Squat-only Experimental AI presentation layer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from ui.app import ExerciseAnalysisApp
from ui.squat_ai_presentation import (
    present_ai_rep,
    present_live_squat_ai,
    present_squat_dashboard,
)


def _rep(
    index: int,
    pass_fail: str | None,
    error: str | None = None,
    probability: float | None = None,
    error_confidence: float | None = None,
) -> dict:
    return {
        "rep_index": index,
        "pass_fail": pass_fail,
        "error_class": error,
        "correct_probability": probability,
        "error_confidence": error_confidence,
        "score": None,
    }


def _aggregate(reps: list[dict]) -> dict:
    correct = sum(rep["pass_fail"] == "PASS" for rep in reps)
    incorrect = sum(rep["pass_fail"] == "FAIL" for rep in reps)
    return {
        "available": True,
        "boundary_available": True,
        "correctness_available": True,
        "error_available": True,
        "ai_detected_reps": len(reps),
        "ai_correct_reps": correct,
        "ai_incorrect_reps": incorrect,
        "pass_count": correct,
        "fail_count": incorrect,
        "ai_pass_rate": correct / len(reps) if reps else None,
        "error_counts": {
            "bad_back": sum(rep["error_class"] == "bad_back" for rep in reps),
            "bad_heel": sum(rep["error_class"] == "bad_heel" for rep in reps),
            "form_issue": sum(rep["error_class"] == "form_issue" for rep in reps),
        },
        "per_rep_results": reps,
        "score": None,
    }


class SquatAIPresentationTests(unittest.TestCase):
    def test_ai_dashboard_is_squat_only_and_counts_are_independent(self) -> None:
        ai = _aggregate(
            [_rep(1, "PASS", probability=0.8), _rep(2, "FAIL", "bad_back", 0.2)]
        )
        self.assertIsNone(present_squat_dashboard("pushup", 9, ai))
        view = present_squat_dashboard("squat", 3, ai)
        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view["official_count"], 3)
        self.assertEqual(view["ai_detected_reps"], 2)
        self.assertEqual(view["difference"], -1)
        self.assertEqual(view["pass_rate"], "50%")

    def test_pass_rep_hides_error_and_score(self) -> None:
        view = present_ai_rep(_rep(4, "PASS", "bad_back", 0.87, 0.99))
        self.assertEqual(view["summary"], "Rep 4 — PASS")
        self.assertEqual(view["detail"], "Correct form")
        self.assertIsNone(view["error"])
        self.assertIsNone(view["error_confidence"])
        self.assertEqual(view["correctness_confidence"], "87%")
        self.assertIsNone(view["score"])

    def test_fail_error_labels_and_confidences(self) -> None:
        back = present_ai_rep(_rep(5, "FAIL", "bad_back", 0.18, 0.91))
        heel = present_ai_rep(_rep(6, "FAIL", "bad_heel", 0.13, 0.93))
        unknown = present_ai_rep(_rep(7, "FAIL", "other", 0.3, 0.77))
        self.assertEqual(back["summary"], "Rep 5 — FAIL — Bad Back")
        self.assertEqual(heel["summary"], "Rep 6 — FAIL — Bad Heel")
        self.assertEqual(unknown["summary"], "Rep 7 — FAIL — Form Issue")
        self.assertEqual(back["correctness_confidence"], "82%")
        self.assertEqual(back["error_confidence"], "91%")
        self.assertIsNone(unknown["error_confidence"])

    def test_per_rep_list_deduplicates_repetition_indices(self) -> None:
        first = _rep(1, "PASS", probability=0.8)
        duplicate = _rep(1, "FAIL", "bad_heel", 0.2)
        view = present_squat_dashboard("squat", 1, _aggregate([first, duplicate]))
        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(len(view["per_rep_results"]), 1)
        self.assertEqual(view["per_rep_results"][0]["pass_fail"], "PASS")

    def test_unavailable_states_do_not_invent_results(self) -> None:
        live = present_live_squat_ai(
            {
                "boundary_available": False,
                "correctness_available": False,
                "error_available": False,
            }
        )
        self.assertEqual(live["detected"], "Not Available")
        self.assertEqual(live["last_rep"], "Not Available")
        rep = present_ai_rep(_rep(1, None, "bad_back", None, 0.99))
        self.assertIsNone(rep["pass_fail"])
        self.assertIsNone(rep["error"])
        self.assertIn("Not Available", rep["summary"])

    def test_live_waiting_analyzing_and_completed_states(self) -> None:
        waiting = present_live_squat_ai(
            {
                "boundary_available": True,
                "ai_detected_reps": 0,
                "analyzing": False,
                "analyzing_rep_index": None,
            }
        )
        self.assertEqual(waiting["last_rep"], "Waiting for Repetition")
        analyzing = present_live_squat_ai(
            {
                "boundary_available": True,
                "ai_detected_reps": 2,
                "analyzing": True,
                "analyzing_rep_index": 3,
            }
        )
        self.assertEqual(analyzing["processing"], "Processing")
        completed = present_live_squat_ai(
            {
                "boundary_available": True,
                "correctness_available": True,
                "error_available": True,
                "ai_detected_reps": 1,
                "analyzing": False,
                "last_ai_rep": _rep(1, "FAIL", "bad_heel", 0.1, 0.92),
            }
        )
        self.assertEqual(completed["last_rep"], "Rep 1 — FAIL — Bad Heel")
        self.assertEqual(completed["confidence"], "Correctness 90% | Error 92%")

    def test_reset_clears_all_ui_only_ai_state(self) -> None:
        app = ExerciseAnalysisApp.__new__(ExerciseAnalysisApp)
        app._last_result = {"old": True}
        app._last_worker_stats = {"old": True}
        app._dashboard_ai_view = {"old": True}
        app._ai_rep_text_widget = SimpleNamespace()
        app._reset_session_presentation_state()
        self.assertIsNone(app._last_result)
        self.assertIsNone(app._last_worker_stats)
        self.assertIsNone(app._dashboard_ai_view)
        self.assertIsNone(app._ai_rep_text_widget)

    def test_model_failure_dashboard_keeps_score_unavailable(self) -> None:
        view = present_squat_dashboard(
            "squat",
            4,
            {
                "available": False,
                "boundary_available": False,
                "correctness_available": False,
                "error_available": False,
                "ai_detected_reps": 0,
                "per_rep_results": [],
                "error_counts": {},
                "score": None,
            },
        )
        self.assertIsNotNone(view)
        assert view is not None
        self.assertFalse(view["available"])
        self.assertIsNone(view["ai_detected_reps"])
        self.assertIsNone(view["score"])

    def test_needs_review_has_no_error_and_is_presented_truthfully(self) -> None:
        view = present_ai_rep(
            {
                "rep_index": 8,
                "assessment": "NEEDS_REVIEW",
                "pass_fail": "NEEDS_REVIEW",
                "correct_probability": 0.2,
                "assessment_confidence": 0.46,
                "error_class": None,
                "error_confidence": None,
                "needs_review_reason": "Unsupported full side view",
            }
        )
        self.assertEqual(view["summary"], "Rep 8 — NEEDS REVIEW")
        self.assertEqual(view["detail"], "Unsupported full side view")
        self.assertIsNone(view["error"])
        self.assertEqual(view["correctness_confidence"], "46%")


if __name__ == "__main__":
    unittest.main()
