from __future__ import annotations

import unittest
from hashlib import sha256
from pathlib import Path

from application.presentation_contract import (
    ALLOWED_PROCESSING_STATES,
    NOT_AVAILABLE,
    PROCESSING,
    READY,
    WAITING_REPETITION,
)
from ui import strings
from ui.family_parity_presentation import (
    normalize_pushup_live_snapshot,
    present_live_family_status,
    present_shared_dashboard,
)
from ui.squat_ai_presentation import present_live_squat_ai
from input_sources.frame_sources import VideoFrameSource
from application.exercise_registry import ExerciseRegistry


ROOT = Path(__file__).resolve().parents[1]


def result_for(family: str, ai: dict | None = None) -> dict:
    return {
        "exercise_id": family,
        "family_id": family,
        "duration_seconds": 12.0,
        "summary": {"total_repetitions": 2, "hold_duration": 6.2 if family == "plank" else None},
        "experimental_ai": ai,
    }


class UIRuntimeStabilizationTests(unittest.TestCase):
    def test_not_available_is_one_exact_shared_value(self) -> None:
        self.assertEqual(NOT_AVAILABLE, "Not Available")
        self.assertEqual(strings.NOT_AVAILABLE, NOT_AVAILABLE)
        unavailable = present_live_family_status("pullup", None)
        self.assertEqual(unavailable["detected"], NOT_AVAILABLE)
        self.assertEqual(unavailable["errors"], NOT_AVAILABLE)
        self.assertNotIn("ground truth", unavailable["errors"].lower())

    def test_live_processing_states_use_only_shared_vocabulary(self) -> None:
        states = {
            READY,
            PROCESSING,
            WAITING_REPETITION,
            present_live_squat_ai(None)["processing"],
            present_live_squat_ai({"analyzing": True})["processing"],
        }
        self.assertTrue(states <= ALLOWED_PROCESSING_STATES)

    def test_all_dashboards_share_the_same_primary_field_order(self) -> None:
        push_ai = {
            "available": True,
            "scope": "Table/Incline Push-up",
            "ai_detected_reps": 1,
            "pass_count": 1,
            "fail_count": 0,
            "performance_score": 100.0,
            "assessment_coverage": 1.0,
            "per_rep_results": [{"rep_index": 1, "assessment": "PASS", "confidence": 0.8}],
        }
        lunge_ai = {
            "available": True,
            "scope": "REHAB24 Ex5 Lunge",
            "ai_detected_reps": 1,
            "per_rep_results": [{"rep_index": 1, "start_frame": 1, "end_frame": 30}],
        }
        squat_ai = {
            "available": True,
            "boundary_available": True,
            "ai_detected_reps": 1,
            "per_rep_results": [{"rep_index": 1, "assessment": "PASS", "correct_probability": 0.8}],
            "pass_count": 1,
            "fail_count": 0,
        }
        views = [
            present_shared_dashboard(result_for("squat", squat_ai)),
            present_shared_dashboard(result_for("pushup", push_ai)),
            present_shared_dashboard(result_for("pullup", {})),
            present_shared_dashboard(result_for("plank", {})),
            present_shared_dashboard(result_for("lunge", lunge_ai)),
        ]
        expected = [
            "Exercise", "Variation", "Official Reps", "Duration",
            "AI Detected Reps", "PASS", "FAIL", "Needs Review",
            "Performance Score", "Assessment Coverage",
        ]
        for index, view in enumerate(views):
            expected_session = expected[:4] if index != 3 else ["Exercise", "Variation", "Official Result", "Duration"]
            self.assertEqual(view["field_order"][:4], expected_session)
            self.assertEqual(view["field_order"][-6:], expected[-6:])

    def test_unavailable_analysis_and_form_fields_are_exact(self) -> None:
        view = present_shared_dashboard(result_for("pullup", {}))
        for _name, value in view["analysis_fields"]:
            self.assertIsNone(value)
        self.assertEqual(dict(view["form_fields"])["Error"], NOT_AVAILABLE)
        self.assertEqual(dict(view["form_fields"])["Confidence"], NOT_AVAILABLE)

    def test_plank_dashboard_exposes_hold_time(self) -> None:
        view = present_shared_dashboard(result_for("plank", {}))
        self.assertEqual(dict(view["session_fields"])["Official Result"], "2 s")
        self.assertEqual(dict(view["session_fields"])["Valid Hold Time"], "6 s")

    def test_floor_variation_does_not_display_incline_scope(self) -> None:
        ai = {"available": False, "scope": "REHAB24 Ex3 table/incline Push-up only"}
        view = present_shared_dashboard(result_for("pushup", ai))
        self.assertEqual(dict(view["session_fields"])["Variation"], "Standard / Floor Push-up")
        self.assertEqual(
            [name for name, _ in view["analysis_fields"]],
            [
                "AI Detected Reps", "PASS", "FAIL", "Needs Review",
                "Performance Score", "Assessment Coverage",
            ],
        )
        self.assertTrue(all(value is None for _, value in view["analysis_fields"]))
        self.assertEqual(dict(view["form_fields"])["Error"], NOT_AVAILABLE)
        self.assertEqual(dict(view["form_fields"])["Confidence"], NOT_AVAILABLE)
        self.assertEqual(view["rows"], [])

    def test_video_source_read_does_not_modify_or_replace_source(self) -> None:
        path = ROOT / "datasets" / "external" / "PushUpDatabase" / "Correct sequence" / "Copy of push up 2.mp4"
        before = (sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.suffix)
        source = VideoFrameSource(path, max_frames=1)
        try:
            packet = source.read()
            self.assertIsNotNone(packet)
        finally:
            source.close()
        after = (sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.suffix)
        self.assertEqual(before, after)

    def test_incline_ai_detected_count_is_boundary_count_and_may_differ(self) -> None:
        ai = {
            "available": True,
            "ai_detected_count": 4,
            "ai_detected_reps": 4,
            "per_rep_results": [
                {"rep_index": 1, "assessment": "PASS", "confidence": 0.82},
                {"rep_index": 2, "assessment": "PASS", "confidence": 0.80},
                {"rep_index": 3, "assessment": "PASS", "confidence": 0.79},
                {"rep_index": 4, "assessment": "FAIL", "confidence": 0.74},
            ],
            "pass_count": 3,
            "fail_count": 1,
            "needs_review_count": 0,
            "performance_score": 75.0,
            "assessment_coverage": 1.0,
        }
        result = result_for("pushup", ai)
        result["exercise_id"] = "table_incline_pushup"
        result["summary"]["total_repetitions"] = 3
        view = present_shared_dashboard(result)
        fields = dict(view["analysis_fields"])
        self.assertEqual(dict(view["session_fields"])["Official Reps"], 3)
        self.assertEqual(fields["AI Detected Reps"], 4)
        self.assertEqual(fields["PASS"], 3)
        self.assertEqual(fields["FAIL"], 1)
        self.assertEqual(fields["Performance Score"], "75 / 100")
        self.assertEqual(fields["Assessment Coverage"], "4 / 4 (100%)")
        self.assertEqual(present_live_family_status("pushup", ai)["detected"], "4")

    def test_floor_pushup_has_no_ex3_ai_count_or_checkpoint(self) -> None:
        registry = ExerciseRegistry()
        self.assertIsNone(registry.get("pushup").assessment_checkpoint)
        self.assertIsNotNone(registry.get("table_incline_pushup").assessment_checkpoint)
        floor = present_shared_dashboard(result_for("pushup", {"available": False}))
        self.assertIsNone(dict(floor["analysis_fields"])["AI Detected Reps"])

    def test_switching_pushup_variations_rebuilds_dashboard_rows_without_stale_ai(self) -> None:
        incline_result = result_for(
            "pushup",
            {
                "available": True,
                "ai_detected_reps": 8,
                "pass_count": 1,
                "fail_count": 7,
                "needs_review_count": 0,
                "performance_score": 12.5,
                "assessment_coverage": 1.0,
                "per_rep_results": [
                    {"rep_index": 8, "assessment": "FAIL", "confidence": 0.67}
                ],
            },
        )
        incline_result["exercise_id"] = "table_incline_pushup"
        incline = present_shared_dashboard(incline_result)
        floor = present_shared_dashboard(result_for("pushup", {"available": False}))
        incline_again = present_shared_dashboard(incline_result)

        self.assertEqual(
            [name for name, _ in incline["analysis_fields"]],
            [
                "AI Detected Reps", "PASS", "FAIL", "Needs Review",
                "Performance Score", "Assessment Coverage",
            ],
        )
        self.assertEqual(dict(incline["analysis_fields"])["PASS"], 1)
        self.assertEqual(dict(incline["analysis_fields"])["FAIL"], 7)
        self.assertEqual(dict(incline["analysis_fields"])["Performance Score"], "12.5 / 100")
        self.assertTrue(all(value is None for _, value in floor["analysis_fields"]))
        self.assertEqual(incline_again["analysis_fields"], incline["analysis_fields"])

    def test_table_incline_live_contract_populates_every_learned_field(self) -> None:
        ai = {
            "available": True,
            "ai_detected_reps": 4,
            "last_ai_rep": {"rep_index": 4, "assessment": "FAIL", "confidence": 0.74},
            "performance_score": 50.0,
            "assessment_coverage": 1.0,
            "analyzing": True,
        }
        view = present_live_family_status(
            "pushup", ai, variation_id="table_incline"
        )
        self.assertEqual(view["heading"], "AI ANALYSIS — EXPERIMENTAL")
        self.assertEqual(view["detected"], "4")
        self.assertEqual(view["processing"], PROCESSING)
        self.assertEqual(view["last"], "Rep 4 — FAIL")
        self.assertEqual(view["confidence"], "Correctness 74%")
        self.assertEqual(view["errors"], "Form Issue")
        self.assertEqual(view["score"], "50 / 100")

        ai["performance_score"] = 16.6666667
        self.assertEqual(
            present_live_family_status(
                "pushup", ai, variation_id="table_incline"
            )["score"],
            "16.7 / 100",
        )

    def test_table_incline_pass_hides_error(self) -> None:
        view = present_live_family_status(
            "pushup",
            {
                "available": True,
                "ai_detected_count": 1,
                "last_ai_rep": {"rep_index": 1, "assessment": "PASS", "confidence": 0.82},
                "performance_score": 100.0,
            },
            variation_id="table_incline",
        )
        self.assertEqual(view["last"], "Rep 1 — PASS")
        self.assertEqual(view["errors"], NOT_AVAILABLE)

    def test_floor_pushup_live_guard_rejects_available_ex3_payload(self) -> None:
        leaked = present_live_family_status(
            "pushup",
            {
                "available": True,
                "ai_detected_reps": 9,
                "last_ai_rep": {"rep_index": 9, "assessment": "PASS", "confidence": 0.99},
                "performance_score": 100.0,
            },
            variation_id="standard",
        )
        self.assertTrue(all(value == NOT_AVAILABLE for value in leaked.values()))

    def test_variation_switch_does_not_leak_live_pushup_result(self) -> None:
        incline = present_live_family_status(
            "pushup",
            {"available": True, "ai_detected_reps": 2},
            variation_id="table_incline",
        )
        floor = present_live_family_status(
            "pushup",
            {"available": True, "ai_detected_reps": 2},
            variation_id="standard",
        )
        self.assertEqual(incline["detected"], "2")
        self.assertEqual(floor["detected"], NOT_AVAILABLE)

    def test_normalized_pushup_session_snapshot_preserves_actual_values(self) -> None:
        snapshot = normalize_pushup_live_snapshot(
            {
                "ai_detected_reps": 8,
                "pass_count": 1,
                "fail_count": 7,
                "per_rep_results": [
                    {"rep_index": 8, "assessment": "FAIL", "confidence": 0.81}
                ],
                "performance_score": 12.5,
                "assessment_coverage": 1.0,
            },
            completed=True,
        )
        self.assertEqual(snapshot["detected"], "8")
        self.assertEqual(snapshot["last"], "Rep 8 — FAIL")
        self.assertEqual(snapshot["confidence"], "Correctness 81%")
        self.assertEqual(snapshot["score"], "12.5 / 100")
        self.assertEqual(snapshot["errors"], "Form Issue")
        self.assertEqual(snapshot["coverage"], "8 / 8 (100%)")
        self.assertEqual(snapshot["processing"], "Completed")

    def test_zero_values_are_not_presented_as_unavailable(self) -> None:
        snapshot = normalize_pushup_live_snapshot(
            {
                "ai_detected_reps": 0,
                "pass_count": 0,
                "fail_count": 0,
                "last_ai_rep": {
                    "rep_index": 1,
                    "assessment": "FAIL",
                    "confidence": 0.0,
                },
                "performance_score": 0.0,
                "assessment_coverage": 0.0,
                "analyzing": False,
            }
        )
        self.assertEqual(snapshot["detected"], "0")
        self.assertEqual(snapshot["confidence"], "Correctness 0%")
        self.assertEqual(snapshot["score"], "0 / 100")
        self.assertEqual(snapshot["coverage"], "0 / 0 (0%)")


if __name__ == "__main__":
    unittest.main()
