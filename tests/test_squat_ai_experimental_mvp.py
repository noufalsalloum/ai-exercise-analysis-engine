"""Focused safety and policy tests for Experimental Squat AI MVP."""

from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from application.exercise_registry import ExerciseRegistry
from application.workers import AnalysisWorker
from inference.squat_ai_mvp import (
    SquatAIExperimentalOrchestrator,
    SquatAIMVPConfig,
    aggregate_ai_results,
    apply_decision_policy,
)
from models.squat_correctness import SquatCorrectnessModel
from models.squat_posture_error import load_squat_posture_error_checkpoint
from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from tests.test_application_worker import FakePose, FakeSource
from training.squat_correctness import load_checkpoint_strict


ROOT = Path(__file__).resolve().parents[1]
CONFIG = SquatAIMVPConfig.load()
CACHE = (
    ROOT
    / "datasets"
    / "window_cache"
    / "rehab24_squat_v1"
    / "full_videos"
    / "PM_022-Camera17_cam17.npz"
)


def decision(probability: float, error: str | None) -> dict:
    return apply_decision_policy(
        rep_index=1,
        start_frame=10,
        end_frame=80,
        correct_probability=probability,
        threshold=0.61,
        raw_error_class=error,
        error_confidence=0.9 if error is not None else None,
        representative_frame=50,
    )


class FakeExperimentalAI:
    def __init__(self) -> None:
        self.recorded: list[int] = []
        self.closed = False

    def record_frame(self, _landmarks: np.ndarray, frame_index: int, _timestamp: float) -> None:
        self.recorded.append(frame_index)

    def request_live_analysis(self) -> None:
        return None

    def live_status(self) -> dict:
        return {
            "experimental": True,
            "ai_detected_reps": 0,
            "last_ai_rep": None,
            "analyzing": False,
            "boundary_available": True,
            "correctness_available": True,
            "error_available": True,
        }

    def finalize_and_write(self, **_kwargs: object) -> tuple[dict, Path]:
        return aggregate_ai_results([], detected_count=0), Path("unused.json")

    def close(self) -> None:
        self.closed = True


class SquatAIExperimentalPolicyTests(unittest.TestCase):
    def test_correct_becomes_pass(self) -> None:
        result = decision(0.75, "good")
        self.assertEqual(result["correctness"], "correct")
        self.assertEqual(result["pass_fail"], "PASS")
        self.assertIsNone(result["error_class"])

    def test_incorrect_becomes_fail(self) -> None:
        result = decision(0.40, "good")
        self.assertEqual(result["correctness"], "incorrect")
        self.assertEqual(result["pass_fail"], "FAIL")
        self.assertEqual(result["error_class"], "form_issue")

    def test_incorrect_error_explanations(self) -> None:
        self.assertEqual(decision(0.40, "bad_back")["error_class"], "bad_back")
        self.assertEqual(decision(0.40, "bad_heel")["error_class"], "bad_heel")

    def test_correct_error_conflict_does_not_flip_pass(self) -> None:
        result = decision(0.80, "bad_back")
        self.assertEqual(result["pass_fail"], "PASS")
        self.assertIsNone(result["error_class"])
        self.assertTrue(result["ai_conflict"])

    def test_aggregation_pass_rate_errors_and_score(self) -> None:
        values = [
            decision(0.80, "good"),
            {**decision(0.40, "bad_back"), "rep_index": 2},
            {**decision(0.30, "bad_heel"), "rep_index": 3},
        ]
        result = aggregate_ai_results(values, detected_count=3)
        self.assertEqual(result["ai_correct_reps"], 1)
        self.assertEqual(result["ai_incorrect_reps"], 2)
        self.assertAlmostEqual(result["ai_pass_rate"], 1 / 3)
        self.assertEqual(result["error_counts"]["bad_back"], 1)
        self.assertEqual(result["error_counts"]["bad_heel"], 1)
        self.assertAlmostEqual(result["score"], 100.0 / 3.0, places=1)
        self.assertEqual(result["needs_review_count"], 0)
        self.assertEqual(result["assessment_coverage"], 1.0)

    def test_duplicate_overlap_is_rejected(self) -> None:
        known = [{"start_sequence_index": 10, "end_sequence_index": 80}]
        self.assertTrue(SquatAIExperimentalOrchestrator._range_matches(12, 82, known))
        self.assertFalse(SquatAIExperimentalOrchestrator._range_matches(100, 160, known))


class SquatAIExperimentalCheckpointTests(unittest.TestCase):
    def test_boundary_v2_loads_strictly(self) -> None:
        checkpoint = torch.load(CONFIG.boundary_checkpoint, map_location="cpu", weights_only=True)
        model = SquatRepBoundaryV2Model()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(checkpoint["experiment"]["architecture"], "boundary_aux_tcn")

    def test_correctness_v3_loads_strictly_at_point_61(self) -> None:
        model = SquatCorrectnessModel(CONFIG.motionbert_checkpoint)
        checkpoint = load_checkpoint_strict(CONFIG.correctness_checkpoint, model, torch.device("cpu"))
        self.assertEqual(float(checkpoint["decision_threshold"]), 0.61)
        self.assertTrue(checkpoint["motionbert_frozen"])

    def test_error_v1_loads_strictly(self) -> None:
        model, checkpoint = load_squat_posture_error_checkpoint(
            CONFIG.error_checkpoint, torch.device("cpu")
        )
        self.assertEqual(tuple(model(torch.zeros(1, checkpoint["input_dim"])).shape), (1, 3))


class SquatAIExperimentalWorkerSafetyTests(unittest.TestCase):
    def test_model_factory_failure_does_not_crash_rule_runtime(self) -> None:
        source = FakeSource([0.0, 0.1, 0.2])
        pose = FakePose([170, 150, 130])
        events: queue.Queue[dict] = queue.Queue(maxsize=20)

        def fail() -> None:
            raise RuntimeError("intentional AI load failure")

        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("squat"),
            input_mode="video",
            camera_view="side",
            source_factory=lambda: source,
            pose_factory=lambda: pose,
            events=events,
            squat_shadow_factory=lambda: None,
            squat_ai_factory=fail,
            preserve_video_timing=False,
        )
        worker.run_sync()
        complete = [item for item in list(events.queue) if item.get("type") == "complete"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]["result"]["summary"]["total_repetitions"], 0)

    def test_countdown_frames_never_reach_ai(self) -> None:
        source = FakeSource([0, 1, 2, 3, 4, 5, 6])
        pose = FakePose([170] * 7)
        ai = FakeExperimentalAI()
        events: queue.Queue[dict] = queue.Queue(maxsize=30)
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("squat"),
            input_mode="realtime",
            camera_view="side",
            source_factory=lambda: source,
            pose_factory=lambda: pose,
            events=events,
            squat_shadow_factory=lambda: None,
            squat_ai_factory=lambda: ai,
            preserve_video_timing=False,
        )
        worker.run_sync()
        self.assertEqual(ai.recorded, [5, 6])
        self.assertTrue(ai.closed)


@unittest.skipUnless(CACHE.is_file(), "Development pose cache unavailable")
class SquatAIExperimentalRealInferenceTests(unittest.TestCase):
    def test_one_detected_repetition_end_to_end(self) -> None:
        with np.load(CACHE, allow_pickle=False) as archive:
            landmarks = np.asarray(archive["landmarks"][:400], dtype=np.float32)
            fps = float(archive["fps"])
        orchestrator = SquatAIExperimentalOrchestrator(CONFIG, allow_partial=False)
        try:
            for index, frame in enumerate(landmarks):
                orchestrator.record_frame(frame, index, index / fps)
            result = orchestrator.finalize()
        finally:
            orchestrator.close()
        self.assertGreaterEqual(result["ai_detected_reps"], 1)
        first = result["per_rep_results"][0]
        self.assertIn(first["pass_fail"], {"PASS", "FAIL"})
        self.assertIsNotNone(first["correct_probability"])
        self.assertIsNotNone(first["representative_frame"])
        self.assertIsNone(first["score"])
        self.assertNotIn("averaged_correctness_experiment", first)


if __name__ == "__main__":
    unittest.main()
