"""Tests for api/pipeline.py's synchronous-request-safety behavior: squat's
experimental AI (boundary_v2/correctness_v3/error_v1) must never block the
synchronous /analyze request, and an over-long video must be rejected before
any real processing starts. See api/config.py's MAX_VIDEO_DURATION_SECONDS
comment and api/pipeline.py's _squat_experimental_ai_skipped_status() for the
reasoning this guards.

Mocks application.workers.AnalysisWorker directly, the same way
tests/test_application_worker.py exercises the worker itself with fakes —
this file is about api/pipeline.py's own orchestration logic, not the
worker's internals (already covered elsewhere), so it does not need real
checkpoints, MediaPipe, or torch to run.
"""

from __future__ import annotations

import queue
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from api.pipeline import (
    _squat_experimental_ai_skipped_status,
    probe_video_duration_seconds,
    run_analysis,
)


def _fake_session_result(exercise_id: str, family_id: str, experimental_ai) -> dict:
    return {
        "session_id": "fake-session",
        "exercise_id": exercise_id,
        "family_id": family_id,
        "summary": {"total_repetitions": 4},
        "experimental_ai": experimental_ai,
    }


class SquatExperimentalAiSkippedStatusTests(unittest.TestCase):
    def test_shape_is_honestly_unavailable_not_a_fabricated_result(self) -> None:
        status = _squat_experimental_ai_skipped_status()
        self.assertFalse(status["available"])
        self.assertFalse(status["boundary_available"])
        self.assertFalse(status["correctness_available"])
        self.assertFalse(status["error_available"])
        self.assertEqual(status["ai_detected_reps"], 0)
        self.assertEqual(status["ai_correct_reps"], 0)
        self.assertEqual(status["ai_incorrect_reps"], 0)
        self.assertIsNone(status["score"])
        self.assertEqual(status["per_rep_results"], [])
        # The reason must say "skipped by design" and explicitly disclaim
        # that this is a failure — it never ran on purpose, it didn't crash.
        self.assertIn("skipped by design", status["reason"])
        self.assertIn("not a load or inference failure", status["reason"].lower())


class RunAnalysisSquatAiSkipTests(unittest.TestCase):
    """Verifies run_analysis() never lets squat_ai block the request, and
    that it honestly fills in the gap AnalysisWorker itself leaves (a bare
    None) when no squat_ai instance was ever constructed."""

    def _run_with_mocked_worker(self, exercise_id: str, family_id: str):
        fake_exercise = MagicMock()
        fake_exercise.family_id = family_id
        fake_exercise.recommended_camera_view = "side"

        mock_registry = MagicMock()
        mock_registry.require_runnable.return_value = fake_exercise

        captured_kwargs = {}

        def fake_worker_ctor(**kwargs):
            captured_kwargs.update(kwargs)
            worker = MagicMock()

            def fake_run_sync():
                kwargs["events"].put(
                    {"type": "complete", "result": _fake_session_result(exercise_id, family_id, None)}
                )

            worker.run_sync.side_effect = fake_run_sync
            return worker

        with patch("api.pipeline._registry", mock_registry), patch(
            "api.pipeline.AnalysisWorker", side_effect=fake_worker_ctor
        ), patch("api.pipeline.VideoFrameSource"), patch("api.pipeline.PoseStreamProcessor"):
            outcome = run_analysis(
                video_path=Path("irrelevant-for-this-test.mp4"),
                exercise_id=exercise_id,
                pose_model_path=Path("irrelevant-for-this-test.task"),
            )
        return outcome, captured_kwargs

    def test_squat_passes_a_factory_that_returns_none(self) -> None:
        _, kwargs = self._run_with_mocked_worker("squat", "squat")
        factory = kwargs.get("squat_ai_factory")
        self.assertIsNotNone(factory, "squat must pass an explicit squat_ai_factory, not leave the default")
        self.assertIsNone(factory(), "the factory itself must return None so AnalysisWorker never constructs squat_ai")

    def test_non_squat_family_does_not_touch_squat_ai_factory(self) -> None:
        _, kwargs = self._run_with_mocked_worker("pushup", "pushup")
        self.assertIsNone(kwargs.get("squat_ai_factory"), "non-squat families must be unaffected by this change")

    def test_squat_response_gets_an_honest_skipped_status_not_a_bare_null(self) -> None:
        outcome, _ = self._run_with_mocked_worker("squat", "squat")
        experimental_ai = outcome["result"]["experimental_ai"]
        self.assertIsNotNone(experimental_ai)
        self.assertFalse(experimental_ai["available"])
        self.assertIn("skipped by design", experimental_ai["reason"])

    def test_non_squat_null_experimental_ai_is_left_alone(self) -> None:
        # Guard against accidentally widening the fill-in logic to every
        # family — only squat's specific bare-None gap is patched.
        outcome, _ = self._run_with_mocked_worker("pushup", "pushup")
        self.assertIsNone(outcome["result"]["experimental_ai"])


class ProbeVideoDurationTests(unittest.TestCase):
    def test_reads_real_duration_from_a_real_small_video(self) -> None:
        import tempfile

        path = Path(tempfile.gettempdir()) / "api_pipeline_duration_probe_test.mp4"
        fps = 10.0
        frame_count = 25  # 2.5 real seconds
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 16))
        try:
            for _ in range(frame_count):
                writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        finally:
            writer.release()
        try:
            duration = probe_video_duration_seconds(path)
            self.assertIsNotNone(duration)
            self.assertAlmostEqual(duration, frame_count / fps, delta=0.2)
        finally:
            path.unlink(missing_ok=True)

    def test_returns_none_for_a_nonexistent_file_rather_than_raising(self) -> None:
        self.assertIsNone(probe_video_duration_seconds(Path("this-file-does-not-exist.mp4")))


if __name__ == "__main__":
    unittest.main()
