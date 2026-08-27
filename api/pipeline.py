"""Stateless, one-video-in-one-result-out wrapper around the engine's own
AnalysisWorker — the same worker application/workers.py uses for the desktop
app's live camera view, run synchronously instead of threaded/paced to
real-time playback. This module intentionally does not modify any engine
logic, threshold, or checkpoint; it only supplies the worker with a video
file's frames instead of a live camera's, exactly like
ui/app.py's run_headless_video() already does for the desktop app's headless
smoke-test path (see that function for the reference this mirrors).

Pose coverage (fraction of frames MediaPipe actually found a person in) is
not part of the engine's own SessionResult contract, so it is not invented
inside the engine — it's observed here, from the outside, by wrapping the
pose_factory the worker already accepts as a public constructor parameter.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import cv2

from application.exercise_registry import ExerciseRegistry
from application.workers import AnalysisWorker
from input_sources.frame_sources import VideoFrameSource
from input_sources.pose_stream import PoseStreamProcessor


class UnsupportedExerciseError(Exception):
    """exercise_id is unknown, or not runnable for video input right now."""


class VideoProcessingError(Exception):
    """The video could not be decoded, or the pipeline raised mid-run."""


def probe_video_duration_seconds(video_path: Path) -> float | None:
    """Cheap, metadata-only duration read (container header, no frame
    decode) — fast regardless of video length. Used to reject a video whose
    analysis would very likely exceed the platform's request-timeout window
    before spending any real processing time on it (see
    api.config.MAX_VIDEO_DURATION_SECONDS for how the cap was derived).
    Returns None if OpenCV can't read usable metadata — the real pipeline's
    own VideoFrameSource raises a clear, honest error in that case instead
    of this probe guessing."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            return None
        return frame_count / fps
    finally:
        capture.release()


_registry = ExerciseRegistry()


class _CoverageTrackingPoseProcessor:
    """Wraps PoseStreamProcessor purely to count detected-vs-processed frames.
    Never alters what lands on the runtime — every call is forwarded as-is."""

    def __init__(self, inner: PoseStreamProcessor) -> None:
        self._inner = inner
        self.processed = 0
        self.detected = 0

    def process(self, bgr_frame: Any, timestamp_seconds: float) -> Any:
        result = self._inner.process(bgr_frame, timestamp_seconds)
        self.processed += 1
        if result is not None:
            self.detected += 1
        return result

    def close(self) -> None:
        self._inner.close()


def _squat_experimental_ai_skipped_status() -> dict[str, Any]:
    """Honest 'not attempted' status for squat's experimental AI (boundary_v2
    / correctness_v3 / error_v1) on the synchronous /analyze request.

    These three models load fine now (see the weights_only=False fix at
    their call sites in inference/squat_ai_mvp.py), but *running* them is
    real, previously-never-executed inference work — on RunPod, a request
    that includes it measured ~111s against a synchronous request-timeout
    window of roughly 60s, for a clip whose rule-based-only analysis
    reliably completes in ~50-58s. So the synchronous request never invokes
    them at all (see run_analysis() below, squat_ai_factory=lambda: None) —
    this keeps the proven, fast core path (rep count, phase, pose coverage)
    completely unaffected by however slow or unavailable the experimental
    models are. Shape matches the engine's own real "unavailable" contract
    (application/workers.py's except-path for squat_ai.finalize_and_write)
    field-for-field, so the client's existing parsing needs no changes —
    this only fills in what the engine leaves as a bare `null` when no
    squat_ai instance was ever constructed at all.
    """

    return {
        "experimental": True,
        "status": "Experimental",
        "available": False,
        "boundary_available": False,
        "correctness_available": False,
        "error_available": False,
        "reason": (
            "Experimental squat AI (boundary_v2/correctness_v3/error_v1) is "
            "skipped by design on the synchronous /analyze request to keep "
            "response time within the platform's request-timeout window — "
            "see api/pipeline.py. This is not a load or inference failure."
        ),
        "ai_detected_reps": 0,
        "ai_correct_reps": 0,
        "ai_incorrect_reps": 0,
        "ai_pass_rate": None,
        "error_counts": {"bad_back": 0, "bad_heel": 0, "form_issue": 0},
        "per_rep_results": [],
        "score": None,
    }


def supported_exercise_ids() -> list[str]:
    """Exercise ids the registry honestly reports as runnable for video today
    — for /health, so it reflects the same registry api/main.py actually uses
    instead of a hand-maintained list that could silently drift from it."""

    return sorted(item.exercise_id for item in _registry.all() if item.can_analyze)


def ensure_runnable(exercise_id: str) -> None:
    """Cheap upfront check so the API can reject an unknown/unsupported
    exercise_id (404) before spending time downloading the video at all."""

    try:
        _registry.require_runnable(exercise_id, "video")
    except (KeyError, RuntimeError) as exc:
        raise UnsupportedExerciseError(str(exc)) from exc


def run_analysis(
    video_path: Path,
    exercise_id: str,
    pose_model_path: Path,
    camera_view: str | None = None,
) -> dict[str, Any]:
    """Run one uploaded video through the real product pipeline.

    Returns {"result": SessionResult.to_dict(), "pose_coverage_rate": float | None}.
    Raises UnsupportedExerciseError / VideoProcessingError on failure — the
    API layer maps these to the HTTP contract the Edge Function expects.
    """

    try:
        exercise = _registry.require_runnable(exercise_id, "video")
    except (KeyError, RuntimeError) as exc:
        raise UnsupportedExerciseError(str(exc)) from exc

    selected_view = camera_view or exercise.recommended_camera_view
    events: queue.Queue[dict] = queue.Queue(maxsize=64)
    coverage_holder: dict[str, _CoverageTrackingPoseProcessor] = {}

    def pose_factory() -> _CoverageTrackingPoseProcessor:
        wrapped = _CoverageTrackingPoseProcessor(PoseStreamProcessor(pose_model_path))
        coverage_holder["processor"] = wrapped
        return wrapped

    worker = AnalysisWorker(
        exercise=exercise,
        input_mode="video",
        camera_view=selected_view,
        source_factory=lambda: VideoFrameSource(video_path),
        pose_factory=pose_factory,
        events=events,
        video_path=video_path,
        # No live UI to pace playback for — process every frame as fast as
        # the pipeline can, exactly like ui/app.py's run_headless_video().
        preserve_video_timing=False,
        # Squat's experimental AI (boundary_v2/correctness_v3/error_v1) is
        # real, previously-hidden-by-checkpoint-load-failure inference work
        # — measured at ~111s on RunPod for a clip whose rule-based-only
        # path takes ~50-58s, well past the platform's synchronous
        # request-timeout window. AnalysisWorker already treats a factory
        # that returns None as "no experimental AI this session" (see
        # application/workers.py) — passing one here means squat_ai is
        # never constructed, so none of record_frame / request_live_analysis
        # / finalize_and_write ever run. This does not touch the rule-based
        # phase/rep counter at all — that is a fully separate code path
        # (runtime.update_landmarks) unaffected by squat_ai's presence.
        squat_ai_factory=(lambda: None) if exercise.family_id == "squat" else None,
    )
    worker.run_sync()

    terminal = [event for event in list(events.queue) if event.get("type") in {"complete", "error"}]
    if not terminal:
        raise VideoProcessingError("Analysis worker produced no terminal event.")
    event = terminal[-1]
    if event["type"] == "error":
        raise VideoProcessingError(str(event.get("message")))

    result = event["result"]
    if exercise.family_id == "squat" and result.get("experimental_ai") is None:
        # AnalysisWorker leaves experimental_ai as a bare None when no
        # squat_ai instance was ever constructed (see the finalization
        # elif-chain in application/workers.py — it has no branch for
        # "squat, but squat_ai is None"). Fill in an honest status instead
        # of relaying null, so the client can tell "skipped by design" apart
        # from "the engine returned nothing about this at all".
        result["experimental_ai"] = _squat_experimental_ai_skipped_status()

    processor = coverage_holder.get("processor")
    coverage_rate = (
        processor.detected / processor.processed
        if processor is not None and processor.processed > 0
        else None
    )

    return {"result": result, "pose_coverage_rate": coverage_rate}
