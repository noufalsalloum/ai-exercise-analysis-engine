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

from application.exercise_registry import ExerciseRegistry
from application.workers import AnalysisWorker
from input_sources.frame_sources import VideoFrameSource
from input_sources.pose_stream import PoseStreamProcessor


class UnsupportedExerciseError(Exception):
    """exercise_id is unknown, or not runnable for video input right now."""


class VideoProcessingError(Exception):
    """The video could not be decoded, or the pipeline raised mid-run."""


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
    )
    worker.run_sync()

    terminal = [event for event in list(events.queue) if event.get("type") in {"complete", "error"}]
    if not terminal:
        raise VideoProcessingError("Analysis worker produced no terminal event.")
    event = terminal[-1]
    if event["type"] == "error":
        raise VideoProcessingError(str(event.get("message")))

    processor = coverage_holder.get("processor")
    coverage_rate = (
        processor.detected / processor.processed
        if processor is not None and processor.processed > 0
        else None
    )

    return {"result": event["result"], "pose_coverage_rate": coverage_rate}
