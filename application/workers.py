from __future__ import annotations

import queue
import threading
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Protocol

from input_sources.frame_sources import FramePacket
from input_sources.pose_stream import draw_pose_skeleton, pose_visible_for_family
from .contracts import RepetitionResult
from .exercise_registry import ExerciseDefinition
from .feedback_events import FeedbackEvent
from .runtime_router import FamilyRuntimeRouter
from .session import SessionResultAggregator
from .presentation_contract import PROCESSING, READY
from inference.squat_ai_mvp import SquatAIExperimentalOrchestrator
from inference.pushup_ai_mvp import PushupTableAIOrchestrator
from inference.lunge_ai_mvp import LungeBoundaryExperimentalOrchestrator
from inference.family_ai_status import unavailable_family_ai, rule_based_parity_status
from .squat_shadow import SquatBoundaryShadow


class FrameSource(Protocol):
    fps: float
    stream_lost: bool
    backend_name: str
    camera_index: int | None

    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


class PoseProcessor(Protocol):
    def process(self, frame: Any, timestamp_seconds: float) -> Any: ...
    def close(self) -> None: ...


class FamilyFrameResult(Protocol):
    phase: str
    repetition_count: int
    elbow_angle: float | None
    selected_side: str | None
    signal_confidence: float | None
    last_repetition_duration: float | None
    current_cycle_valid: bool
    completed_cycle: Any
    valid_frame: bool


class AnalysisWorker:
    """Background processing with paired frames/metrics and safe resource cleanup."""

    def __init__(
        self,
        exercise: ExerciseDefinition,
        input_mode: str,
        camera_view: str,
        source_factory: Callable[[], FrameSource],
        pose_factory: Callable[[], PoseProcessor],
        events: queue.Queue[dict] | None = None,
        video_path: str | Path | None = None,
        runtime_router: FamilyRuntimeRouter | None = None,
        preserve_video_timing: bool = True,
        squat_shadow_factory: Callable[[], Any | None] | None = None,
        squat_ai_factory: Callable[[], Any | None] | None = None,
        pushup_ai_factory: Callable[[], Any | None] | None = None,
        lunge_ai_factory: Callable[[], Any | None] | None = None,
        audio_feedback_factory: Callable[[], Any | None] | None = None,
    ) -> None:
        self.exercise = exercise
        self.input_mode = input_mode
        self.camera_view = camera_view
        self.source_factory = source_factory
        self.pose_factory = pose_factory
        self.events: queue.Queue[dict] = events or queue.Queue(maxsize=4)
        self.video_path = Path(video_path) if video_path is not None else None
        self.runtime_router = runtime_router or FamilyRuntimeRouter()
        self.preserve_video_timing = bool(preserve_video_timing)
        self.squat_shadow_factory = squat_shadow_factory
        self.squat_ai_factory = squat_ai_factory
        self.pushup_ai_factory = pushup_ai_factory
        self.lunge_ai_factory = lunge_ai_factory
        self.audio_feedback_factory = audio_feedback_factory
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._resource_lock = threading.Lock()
        self._source: FrameSource | None = None
        self._pose: PoseProcessor | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            raise RuntimeError("Analysis worker is already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="exercise-analysis",
            daemon=True,
        )
        self._thread.start()

    def run_sync(self) -> None:
        self._stop_event.clear()
        self._run()

    def stop(self) -> None:
        self._stop_event.set()
        with self._resource_lock:
            if self._source is not None:
                self._source.close()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _put_control(self, event: dict) -> None:
        while True:
            try:
                self.events.put(event, timeout=0.1)
                return
            except queue.Full:
                try:
                    discarded = self.events.get_nowait()
                    if discarded.get("type") != "frame":
                        self.events.put_nowait(discarded)
                except queue.Empty:
                    pass

    def _put_frame(self, event: dict) -> None:
        """Keep latency bounded by replacing an old display frame when full."""

        try:
            self.events.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            oldest = self.events.get_nowait()
        except queue.Empty:
            return
        if oldest.get("type") != "frame":
            try:
                self.events.put_nowait(oldest)
            except queue.Full:
                pass
            return
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def _pace_uploaded_video(
        self,
        packet: FramePacket,
        playback_wall_start: float,
        playback_timestamp_start: float,
    ) -> bool:
        if self.input_mode != "video" or not self.preserve_video_timing:
            return True
        target = playback_wall_start + packet.timestamp_seconds - playback_timestamp_start
        delay = target - monotonic()
        return not (delay > 0.0 and self._stop_event.wait(delay))

    def _run(self) -> None:
        source: FrameSource | None = None
        pose: PoseProcessor | None = None
        aggregator = SessionResultAggregator(
            self.exercise,
            self.input_mode,
            self.camera_view,
            self.video_path,
        )
        runtime = self.runtime_router.create(
            self.exercise.exercise_id,
            self.input_mode,
            camera_view=self.camera_view,
        )
        squat_shadow: Any | None = None
        squat_ai: Any | None = None
        pushup_ai: Any | None = None
        lunge_ai: Any | None = None
        audio_feedback: Any | None = None
        try:
            audio_feedback = (
                self.audio_feedback_factory()
                if self.audio_feedback_factory is not None
                else None
            )
        except Exception:
            # Audio is an optional output channel. Initialization, pre-warm, or
            # dependency failures must never affect video analysis or counting.
            audio_feedback = None
        live_ai_status: dict[str, Any] | None = None
        if self.exercise.family_id == "squat":
            try:
                squat_shadow = (
                    self.squat_shadow_factory()
                    if self.squat_shadow_factory is not None
                    else SquatBoundaryShadow.from_default_config()
                )
            except Exception:
                # Shadow inference is observational and must never break the
                # existing user-facing rule-based Squat session.
                squat_shadow = None
            try:
                squat_ai = (
                    self.squat_ai_factory()
                    if self.squat_ai_factory is not None
                    else SquatAIExperimentalOrchestrator.from_default_config(
                        camera_view=self.camera_view
                    )
                )
                if squat_ai is not None:
                    live_ai_status = squat_ai.live_status()
            except Exception as exc:
                # All experimental models are fail-open: official rule-based
                # phase/counting must remain usable even when AI cannot load.
                squat_ai = None
                live_ai_status = {
                    "experimental": True,
                    "ai_detected_reps": 0,
                    "last_ai_rep": None,
                    "analyzing": False,
                    "analyzing_rep_index": None,
                    "boundary_available": False,
                    "correctness_available": False,
                    "error_available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        elif self.exercise.family_id == "pushup" and self.exercise.variation_id == "table_incline":
            try:
                pushup_ai = (
                    self.pushup_ai_factory()
                    if self.pushup_ai_factory is not None
                    else PushupTableAIOrchestrator.from_default_config()
                )
                live_ai_status = pushup_ai.live_status()
            except Exception as exc:
                pushup_ai = None
                live_ai_status = unavailable_family_ai("pushup", 0)
                if live_ai_status is not None:
                    live_ai_status["reason"] = f"Push-up learned models unavailable: {type(exc).__name__}: {exc}"
        elif self.exercise.family_id in {"pullup", "plank"}:
            live_ai_status = rule_based_parity_status(self.exercise.family_id, 0)
        elif self.exercise.family_id == "lunge":
            try:
                lunge_ai=(self.lunge_ai_factory() if self.lunge_ai_factory is not None else LungeBoundaryExperimentalOrchestrator.from_default_config())
                live_ai_status=lunge_ai.live_status()
            except Exception as exc:
                lunge_ai=None; live_ai_status={"exercise":"lunge","available":False,"experimental":True,"scope":"REHAB24 Ex5 Lunge","reason":f"Boundary unavailable: {type(exc).__name__}: {exc}","ai_detected_reps":None,"performance_score":None,"assessment_coverage":None,"per_rep_results":[],"model_status":{"detection":"Unavailable","correctness":"Not Enabled","errors":"Not Available","score":"Not Available"}}
        else:
            live_ai_status = unavailable_family_ai(self.exercise.family_id, 0)
        last_packet: FramePacket | None = None
        analysis_start_timestamp: float | None = None
        countdown_start: float | None = None
        playback_wall_start: float | None = None
        playback_timestamp_start: float | None = None
        saw_frame = False
        processed_frame_count = 0
        try:
            source = self.source_factory()
            pose = self.pose_factory()
            with self._resource_lock:
                self._source = source
                self._pose = pose
            self._put_control(
                {
                    "type": "started",
                    "input_mode": self.input_mode,
                    "backend": getattr(source, "backend_name", None),
                    "camera_index": getattr(source, "camera_index", None),
                    "first_frame_verified": bool(
                        getattr(source, "first_frame_verified", self.input_mode == "video")
                    ),
                }
            )
            while not self._stop_event.is_set():
                packet = source.read()
                if packet is None:
                    if (
                        self.input_mode == "realtime"
                        and bool(getattr(source, "continuous", False))
                        and not self._stop_event.is_set()
                    ):
                        if bool(getattr(source, "stream_lost", False)):
                            raise RuntimeError(
                                "Camera stream was lost after repeated read failures. "
                                "The session was closed safely."
                            )
                        self._put_control(
                            {"type": "status", "message": "Reconnecting to camera stream…"}
                        )
                        continue
                    break
                saw_frame = True
                last_packet = packet
                if playback_wall_start is None:
                    playback_wall_start = monotonic()
                    playback_timestamp_start = packet.timestamp_seconds
                landmarks = pose.process(packet.frame, packet.timestamp_seconds)
                processed_frame_count += 1
                is_realtime = self.input_mode == "realtime"
                if is_realtime and countdown_start is None:
                    countdown_start = packet.timestamp_seconds
                elapsed_countdown = (
                    packet.timestamp_seconds - countdown_start
                    if is_realtime and countdown_start is not None
                    else self.exercise.preparation_seconds
                )
                preparing = is_realtime and elapsed_countdown < self.exercise.preparation_seconds
                if preparing:
                    remaining = max(
                        1,
                        int(ceil(self.exercise.preparation_seconds - elapsed_countdown)),
                    )
                    countdown = str(remaining)
                    result: FamilyFrameResult = runtime.update_landmarks(
                        landmarks,
                        packet.timestamp_seconds,
                        packet.frame_index,
                        counting_enabled=False,
                    )
                    status = (
                        "Please reposition until the required joints are visible."
                        if not pose_visible_for_family(
                            landmarks,
                            self.exercise.family_id,
                            self.camera_view,
                        )
                        else "Get ready"
                    )
                    session_time = 0.0
                else:
                    if analysis_start_timestamp is None:
                        analysis_start_timestamp = packet.timestamp_seconds
                        runtime.reset()
                    relative_timestamp = packet.timestamp_seconds - analysis_start_timestamp
                    result = runtime.update_landmarks(
                        landmarks,
                        relative_timestamp,
                        packet.frame_index,
                        counting_enabled=True,
                    )
                    if squat_shadow is not None:
                        try:
                            squat_shadow.record_frame(
                                landmarks,
                                packet.frame_index,
                                relative_timestamp,
                            )
                        except Exception:
                            squat_shadow = None
                    if squat_ai is not None:
                        try:
                            squat_ai.record_frame(
                                landmarks,
                                packet.frame_index,
                                relative_timestamp,
                            )
                            squat_ai.request_live_analysis()
                            live_ai_status = squat_ai.live_status()
                        except Exception:
                            squat_ai.close()
                            squat_ai = None
                            live_ai_status = {
                                "experimental": True,
                                "ai_detected_reps": 0,
                                "last_ai_rep": None,
                                "analyzing": False,
                                "analyzing_rep_index": None,
                                "boundary_available": False,
                                "correctness_available": False,
                                "error_available": False,
                            }
                    if pushup_ai is not None:
                        try:
                            pushup_ai.record_frame(landmarks, packet.frame_index, relative_timestamp)
                            pushup_ai.request_live_analysis()
                            live_ai_status = pushup_ai.live_status()
                        except Exception as exc:
                            pushup_ai.close()
                            pushup_ai = None
                            live_ai_status = unavailable_family_ai("pushup", len(aggregator.repetitions))
                            if live_ai_status is not None:
                                live_ai_status["reason"] = f"Correctness analysis unavailable: {type(exc).__name__}: {exc}"
                    if lunge_ai is not None:
                        try:
                            lunge_ai.record_frame(landmarks,packet.frame_index,relative_timestamp); lunge_ai.request_live_analysis(); live_ai_status=lunge_ai.live_status()
                        except Exception as exc:
                            lunge_ai.close(); lunge_ai=None; live_ai_status={"exercise":"lunge","available":False,"experimental":True,"reason":f"Boundary unavailable: {type(exc).__name__}: {exc}","performance_score":None,"assessment_coverage":None,"per_rep_results":[]}
                    countdown = (
                        "Start"
                        if is_realtime and relative_timestamp < 0.5
                        else None
                    )
                    status = PROCESSING
                    session_time = max(0.0, relative_timestamp)
                    aggregator.record_phase(result.phase)
                    if result.completed_cycle is not None:
                        cycle = result.completed_cycle
                        if squat_shadow is not None:
                            squat_shadow.record_rule_cycle(cycle)
                        aggregator.add_repetition(
                            RepetitionResult(
                                repetition_index=result.repetition_count,
                                start_frame=cycle.start_frame,
                                end_frame=cycle.end_frame,
                                start_time=cycle.start_time,
                                end_time=cycle.end_time,
                                duration_seconds=cycle.duration_seconds,
                                phase_sequence=cycle.phase_sequence,
                                score=None,
                                pass_fail=None,
                                errors=[],
                                confidence=cycle.confidence,
                                assessment_source="rule_based_repetition_counting",
                            )
                        )
                        if len(aggregator.repetitions) == 1 and audio_feedback is not None:
                            try:
                                audio_feedback.submit(
                                    FeedbackEvent.first_repetition_completed(
                                        session_id=aggregator.session_id,
                                        exercise_id=self.exercise.exercise_id,
                                        family_id=self.exercise.family_id,
                                        timestamp_seconds=cycle.end_time,
                                    )
                                )
                            except Exception:
                                # Speech failure is fail-open and cannot mutate
                                # official results or the learned-AI contract.
                                try:
                                    audio_feedback.shutdown()
                                except Exception:
                                    pass
                                audio_feedback = None
                        if self.exercise.family_id in {"pullup", "plank"}:
                            live_ai_status = rule_based_parity_status(
                                self.exercise.family_id,
                                len(aggregator.repetitions),
                                [item.to_dict() for item in aggregator.repetitions],
                                current_phase=result.phase,
                            )
                aggregator.set_incomplete_cycles(runtime.incomplete_cycles)
                if self.exercise.family_id == "plank":
                    aggregator.set_hold_duration(getattr(result, "hold_time_seconds", 0.0))
                annotated = draw_pose_skeleton(packet.frame, landmarks)
                if not self._pace_uploaded_video(
                    packet,
                    playback_wall_start or monotonic(),
                    playback_timestamp_start or 0.0,
                ):
                    break
                self._put_frame(
                    {
                        "type": "frame",
                        "frame": annotated,
                        "frame_index": packet.frame_index,
                        "timestamp_seconds": packet.timestamp_seconds,
                        "metrics": {
                            "exercise": self.exercise.display_name,
                            "family_id": self.exercise.family_id,
                            "camera_view": self.camera_view,
                            "repetitions": result.repetition_count,
                            "phase": result.phase,
                            "current_angle": result.elbow_angle,
                            "selected_side": result.selected_side,
                            "vertical_body_motion": getattr(result, "vertical_body_motion", None),
                            "confidence": result.signal_confidence,
                            "last_repetition_duration": result.last_repetition_duration,
                            "hold_time_seconds": getattr(result, "hold_time_seconds", None),
                            "session_time": session_time,
                            "status": status,
                            "countdown": countdown,
                            "frame_index": packet.frame_index,
                            "experimental_ai": live_ai_status,
                        },
                        "progress": packet.progress,
                    }
                )
            if not saw_frame:
                raise ValueError("The input source returned no decodable frames.")
            finish_runtime = getattr(runtime, "finish", None)
            if callable(finish_runtime):
                finish_runtime()
            if analysis_start_timestamp is None:
                duration = 0.0
            elif self.input_mode == "video" and last_packet is not None:
                duration = last_packet.timestamp_seconds + 1.0 / max(source.fps, 1.0)
            elif last_packet is not None:
                duration = max(0.0, last_packet.timestamp_seconds - analysis_start_timestamp)
            else:
                duration = 0.0
            aggregator.set_incomplete_cycles(runtime.incomplete_cycles)
            if squat_ai is not None:
                self._put_control(
                    {"type": "status", "message": "Finalizing experimental Squat AI analysis"}
                )
                try:
                    ai_result, _ = squat_ai.finalize_and_write(
                        session_id=aggregator.session_id,
                        rule_based_count=len(aggregator.repetitions),
                        camera_view=self.camera_view,
                        input_mode=self.input_mode,
                        duration_seconds=duration,
                        cancelled=self._stop_event.is_set(),
                    )
                    aggregator.set_experimental_ai(ai_result)
                except Exception as exc:
                    aggregator.set_experimental_ai(
                        {
                            "experimental": True,
                            "status": "Experimental",
                            "available": False,
                            "boundary_available": False,
                            "correctness_available": False,
                            "error_available": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "ai_detected_reps": 0,
                            "ai_correct_reps": 0,
                            "ai_incorrect_reps": 0,
                            "ai_pass_rate": None,
                            "error_counts": {"bad_back": 0, "bad_heel": 0, "form_issue": 0},
                            "per_rep_results": [],
                            "score": None,
                        }
                    )
            elif pushup_ai is not None:
                self._put_control({"type": "status", "message": "Finalizing experimental Push-up AI analysis"})
                try:
                    ai_result = pushup_ai.finalize()
                    ai_result["official_count"] = len(aggregator.repetitions)
                    aggregator.set_experimental_ai(ai_result)
                except Exception as exc:
                    status = unavailable_family_ai("pushup", len(aggregator.repetitions))
                    if status is not None:
                        status["reason"] = f"Push-up AI finalization unavailable: {type(exc).__name__}: {exc}"
                    aggregator.set_experimental_ai(status)
            elif lunge_ai is not None:
                try:
                    ai_result=lunge_ai.finalize(); ai_result["official_count"]=len(aggregator.repetitions); aggregator.set_experimental_ai(ai_result)
                except Exception as exc:
                    aggregator.set_experimental_ai({"exercise":"lunge","available":False,"experimental":True,"reason":f"Boundary finalization unavailable: {type(exc).__name__}: {exc}","performance_score":None,"assessment_coverage":None,"per_rep_results":[]})
            elif self.exercise.family_id in {"pullup", "plank"}:
                aggregator.set_experimental_ai(
                    rule_based_parity_status(
                        self.exercise.family_id,
                        len(aggregator.repetitions),
                        [item.to_dict() for item in aggregator.repetitions],
                        current_phase=getattr(runtime, "phase", None),
                    )
                )
            elif self.exercise.family_id != "squat":
                aggregator.set_experimental_ai(
                    unavailable_family_ai(
                        self.exercise.family_id,
                        len(aggregator.repetitions),
                    )
                )
            session_result = aggregator.finalize(duration)
            source_stats = {
                "processed_frames": processed_frame_count,
                "captured_frames": getattr(source, "captured_frame_count", processed_frame_count),
                "delivered_frames": getattr(source, "delivered_frame_count", processed_frame_count),
                "dropped_frames": getattr(source, "dropped_frame_count", 0),
                "backend": getattr(source, "backend_name", None),
                "camera_index": getattr(source, "camera_index", None),
            }
            self._put_control(
                {
                    "type": "complete",
                    "cancelled": self._stop_event.is_set(),
                    "result": session_result.to_dict(),
                    "worker_stats": source_stats,
                }
            )
            # The complete event is queued first, so the dashboard is not
            # blocked by the observational Boundary V2 pass. This still runs
            # on the existing background worker and never mutates the result.
            if squat_shadow is not None:
                try:
                    squat_shadow.finalize_and_write(
                        session_id=session_result.session_id,
                        rule_based_total_reps=len(session_result.repetitions),
                        camera_view=self.camera_view,
                        input_mode=self.input_mode,
                        duration_seconds=duration,
                        video_path=(
                            None if self.video_path is None else str(self.video_path)
                        ),
                        cancelled=self._stop_event.is_set(),
                    )
                except Exception:
                    # Shadow failures remain isolated from production results.
                    pass
        except Exception as exc:
            self._put_control(
                {"type": "error", "message": str(exc), "exception": type(exc).__name__}
            )
        finally:
            if pose is not None:
                pose.close()
            if source is not None:
                source.close()
            if squat_ai is not None:
                squat_ai.close()
            if pushup_ai is not None:
                pushup_ai.close()
            if lunge_ai is not None:
                lunge_ai.close()
            if audio_feedback is not None:
                audio_feedback.shutdown()
            with self._resource_lock:
                self._pose = None
                self._source = None
