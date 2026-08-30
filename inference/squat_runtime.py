from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


@dataclass(frozen=True)
class SquatRepConfig:
    """Independent thresholds for side-view Air Squat counting."""

    min_landmark_confidence: float = 0.55
    angle_smoothing_window: int = 5
    pelvis_smoothing_window: int = 5
    side_confidence_window: int = 12
    side_switch_margin: float = 0.15
    side_switch_confirm_frames: int = 5
    standing_ready_angle: float = 155.0
    descent_enter_angle: float = 145.0
    bottom_enter_angle: float = 110.0
    bottom_exit_angle: float = 122.0
    standing_return_angle: float = 155.0
    descent_start_displacement: float = 0.015
    # Lowered from 0.070 (2026-08-30 root-cause investigation): a moderate-
    # but-genuine squat depth (knee angle in the low 70s, well past
    # bottom_enter_angle=110) was observed producing normalized pelvis-Y
    # displacement of only ~0.0672-0.0687 on real video — short of the old
    # 0.070 floor by ~2-4% despite the movement being a complete, honest
    # squat cycle (confirmed by frame-by-frame visual inspection and
    # comparison against a known-good reference clip at matched knee angle).
    # 0.060 keeps a wide margin above both return_pelvis_tolerance (0.055,
    # so the "gave up before reaching depth" cancellation path stays
    # non-overlapping) and the deliberately-incomplete/ambiguous reference
    # clip's observed ceiling (0.0338, ~1.8x below this floor), while
    # comfortably admitting every real squat depth measured so far
    # (0.0672-0.1537 across 6 clips). See tests/test_squat_runtime.py's
    # "moderate depth" tests and the regression matrix in the AI-engine
    # investigation notes for the full evidence.
    minimum_pelvis_displacement: float = 0.060
    return_pelvis_tolerance: float = 0.055
    ascent_displacement_hysteresis: float = 0.015
    minimum_knee_rom: float = 45.0
    minimum_hip_rom: float = 25.0
    minimum_repetition_duration: float = 0.45
    maximum_repetition_duration: float = 8.0
    cooldown_seconds: float = 0.25
    max_low_confidence_gap_frames: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SquatRepConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Squat counter settings: {unknown}")
        config = cls(**payload)
        if not 0.0 <= config.min_landmark_confidence <= 1.0:
            raise ValueError("min_landmark_confidence must be within [0, 1].")
        if not (
            config.bottom_enter_angle
            < config.bottom_exit_angle
            < config.descent_enter_angle
            < config.standing_return_angle
        ):
            raise ValueError("Squat knee-angle hysteresis thresholds are not ordered.")
        if config.minimum_pelvis_displacement <= config.descent_start_displacement:
            raise ValueError("minimum_pelvis_displacement must exceed descent start.")
        if min(
            config.angle_smoothing_window,
            config.pelvis_smoothing_window,
            config.side_confidence_window,
            config.side_switch_confirm_frames,
        ) <= 0:
            raise ValueError("Squat smoothing and side-lock windows must be positive.")
        return config


@dataclass
class CompletedSquatCycle:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    phase_sequence: list[str]
    confidence: float
    selected_side: str | None
    knee_angle_range: float
    hip_angle_range: float
    pelvis_displacement: float

    @property
    def details(self) -> dict[str, Any]:
        return {
            "variation": "air_squat",
            "selected_side": self.selected_side,
            "knee_angle_range": self.knee_angle_range,
            "hip_angle_range": self.hip_angle_range,
            "pelvis_displacement": self.pelvis_displacement,
        }


@dataclass
class SquatFrameResult:
    phase: str
    repetition_count: int
    elbow_angle: float | None
    selected_side: str | None
    signal_confidence: float | None
    last_repetition_duration: float | None
    current_cycle_valid: bool
    vertical_body_motion: float | None
    hip_angle: float | None
    completed_cycle: CompletedSquatCycle | None = None
    valid_frame: bool = False

    @property
    def knee_angle(self) -> float | None:
        return self.elbow_angle

    @property
    def pelvis_vertical_motion(self) -> float | None:
        return self.vertical_body_motion


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    first = np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    second = np.asarray(c, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return None
    angle = float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(first, second) / denominator), -1.0, 1.0))
        )
    )
    return angle if np.isfinite(angle) else None


class SquatRepetitionRuntime:
    """Side-view Air Squat state machine with stable confidence-based side lock."""

    SIDE_NAMES = {
        "left": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "right": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    }

    def __init__(
        self,
        config: SquatRepConfig | None = None,
        camera_view: str = "side",
    ) -> None:
        if camera_view != "side":
            raise ValueError("The validated Squat runtime currently supports side view only.")
        self.config = config or SquatRepConfig()
        self.camera_view = camera_view
        self.reset()

    def reset(self) -> None:
        self.phase = "READY"
        self.repetition_count = 0
        self.incomplete_cycles = 0
        self.selected_side: str | None = None
        self.last_angle: float | None = None
        self.last_hip_angle: float | None = None
        self.last_vertical_body_motion: float | None = None
        self.last_repetition_duration: float | None = None
        self.low_confidence_gap = 0
        self._knee_history: deque[float] = deque(maxlen=self.config.angle_smoothing_window)
        self._hip_history: deque[float] = deque(maxlen=self.config.angle_smoothing_window)
        self._pelvis_history: deque[float] = deque(maxlen=self.config.pelvis_smoothing_window)
        self._side_confidence = {
            side: deque(maxlen=self.config.side_confidence_window)
            for side in ("left", "right")
        }
        self._switch_candidate_frames = 0
        self._standing_baseline: float | None = None
        self._cycle_start_time: float | None = None
        self._cycle_start_frame: int | None = None
        self._cycle_min_knee: float | None = None
        self._cycle_max_knee: float | None = None
        self._cycle_min_hip: float | None = None
        self._cycle_max_hip: float | None = None
        self._cycle_max_pelvis = 0.0
        self._cycle_confidences: list[float] = []
        self._phase_sequence: list[str] = []
        self._bottom_seen = False
        self._last_complete_time = float("-inf")

    @property
    def current_cycle_valid(self) -> bool:
        return (
            self._cycle_start_time is not None
            and self.low_confidence_gap <= self.config.max_low_confidence_gap_frames
        )

    def _reset_cycle(self) -> None:
        self._cycle_start_time = None
        self._cycle_start_frame = None
        self._cycle_min_knee = None
        self._cycle_max_knee = None
        self._cycle_min_hip = None
        self._cycle_max_hip = None
        self._cycle_max_pelvis = 0.0
        self._cycle_confidences = []
        self._phase_sequence = []
        self._bottom_seen = False

    def _invalidate_cycle(self, next_phase: str = "READY") -> None:
        if self._cycle_start_time is not None:
            self.incomplete_cycles += 1
        self._reset_cycle()
        self.phase = next_phase

    def finish(self) -> None:
        if self._cycle_start_time is not None:
            self._invalidate_cycle("STANDING")

    def _append_phase(self, phase: str) -> None:
        self.phase = phase
        if not self._phase_sequence or self._phase_sequence[-1] != phase:
            self._phase_sequence.append(phase)

    def _side_confidence_value(self, landmarks: np.ndarray, side: str) -> float:
        values = landmarks[
            [MEDIAPIPE_LANDMARKS[name] for name in self.SIDE_NAMES[side]], 3
        ]
        return float(np.clip(np.min(values), 0.0, 1.0)) if np.isfinite(values).all() else 0.0

    def _select_side(self, landmarks: np.ndarray) -> tuple[str | None, float]:
        current = {
            side: self._side_confidence_value(landmarks, side)
            for side in ("left", "right")
        }
        for side, value in current.items():
            self._side_confidence[side].append(value)
        rolling = {
            side: float(np.mean(values)) if values else 0.0
            for side, values in self._side_confidence.items()
        }
        if self.selected_side is None:
            candidate = max(rolling, key=rolling.get)  # type: ignore[arg-type]
            if rolling[candidate] >= self.config.min_landmark_confidence:
                self.selected_side = candidate
        else:
            other = "right" if self.selected_side == "left" else "left"
            if rolling[other] >= rolling[self.selected_side] + self.config.side_switch_margin:
                self._switch_candidate_frames += 1
                if self._switch_candidate_frames >= self.config.side_switch_confirm_frames:
                    if self.current_cycle_valid:
                        self._invalidate_cycle("READY")
                    self.selected_side = other
                    self._switch_candidate_frames = 0
                    self._knee_history.clear()
                    self._hip_history.clear()
                    self._pelvis_history.clear()
                    self._standing_baseline = None
            else:
                self._switch_candidate_frames = 0
        if self.selected_side is None:
            return None, 0.0
        return self.selected_side, current[self.selected_side]

    def update_landmarks(
        self,
        landmarks_33: np.ndarray | None,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> SquatFrameResult:
        if landmarks_33 is None:
            return self.update_signals(None, None, None, 0.0, timestamp, frame_index, counting_enabled)
        landmarks = np.asarray(landmarks_33, dtype=np.float32)
        if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
            return self.update_signals(None, None, None, 0.0, timestamp, frame_index, counting_enabled)
        side, confidence = self._select_side(landmarks)
        if side is None:
            return self.update_signals(None, None, None, confidence, timestamp, frame_index, counting_enabled)
        shoulder_name, hip_name, knee_name, ankle_name = self.SIDE_NAMES[side]
        shoulder = landmarks[MEDIAPIPE_LANDMARKS[shoulder_name]]
        hip = landmarks[MEDIAPIPE_LANDMARKS[hip_name]]
        knee = landmarks[MEDIAPIPE_LANDMARKS[knee_name]]
        ankle = landmarks[MEDIAPIPE_LANDMARKS[ankle_name]]
        return self.update_signals(
            _joint_angle(hip, knee, ankle),
            _joint_angle(shoulder, hip, knee),
            float(hip[1]),
            confidence,
            timestamp,
            frame_index,
            counting_enabled,
        )

    def update_signals(
        self,
        knee_angle: float | None,
        hip_angle: float | None,
        pelvis_y: float | None,
        signal_confidence: float,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> SquatFrameResult:
        confidence = float(np.clip(signal_confidence, 0.0, 1.0))
        valid = (
            knee_angle is not None
            and hip_angle is not None
            and pelvis_y is not None
            and np.isfinite(knee_angle)
            and np.isfinite(hip_angle)
            and np.isfinite(pelvis_y)
            and confidence >= self.config.min_landmark_confidence
        )
        if not counting_enabled:
            return self._result(
                phase="PREPARING",
                knee_angle=knee_angle,
                hip_angle=hip_angle,
                confidence=confidence,
                valid=valid,
            )
        if not valid:
            self.low_confidence_gap += 1
            if self.low_confidence_gap > self.config.max_low_confidence_gap_frames:
                self._invalidate_cycle("UNKNOWN")
                self._knee_history.clear()
                self._hip_history.clear()
                self._pelvis_history.clear()
            return self._result(confidence=confidence, valid=False)

        self.low_confidence_gap = 0
        self._knee_history.append(float(knee_angle))
        self._hip_history.append(float(hip_angle))
        self._pelvis_history.append(float(pelvis_y))
        knee = float(np.median(self._knee_history))
        hip = float(np.median(self._hip_history))
        pelvis = float(np.median(self._pelvis_history))
        self.last_angle = knee
        self.last_hip_angle = hip
        displacement = 0.0 if self._standing_baseline is None else pelvis - self._standing_baseline
        self.last_vertical_body_motion = displacement
        completed: CompletedSquatCycle | None = None

        if self._cycle_start_time is not None:
            self._cycle_min_knee = min(self._cycle_min_knee or knee, knee)
            self._cycle_max_knee = max(self._cycle_max_knee or knee, knee)
            self._cycle_min_hip = min(self._cycle_min_hip or hip, hip)
            self._cycle_max_hip = max(self._cycle_max_hip or hip, hip)
            self._cycle_max_pelvis = max(self._cycle_max_pelvis, displacement)
            self._cycle_confidences.append(confidence)
            if timestamp - self._cycle_start_time > self.config.maximum_repetition_duration:
                self._invalidate_cycle("READY")

        standing = knee >= self.config.standing_ready_angle
        if self.phase in {"READY", "UNKNOWN"}:
            if standing:
                self._standing_baseline = pelvis
                self.phase = "STANDING"
        elif self.phase == "STANDING":
            if standing and self._cycle_start_time is None:
                self._standing_baseline = 0.9 * float(self._standing_baseline or pelvis) + 0.1 * pelvis
                displacement = pelvis - self._standing_baseline
                self.last_vertical_body_motion = displacement
            if (
                knee <= self.config.descent_enter_angle
                and displacement >= self.config.descent_start_displacement
                and timestamp - self._last_complete_time >= self.config.cooldown_seconds
            ):
                self._cycle_start_time = float(timestamp)
                self._cycle_start_frame = int(frame_index)
                self._cycle_min_knee = knee
                self._cycle_max_knee = knee
                self._cycle_min_hip = hip
                self._cycle_max_hip = hip
                self._cycle_max_pelvis = displacement
                self._cycle_confidences = [confidence]
                self._phase_sequence = ["STANDING", "DESCENDING"]
                self._bottom_seen = False
                self.phase = "DESCENDING"
        elif self.phase == "DESCENDING":
            if (
                knee <= self.config.bottom_enter_angle
                and displacement >= self.config.minimum_pelvis_displacement
            ):
                self._bottom_seen = True
                self._append_phase("BOTTOM")
            elif standing and displacement <= self.config.return_pelvis_tolerance:
                self._invalidate_cycle("STANDING")
        elif self.phase == "BOTTOM":
            if (
                knee >= self.config.bottom_exit_angle
                and displacement <= self._cycle_max_pelvis - self.config.ascent_displacement_hysteresis
            ):
                self._append_phase("ASCENDING")
        elif self.phase == "ASCENDING":
            if (
                knee >= self.config.standing_return_angle
                and displacement <= self.config.return_pelvis_tolerance
            ):
                start_time = self._cycle_start_time
                start_frame = self._cycle_start_frame
                duration = float(timestamp - start_time) if start_time is not None else -1.0
                knee_range = float((self._cycle_max_knee or knee) - (self._cycle_min_knee or knee))
                hip_range = float((self._cycle_max_hip or hip) - (self._cycle_min_hip or hip))
                valid_cycle = (
                    start_time is not None
                    and start_frame is not None
                    and self._bottom_seen
                    and self.config.minimum_repetition_duration <= duration <= self.config.maximum_repetition_duration
                    and knee_range >= self.config.minimum_knee_rom
                    and hip_range >= self.config.minimum_hip_rom
                    and self._cycle_max_pelvis >= self.config.minimum_pelvis_displacement
                    and self._phase_sequence == ["STANDING", "DESCENDING", "BOTTOM", "ASCENDING"]
                )
                if valid_cycle:
                    self.repetition_count += 1
                    self.last_repetition_duration = duration
                    completed = CompletedSquatCycle(
                        start_frame=start_frame,
                        end_frame=int(frame_index),
                        start_time=start_time,
                        end_time=float(timestamp),
                        duration_seconds=duration,
                        phase_sequence=[*self._phase_sequence, "STANDING"],
                        confidence=float(np.mean(self._cycle_confidences)),
                        selected_side=self.selected_side,
                        knee_angle_range=knee_range,
                        hip_angle_range=hip_range,
                        pelvis_displacement=self._cycle_max_pelvis,
                    )
                    self._last_complete_time = float(timestamp)
                    self._reset_cycle()
                    self.phase = "STANDING"
                    self._standing_baseline = pelvis
                else:
                    self._invalidate_cycle("STANDING")

        return self._result(knee_angle=knee, hip_angle=hip, confidence=confidence, completed=completed, valid=True)

    def _result(
        self,
        phase: str | None = None,
        knee_angle: float | None = None,
        hip_angle: float | None = None,
        confidence: float | None = None,
        completed: CompletedSquatCycle | None = None,
        valid: bool = False,
    ) -> SquatFrameResult:
        return SquatFrameResult(
            phase=phase or self.phase,
            repetition_count=self.repetition_count,
            elbow_angle=self.last_angle if knee_angle is None else float(knee_angle),
            selected_side=self.selected_side,
            signal_confidence=confidence,
            last_repetition_duration=self.last_repetition_duration,
            current_cycle_valid=self.current_cycle_valid,
            vertical_body_motion=self.last_vertical_body_motion,
            hip_angle=self.last_hip_angle if hip_angle is None else float(hip_angle),
            completed_cycle=completed,
            valid_frame=valid,
        )
