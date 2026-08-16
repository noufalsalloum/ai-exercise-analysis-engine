from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


@dataclass(frozen=True)
class PushupRepConfig:
    """Configurable thresholds for the standard side-view Push-up MVP."""

    min_landmark_confidence: float = 0.55
    side_confidence_window: int = 12
    side_switch_margin: float = 0.15
    side_switch_confirm_frames: int = 5
    angle_smoothing_window: int = 5
    top_enter_angle: float = 155.0
    descent_enter_angle: float = 145.0
    bottom_enter_angle: float = 115.0
    bottom_exit_angle: float = 125.0
    min_range_of_motion: float = 45.0
    min_repetition_duration: float = 0.35
    max_repetition_duration: float = 8.0
    cooldown_seconds: float = 0.25
    max_low_confidence_gap_frames: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PushupRepConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Push-up counter settings: {unknown}")
        config = cls(**payload)
        if not 0.0 <= config.min_landmark_confidence <= 1.0:
            raise ValueError("min_landmark_confidence must be within [0, 1].")
        if config.bottom_enter_angle >= config.descent_enter_angle:
            raise ValueError("bottom_enter_angle must be below descent_enter_angle.")
        if config.descent_enter_angle >= config.top_enter_angle:
            raise ValueError("descent_enter_angle must be below top_enter_angle.")
        return config


@dataclass
class CompletedPushupCycle:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    phase_sequence: list[str]
    confidence: float


@dataclass
class PushupFrameResult:
    phase: str
    repetition_count: int
    elbow_angle: float | None
    selected_side: str | None
    signal_confidence: float | None
    last_repetition_duration: float | None
    current_cycle_valid: bool
    completed_cycle: CompletedPushupCycle | None = None
    valid_frame: bool = False


def calculate_joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Calculate the finite 2-D angle ABC in degrees."""

    first = np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    second = np.asarray(c, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return None
    cosine = float(np.dot(first, second) / denominator)
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return angle if np.isfinite(angle) else None


class PushupRepetitionRuntime:
    """Stateful side-view phase detector and standard Push-up counter.

    Confidence is used only to reject unreliable landmarks. State transitions
    are driven by the smoothed elbow-angle trajectory and required phase order.
    """

    SIDE_JOINTS = {
        "left": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "right": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    }

    def __init__(self, config: PushupRepConfig | None = None) -> None:
        self.config = config or PushupRepConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = "READY"
        self.repetition_count = 0
        self.selected_side: str | None = None
        self.last_angle: float | None = None
        self.last_repetition_duration: float | None = None
        self.incomplete_cycles = 0
        self.low_confidence_gap = 0
        self._angle_history: deque[float] = deque(maxlen=self.config.angle_smoothing_window)
        self._side_confidence = {
            "left": deque(maxlen=self.config.side_confidence_window),
            "right": deque(maxlen=self.config.side_confidence_window),
        }
        self._switch_candidate_frames = 0
        self._cycle_start_time: float | None = None
        self._cycle_start_frame: int | None = None
        self._cycle_min_angle: float | None = None
        self._cycle_max_angle: float | None = None
        self._cycle_confidences: list[float] = []
        self._phase_sequence: list[str] = []
        self._bottom_seen = False
        self._last_complete_time = float("-inf")

    @property
    def current_cycle_valid(self) -> bool:
        return self._cycle_start_time is not None and self.low_confidence_gap <= self.config.max_low_confidence_gap_frames

    def _append_phase(self, phase: str) -> None:
        self.phase = phase
        if not self._phase_sequence or self._phase_sequence[-1] != phase:
            self._phase_sequence.append(phase)

    def _reset_cycle(self) -> None:
        self._cycle_start_time = None
        self._cycle_start_frame = None
        self._cycle_min_angle = None
        self._cycle_max_angle = None
        self._cycle_confidences = []
        self._phase_sequence = []
        self._bottom_seen = False

    def _invalidate_cycle(self, next_phase: str = "READY") -> None:
        if self._cycle_start_time is not None:
            self.incomplete_cycles += 1
        self._reset_cycle()
        self.phase = next_phase

    def _side_confidence_value(self, landmarks: np.ndarray, side: str) -> float:
        indices = [MEDIAPIPE_LANDMARKS[name] for name in self.SIDE_JOINTS[side]]
        values = np.asarray(landmarks[indices, 3], dtype=np.float64)
        if not np.isfinite(values).all():
            return 0.0
        return float(np.clip(values.min(), 0.0, 1.0))

    def _select_side(self, landmarks: np.ndarray) -> tuple[str | None, float | None]:
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
                    self._angle_history.clear()
            else:
                self._switch_candidate_frames = 0
        if self.selected_side is None:
            return None, None
        return self.selected_side, current[self.selected_side]

    def update_landmarks(
        self,
        landmarks_33: np.ndarray | None,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> PushupFrameResult:
        """Update from one MediaPipe ``(33,4)`` frame."""

        if landmarks_33 is None:
            return self.update_signal(None, 0.0, timestamp, frame_index, counting_enabled)
        landmarks = np.asarray(landmarks_33, dtype=np.float32)
        if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
            return self.update_signal(None, 0.0, timestamp, frame_index, counting_enabled)
        side, confidence = self._select_side(landmarks)
        angle: float | None = None
        if side is not None:
            names = self.SIDE_JOINTS[side]
            points = [landmarks[MEDIAPIPE_LANDMARKS[name], :2] for name in names]
            angle = calculate_joint_angle(points[0], points[1], points[2])
        return self.update_signal(
            angle,
            confidence or 0.0,
            timestamp,
            frame_index,
            counting_enabled,
        )

    def update_signal(
        self,
        elbow_angle: float | None,
        signal_confidence: float,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> PushupFrameResult:
        """Update from an elbow signal; useful for deterministic runtime tests."""

        confidence = float(np.clip(signal_confidence, 0.0, 1.0))
        valid = (
            elbow_angle is not None
            and np.isfinite(elbow_angle)
            and confidence >= self.config.min_landmark_confidence
        )
        if not counting_enabled:
            return self._result(
                phase="PREPARING",
                angle=float(elbow_angle) if elbow_angle is not None else None,
                confidence=confidence,
                valid=valid,
            )
        if not valid:
            self.low_confidence_gap += 1
            if self.low_confidence_gap > self.config.max_low_confidence_gap_frames:
                self._invalidate_cycle("UNKNOWN")
                self._angle_history.clear()
            return self._result(confidence=confidence, valid=False)

        self.low_confidence_gap = 0
        self._angle_history.append(float(elbow_angle))
        angle = float(np.median(np.asarray(self._angle_history, dtype=np.float64)))
        self.last_angle = angle
        completed: CompletedPushupCycle | None = None

        if self._cycle_start_time is not None:
            self._cycle_min_angle = min(self._cycle_min_angle or angle, angle)
            self._cycle_max_angle = max(self._cycle_max_angle or angle, angle)
            self._cycle_confidences.append(confidence)
            if timestamp - self._cycle_start_time > self.config.max_repetition_duration:
                self._invalidate_cycle("READY")

        if self.phase in {"READY", "UNKNOWN"}:
            if angle >= self.config.top_enter_angle:
                self.phase = "TOP"
        elif self.phase == "TOP":
            if angle <= self.config.descent_enter_angle and timestamp - self._last_complete_time >= self.config.cooldown_seconds:
                self._cycle_start_time = float(timestamp)
                self._cycle_start_frame = int(frame_index)
                self._cycle_min_angle = angle
                self._cycle_max_angle = max(angle, self.last_angle or angle)
                self._cycle_confidences = [confidence]
                self._phase_sequence = ["TOP", "DESCENDING"]
                self._bottom_seen = False
                self.phase = "DESCENDING"
        elif self.phase == "DESCENDING":
            if angle <= self.config.bottom_enter_angle:
                self._bottom_seen = True
                self._append_phase("BOTTOM")
            elif angle >= self.config.top_enter_angle:
                self._invalidate_cycle("TOP")
        elif self.phase == "BOTTOM":
            if angle >= self.config.bottom_exit_angle:
                self._append_phase("ASCENDING")
        elif self.phase == "ASCENDING":
            if angle <= self.config.bottom_enter_angle:
                self._append_phase("BOTTOM")
            elif angle >= self.config.top_enter_angle:
                start_time = self._cycle_start_time
                start_frame = self._cycle_start_frame
                duration = float(timestamp - start_time) if start_time is not None else -1.0
                motion_range = (
                    float((self._cycle_max_angle or angle) - (self._cycle_min_angle or angle))
                )
                valid_cycle = (
                    start_time is not None
                    and start_frame is not None
                    and self._bottom_seen
                    and self.config.min_repetition_duration <= duration <= self.config.max_repetition_duration
                    and motion_range >= self.config.min_range_of_motion
                    and self._phase_sequence[:4] == ["TOP", "DESCENDING", "BOTTOM", "ASCENDING"]
                )
                if valid_cycle:
                    self.repetition_count += 1
                    self.last_repetition_duration = duration
                    completed = CompletedPushupCycle(
                        start_frame=start_frame,
                        end_frame=int(frame_index),
                        start_time=start_time,
                        end_time=float(timestamp),
                        duration_seconds=duration,
                        phase_sequence=[*self._phase_sequence, "TOP"],
                        confidence=float(np.mean(self._cycle_confidences)),
                    )
                    self._last_complete_time = float(timestamp)
                    self._reset_cycle()
                    self.phase = "TOP"
                else:
                    self._invalidate_cycle("TOP")

        return self._result(
            angle=angle,
            confidence=confidence,
            completed=completed,
            valid=True,
        )

    def _result(
        self,
        phase: str | None = None,
        angle: float | None = None,
        confidence: float | None = None,
        completed: CompletedPushupCycle | None = None,
        valid: bool = False,
    ) -> PushupFrameResult:
        return PushupFrameResult(
            phase=phase or self.phase,
            repetition_count=self.repetition_count,
            elbow_angle=self.last_angle if angle is None else angle,
            selected_side=self.selected_side,
            signal_confidence=confidence,
            last_repetition_duration=self.last_repetition_duration,
            current_cycle_valid=self.current_cycle_valid,
            completed_cycle=completed,
            valid_frame=valid,
        )
