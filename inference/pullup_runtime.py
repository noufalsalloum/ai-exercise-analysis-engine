from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


@dataclass(frozen=True)
class PullupRepConfig:
    """Independent thresholds for front/side-view Pull-up repetition logic."""

    min_landmark_confidence: float = 0.55
    angle_smoothing_window: int = 5
    vertical_smoothing_window: int = 5
    side_confidence_window: int = 12
    side_switch_margin: float = 0.15
    side_switch_confirm_frames: int = 5
    hang_ready_angle: float = 150.0
    ascent_enter_angle: float = 140.0
    top_enter_angle: float = 95.0
    top_exit_angle: float = 112.0
    hang_return_angle: float = 150.0
    ascent_start_displacement: float = 0.015
    minimum_vertical_displacement: float = 0.055
    return_vertical_tolerance: float = 0.080
    descent_displacement_hysteresis: float = 0.015
    minimum_elbow_range: float = 55.0
    minimum_repetition_duration: float = 0.60
    maximum_repetition_duration: float = 10.0
    cooldown_seconds: float = 0.30
    wrist_above_shoulder_tolerance: float = 0.08
    max_low_confidence_gap_frames: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PullupRepConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Pull-up counter settings: {unknown}")
        config = cls(**payload)
        if not 0.0 <= config.min_landmark_confidence <= 1.0:
            raise ValueError("min_landmark_confidence must be within [0, 1].")
        if not (
            config.top_enter_angle
            < config.top_exit_angle
            < config.hang_return_angle
        ):
            raise ValueError("Pull-up elbow hysteresis thresholds are not ordered.")
        if config.minimum_vertical_displacement <= config.ascent_start_displacement:
            raise ValueError(
                "minimum_vertical_displacement must exceed ascent_start_displacement."
            )
        return config


@dataclass
class CompletedPullupCycle:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    phase_sequence: list[str]
    confidence: float


@dataclass
class PullupFrameResult:
    phase: str
    repetition_count: int
    elbow_angle: float | None
    selected_side: str | None
    signal_confidence: float | None
    last_repetition_duration: float | None
    current_cycle_valid: bool
    vertical_body_motion: float | None
    completed_cycle: CompletedPullupCycle | None = None
    valid_frame: bool = False


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    first = np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    second = np.asarray(c, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return None
    value = float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(first, second) / denominator), -1.0, 1.0))
        )
    )
    return value if np.isfinite(value) else None


class PullupRepetitionRuntime:
    """Stateful Pull-up phases driven by elbow flexion and body elevation.

    The detector does not claim that the chin crossed a bar because the bar is
    not detected. ``TOP`` means sufficient elbow flexion plus shoulder/torso
    elevation relative to the established hang baseline.
    """

    ARM_NAMES = {
        "left": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP"),
        "right": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HIP"),
    }

    def __init__(
        self,
        config: PullupRepConfig | None = None,
        camera_view: str = "front",
    ) -> None:
        self.config = config or PullupRepConfig()
        if camera_view not in {"front", "side"}:
            raise ValueError("Pull-up camera_view must be 'front' or 'side'.")
        self.camera_view = camera_view
        self.reset()

    def reset(self) -> None:
        self.phase = "READY"
        self.repetition_count = 0
        self.selected_side: str | None = None
        self.last_angle: float | None = None
        self.last_vertical_body_motion: float | None = None
        self.last_repetition_duration: float | None = None
        self.incomplete_cycles = 0
        self.low_confidence_gap = 0
        self._angle_history: deque[float] = deque(maxlen=self.config.angle_smoothing_window)
        self._body_history: deque[float] = deque(maxlen=self.config.vertical_smoothing_window)
        self._side_confidence = {
            "left": deque(maxlen=self.config.side_confidence_window),
            "right": deque(maxlen=self.config.side_confidence_window),
        }
        self._switch_candidate_frames = 0
        self._hang_baseline: float | None = None
        self._cycle_start_time: float | None = None
        self._cycle_start_frame: int | None = None
        self._cycle_max_angle: float | None = None
        self._cycle_min_angle: float | None = None
        self._cycle_max_elevation = 0.0
        self._cycle_confidences: list[float] = []
        self._phase_sequence: list[str] = []
        self._top_seen = False
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
        self._cycle_max_angle = None
        self._cycle_min_angle = None
        self._cycle_max_elevation = 0.0
        self._cycle_confidences = []
        self._phase_sequence = []
        self._top_seen = False

    def _invalidate_cycle(self, next_phase: str = "READY") -> None:
        if self._cycle_start_time is not None:
            self.incomplete_cycles += 1
        self._reset_cycle()
        self.phase = next_phase

    def finish(self) -> None:
        """Close an unfinished ascent/descent when a stream ends or is stopped."""

        if self._cycle_start_time is not None:
            self._invalidate_cycle("HANG")

    def _append_phase(self, phase: str) -> None:
        self.phase = phase
        if not self._phase_sequence or self._phase_sequence[-1] != phase:
            self._phase_sequence.append(phase)

    def _side_confidence_value(self, landmarks: np.ndarray, side: str) -> float:
        indices = [MEDIAPIPE_LANDMARKS[name] for name in self.ARM_NAMES[side]]
        values = landmarks[indices, 3]
        return float(np.clip(np.min(values), 0.0, 1.0)) if np.isfinite(values).all() else 0.0

    def _select_landmark_sides(self, landmarks: np.ndarray) -> tuple[list[str], float]:
        current = {
            side: self._side_confidence_value(landmarks, side)
            for side in ("left", "right")
        }
        for side, value in current.items():
            self._side_confidence[side].append(value)
        if self.camera_view == "front" and all(
            value >= self.config.min_landmark_confidence for value in current.values()
        ):
            self.selected_side = "bilateral"
            return ["left", "right"], min(current.values())

        rolling = {
            side: float(np.mean(values)) if values else 0.0
            for side, values in self._side_confidence.items()
        }
        if self.selected_side not in {"left", "right"}:
            candidate = max(rolling, key=rolling.get)  # type: ignore[arg-type]
            self.selected_side = (
                candidate
                if rolling[candidate] >= self.config.min_landmark_confidence
                else None
            )
        elif self.selected_side is not None:
            other = "right" if self.selected_side == "left" else "left"
            if rolling[other] >= rolling[self.selected_side] + self.config.side_switch_margin:
                self._switch_candidate_frames += 1
                if self._switch_candidate_frames >= self.config.side_switch_confirm_frames:
                    if self.current_cycle_valid:
                        self._invalidate_cycle("READY")
                    self.selected_side = other
                    self._switch_candidate_frames = 0
                    self._angle_history.clear()
                    self._body_history.clear()
            else:
                self._switch_candidate_frames = 0
        if self.selected_side is None:
            return [], 0.0
        return [self.selected_side], current[self.selected_side]

    def update_landmarks(
        self,
        landmarks_33: np.ndarray | None,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> PullupFrameResult:
        if landmarks_33 is None:
            return self.update_signals(
                None, None, False, 0.0, timestamp, frame_index, counting_enabled
            )
        landmarks = np.asarray(landmarks_33, dtype=np.float32)
        if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
            return self.update_signals(
                None, None, False, 0.0, timestamp, frame_index, counting_enabled
            )
        sides, confidence = self._select_landmark_sides(landmarks)
        if not sides:
            return self.update_signals(
                None, None, False, confidence, timestamp, frame_index, counting_enabled
            )

        angles: list[float] = []
        shoulders: list[float] = []
        wrists: list[float] = []
        hips: list[float] = []
        for side in sides:
            shoulder_name, elbow_name, wrist_name, hip_name = self.ARM_NAMES[side]
            shoulder = landmarks[MEDIAPIPE_LANDMARKS[shoulder_name]]
            elbow = landmarks[MEDIAPIPE_LANDMARKS[elbow_name]]
            wrist = landmarks[MEDIAPIPE_LANDMARKS[wrist_name]]
            hip = landmarks[MEDIAPIPE_LANDMARKS[hip_name]]
            angle = _joint_angle(shoulder, elbow, wrist)
            if angle is not None:
                angles.append(angle)
            shoulders.append(float(shoulder[1]))
            wrists.append(float(wrist[1]))
            hips.append(float(hip[1]))
        elbow_angle = float(np.median(angles)) if angles else None
        shoulder_y = float(np.mean(shoulders))
        torso_center_y = float(np.mean([shoulder_y, float(np.mean(hips))]))
        wrist_y = float(np.mean(wrists))
        wrists_above_shoulders = (
            wrist_y <= shoulder_y + self.config.wrist_above_shoulder_tolerance
        )
        return self.update_signals(
            elbow_angle,
            torso_center_y,
            wrists_above_shoulders,
            confidence,
            timestamp,
            frame_index,
            counting_enabled,
        )

    def update_signals(
        self,
        elbow_angle: float | None,
        torso_center_y: float | None,
        wrists_above_shoulders: bool,
        signal_confidence: float,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> PullupFrameResult:
        """Update independent elbow/body-elevation signals for one frame."""

        confidence = float(np.clip(signal_confidence, 0.0, 1.0))
        valid = (
            elbow_angle is not None
            and torso_center_y is not None
            and np.isfinite(elbow_angle)
            and np.isfinite(torso_center_y)
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
                self._body_history.clear()
            return self._result(confidence=confidence, valid=False)

        self.low_confidence_gap = 0
        self._angle_history.append(float(elbow_angle))
        self._body_history.append(float(torso_center_y))
        angle = float(np.median(np.asarray(self._angle_history, dtype=np.float64)))
        body_y = float(np.median(np.asarray(self._body_history, dtype=np.float64)))
        self.last_angle = angle

        if self._hang_baseline is None:
            elevation = 0.0
        else:
            elevation = float(self._hang_baseline - body_y)
        self.last_vertical_body_motion = elevation
        completed: CompletedPullupCycle | None = None

        if self._cycle_start_time is not None:
            self._cycle_min_angle = min(self._cycle_min_angle or angle, angle)
            self._cycle_max_angle = max(self._cycle_max_angle or angle, angle)
            self._cycle_max_elevation = max(self._cycle_max_elevation, elevation)
            self._cycle_confidences.append(confidence)
            if timestamp - self._cycle_start_time > self.config.maximum_repetition_duration:
                self._invalidate_cycle("READY")

        hang_ready = angle >= self.config.hang_ready_angle and wrists_above_shoulders
        if self.phase in {"READY", "UNKNOWN"}:
            if hang_ready:
                self._hang_baseline = body_y
                self.phase = "HANG"
        elif self.phase == "HANG":
            if hang_ready and self._cycle_start_time is None:
                self._hang_baseline = 0.9 * float(self._hang_baseline or body_y) + 0.1 * body_y
                elevation = float(self._hang_baseline - body_y)
                self.last_vertical_body_motion = elevation
            if (
                angle <= self.config.ascent_enter_angle
                and elevation >= self.config.ascent_start_displacement
                and timestamp - self._last_complete_time >= self.config.cooldown_seconds
            ):
                self._cycle_start_time = float(timestamp)
                self._cycle_start_frame = int(frame_index)
                self._cycle_max_angle = angle
                self._cycle_min_angle = angle
                self._cycle_max_elevation = elevation
                self._cycle_confidences = [confidence]
                self._phase_sequence = ["HANG", "ASCENDING"]
                self._top_seen = False
                self.phase = "ASCENDING"
        elif self.phase == "ASCENDING":
            if (
                angle <= self.config.top_enter_angle
                and elevation >= self.config.minimum_vertical_displacement
            ):
                self._top_seen = True
                self._append_phase("TOP")
            elif hang_ready and elevation <= self.config.return_vertical_tolerance:
                self._invalidate_cycle("HANG")
        elif self.phase == "TOP":
            if (
                angle >= self.config.top_exit_angle
                or elevation
                <= self._cycle_max_elevation - self.config.descent_displacement_hysteresis
            ):
                self._append_phase("DESCENDING")
        elif self.phase == "DESCENDING":
            if (
                angle >= self.config.hang_return_angle
                and elevation <= self.config.return_vertical_tolerance
                and wrists_above_shoulders
            ):
                start_time = self._cycle_start_time
                start_frame = self._cycle_start_frame
                duration = float(timestamp - start_time) if start_time is not None else -1.0
                elbow_range = float(
                    (self._cycle_max_angle or angle) - (self._cycle_min_angle or angle)
                )
                valid_cycle = (
                    start_time is not None
                    and start_frame is not None
                    and self._top_seen
                    and self.config.minimum_repetition_duration
                    <= duration
                    <= self.config.maximum_repetition_duration
                    and elbow_range >= self.config.minimum_elbow_range
                    and self._cycle_max_elevation
                    >= self.config.minimum_vertical_displacement
                    and self._phase_sequence[:4]
                    == ["HANG", "ASCENDING", "TOP", "DESCENDING"]
                )
                if valid_cycle:
                    self.repetition_count += 1
                    self.last_repetition_duration = duration
                    completed = CompletedPullupCycle(
                        start_frame=start_frame,
                        end_frame=int(frame_index),
                        start_time=start_time,
                        end_time=float(timestamp),
                        duration_seconds=duration,
                        phase_sequence=[*self._phase_sequence, "HANG"],
                        confidence=float(np.mean(self._cycle_confidences)),
                    )
                    self._last_complete_time = float(timestamp)
                    self._reset_cycle()
                    self.phase = "HANG"
                    self._hang_baseline = body_y
                else:
                    self._invalidate_cycle("HANG")

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
        completed: CompletedPullupCycle | None = None,
        valid: bool = False,
    ) -> PullupFrameResult:
        return PullupFrameResult(
            phase=phase or self.phase,
            repetition_count=self.repetition_count,
            elbow_angle=self.last_angle if angle is None else angle,
            selected_side=self.selected_side,
            signal_confidence=confidence,
            last_repetition_duration=self.last_repetition_duration,
            current_cycle_valid=self.current_cycle_valid,
            vertical_body_motion=self.last_vertical_body_motion,
            completed_cycle=completed,
            valid_frame=valid,
        )
