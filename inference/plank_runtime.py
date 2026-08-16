from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


@dataclass(frozen=True)
class MarchingPlankConfig:
    """Independent, uncalibrated Marching Plank counter settings."""

    min_landmark_confidence: float = 0.55
    signal_smoothing_window: int = 5
    neutral_baseline_window: int = 15
    lift_start_displacement: float = 0.025
    leg_up_displacement: float = 0.060
    return_tolerance: float = 0.025
    return_hysteresis: float = 0.015
    maximum_other_leg_displacement: float = 0.035
    maximum_pelvis_motion: float = 0.055
    minimum_repetition_duration: float = 0.35
    maximum_repetition_duration: float = 5.0
    cooldown_seconds: float = 0.20
    max_low_confidence_gap_frames: int = 4
    hold_unit_seconds: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarchingPlankConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Marching Plank settings: {unknown}")
        config = cls(**payload)
        if not 0.0 <= config.min_landmark_confidence <= 1.0:
            raise ValueError("min_landmark_confidence must be within [0, 1].")
        # The active project config intentionally uses the neutral-return
        # tolerance as the lift-entry boundary (both 0.025).  Equality is a
        # valid contiguous hysteresis contract and does not change thresholds.
        if not config.return_tolerance <= config.lift_start_displacement < config.leg_up_displacement:
            raise ValueError("Marching Plank displacement hysteresis is not ordered.")
        if min(config.signal_smoothing_window, config.neutral_baseline_window) <= 0:
            raise ValueError("Marching Plank smoothing windows must be positive.")
        if config.hold_unit_seconds <= 0.0:
            raise ValueError("hold_unit_seconds must be positive.")
        return config


@dataclass
class CompletedMarchingPlankCycle:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    phase_sequence: list[str]
    confidence: float
    active_leg: str
    maximum_leg_displacement: float
    maximum_pelvis_motion: float

    @property
    def details(self) -> dict[str, Any]:
        return {
            "variation": "marching_plank",
            "active_leg": self.active_leg,
            "maximum_leg_displacement": self.maximum_leg_displacement,
            "maximum_pelvis_motion": self.maximum_pelvis_motion,
        }


@dataclass
class MarchingPlankFrameResult:
    phase: str
    repetition_count: int
    elbow_angle: float | None
    selected_side: str | None
    signal_confidence: float | None
    last_repetition_duration: float | None
    current_cycle_valid: bool
    vertical_body_motion: float | None
    completed_cycle: CompletedMarchingPlankCycle | None = None
    valid_frame: bool = False
    active_leg: str | None = None
    left_leg_displacement: float | None = None
    right_leg_displacement: float | None = None
    pelvis_stability: float | None = None
    left_repetitions: int = 0
    right_repetitions: int = 0
    hold_time_seconds: float = 0.0


class MarchingPlankRepetitionRuntime:
    """Marching Plank leg-lift counter, awaiting positive-video calibration."""

    REQUIRED_NAMES = (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    )

    def __init__(
        self,
        config: MarchingPlankConfig | None = None,
        camera_view: str = "side",
    ) -> None:
        if camera_view not in {"side", "half_profile"}:
            raise ValueError("Marching Plank supports side or half_profile view only.")
        self.config = config or MarchingPlankConfig()
        self.camera_view = camera_view
        self.reset()

    def reset(self) -> None:
        self.phase = "READY"
        self.repetition_count = 0
        self.left_repetitions = 0
        self.right_repetitions = 0
        self.incomplete_cycles = 0
        self.active_leg: str | None = None
        self.selected_side: str | None = None
        self.last_angle: float | None = None
        self.last_repetition_duration: float | None = None
        self.last_vertical_body_motion: float | None = None
        self.low_confidence_gap = 0
        self._left_history: deque[float] = deque(maxlen=self.config.signal_smoothing_window)
        self._right_history: deque[float] = deque(maxlen=self.config.signal_smoothing_window)
        self._pelvis_history: deque[float] = deque(maxlen=self.config.signal_smoothing_window)
        self._neutral_left: deque[float] = deque(maxlen=self.config.neutral_baseline_window)
        self._neutral_right: deque[float] = deque(maxlen=self.config.neutral_baseline_window)
        self._neutral_pelvis: deque[float] = deque(maxlen=self.config.neutral_baseline_window)
        self._cycle_start_time: float | None = None
        self._cycle_start_frame: int | None = None
        self._cycle_max_displacement = 0.0
        self._cycle_max_pelvis = 0.0
        self._cycle_confidences: list[float] = []
        self._phase_sequence: list[str] = []
        self._leg_up_seen = False
        self._last_complete_time = float("-inf")
        self.valid_hold_seconds = 0.0
        self._last_valid_hold_timestamp: float | None = None
        self._hold_unit_start_frame: int | None = None
        self._hold_unit_start_time: float | None = None
        self._hold_confidences: list[float] = []

    @property
    def current_cycle_valid(self) -> bool:
        return (
            self._cycle_start_time is not None
            and self.low_confidence_gap <= self.config.max_low_confidence_gap_frames
        )

    def _reset_cycle(self) -> None:
        self._cycle_start_time = None
        self._cycle_start_frame = None
        self._cycle_max_displacement = 0.0
        self._cycle_max_pelvis = 0.0
        self._cycle_confidences = []
        self._phase_sequence = []
        self._leg_up_seen = False
        self.active_leg = None
        self.selected_side = None

    def _invalidate_cycle(self, next_phase: str = "READY") -> None:
        if self._cycle_start_time is not None:
            self.incomplete_cycles += 1
        self._reset_cycle()
        self.phase = next_phase

    def finish(self) -> None:
        if self._cycle_start_time is not None:
            self._invalidate_cycle("SUPPORT")

    def _append_phase(self, phase: str) -> None:
        self.phase = phase
        if not self._phase_sequence or self._phase_sequence[-1] != phase:
            self._phase_sequence.append(phase)

    def update_landmarks(
        self,
        landmarks_33: np.ndarray | None,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> MarchingPlankFrameResult:
        if landmarks_33 is None:
            return self.update_signals(None, None, None, 0.0, timestamp, frame_index, counting_enabled)
        landmarks = np.asarray(landmarks_33, dtype=np.float32)
        if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
            return self.update_signals(None, None, None, 0.0, timestamp, frame_index, counting_enabled)
        indices = [MEDIAPIPE_LANDMARKS[name] for name in self.REQUIRED_NAMES]
        confidence = float(np.clip(np.min(landmarks[indices, 3]), 0.0, 1.0))
        left_ankle_y = float(landmarks[MEDIAPIPE_LANDMARKS["LEFT_ANKLE"], 1])
        right_ankle_y = float(landmarks[MEDIAPIPE_LANDMARKS["RIGHT_ANKLE"], 1])
        pelvis_y = float(np.mean([
            landmarks[MEDIAPIPE_LANDMARKS["LEFT_HIP"], 1],
            landmarks[MEDIAPIPE_LANDMARKS["RIGHT_HIP"], 1],
        ]))
        return self.update_signals(
            left_ankle_y,
            right_ankle_y,
            pelvis_y,
            confidence,
            timestamp,
            frame_index,
            counting_enabled,
        )

    def update_signals(
        self,
        left_ankle_y: float | None,
        right_ankle_y: float | None,
        pelvis_y: float | None,
        signal_confidence: float,
        timestamp: float,
        frame_index: int,
        counting_enabled: bool = True,
    ) -> MarchingPlankFrameResult:
        confidence = float(np.clip(signal_confidence, 0.0, 1.0))
        valid = (
            left_ankle_y is not None
            and right_ankle_y is not None
            and pelvis_y is not None
            and np.isfinite(left_ankle_y)
            and np.isfinite(right_ankle_y)
            and np.isfinite(pelvis_y)
            and confidence >= self.config.min_landmark_confidence
        )
        if not counting_enabled:
            self._last_valid_hold_timestamp = None
            return self._result(phase="PREPARING", confidence=confidence, valid=valid)
        if not valid:
            self._last_valid_hold_timestamp = None
            self.low_confidence_gap += 1
            if self.low_confidence_gap > self.config.max_low_confidence_gap_frames:
                self._invalidate_cycle("UNKNOWN")
                self._left_history.clear()
                self._right_history.clear()
                self._pelvis_history.clear()
            return self._result(confidence=confidence, valid=False)

        self.low_confidence_gap = 0
        self._left_history.append(float(left_ankle_y))
        self._right_history.append(float(right_ankle_y))
        self._pelvis_history.append(float(pelvis_y))
        left = float(np.median(self._left_history))
        right = float(np.median(self._right_history))
        pelvis = float(np.median(self._pelvis_history))

        if not self._neutral_left:
            self._neutral_left.append(left)
            self._neutral_right.append(right)
            self._neutral_pelvis.append(pelvis)
        left_displacement = float(np.median(self._neutral_left) - left)
        right_displacement = float(np.median(self._neutral_right) - right)
        pelvis_motion = abs(pelvis - float(np.median(self._neutral_pelvis)))
        self.last_vertical_body_motion = max(left_displacement, right_displacement)
        completed: CompletedMarchingPlankCycle | None = None

        if self._cycle_start_time is not None and self.active_leg is not None:
            active = left_displacement if self.active_leg == "LEFT" else right_displacement
            other = right_displacement if self.active_leg == "LEFT" else left_displacement
            self._cycle_max_displacement = max(self._cycle_max_displacement, active)
            self._cycle_max_pelvis = max(self._cycle_max_pelvis, pelvis_motion)
            self._cycle_confidences.append(confidence)
            if other > self.config.maximum_other_leg_displacement:
                self._invalidate_cycle("SUPPORT")
            elif timestamp - self._cycle_start_time > self.config.maximum_repetition_duration:
                self._invalidate_cycle("SUPPORT")

        if self.phase in {"READY", "UNKNOWN"}:
            if max(abs(left_displacement), abs(right_displacement)) <= self.config.return_tolerance:
                self.phase = "SUPPORT"
        elif self.phase == "SUPPORT":
            neutral = max(abs(left_displacement), abs(right_displacement)) <= self.config.return_tolerance
            if neutral and pelvis_motion <= self.config.maximum_pelvis_motion:
                self._neutral_left.append(left)
                self._neutral_right.append(right)
                self._neutral_pelvis.append(pelvis)
            candidate: str | None = None
            if left_displacement >= self.config.lift_start_displacement and right_displacement < self.config.maximum_other_leg_displacement:
                candidate = "LEFT"
            elif right_displacement >= self.config.lift_start_displacement and left_displacement < self.config.maximum_other_leg_displacement:
                candidate = "RIGHT"
            if (
                candidate is not None
                and pelvis_motion <= self.config.maximum_pelvis_motion
                and timestamp - self._last_complete_time >= self.config.cooldown_seconds
            ):
                self.active_leg = candidate
                self.selected_side = candidate.lower()
                self._cycle_start_time = float(timestamp)
                self._cycle_start_frame = int(frame_index)
                self._cycle_max_displacement = max(left_displacement, right_displacement)
                self._cycle_max_pelvis = pelvis_motion
                self._cycle_confidences = [confidence]
                self._phase_sequence = ["SUPPORT", "LEG_LIFTING"]
                self._leg_up_seen = False
                self.phase = "LEG_LIFTING"
        elif self.phase == "LEG_LIFTING" and self.active_leg is not None:
            active = left_displacement if self.active_leg == "LEFT" else right_displacement
            if active >= self.config.leg_up_displacement:
                self._leg_up_seen = True
                self._append_phase("LEG_UP")
            elif active <= self.config.return_tolerance:
                self._invalidate_cycle("SUPPORT")
        elif self.phase == "LEG_UP" and self.active_leg is not None:
            active = left_displacement if self.active_leg == "LEFT" else right_displacement
            if active <= self._cycle_max_displacement - self.config.return_hysteresis:
                self._append_phase("RETURNING")
        elif self.phase == "RETURNING" and self.active_leg is not None:
            active = left_displacement if self.active_leg == "LEFT" else right_displacement
            other = right_displacement if self.active_leg == "LEFT" else left_displacement
            if active <= self.config.return_tolerance and abs(other) <= self.config.return_tolerance:
                start_time = self._cycle_start_time
                start_frame = self._cycle_start_frame
                duration = float(timestamp - start_time) if start_time is not None else -1.0
                valid_cycle = (
                    start_time is not None
                    and start_frame is not None
                    and self._leg_up_seen
                    and self.config.minimum_repetition_duration <= duration <= self.config.maximum_repetition_duration
                    and self._cycle_max_displacement >= self.config.leg_up_displacement
                    and self._cycle_max_pelvis <= self.config.maximum_pelvis_motion
                    and self._phase_sequence == ["SUPPORT", "LEG_LIFTING", "LEG_UP", "RETURNING"]
                )
                if valid_cycle:
                    leg = self.active_leg
                    if leg == "LEFT":
                        self.left_repetitions += 1
                    else:
                        self.right_repetitions += 1
                    self.last_repetition_duration = duration
                    self._last_complete_time = float(timestamp)
                    self._reset_cycle()
                    self.phase = "SUPPORT"
                else:
                    self._invalidate_cycle("SUPPORT")

        valid_hold = (
            pelvis_motion <= self.config.maximum_pelvis_motion
            and self.phase in {"SUPPORT", "LEG_LIFTING", "LEG_UP", "RETURNING"}
        )
        if valid_hold:
            if self._last_valid_hold_timestamp is None:
                self._hold_unit_start_time = float(timestamp)
                self._hold_unit_start_frame = int(frame_index)
            else:
                delta = max(0.0, float(timestamp) - self._last_valid_hold_timestamp)
                self.valid_hold_seconds += delta
            self._last_valid_hold_timestamp = float(timestamp)
            self._hold_confidences.append(confidence)
            target_count = int((self.valid_hold_seconds + 1e-9) // self.config.hold_unit_seconds)
            if target_count > self.repetition_count:
                self.repetition_count = target_count
                self.last_repetition_duration = self.config.hold_unit_seconds
                completed = CompletedMarchingPlankCycle(
                    start_frame=self._hold_unit_start_frame if self._hold_unit_start_frame is not None else int(frame_index),
                    end_frame=int(frame_index),
                    start_time=self._hold_unit_start_time if self._hold_unit_start_time is not None else float(timestamp),
                    end_time=float(timestamp),
                    duration_seconds=self.config.hold_unit_seconds,
                    phase_sequence=["VALID_MARCHING_PLANK_HOLD"],
                    confidence=float(np.mean(self._hold_confidences)) if self._hold_confidences else confidence,
                    active_leg="HOLD",
                    maximum_leg_displacement=max(abs(left_displacement), abs(right_displacement)),
                    maximum_pelvis_motion=pelvis_motion,
                )
                self._hold_unit_start_time = float(timestamp)
                self._hold_unit_start_frame = int(frame_index)
                self._hold_confidences = []
        else:
            self._last_valid_hold_timestamp = None

        return self._result(
            confidence=confidence,
            valid=True,
            completed=completed,
            left_displacement=left_displacement,
            right_displacement=right_displacement,
            pelvis_stability=pelvis_motion,
        )

    def _result(
        self,
        phase: str | None = None,
        confidence: float | None = None,
        valid: bool = False,
        completed: CompletedMarchingPlankCycle | None = None,
        left_displacement: float | None = None,
        right_displacement: float | None = None,
        pelvis_stability: float | None = None,
    ) -> MarchingPlankFrameResult:
        return MarchingPlankFrameResult(
            phase=phase or self.phase,
            repetition_count=self.repetition_count,
            elbow_angle=None,
            selected_side=self.selected_side,
            signal_confidence=confidence,
            last_repetition_duration=self.last_repetition_duration,
            current_cycle_valid=self.current_cycle_valid,
            vertical_body_motion=self.last_vertical_body_motion,
            completed_cycle=completed,
            valid_frame=valid,
            active_leg=self.active_leg,
            left_leg_displacement=left_displacement,
            right_leg_displacement=right_displacement,
            pelvis_stability=pelvis_stability,
            left_repetitions=self.left_repetitions,
            right_repetitions=self.right_repetitions,
            hold_time_seconds=self.valid_hold_seconds,
        )
