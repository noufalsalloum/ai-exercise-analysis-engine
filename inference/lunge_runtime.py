"""Independent rule-based Lunge repetition runtime calibrated on REHAB24 Ex5 signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inference.squat_runtime import SquatRepConfig, SquatRepetitionRuntime, _joint_angle
from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


@dataclass(frozen=True)
class LungeRepConfig(SquatRepConfig):
    standing_ready_angle: float = 160.0
    descent_enter_angle: float = 145.0
    bottom_enter_angle: float = 120.0
    bottom_exit_angle: float = 132.0
    standing_return_angle: float = 160.0
    descent_start_displacement: float = 0.010
    minimum_pelvis_displacement: float = 0.040
    return_pelvis_tolerance: float = 0.035
    ascent_displacement_hysteresis: float = 0.010
    minimum_knee_rom: float = 35.0
    minimum_hip_rom: float = 12.0
    minimum_repetition_duration: float = 0.40
    maximum_repetition_duration: float = 6.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LungeRepConfig":
        return super().from_dict(payload)  # type: ignore[return-value]


class LungeRepetitionRuntime(SquatRepetitionRuntime):
    """Standing→descending→bottom→ascending→standing Lunge counter.

    The active/front leg is selected from the more flexed reliable knee at
    descent and remains locked for the cycle. Thresholds were selected from
    REHAB24 Ex5 GT signal percentiles, not copied from the Squat config.
    """

    def __init__(self, config: LungeRepConfig | None = None, camera_view: str = "side") -> None:
        if camera_view not in {"side", "half_profile"}:
            raise ValueError("Lunge rule runtime supports side or half-profile views.")
        self.config=config or LungeRepConfig(); self.camera_view=camera_view; self.reset()

    def update_landmarks(self, landmarks_33, timestamp, frame_index, counting_enabled=True):
        if landmarks_33 is None:
            return self.update_signals(None,None,None,0.0,timestamp,frame_index,counting_enabled)
        landmarks=np.asarray(landmarks_33,dtype=np.float32)
        if landmarks.shape != (33,4) or not np.isfinite(landmarks).all():
            return self.update_signals(None,None,None,0.0,timestamp,frame_index,counting_enabled)
        candidates={}
        for side in ("left","right"):
            names=self.SIDE_NAMES[side]; points=[landmarks[MEDIAPIPE_LANDMARKS[name]] for name in names]
            confidence=float(np.min([point[3] for point in points])); knee=_joint_angle(points[1],points[2],points[3]); hip=_joint_angle(points[0],points[1],points[2])
            candidates[side]=(knee,hip,float(points[1][1]),confidence)
        reliable=[side for side,value in candidates.items() if value[0] is not None and value[3]>=self.config.min_landmark_confidence]
        if not reliable:
            return self.update_signals(None,None,None,0.0,timestamp,frame_index,counting_enabled)
        if self.current_cycle_valid and self.selected_side in reliable:
            side=self.selected_side
        else:
            flexing=[side for side in reliable if float(candidates[side][0]) <= self.config.descent_enter_angle]
            side=min(flexing,key=lambda value:float(candidates[value][0])) if flexing else max(reliable,key=lambda value:candidates[value][3])
            self.selected_side=side
        knee,hip,pelvis,confidence=candidates[side]
        return self.update_signals(knee,hip,pelvis,confidence,timestamp,frame_index,counting_enabled)
