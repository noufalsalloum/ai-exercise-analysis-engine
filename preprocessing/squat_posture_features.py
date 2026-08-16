"""Static MediaPipe pose and geometric features for Squat Error V1."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np

from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


POSTURE_JOINTS: Final[tuple[str, ...]] = (
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "HEEL_LEFT", "HEEL_RIGHT",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    left = a - b; right = c - b
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _vertical_inclination(top: np.ndarray, bottom: np.ndarray) -> float:
    vector = top - bottom
    return float(np.degrees(np.arctan2(abs(float(vector[0])), abs(float(vector[1])) + 1e-8)))


def _line_tilt(left: np.ndarray, right: np.ndarray) -> float:
    vector = right - left
    return float(np.degrees(np.arctan2(abs(float(vector[1])), abs(float(vector[0])) + 1e-8)))


class SquatPostureFeatureExtractor:
    """Create finite, label-independent static squat posture features."""

    VERSION = "squat_static_mp12_geometry_v1"

    def __init__(self, coordinate_clip: float = 4.0) -> None:
        self.coordinate_clip = float(coordinate_clip)
        coordinate_names = [f"{joint.lower()}_{axis}_normalized" for joint in POSTURE_JOINTS for axis in ("x", "y")]
        confidence_names = [f"{joint.lower()}_confidence" for joint in POSTURE_JOINTS]
        geometry_names = [
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle",
            "left_torso_inclination", "right_torso_inclination",
            "center_torso_inclination", "shoulder_line_tilt", "hip_line_tilt",
            "left_knee_ankle_horizontal_offset", "right_knee_ankle_horizontal_offset",
            "left_heel_foot_tilt", "right_heel_foot_tilt",
            "left_heel_foot_length", "right_heel_foot_length",
            "left_shoulder_hip_horizontal_offset", "right_shoulder_hip_horizontal_offset",
            "knee_angle_asymmetry", "hip_angle_asymmetry",
            "torso_inclination_asymmetry", "ankle_height_asymmetry",
            "knee_ankle_offset_asymmetry", "mean_required_confidence",
            "minimum_required_confidence", "left_chain_confidence",
            "right_chain_confidence", "body_scale", "pose_success",
        ]
        self.feature_names = tuple(coordinate_names + confidence_names + geometry_names)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def extract(self, landmarks_33: np.ndarray | None) -> np.ndarray:
        """Return a finite vector; pose failures use a documented zero sentinel."""

        if landmarks_33 is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
        landmarks = np.asarray(landmarks_33, dtype=np.float32)
        if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
            raise ValueError("Expected finite MediaPipe landmarks (33,4).")
        values = landmarks[[MEDIAPIPE_LANDMARKS[name] for name in POSTURE_JOINTS]].copy()
        values[:, 3] = np.clip(values[:, 3], 0.0, 1.0)
        xy = values[:, :2]
        point = {name: xy[index] for index, name in enumerate(POSTURE_JOINTS)}
        confidence = {name: float(values[index, 3]) for index, name in enumerate(POSTURE_JOINTS)}
        hip_center = 0.5 * (point["LEFT_HIP"] + point["RIGHT_HIP"])
        shoulder_center = 0.5 * (point["LEFT_SHOULDER"] + point["RIGHT_SHOULDER"])
        candidates = np.asarray(
            [
                np.linalg.norm(point["LEFT_SHOULDER"] - point["RIGHT_SHOULDER"]),
                np.linalg.norm(point["LEFT_HIP"] - point["RIGHT_HIP"]),
                np.linalg.norm(shoulder_center - hip_center),
                np.linalg.norm(point["LEFT_KNEE"] - point["RIGHT_KNEE"]),
            ],
            dtype=np.float32,
        )
        valid_scales = candidates[candidates > 1e-6]
        scale = float(np.max(valid_scales)) if len(valid_scales) else 1.0
        normalized = np.clip((xy - hip_center) / scale, -self.coordinate_clip, self.coordinate_clip)

        left_knee = _angle(point["LEFT_HIP"], point["LEFT_KNEE"], point["LEFT_ANKLE"])
        right_knee = _angle(point["RIGHT_HIP"], point["RIGHT_KNEE"], point["RIGHT_ANKLE"])
        left_hip = _angle(point["LEFT_SHOULDER"], point["LEFT_HIP"], point["LEFT_KNEE"])
        right_hip = _angle(point["RIGHT_SHOULDER"], point["RIGHT_HIP"], point["RIGHT_KNEE"])
        left_torso = _vertical_inclination(point["LEFT_SHOULDER"], point["LEFT_HIP"])
        right_torso = _vertical_inclination(point["RIGHT_SHOULDER"], point["RIGHT_HIP"])
        center_torso = _vertical_inclination(shoulder_center, hip_center)
        left_knee_offset = float((point["LEFT_KNEE"][0] - point["LEFT_ANKLE"][0]) / scale)
        right_knee_offset = float((point["RIGHT_KNEE"][0] - point["RIGHT_ANKLE"][0]) / scale)
        left_chain = np.mean([confidence[name] for name in ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "HEEL_LEFT", "LEFT_FOOT_INDEX")])
        right_chain = np.mean([confidence[name] for name in ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "HEEL_RIGHT", "RIGHT_FOOT_INDEX")])
        geometry = np.asarray(
            [
                left_knee, right_knee, left_hip, right_hip,
                left_torso, right_torso, center_torso,
                _line_tilt(point["LEFT_SHOULDER"], point["RIGHT_SHOULDER"]),
                _line_tilt(point["LEFT_HIP"], point["RIGHT_HIP"]),
                left_knee_offset, right_knee_offset,
                _line_tilt(point["HEEL_LEFT"], point["LEFT_FOOT_INDEX"]),
                _line_tilt(point["HEEL_RIGHT"], point["RIGHT_FOOT_INDEX"]),
                float(np.linalg.norm(point["HEEL_LEFT"] - point["LEFT_FOOT_INDEX"]) / scale),
                float(np.linalg.norm(point["HEEL_RIGHT"] - point["RIGHT_FOOT_INDEX"]) / scale),
                float((point["LEFT_SHOULDER"][0] - point["LEFT_HIP"][0]) / scale),
                float((point["RIGHT_SHOULDER"][0] - point["RIGHT_HIP"][0]) / scale),
                abs(left_knee - right_knee), abs(left_hip - right_hip),
                abs(left_torso - right_torso),
                float(abs(point["LEFT_ANKLE"][1] - point["RIGHT_ANKLE"][1]) / scale),
                abs(left_knee_offset - right_knee_offset),
                float(values[:, 3].mean()), float(values[:, 3].min()),
                float(left_chain), float(right_chain), scale, 1.0,
            ],
            dtype=np.float32,
        )
        output = np.concatenate((normalized.reshape(-1), values[:, 3], geometry)).astype(np.float32)
        if output.shape != (self.feature_dim,) or not np.isfinite(output).all():
            raise FloatingPointError("Static squat posture features are invalid.")
        return output


class MediaPipeImagePoseExtractor:
    """Independent IMAGE-mode pose detection for unrelated static images."""

    def __init__(self, model_path: str | Path, max_dimension: int = 1024) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model is missing: {path}")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.max_dimension = int(max_dimension)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.closed = False

    def process(self, bgr_image: np.ndarray) -> np.ndarray | None:
        if self.closed:
            raise RuntimeError("Pose extractor is closed.")
        image = np.asarray(bgr_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected BGR image with three channels.")
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > self.max_dimension:
            factor = self.max_dimension / longest
            image = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return None
        landmarks = np.asarray(
            [[item.x, item.y, item.z, item.visibility] for item in result.pose_landmarks[0]],
            dtype=np.float32,
        )
        if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
            return None
        landmarks[:, 3] = np.clip(landmarks[:, 3], 0.0, 1.0)
        return landmarks

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._landmarker.close()

    def __enter__(self) -> "MediaPipeImagePoseExtractor":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

