from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


MEDIAPIPE_POSE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29),
    (29, 31), (27, 31), (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
)


class PoseStreamProcessor:
    """Blocking MediaPipe Tasks VIDEO-mode detector owned by one worker thread."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model is missing: {path}")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        self.closed = False

    def process(self, bgr_frame: np.ndarray, timestamp_seconds: float) -> np.ndarray | None:
        if self.closed:
            return None
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            self._last_timestamp_ms + 1,
            int(round(max(0.0, float(timestamp_seconds)) * 1000.0)),
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
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


def full_body_visible(landmarks_33: np.ndarray | None, threshold: float = 0.45) -> bool:
    """Check whether at least one full side chain is reliably inside the frame."""

    if landmarks_33 is None:
        return False
    landmarks = np.asarray(landmarks_33)
    if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
        return False
    chains = (
        ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    )
    return any(
        min(float(landmarks[MEDIAPIPE_LANDMARKS[name], 3]) for name in chain) >= threshold
        for chain in chains
    )


def pose_visible_for_family(
    landmarks_33: np.ndarray | None,
    family_id: str,
    camera_view: str,
    threshold: float = 0.45,
) -> bool:
    """Use family-specific preparation visibility without changing analysis signals."""

    if family_id == "pushup":
        return full_body_visible(landmarks_33, threshold=threshold)
    if landmarks_33 is None:
        return False
    landmarks = np.asarray(landmarks_33)
    if landmarks.shape != (33, 4) or not np.isfinite(landmarks).all():
        return False
    if family_id == "pullup":
        arm_chains = (
            ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HIP"),
        )
        reliable = [
            min(float(landmarks[MEDIAPIPE_LANDMARKS[name], 3]) for name in chain)
            >= threshold
            for chain in arm_chains
        ]
        nose_visible = (
            float(landmarks[MEDIAPIPE_LANDMARKS["NOSE"], 3]) >= threshold
        )
        return nose_visible and (all(reliable) if camera_view == "front" else any(reliable))
    if family_id == "squat":
        chains = (
            ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        )
        return any(min(float(landmarks[MEDIAPIPE_LANDMARKS[name], 3]) for name in chain) >= threshold for chain in chains)
    return full_body_visible(landmarks_33, threshold=threshold)


def draw_pose_skeleton(
    frame: np.ndarray,
    landmarks_33: np.ndarray | None,
    confidence_threshold: float = 0.35,
) -> np.ndarray:
    """Draw MediaPipe Pose connections without mutating the source frame."""

    output = frame.copy()
    if landmarks_33 is None:
        return output
    landmarks = np.asarray(landmarks_33)
    if landmarks.shape != (33, 4):
        return output
    height, width = output.shape[:2]
    points = np.rint(landmarks[:, :2] * np.asarray([width, height])).astype(np.int32)
    for start, end in MEDIAPIPE_POSE_EDGES:
        if min(float(landmarks[start, 3]), float(landmarks[end, 3])) < confidence_threshold:
            continue
        cv2.line(output, tuple(points[start]), tuple(points[end]), (72, 214, 154), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        if float(landmarks[index, 3]) >= confidence_threshold:
            cv2.circle(output, tuple(point), 3, (255, 210, 80), -1, cv2.LINE_AA)
    return output
