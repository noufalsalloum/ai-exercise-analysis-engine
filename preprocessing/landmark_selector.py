from __future__ import annotations

from typing import Final

import numpy as np


MEDIAPIPE_LANDMARKS: Final[dict[str, int]] = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "HEEL_LEFT": 29,
    "HEEL_RIGHT": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}


H36M_JOINT_NAMES: Final[tuple[str, ...]] = (
    "root",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "belly",
    "neck",
    "nose",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

H36M_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)


class LandmarkSelector:
    """Select MediaPipe landmarks and build MotionBERT H36M inputs.

    MediaPipe data remains in its original ``(x, y, z, visibility)`` form
    when returned by :meth:`select_landmarks`. The MotionBERT conversion is a
    separate representation whose channels are ``(x, y, confidence)``.
    MediaPipe ``z`` is deliberately not passed to the pretrained MotionBERT
    model because the current checkpoint was trained with confidence as its
    third input channel.
    """

    def __init__(self, exercise_spec: dict) -> None:
        self.selected_names = exercise_spec["landmarks"]["selected_landmarks"]

        unknown_names = [
            name for name in self.selected_names if name not in MEDIAPIPE_LANDMARKS
        ]
        if unknown_names:
            raise ValueError(
                "Unknown MediaPipe landmark names in exercise config: "
                f"{unknown_names}"
            )

        self.selected_indices = [
            MEDIAPIPE_LANDMARKS[name] for name in self.selected_names
        ]

    def load_selected_landmarks(self) -> list[str]:
        """Return the configured MediaPipe landmark names."""

        return self.selected_names

    def select_landmarks(self, landmarks_33: np.ndarray) -> np.ndarray:
        """Select configured landmarks without dropping ``z`` or visibility."""

        landmarks = np.asarray(landmarks_33)

        if landmarks.ndim == 3:
            return landmarks[:, self.selected_indices, :]
        if landmarks.ndim == 2:
            return landmarks[self.selected_indices, :]

        raise ValueError(
            "Expected MediaPipe landmarks with shape (33, D) or (T, 33, D), "
            f"got {tuple(landmarks.shape)}"
        )

    def extract_landmark(self, landmarks_33: np.ndarray, name: str) -> np.ndarray:
        """Extract one named MediaPipe landmark from a frame or sequence."""

        idx = MEDIAPIPE_LANDMARKS.get(name)
        if idx is None:
            raise KeyError(f"Landmark {name} not found.")

        landmarks = np.asarray(landmarks_33)

        if landmarks.ndim == 3:
            return landmarks[:, idx, :]
        if landmarks.ndim == 2:
            return landmarks[idx, :]

        raise ValueError(
            "Expected MediaPipe landmarks with shape (33, D) or (T, 33, D), "
            f"got {tuple(landmarks.shape)}"
        )

    def to_h36m_17(self, landmarks_33: np.ndarray) -> np.ndarray:
        """Convert MediaPipe Pose landmarks to MotionBERT's H36M contract.

        Parameters
        ----------
        landmarks_33:
            A single MediaPipe frame with shape ``(33, 4)`` or a sequence
            with shape ``(T, 33, 4)``. Channels must be ordered as
            ``(x, y, z, visibility)``.

        Returns
        -------
        np.ndarray
            A ``float32`` array with shape ``(17, 3)`` for a single frame or
            ``(T, 17, 3)`` for a sequence. Output channels are
            ``(x, y, confidence)``, where MediaPipe visibility is used as the
            detector-confidence proxy expected by the pretrained MotionBERT
            checkpoint.

        Notes
        -----
        This method never mutates ``landmarks_33`` and never sends MediaPipe
        ``z`` to MotionBERT. The H36M order matches MotionBERT's official
        COCO/Halpe conversion: root, right leg, left leg, torso, head, left
        arm, then right arm.
        """

        landmarks = np.asarray(landmarks_33)
        single_frame = landmarks.ndim == 2

        if landmarks.ndim not in (2, 3):
            raise ValueError(
                "Expected MediaPipe landmarks with shape (33, 4) or "
                f"(T, 33, 4), got {tuple(landmarks.shape)}"
            )

        if landmarks.shape[-2] != 33:
            raise ValueError(
                "Expected exactly 33 MediaPipe landmarks, "
                f"got {landmarks.shape[-2]}"
            )

        if landmarks.shape[-1] < 4:
            raise ValueError(
                "MotionBERT conversion requires MediaPipe channels "
                "(x, y, z, visibility); visibility must not be replaced by z."
            )

        sequence = landmarks[np.newaxis, ...] if single_frame else landmarks

        # MotionBERT was trained with (x, y, confidence). Keep MediaPipe z in
        # the separate geometry representation returned by select_landmarks.
        motionbert_source = np.asarray(
            sequence[..., [0, 1, 3]],
            dtype=np.float32,
        )

        if not np.isfinite(motionbert_source).all():
            raise ValueError(
                "MediaPipe x, y, and visibility values must all be finite."
            )

        confidence = motionbert_source[..., 2]
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("MediaPipe visibility must be within [0, 1].")

        output = np.empty(
            (sequence.shape[0], len(H36M_JOINT_NAMES), 3),
            dtype=np.float32,
        )

        left_hip = motionbert_source[:, MEDIAPIPE_LANDMARKS["LEFT_HIP"], :]
        right_hip = motionbert_source[:, MEDIAPIPE_LANDMARKS["RIGHT_HIP"], :]
        left_shoulder = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["LEFT_SHOULDER"], :
        ]
        right_shoulder = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["RIGHT_SHOULDER"], :
        ]

        def derived_joint(*sources: np.ndarray) -> np.ndarray:
            """Average XY but propagate confidence conservatively by minimum."""

            stacked = np.stack(sources, axis=0)
            result = np.empty_like(sources[0])
            result[..., :2] = np.mean(stacked[..., :2], axis=0)
            result[..., 2] = np.min(stacked[..., 2], axis=0)
            return result

        root = derived_joint(left_hip, right_hip)
        neck = derived_joint(left_shoulder, right_shoulder)
        belly = derived_joint(root, neck)
        head = derived_joint(
            motionbert_source[:, MEDIAPIPE_LANDMARKS["LEFT_EYE"], :],
            motionbert_source[:, MEDIAPIPE_LANDMARKS["RIGHT_EYE"], :],
        )

        output[:, 0, :] = root
        output[:, 1, :] = right_hip
        output[:, 2, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["RIGHT_KNEE"], :
        ]
        output[:, 3, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["RIGHT_ANKLE"], :
        ]
        output[:, 4, :] = left_hip
        output[:, 5, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["LEFT_KNEE"], :
        ]
        output[:, 6, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["LEFT_ANKLE"], :
        ]
        output[:, 7, :] = belly
        output[:, 8, :] = neck
        output[:, 9, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["NOSE"], :
        ]
        output[:, 10, :] = head
        output[:, 11, :] = left_shoulder
        output[:, 12, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["LEFT_ELBOW"], :
        ]
        output[:, 13, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["LEFT_WRIST"], :
        ]
        output[:, 14, :] = right_shoulder
        output[:, 15, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["RIGHT_ELBOW"], :
        ]
        output[:, 16, :] = motionbert_source[
            :, MEDIAPIPE_LANDMARKS["RIGHT_WRIST"], :
        ]

        return output[0] if single_frame else output
