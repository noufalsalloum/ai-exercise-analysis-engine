import numpy as np


class Normalizer:
    def __init__(self, exercise_spec: dict):
        self.exercise_id = exercise_spec["exercise"]["id"]

    def _coord_dims(self, frame: np.ndarray) -> int:
        if frame.ndim != 2:
            raise ValueError("Frame must be 2D: (num_landmarks, num_dims)")
        return 3 if frame.shape[-1] >= 3 else frame.shape[-1]

    def center_pose(self, frame: np.ndarray) -> np.ndarray:
        coord_dims = self._coord_dims(frame)
        coords = frame[:, :coord_dims].astype(np.float32, copy=True)

        centroid = np.mean(coords, axis=0)
        coords -= centroid

        centered = frame.copy()
        centered[:, :coord_dims] = coords
        return centered

    def scale_pose(self, frame: np.ndarray) -> np.ndarray:
        coord_dims = self._coord_dims(frame)
        coords = frame[:, :coord_dims].astype(np.float32, copy=True)

        max_dist = np.max(np.linalg.norm(coords, axis=1))
        scaled = frame.copy()

        if max_dist > 0:
            scaled[:, :coord_dims] = coords / max_dist

        return scaled

    def normalize_coordinates(self, sequence: np.ndarray) -> np.ndarray:
        batch_mode = sequence.ndim == 3
        if not batch_mode:
            sequence = np.expand_dims(sequence, axis=0)

        normalized_batch = []
        for frame in sequence:
            centered = self.center_pose(frame)
            scaled = self.scale_pose(centered)
            normalized_batch.append(scaled)

        res = np.array(normalized_batch)
        return res if batch_mode else res[0]