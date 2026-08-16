import numpy as np


class FeatureExtractor:
    def __init__(self, exercise_spec: dict):
        self.angle_defs = exercise_spec["landmarks"]["angle_definitions"]
        self.selected_landmarks = exercise_spec["landmarks"]["selected_landmarks"]
        self.landmark_map = {name: idx for idx, name in enumerate(self.selected_landmarks)}

    def _coords(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 3:
            raise ValueError("Sequence must be 3D: (frames, landmarks, dims)")
        if sequence.shape[-1] >= 3:
            return sequence[:, :, :3]
        return sequence

    def calculate_angles(self, frame: np.ndarray) -> dict:
        angles = {}
        for angle_name, pts in self.angle_defs.items():
            if all(p in self.landmark_map for p in pts):
                p1 = frame[self.landmark_map[pts[0]], :2]
                p2 = frame[self.landmark_map[pts[1]], :2]
                p3 = frame[self.landmark_map[pts[2]], :2]

                v1 = p1 - p2
                v2 = p3 - p2

                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                angles[angle_name] = float(angle)

        return angles

    def calculate_velocity(self, sequence: np.ndarray) -> np.ndarray:
        coords = self._coords(sequence)
        if len(coords) < 2:
            return np.zeros_like(coords)
        return np.gradient(coords, axis=0)

    def calculate_rom(self, angles_sequence: list) -> dict:
        rom = {}
        if not angles_sequence:
            return rom

        keys = sorted({k for frame_angles in angles_sequence for k in frame_angles.keys()})
        for key in keys:
            vals = [frame_angles[key] for frame_angles in angles_sequence if key in frame_angles]
            if vals:
                rom[key] = {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "range": float(np.max(vals) - np.min(vals)),
                }
        return rom

    def calculate_trajectory(self, sequence: np.ndarray) -> np.ndarray:
        coords = self._coords(sequence)
        return coords[:, :, :2]

    def extract_features(self, sequence: np.ndarray) -> dict:
        num_frames = sequence.shape[0]

        velocity = self.calculate_velocity(sequence)
        angular_velocity = np.gradient(velocity, axis=0) if num_frames > 1 else np.zeros_like(velocity)
        trajectory = self.calculate_trajectory(sequence)

        angles_list = [self.calculate_angles(sequence[i]) for i in range(num_frames)]

        return {
            "landmarks": sequence,
            "angles": angles_list,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "rom": self.calculate_rom(angles_list),
            "trajectory": trajectory,
        }