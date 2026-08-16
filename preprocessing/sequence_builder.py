from copy import deepcopy
import numpy as np


class SequenceBuilder:
    def __init__(self, window_size: int = 30, step_size: int = 5):
        self.window_size = window_size
        self.step_size = step_size

    def pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        current_len = len(sequence)
        if current_len >= target_length:
            return sequence[:target_length]

        pad_size = target_length - current_len
        padding = np.tile(sequence[-1:], (pad_size, 1, 1))
        return np.concatenate([sequence, padding], axis=0)

    def _pad_list(self, values: list, target_length: int) -> list:
        if not values:
            return [None] * target_length

        if len(values) >= target_length:
            return values[:target_length]

        return values + [deepcopy(values[-1]) for _ in range(target_length - len(values))]

    def _pad_value(self, value, target_length: int):
        if isinstance(value, np.ndarray):
            return self.pad_sequence(value, target_length)
        if isinstance(value, list):
            return self._pad_list(value, target_length)
        return value

    def build_sequence(
        self,
        motionbert_input: np.ndarray,
        selected_landmarks: np.ndarray,
    ) -> list:
        """
        Build aligned windows for:
        - motionbert_input: H36M-like 17 joints
        - selected_landmarks: selected MediaPipe landmarks
        """
        if motionbert_input.shape[0] != selected_landmarks.shape[0]:
            raise ValueError("motionbert_input and selected_landmarks must have the same number of frames.")

        num_frames = selected_landmarks.shape[0]
        windows = []

        if num_frames <= self.window_size:
            windows.append({
                "motionbert_input": self.pad_sequence(motionbert_input, self.window_size),
                "landmarks": self.pad_sequence(selected_landmarks, self.window_size),
                "window_start": 0,
                "window_end": num_frames,
            })
            return windows

        for start in range(0, num_frames - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append({
                "motionbert_input": motionbert_input[start:end],
                "landmarks": selected_landmarks[start:end],
                "window_start": start,
                "window_end": end,
            })

        return windows