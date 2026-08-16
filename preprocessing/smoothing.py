from __future__ import annotations

import numpy as np


class Smoother:
    """Nearest-edge moving average without an external SciPy dependency."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")
        self.window_size = window_size

    def moving_average(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 3:
            return sequence
        smoothed = sequence.copy()
        coord_dims = 3 if sequence.shape[-1] >= 3 else sequence.shape[-1]
        radius = self.window_size // 2
        coordinates = np.asarray(sequence[:, :, :coord_dims], dtype=np.float32)
        padded = np.pad(coordinates, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=self.window_size,
            axis=0,
        )
        smoothed[:, :, :coord_dims] = windows.mean(axis=-1, dtype=np.float32)
        if sequence.shape[-1] >= 4:
            smoothed[:, :, 3] = sequence[:, :, 3]
        return smoothed

    def kalman_filter(self, sequence: np.ndarray) -> np.ndarray:
        """Compatibility fallback until a trained/stateful Kalman model exists."""

        return self.moving_average(sequence)

    def smooth_sequence(
        self,
        sequence: np.ndarray,
        method: str = "moving_average",
    ) -> np.ndarray:
        if method == "kalman":
            return self.kalman_filter(sequence)
        if method != "moving_average":
            raise ValueError(f"Unsupported smoothing method: {method}")
        return self.moving_average(sequence)
