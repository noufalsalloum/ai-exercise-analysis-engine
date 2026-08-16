import numpy as np
import pandas as pd


class Interpolator:
    def __init__(self, visibility_threshold: float = 0.5):
        self.visibility_threshold = visibility_threshold

    def find_missing(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 3:
            return np.array([])

        if sequence.shape[-1] >= 4:
            visibility = sequence[:, :, 3]
            return visibility < self.visibility_threshold

        return np.isnan(sequence) | (sequence == 0)

    def linear_interpolation(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 3:
            return sequence

        # If visibility channel exists: (frames, landmarks, 4)
        if sequence.shape[-1] >= 4:
            out = sequence.copy()
            coords = out[:, :, :3]
            visibility = out[:, :, 3]

            num_frames, num_landmarks, _ = coords.shape

            for lm_idx in range(num_landmarks):
                low_vis_mask = visibility[:, lm_idx] < self.visibility_threshold

                for dim in range(3):
                    series = pd.Series(coords[:, lm_idx, dim].astype(float))
                    series[low_vis_mask] = np.nan
                    series = (
                        series.interpolate(method="linear", limit_direction="both")
                        .bfill()
                        .ffill()
                        .fillna(0.0)
                    )
                    coords[:, lm_idx, dim] = series.to_numpy(dtype=np.float32)

            out[:, :, :3] = coords
            return out

        # Fallback for 3D arrays without visibility
        num_frames, num_landmarks, num_dims = sequence.shape
        reshaped = sequence.reshape(num_frames, -1)

        reshaped[reshaped == 0] = np.nan
        df = pd.DataFrame(reshaped)
        df = df.interpolate(method="linear", limit_direction="both").bfill().ffill()
        df = df.fillna(0.0)

        return df.to_numpy().reshape(num_frames, num_landmarks, num_dims)

    def interpolate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return self.linear_interpolation(sequence)