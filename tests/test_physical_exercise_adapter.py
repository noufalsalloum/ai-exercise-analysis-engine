from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.adapters.physical_exercise_recognition_adapter import (
    PhysicalExerciseRecognitionAdapter,
)
from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


def make_physical_fixture(root: Path, frames: list[int]) -> None:
    pd.DataFrame({"vid_id": [7], "class": ["push_up"]}).to_csv(root / "labels.csv", index=False)
    names = [name for name, _ in sorted(MEDIAPIPE_LANDMARKS.items(), key=lambda item: item[1])]
    columns = [axis + "_" + name for name in names for axis in ("x", "y", "z")]
    rows = []
    for frame in frames:
        values = []
        for index in range(33):
            values.extend(
                (
                    index % 5 + frame * 0.01,
                    index // 5 + frame * 0.02,
                    1000 + index,
                )
            )
        rows.append([7, frame, *values])
    pd.DataFrame(rows, columns=["vid_id", "frame_order", *columns]).to_csv(
        root / "landmarks.csv", index=False
    )


class PhysicalExerciseAdapterTests(unittest.TestCase):
    def test_frame_order_gap_confidence_and_motionbert_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_physical_fixture(root, [value for value in range(35) if value != 10])
            adapter = PhysicalExerciseRecognitionAdapter(root)
            segment = next(adapter.iter_processed_segments(7))

            self.assertEqual(segment["geometry"].shape, (35, 33, 4))
            self.assertEqual(segment["motionbert_input"].shape, (35, 17, 3))
            self.assertEqual(segment["geometry"].dtype, np.float32)
            self.assertEqual(segment["motionbert_input"].dtype, np.float32)
            np.testing.assert_array_equal(segment["frame_indices"], np.arange(35))
            self.assertFalse(segment["observed_mask"][10])
            np.testing.assert_allclose(segment["geometry"][10, :, 3], 0.5)
            np.testing.assert_allclose(segment["motionbert_input"][10, :, 2], 0.5)
            np.testing.assert_allclose(segment["motionbert_input"][0, :, 2], 1.0)
            self.assertTrue(np.isfinite(segment["motionbert_input"]).all())
            self.assertFalse(np.allclose(segment["motionbert_input"][0, :, 2], segment["geometry"][0, :17, 2]))
            windows = list(adapter.iter_windows(7))
            self.assertGreaterEqual(len(windows), 1)
            self.assertEqual(windows[0]["motionbert_input"].shape, (30, 17, 3))

    def test_long_gap_splits_instead_of_interpolating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_physical_fixture(root, list(range(5)) + list(range(20, 25)))
            segments = list(PhysicalExerciseRecognitionAdapter(root).iter_processed_segments(7))
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["frame_indices"].tolist(), list(range(5)))
            self.assertEqual(segments[1]["frame_indices"].tolist(), list(range(20, 25)))


if __name__ == "__main__":
    unittest.main()
