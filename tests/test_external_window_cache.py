from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.tools.build_external_caches import build_physical_cache
from datasets.adapters.physical_exercise_recognition_adapter import PhysicalExerciseRecognitionAdapter
from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


def write_sequence(root: Path, frames: int = 35) -> None:
    pd.DataFrame({"vid_id": [7], "class": ["push_up"]}).to_csv(root / "labels.csv", index=False)
    names = [name for name, _ in sorted(MEDIAPIPE_LANDMARKS.items(), key=lambda item: item[1])]
    columns = [axis + "_" + name for name in names for axis in ("x", "y", "z")]
    rows = []
    for frame in range(frames):
        coordinates = [value for joint in range(33) for value in (joint + frame, joint - frame, joint * 3)]
        rows.append([7, frame, *coordinates])
    pd.DataFrame(rows, columns=["vid_id", "frame_order", *columns]).to_csv(
        root / "landmarks.csv", index=False
    )


class ExternalWindowCacheTests(unittest.TestCase):
    def test_cache_shapes_finite_channels_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_sequence(source)
            output = root / "cache"
            statistics = build_physical_cache(
                PhysicalExerciseRecognitionAdapter(source), output, {"7": "train"}
            )
            self.assertEqual(statistics["cached_videos"], 1)
            with np.load(output / "video_0007.npz", allow_pickle=False) as archive:
                motion = archive["motionbert_input"]
                geometry = archive["geometry"]
                metadata = json.loads(str(archive["metadata_json"]))
            self.assertEqual(motion.shape[1:], (30, 17, 3))
            self.assertEqual(geometry.shape[1:], (30, 33, 4))
            self.assertEqual(motion.dtype, np.float32)
            self.assertTrue(np.isfinite(motion).all())
            self.assertEqual(metadata["motionbert_channels"], ["x", "y", "confidence"])
            self.assertEqual(metadata["geometry_channels"], ["raw_x", "raw_y", "raw_z", "confidence"])
            self.assertEqual(metadata["root_joint"], 0)
            self.assertIn("body scale", metadata["coordinate_normalization"])

    def test_preprocessing_version_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_sequence(source)
            output = root / "cache"
            output.mkdir()
            (output / "cache_statistics.json").write_text(
                json.dumps({"preprocessing_version": "obsolete_v1"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "incompatible cache"):
                build_physical_cache(
                    PhysicalExerciseRecognitionAdapter(source), output, {"7": "train"}
                )

    def test_short_video_is_padded_with_invalid_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_sequence(source, frames=20)
            output = root / "cache"
            statistics = build_physical_cache(
                PhysicalExerciseRecognitionAdapter(source), output, {"7": "test"}
            )
            self.assertEqual(statistics["cached_videos"], 1)
            self.assertEqual(statistics["dropped_videos_shorter_than_window"], [])
            with np.load(output / "video_0007.npz", allow_pickle=False) as archive:
                motion = archive["motionbert_input"]
                padding = archive["padding_mask"]
                frames = archive["frame_indices"]
            self.assertEqual(motion.shape, (1, 30, 17, 3))
            self.assertTrue(padding[0, 20:].all())
            self.assertTrue((frames[0, 20:] == -1).all())
            np.testing.assert_array_equal(motion[0, 20:, :, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
