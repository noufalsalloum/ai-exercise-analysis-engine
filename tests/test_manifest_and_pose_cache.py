from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.build_manifest import build_manifest_rows, normalize_exercise_id, write_manifest
from preprocessing.pose_cache import PoseCache
from preprocessing.smoothing import Smoother


class ManifestAndCacheTests(unittest.TestCase):
    def test_aliases_and_missing_labels(self) -> None:
        self.assertEqual(normalize_exercise_id("push-up"), "pushup")
        self.assertEqual(normalize_exercise_id("pull Up"), "pullup")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "pull Up" / "sample.mp4"
            video.parent.mkdir()
            video.touch()
            rows = build_manifest_rows(root)
            self.assertEqual(rows[0]["exercise_id"], "pullup")
            self.assertEqual(rows[0]["form_label"], "")
            self.assertEqual(rows[0]["split_group"], rows[0]["video_id"])
            output = write_manifest(rows, root / "manifest.csv")
            self.assertTrue(output.is_file())

    def test_pose_cache_round_trip(self) -> None:
        landmarks = np.arange(4 * 33 * 4, dtype=np.float32).reshape(4, 33, 4)
        with tempfile.TemporaryDirectory() as folder:
            path = PoseCache.save(Path(folder) / "pose.npz", landmarks, {"video": "x.mp4"})
            loaded, metadata = PoseCache.load(path)
        np.testing.assert_array_equal(loaded, landmarks)
        self.assertEqual(metadata["video"], "x.mp4")

    def test_nearest_edge_smoothing_preserves_visibility(self) -> None:
        sequence = np.zeros((3, 1, 4), dtype=np.float32)
        sequence[:, 0, 0] = [0.0, 3.0, 6.0]
        sequence[:, 0, 3] = [0.1, 0.5, 0.9]
        smoothed = Smoother(window_size=3).moving_average(sequence)
        np.testing.assert_allclose(smoothed[:, 0, 0], [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(smoothed[:, 0, 3], sequence[:, 0, 3])


if __name__ == "__main__":
    unittest.main()
