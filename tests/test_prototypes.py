from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototypes.builder import PrototypeBuilder
from prototypes.similarity import SimilarityEvaluator
from prototypes.store import PrototypeStore


class PrototypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.videos = {
            "a": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            "b": np.asarray([[0.9, 0.1]], dtype=np.float32),
            "c": np.asarray([[0.8, 0.2]], dtype=np.float32),
        }

    def test_mean_medoid_and_trimmed_mean_are_normalized(self) -> None:
        for strategy in ("mean", "medoid", "trimmed_mean"):
            artifact = PrototypeBuilder(
                strategy=strategy,
                min_reference_videos=3,
                trim_fraction=0.2,
                reject_outliers=False,
            ).build("squat", self.videos)
            self.assertAlmostEqual(float(np.linalg.norm(artifact.prototype)), 1.0, places=6)
            self.assertEqual(artifact.metadata["prototype_strategy"], strategy)
            self.assertEqual(artifact.metadata["number_of_videos"], 3)

    def test_outlier_rejection(self) -> None:
        videos = dict(self.videos)
        videos["d"] = np.asarray([[-1.0, 0.0]], dtype=np.float32)
        artifact = PrototypeBuilder(
            min_reference_videos=3,
            reject_outliers=True,
            outlier_mad_threshold=2.0,
        ).build("squat", videos)
        self.assertIn("d", artifact.metadata["rejected_video_ids"])

    def test_safe_save_load_preserves_metadata(self) -> None:
        artifact = PrototypeBuilder(min_reference_videos=3, reject_outliers=False).build(
            "squat", self.videos, preprocessing_version="h36m_xy_conf_v1"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = PrototypeStore.save(Path(folder) / "squat.npz", artifact)
            loaded = PrototypeStore.load(path)
        np.testing.assert_allclose(loaded.prototype, artifact.prototype)
        self.assertEqual(loaded.metadata["exercise_id"], "squat")
        self.assertEqual(loaded.metadata["preprocessing_version"], "h36m_xy_conf_v1")

    def test_cosine_reference_cases(self) -> None:
        artifact = PrototypeBuilder(min_reference_videos=3, reject_outliers=False).build(
            "test",
            {"a": np.asarray([[1.0, 0.0]]), "b": np.asarray([[1.0, 0.0]]), "c": np.asarray([[1.0, 0.0]])},
        )
        evaluator = SimilarityEvaluator()
        self.assertAlmostEqual(evaluator.evaluate(np.asarray([1.0, 0.0]), artifact)["similarity"], 1.0, places=6)
        self.assertAlmostEqual(evaluator.evaluate(np.asarray([0.0, 1.0]), artifact)["similarity"], 0.0, places=6)
        self.assertAlmostEqual(evaluator.evaluate(np.asarray([-1.0, 0.0]), artifact)["similarity"], -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
