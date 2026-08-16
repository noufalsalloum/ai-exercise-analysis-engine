from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from training.exercise_representation_pilot import (
    PilotSplitGuard,
    PilotWindowDataset,
    VideoBalancedSampler,
    aggregate_video_logits,
)


class _SyntheticVideoDataset:
    video_ids = ("1", "2", "3")
    video_to_indices = {"1": [0], "2": [1, 2, 3], "3": [4, 5]}


class ExerciseRepresentationPilotTests(unittest.TestCase):
    def test_video_level_mean_logit_aggregation(self) -> None:
        logits = np.asarray([[2.0, 0.0], [0.0, 4.0], [3.0, 1.0]], dtype=np.float32)
        means, labels, video_ids = aggregate_video_logits(
            logits, [1, 1, 0], ["7", "7", "9"]
        )
        np.testing.assert_allclose(means[0], [1.0, 2.0])
        np.testing.assert_allclose(means[1], [3.0, 1.0])
        self.assertEqual(labels, [1, 0])
        self.assertEqual(video_ids, ["7", "9"])

    def test_video_balanced_sampler_equalizes_contribution(self) -> None:
        sampler = VideoBalancedSampler(_SyntheticVideoDataset(), windows_per_video=2, seed=42)  # type: ignore[arg-type]
        indices = list(iter(sampler))
        counts = {
            video_id: sum(index in candidates for index in indices)
            for video_id, candidates in _SyntheticVideoDataset.video_to_indices.items()
        }
        self.assertEqual(counts, {"1": 2, "2": 2, "3": 2})
        self.assertEqual(len(indices), 6)

    def test_test_split_is_blocked_until_training_finishes(self) -> None:
        guard = PilotSplitGuard()
        guard.assert_evaluation_allowed("validation")
        with self.assertRaisesRegex(RuntimeError, "prohibited"):
            guard.assert_evaluation_allowed("test")
        guard.mark_training_complete()
        guard.assert_evaluation_allowed("test")

    def test_real_v4_cache_contains_full_67_test_videos(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cache_dir = project_root / "datasets" / "window_cache" / "physical_exercise_recognition_v4"
        if not (cache_dir / "cache_manifest.csv").is_file():
            self.skipTest("v4 external cache is not present.")
        dataset = PilotWindowDataset(cache_dir, "test")
        self.assertEqual(len(dataset.video_ids), 67)
        self.assertEqual(len(dataset), 1177)


if __name__ == "__main__":
    unittest.main()
