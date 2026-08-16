from __future__ import annotations

import unittest

import numpy as np
import torch

from models.squat_rep_boundary import SquatRepBoundaryModel
from training.squat_rep_boundary import match_segments, probabilities_to_segments


class SquatRepBoundaryTests(unittest.TestCase):
    def test_model_shape_and_gradient(self) -> None:
        model = SquatRepBoundaryModel(channels=16, dropout=0.0)
        inputs = torch.randn(2, 64, 17, 3)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (2, 64))
        output.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_segment_postprocessing_and_matching(self) -> None:
        probabilities = np.zeros(160, dtype=np.float32)
        probabilities[20:61] = 0.95
        probabilities[90:141] = 0.9
        segments = probabilities_to_segments(
            probabilities, threshold=0.5, smoothing_kernel=1, min_length=15, merge_gap=2
        )
        self.assertEqual(segments, [(20, 60), (90, 140)])
        matches = match_segments(segments, [(19, 61), (92, 142)], iou_threshold=0.5)
        self.assertEqual(len(matches), 2)

    def test_short_noise_is_not_a_repetition(self) -> None:
        probabilities = np.zeros(100, dtype=np.float32)
        probabilities[30:34] = 1.0
        self.assertEqual(
            probabilities_to_segments(probabilities, 0.5, 1, min_length=10, merge_gap=0), []
        )


if __name__ == "__main__":
    unittest.main()

