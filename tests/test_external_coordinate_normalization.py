from __future__ import annotations

import unittest

import numpy as np

from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer


def make_h36m(frames: int = 4) -> np.ndarray:
    values = np.zeros((frames, 17, 3), dtype=np.float32)
    template = np.asarray(
        [
            [0, 0], [1, 0], [1, -2], [1, -4], [-1, 0], [-1, -2], [-1, -4],
            [0, 1], [0, 2], [0, 2.5], [0, 3], [-2, 2], [-3, 1], [-4, 0],
            [2, 2], [3, 1], [4, 0],
        ],
        dtype=np.float32,
    )
    for frame in range(frames):
        values[frame, :, :2] = template + np.asarray([100 + frame, -50 + frame])
    values[..., 2] = 1.0
    return values


class ExternalCoordinateNormalizationTests(unittest.TestCase):
    def test_root_centering_scale_invariance_and_finite_values(self) -> None:
        normalizer = H36MCoordinateNormalizer()
        source = make_h36m()
        normalized, diagnostics = normalizer.normalize(source)
        scaled = source.copy()
        scaled[..., :2] = scaled[..., :2] * 10.0 + 500.0
        normalized_scaled, _ = normalizer.normalize(scaled)

        np.testing.assert_allclose(normalized[:, 0, :2], 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized[..., :2], normalized_scaled[..., :2], atol=2e-5)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertGreater(diagnostics.sequence_scale, 1e-6)
        self.assertFalse(diagnostics.near_zero_scale_mask.any())

    def test_outlier_is_clipped_logged_and_lower_confidence(self) -> None:
        source = make_h36m(5)
        source[3, 16, 0] += 10_000.0
        normalized, diagnostics = H36MCoordinateNormalizer().normalize(source)

        self.assertTrue(diagnostics.clipped_mask[3])
        self.assertTrue(diagnostics.outlier_mask[3])
        self.assertLessEqual(float(np.abs(normalized[..., :2]).max()), 4.0)
        self.assertTrue(np.all(normalized[3, :, 2] <= 0.25))

    def test_near_zero_body_scale_marks_invalid_confidence(self) -> None:
        source = np.zeros((2, 17, 3), dtype=np.float32)
        source[..., 2] = 1.0
        normalized, diagnostics = H36MCoordinateNormalizer().normalize(source)

        self.assertTrue(diagnostics.near_zero_scale_mask.all())
        np.testing.assert_array_equal(normalized[..., 2], 0.0)
        self.assertTrue(np.isfinite(normalized).all())


if __name__ == "__main__":
    unittest.main()
