from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.landmark_selector import (  # noqa: E402
    H36M_JOINT_NAMES,
    LandmarkSelector,
)


def _make_synthetic_frame(frame_offset: float = 0.0) -> np.ndarray:
    """Give every MediaPipe landmark a unique, traceable value."""

    indices = np.arange(33, dtype=np.float32)
    frame = np.empty((33, 4), dtype=np.float32)
    frame[:, 0] = 100.0 + indices * 3.0 + frame_offset
    frame[:, 1] = -200.0 - indices * 5.0 - frame_offset
    frame[:, 2] = 10_000.0 + indices * 11.0 + frame_offset
    frame[:, 3] = 0.01 + indices * 0.02
    return frame


def _expected_h36m(frame: np.ndarray) -> np.ndarray:
    """Build the official H36M order independently from the implementation."""

    xy_conf = frame[:, [0, 1, 3]].astype(np.float32)

    def derived(*sources: np.ndarray) -> np.ndarray:
        stacked = np.stack(sources)
        result = np.empty(3, dtype=np.float32)
        result[:2] = stacked[:, :2].mean(axis=0)
        result[2] = stacked[:, 2].min()
        return result

    root = derived(xy_conf[23], xy_conf[24])
    neck = derived(xy_conf[11], xy_conf[12])

    return np.stack(
        [
            root,
            xy_conf[24],
            xy_conf[26],
            xy_conf[28],
            xy_conf[23],
            xy_conf[25],
            xy_conf[27],
            derived(root, neck),
            neck,
            xy_conf[0],
            derived(xy_conf[2], xy_conf[5]),
            xy_conf[11],
            xy_conf[13],
            xy_conf[15],
            xy_conf[12],
            xy_conf[14],
            xy_conf[16],
        ],
        axis=0,
    ).astype(np.float32)


class MotionBERTLandmarkContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = LandmarkSelector(
            {
                "landmarks": {
                    "selected_landmarks": [
                        "LEFT_SHOULDER",
                        "RIGHT_SHOULDER",
                        "LEFT_HIP",
                        "RIGHT_HIP",
                    ]
                }
            }
        )

    def assert_contract(self, actual: np.ndarray, expected: np.ndarray) -> None:
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(actual).all())

        for joint_index, joint_name in enumerate(H36M_JOINT_NAMES):
            with self.subTest(joint_index=joint_index, joint_name=joint_name):
                np.testing.assert_allclose(
                    actual[..., joint_index, :],
                    expected[..., joint_index, :],
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_single_frame_maps_every_h36m_joint(self) -> None:
        frame = _make_synthetic_frame()
        original = frame.copy()

        actual = self.selector.to_h36m_17(frame)
        expected = _expected_h36m(frame)

        self.assert_contract(actual, expected)
        self.assertEqual(actual.shape, (17, 3))
        np.testing.assert_array_equal(frame, original)

        # Unique source values make an accidental duplicate joint detectable.
        self.assertEqual(np.unique(actual, axis=0).shape[0], 17)

        # The third MotionBERT channel must be visibility/confidence, never z.
        self.assertTrue(np.all((actual[:, 2] >= 0.0) & (actual[:, 2] <= 1.0)))
        self.assertFalse(np.isin(actual[:, 2], frame[:, 2]).any())

    def test_sequence_maps_every_frame_and_joint(self) -> None:
        sequence = np.stack(
            [_make_synthetic_frame(0.0), _make_synthetic_frame(7.0)],
            axis=0,
        )
        original = sequence.copy()

        actual = self.selector.to_h36m_17(sequence)
        expected = np.stack([_expected_h36m(frame) for frame in sequence], axis=0)

        self.assert_contract(actual, expected)
        self.assertEqual(actual.shape, (2, 17, 3))
        np.testing.assert_array_equal(sequence, original)

    def test_geometry_selection_preserves_xy_z_and_visibility(self) -> None:
        frame = _make_synthetic_frame()

        selected = self.selector.select_landmarks(frame)

        np.testing.assert_array_equal(selected, frame[[11, 12, 23, 24], :])
        self.assertEqual(selected.shape, (4, 4))
        np.testing.assert_array_equal(selected[:, 2], frame[[11, 12, 23, 24], 2])

    def test_conversion_rejects_missing_confidence_channel(self) -> None:
        frame_without_visibility = _make_synthetic_frame()[:, :3]

        with self.assertRaisesRegex(ValueError, "visibility must not be replaced by z"):
            self.selector.to_h36m_17(frame_without_visibility)

    def test_derived_joint_confidence_uses_source_minimum(self) -> None:
        frame = _make_synthetic_frame()
        frame[:, 3] = 1.0
        frame[23, 3], frame[24, 3] = 0.2, 0.8
        frame[11, 3], frame[12, 3] = 0.4, 0.9
        frame[2, 3], frame[5, 3] = 0.3, 0.7

        actual = self.selector.to_h36m_17(frame)

        self.assertAlmostEqual(float(actual[0, 2]), 0.2)   # root
        self.assertAlmostEqual(float(actual[8, 2]), 0.4)   # neck
        self.assertAlmostEqual(float(actual[7, 2]), 0.2)   # belly(root, neck)
        self.assertAlmostEqual(float(actual[10, 2]), 0.3)  # head from both eyes


if __name__ == "__main__":
    unittest.main()
