from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.phase_rep_counter import PhaseRepCounter


def repeated(phases: list[str], count: int = 2) -> list[str]:
    return [phase for phase in phases for _ in range(count)]


class PhaseRepCounterTests(unittest.TestCase):
    def test_complete_squat_cycle_counts_one(self) -> None:
        phases = repeated(["REST", "DESCENDING", "BOTTOM", "ASCENDING", "REST"])
        result = PhaseRepCounter("squat", smoothing_window=1).evaluate(phases)
        self.assertEqual(result["count"], 1)

    def test_invalid_jump_does_not_count(self) -> None:
        phases = repeated(["REST", "BOTTOM", "REST"])
        result = PhaseRepCounter("squat", smoothing_window=1).evaluate(phases)
        self.assertEqual(result["count"], 0)

    def test_probability_input_and_plank_hold(self) -> None:
        vocabulary = ("REST", "HOLD")
        probabilities = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32)
        result = PhaseRepCounter(
            "plank", vocabulary, smoothing_window=1, min_phase_frames=2
        ).evaluate(probabilities, timestamps=[0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(result["mode"], "hold")
        self.assertAlmostEqual(result["hold_duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
