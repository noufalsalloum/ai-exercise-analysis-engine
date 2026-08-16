from __future__ import annotations

from collections import Counter
import unittest
from pathlib import Path

from datasets.adapters.rehab24 import Rehab24SquatAdapter
from datasets.adapters.rehab24_split import balanced_subject_split


class Rehab24SquatAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1] / "datasets/external"
        cls.samples = Rehab24SquatAdapter(root).samples()

    def test_contract_counts_pairing_and_boundaries(self) -> None:
        self.assertEqual(len(self.samples), 390)
        pairs = {}
        for sample in self.samples:
            pairs.setdefault(sample.pair_id, []).append(sample)
            self.assertIn(sample.correctness, (0, 1))
            self.assertGreaterEqual(sample.start_frame, 1)
            self.assertLessEqual(sample.end_frame, sample.frame_count)
            self.assertEqual(sample.fps, 30.0)
        self.assertEqual(len(pairs), 195)
        self.assertTrue(all({item.camera_id for item in values} == {"Camera17", "Camera18"} for values in pairs.values()))

    def test_subject_split_has_no_leakage(self) -> None:
        assignment, evidence = balanced_subject_split(self.samples, 42)
        self.assertEqual(Counter(assignment.values()), Counter({"train": 6, "validation": 1, "test": 2}))
        self.assertEqual(set(assignment), {str(value) for value in range(1, 10)})
        for stats in evidence["splits"].values():
            self.assertGreater(stats["correct"], 0)
            self.assertGreater(stats["incorrect"], 0)


if __name__ == "__main__":
    unittest.main()
