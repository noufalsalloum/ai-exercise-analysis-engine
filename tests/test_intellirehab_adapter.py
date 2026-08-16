from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets.adapters.intellirehabds_adapter import (
    INTELLIREHAB_TO_H36M,
    IntelliRehabDSAdapter,
)


class IntelliRehabAdapterTests(unittest.TestCase):
    def test_filename_labels_shape_and_unsafe_h36m_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            simplified = Path(directory) / "SkeletonData" / "Simplified"
            simplified.mkdir(parents=True)
            path = simplified / "101_18_2_4_2_stand.txt"
            np.savetxt(path, np.arange(225, dtype=np.float32).reshape(3, 75), delimiter=",")
            adapter = IntelliRehabDSAdapter(Path(directory))

            sequence = adapter.load_sequence(path)
            sample = next(iter(adapter.iter_samples()))

            self.assertEqual(sequence.shape, (3, 25, 3))
            self.assertEqual(sample.subject_id, "101")
            self.assertEqual(sample.exercise_label, "shoulder_flexion_left")
            self.assertEqual(sample.correctness_label, 0)
            self.assertEqual(sample.rep_id, "4")
            missing = [row for row in INTELLIREHAB_TO_H36M if row["mapping"] == "missing"]
            self.assertEqual(missing, [{"h36m_joint": "nose", "source_joint": "", "mapping": "missing"}])


if __name__ == "__main__":
    unittest.main()
