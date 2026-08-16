from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.tools.audit_external_datasets import discover_dataset_roots


class ExternalDatasetAuditTests(unittest.TestCase):
    def test_signature_based_discovery_ignores_folder_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / "unexpected_physical_name"
            physical.mkdir()
            (physical / "labels.csv").write_text("vid_id,class\n", encoding="utf-8")
            (physical / "landmarks.csv").write_text("vid_id,frame_order\n", encoding="utf-8")
            simplified = root / "rehab_any_name" / "SkeletonData" / "Simplified"
            simplified.mkdir(parents=True)
            (simplified / "1_18_0_1_1_stand.txt").write_text("0\n", encoding="utf-8")
            squat = root / "postures"
            (squat / "train" / "Good").mkdir(parents=True)
            (squat / "test").mkdir()
            (squat / "train" / "Good" / "sample.jpg").write_bytes(b"jpg")

            discovered = discover_dataset_roots(root)

            self.assertEqual(discovered["physical_exercise_recognition"], physical.resolve())
            self.assertEqual(discovered["intellirehabds"], simplified.parent.parent.resolve())
            self.assertEqual(discovered["squat_dataset"], squat.resolve())


if __name__ == "__main__":
    unittest.main()
