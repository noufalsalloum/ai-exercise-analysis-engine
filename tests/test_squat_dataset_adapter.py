from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from datasets.adapters.squat_dataset_adapter import SquatDatasetAdapter


class SquatDatasetAdapterTests(unittest.TestCase):
    def test_static_mutually_exclusive_labels_and_source_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, label in (("train", "Good"), ("test", "Bad Back"), ("test", "Bad Heel")):
                folder = root / split / label
                folder.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", np.zeros((8, 9, 3), dtype=np.uint8))
                self.assertTrue(ok)
                encoded.tofile(folder / f"{label}.jpg")
            adapter = SquatDatasetAdapter(root)
            samples = list(adapter.iter_samples())

            self.assertEqual(len(samples), 3)
            self.assertTrue(all(sample.input_type == "static_rgb_image" for sample in samples))
            self.assertEqual({sample.source_split for sample in samples}, {"train", "test"})
            labels = {tuple(sample.error_labels) for sample in samples}
            self.assertEqual(labels, {(), ("bad_back",), ("bad_heel",)})
            self.assertEqual(adapter.audit()["quality_checks"]["invalid_images"], [])


if __name__ == "__main__":
    unittest.main()
