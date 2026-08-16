"""Contracts for the independent static Squat Error V1 pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.adapters.squat_dataset_adapter import SquatDatasetAdapter
from inference.squat_error_inference import SquatErrorImageInference
from models.squat_posture_error import (
    ERROR_CLASSES,
    SquatPostureErrorModel,
    load_squat_posture_error_checkpoint,
)
from preprocessing.squat_posture_features import SquatPostureFeatureExtractor
from training.squat_error_v1 import make_development_split


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "external" / "SquatDataset"
CHECKPOINT = ROOT / "checkpoints" / "squat_error_v1" / "best.pt"
POSE_MODEL = Path(r"C:\MediaPipe\pose_landmarker_full.task")


class SquatErrorV1UnitTests(unittest.TestCase):
    def test_class_normalization(self) -> None:
        adapter = SquatDatasetAdapter(DATASET)
        expected = {
            "Good": "good",
            "Bad back": "bad_back",
            "Bad Back": "bad_back",
            "Bad heel": "bad_heel",
            "Bad Heel": "bad_heel",
        }
        for folder, label in expected.items():
            self.assertEqual(adapter.normalized_class(Path(folder) / "sample.jpg"), label)

    def test_pose_feature_shape_and_finite_contract(self) -> None:
        extractor = SquatPostureFeatureExtractor()
        landmarks = np.zeros((33, 4), dtype=np.float32)
        landmarks[:, 0] = np.linspace(0.1, 0.9, 33)
        landmarks[:, 1] = np.linspace(0.9, 0.1, 33)
        landmarks[:, 3] = 0.8
        features = extractor.extract(landmarks)
        self.assertEqual(features.shape, (64,))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.isfinite(extractor.extract(None)).all())

    def test_split_is_disjoint_and_official_test_is_locked(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": [f"train-{i}" for i in range(30)] + [f"test-{i}" for i in range(6)],
                "source_split": ["train"] * 30 + ["test"] * 6,
                "label_index": ([0, 1, 2] * 10) + ([0, 1, 2] * 2),
            }
        )
        split = make_development_split(frame, seed=42)
        train_ids = set(split.loc[split.development_split == "train", "sample_id"])
        validation_ids = set(split.loc[split.development_split == "validation", "sample_id"])
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(
            set(split.loc[split.source_split == "test", "development_split"]),
            {"test_locked"},
        )

    def test_output_has_exactly_three_classes_and_no_score(self) -> None:
        model = SquatPostureErrorModel(input_dim=64)
        output = model(torch.zeros(2, 64))
        self.assertEqual(tuple(output.shape), (2, 3))
        prediction = model.predict(torch.zeros(1, 64))
        self.assertEqual(tuple(ERROR_CLASSES), ("good", "bad_back", "bad_heel"))
        self.assertIsNone(prediction["score"])

    def test_checkpoint_roundtrip_is_strict(self) -> None:
        model = SquatPostureErrorModel(input_dim=64, hidden_dim=16, dropout=0.1)
        payload = {
            "model_type": "small_mlp",
            "model_state_dict": model.state_dict(),
            "input_dim": 64,
            "hidden_dim": 16,
            "dropout": 0.1,
            "class_vocabulary": {str(i): name for i, name in enumerate(ERROR_CLASSES)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(payload, path)
            loaded, _ = load_squat_posture_error_checkpoint(path, torch.device("cpu"))
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            broken = dict(payload)
            broken["model_state_dict"] = dict(payload["model_state_dict"])
            broken["model_state_dict"].pop(next(iter(broken["model_state_dict"])))
            torch.save(broken, path)
            with self.assertRaises(RuntimeError):
                load_squat_posture_error_checkpoint(path, torch.device("cpu"))


@unittest.skipUnless(CHECKPOINT.is_file() and POSE_MODEL.is_file() and DATASET.is_dir(), "Squat Error V1 artifacts unavailable")
class SquatErrorV1ImageInferenceTests(unittest.TestCase):
    def test_inference_on_one_image_from_each_class(self) -> None:
        manifest_path = ROOT / "results" / "squat_error_v1" / "data" / "feature_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        samples = []
        for name in ERROR_CLASSES:
            row = manifest[(manifest.canonical_label == name) & manifest.pose_success].iloc[0]
            samples.append((name, Path(row.image_path)))
        with SquatErrorImageInference(CHECKPOINT, POSE_MODEL) as inference:
            for expected, path in samples:
                result = inference.predict_image(path)
                self.assertTrue(result["available"], msg=f"{expected}: {path}")
                self.assertEqual(set(result["probabilities"]), set(ERROR_CLASSES))
                self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
                self.assertIn(result["predicted_error"], ERROR_CLASSES)
                self.assertIsNone(result["score"])


if __name__ == "__main__":
    unittest.main()
