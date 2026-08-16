from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from models.exercise_representation import ExerciseRepresentationModel


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class ExternalCheckpointRoundtripTests(unittest.TestCase):
    def test_weights_only_strict_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ExerciseRepresentationModel(
                5, backbone=TinyBackbone(), temporal_dim=32, global_dim=64,
                dropout=0.0, freeze_backbone=True,
            )
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "class_vocabulary": ["a", "b", "c", "d", "e"]}, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored = ExerciseRepresentationModel(
                5, backbone=TinyBackbone(), temporal_dim=32, global_dim=64,
                dropout=0.0, freeze_backbone=True,
            )
            incompatible = restored.load_state_dict(payload["model_state_dict"], strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            for original, loaded in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(original, loaded)


if __name__ == "__main__":
    unittest.main()
