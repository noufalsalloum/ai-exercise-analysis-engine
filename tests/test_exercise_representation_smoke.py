from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.exercise_representation import ExerciseRepresentationModel


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class ExerciseRepresentationSmokeTests(unittest.TestCase):
    def test_finite_loss_shapes_and_gradient_flow(self) -> None:
        torch.manual_seed(42)
        model = ExerciseRepresentationModel(
            5, backbone=TinyBackbone(), temporal_dim=64, global_dim=128,
            dropout=0.0, freeze_backbone=True,
        )
        inputs = torch.randn(3, 30, 17, 3)
        labels = torch.tensor([0, 1, 2])
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs["logits"], labels)
        loss.backward()

        self.assertEqual(outputs["logits"].shape, (3, 5))
        self.assertEqual(outputs["temporal_embedding"].shape, (3, 30, 64))
        self.assertEqual(outputs["global_embedding"].shape, (3, 128))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.shared_expert.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.classifier.parameters()))

    def test_precomputed_backbone_features_follow_same_output_contract(self) -> None:
        model = ExerciseRepresentationModel(
            5, backbone=TinyBackbone(), temporal_dim=64, global_dim=128,
            dropout=0.0, freeze_backbone=True,
        )
        outputs = model.forward_features(torch.randn(2, 30, 17, 512))
        self.assertEqual(outputs["logits"].shape, (2, 5))
        self.assertEqual(outputs["temporal_embedding"].shape, (2, 30, 64))


if __name__ == "__main__":
    unittest.main()
