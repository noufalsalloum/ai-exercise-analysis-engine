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


class FrozenMotionBERTTrainingTests(unittest.TestCase):
    def test_backbone_stays_eval_frozen_and_has_no_gradients(self) -> None:
        model = ExerciseRepresentationModel(
            3, backbone=TinyBackbone(), temporal_dim=32, global_dim=64,
            dropout=0.0, freeze_backbone=True,
        )
        model.train()
        loss = model(torch.randn(2, 30, 17, 3))["logits"].sum()
        loss.backward()

        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
