from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from models.squat_correctness import SquatCorrectnessModel


class FakeMotionBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class SquatCorrectnessModelTests(unittest.TestCase):
    def test_contract_and_frozen_backbone(self) -> None:
        model = SquatCorrectnessModel(backbone=FakeMotionBERT(), dropout=0.0)
        inputs = torch.randn(2, 12, 17, 3)
        mask = torch.ones(2, 12, dtype=torch.bool)
        output = model(inputs, mask)
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertEqual(tuple(output["correct_probability"].shape), (2,))
        self.assertEqual(tuple(output["temporal_embedding"].shape), (2, 12, 512))
        self.assertEqual(tuple(output["global_embedding"].shape), (2, 1024))
        output["logits"].sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.expert.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.correctness_head.parameters()))

    def test_shape_validation(self) -> None:
        model = SquatCorrectnessModel(backbone=FakeMotionBERT())
        with self.assertRaises(ValueError):
            model(torch.randn(2, 12, 16, 3))
        with self.assertRaises(ValueError):
            model.forward_features(torch.randn(2, 12, 17, 511))


if __name__ == "__main__":
    unittest.main()

