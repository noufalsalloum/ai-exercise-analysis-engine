from __future__ import annotations

import unittest
from pathlib import Path

import torch

from datasets.adapters.rehab24_pushup import Rehab24PushupAdapter
from datasets.adapters.rehab24_pushup_split import balanced_pushup_subject_split
from models.pushup_correctness import PushupCorrectnessModel
from models.pushup_rep_boundary import PushupRepBoundaryModel


ROOT = Path(__file__).resolve().parents[1]


class PushupAIV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = Rehab24PushupAdapter(ROOT / "datasets" / "external").samples()

    def test_adapter_scope_and_counts(self) -> None:
        self.assertEqual(len(self.samples), 214)
        self.assertEqual(len({value.pair_id for value in self.samples}), 107)
        self.assertEqual({value.exercise_variation for value in self.samples}, {"table_incline_pushup"})
        labels = {value.correctness for value in self.samples}
        self.assertEqual(labels, {0, 1})

    def test_subject_split_has_no_leakage_and_locks_test(self) -> None:
        assignment, evidence = balanced_pushup_subject_split(self.samples, 42)
        self.assertEqual(sorted(evidence["splits"]["train"]["subjects"], key=int), sorted([key for key, value in assignment.items() if value == "train"], key=int))
        self.assertEqual([len(evidence["splits"][name]["subjects"]) for name in ("train", "validation", "test")], [7, 1, 2])
        self.assertTrue(evidence["test_locked_during_development"])

    def test_boundary_contract_and_finite_values(self) -> None:
        model = PushupRepBoundaryModel(channels=24, dropout=0.0)
        output = model(torch.zeros(2, 61, 17, 3))
        self.assertEqual(tuple(output["active_logits"].shape), (2, 61))
        self.assertEqual(tuple(output["boundary_logits"].shape), (2, 61))
        self.assertTrue(torch.isfinite(output["active_logits"]).all())

    def test_correctness_uses_pushup_expert_and_frozen_backbone(self) -> None:
        class Stub(torch.nn.Module):
            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return torch.zeros(*values.shape[:3], 512)

        model = PushupCorrectnessModel(backbone=Stub(), dropout=0.0)
        output = model(torch.zeros(2, 60, 17, 3), torch.ones(2, 60, dtype=torch.bool))
        self.assertEqual(model.expert.exercise_id, "pushup")
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))

    def test_final_checkpoints_load_strictly_and_preserve_scope(self) -> None:
        boundary_path = ROOT / "checkpoints/pushup_ai_v1/boundary/best.pt"
        correctness_path = ROOT / "checkpoints/pushup_ai_v1/correctness/final_dev.pt"
        boundary_state = torch.load(boundary_path, map_location="cpu", weights_only=True)
        boundary = PushupRepBoundaryModel(); boundary.load_state_dict(boundary_state["model_state_dict"], strict=True)
        correctness_state = torch.load(correctness_path, map_location="cpu", weights_only=True)
        model = PushupCorrectnessModel(ROOT / "models/latest_epoch.bin")
        model.expert.load_state_dict(correctness_state["expert_state_dict"], strict=True)
        model.correctness_head.load_state_dict(correctness_state["correctness_head_state_dict"], strict=True)
        self.assertTrue(boundary_state["test_locked_during_selection"])
        self.assertTrue(correctness_state["test_locked_during_selection"])
        self.assertIn("table/incline", correctness_state["exercise_scope"])

    def test_end_to_end_result_uses_predicted_boundaries(self) -> None:
        import json
        metrics = json.loads((ROOT / "results/pushup_ai_v1/end_to_end/metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["boundary_source"], "predicted")
        self.assertTrue(metrics["test_opened_once"])
        self.assertGreater(metrics["matched_repetitions"], 0)


if __name__ == "__main__":
    unittest.main()
