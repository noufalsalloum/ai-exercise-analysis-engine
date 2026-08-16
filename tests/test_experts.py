from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from experts.plank_expert import PlankExpert
    from experts.registry import ExpertRegistry


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter.")
class ExpertOutputTests(unittest.TestCase):
    def test_multiple_outputs_attention_and_masks(self) -> None:
        model = PlankExpert(dropout=0.1).eval()
        inputs = torch.randn(2, 8, 17, 512)
        mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
        with torch.no_grad():
            output = model(inputs, temporal_mask=mask)
        self.assertEqual(output["temporal_embedding"].shape, (2, 8, 512))
        self.assertEqual(output["global_embedding"].shape, (2, 1024))
        self.assertEqual(output["joint_attention"].shape, (2, 8, 17))
        self.assertEqual(output["temporal_attention"].shape, (2, 8))
        self.assertTrue(torch.allclose(output["temporal_attention"][1, 5:], torch.zeros(3)))
        self.assertTrue(torch.allclose(output["temporal_embedding"][1, 5:], torch.zeros(3, 512)))
        self.assertTrue(torch.isfinite(output["global_embedding"]).all())

    def test_gradient_flow_reaches_adapter_and_attention(self) -> None:
        model = PlankExpert(dropout=0.0)
        output = model(torch.randn(2, 5, 17, 512))
        loss = output["global_embedding"].square().mean() + output["temporal_embedding"].abs().mean()
        loss.backward()
        self.assertIsNotNone(model.exercise_adapter.exercise_token.grad)
        self.assertGreater(float(model.exercise_adapter.exercise_token.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.joint_attention[0].weight.grad)

    def test_eval_is_deterministic(self) -> None:
        torch.manual_seed(3)
        model = PlankExpert(dropout=0.4).eval()
        inputs = torch.randn(1, 6, 17, 512)
        with torch.no_grad():
            first = model(inputs)["global_embedding"]
            second = model(inputs)["global_embedding"]
        self.assertTrue(torch.equal(first, second))

    def test_registry_is_persistent_module_dict(self) -> None:
        registry = ExpertRegistry(dropout=0.0)
        self.assertIsInstance(registry.experts, torch.nn.ModuleDict)
        first = registry.get("push-up")
        second = registry.get("pushup")
        self.assertIs(first, second)
        names = dict(registry.named_parameters())
        self.assertTrue(any(name.startswith("experts.pushup") for name in names))
        self.assertIsNot(
            registry.get("pushup").exercise_adapter.exercise_token,
            registry.get("squat").exercise_adapter.exercise_token,
        )


if __name__ == "__main__":
    unittest.main()
