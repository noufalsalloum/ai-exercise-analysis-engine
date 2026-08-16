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
    from training.losses import MultiTaskLoss


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter.")
class TrainingComponentTests(unittest.TestCase):
    def test_loss_uses_only_available_tasks(self) -> None:
        criterion = MultiTaskLoss()
        outputs = {
            "global_embedding": torch.randn(2, 1024),
            "passfail_logits": torch.randn(2, 2, requires_grad=True),
        }
        batch = {
            "pass_fail_available": torch.tensor([True, False]),
            "pass_fail_labels": torch.tensor([0, 1]),
            "phase_available": torch.tensor([False, False]),
            "errors_available": torch.tensor([False, False]),
        }
        total, losses = criterion(outputs, batch)
        self.assertIn("pass_fail", losses)
        self.assertNotIn("phase", losses)
        self.assertNotIn("errors", losses)
        total.backward()
        self.assertIsNotNone(outputs["passfail_logits"].grad)


if __name__ == "__main__":
    unittest.main()
