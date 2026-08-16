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
    from models.exercise_aqa_model import ExerciseAQAModel


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter.")
class UnifiedAQAModelTests(unittest.TestCase):
    def test_backbone_expert_and_requested_phase_head(self) -> None:
        model = ExerciseAQAModel(dropout=0.0).eval()
        inputs = torch.randn(1, 4, 17, 3)
        with torch.no_grad():
            output = model(inputs, "squat", tasks={"phase"})
        self.assertEqual(output["motionbert_features"].shape, (1, 4, 17, 512))
        self.assertEqual(output["temporal_embedding"].shape, (1, 4, 512))
        self.assertEqual(output["global_embedding"].shape, (1, 1024))
        self.assertEqual(output["phase_logits"].shape[:2], (1, 4))
        self.assertFalse(output["head_status"]["phase"])


if __name__ == "__main__":
    unittest.main()
