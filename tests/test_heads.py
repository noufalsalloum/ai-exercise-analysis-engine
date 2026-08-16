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
    from heads.error_head import ERROR_VOCABULARY, ErrorHead
    from heads.passfail_head import PassFailHead
    from heads.phase_head import PHASE_VOCABULARY, PhaseHead


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter.")
class HeadTests(unittest.TestCase):
    def test_phase_shape_and_valid_mask(self) -> None:
        head = PhaseHead(input_dim=32, hidden_dim=16, dropout=0.0).eval()
        inputs = torch.randn(2, 7, 32)
        mask = PhaseHead.build_valid_phase_mask(["REST", "DESCENDING"])
        logits = head(inputs, valid_phase_mask=mask)
        self.assertEqual(logits.shape, (2, 7, len(PHASE_VOCABULARY)))
        invalid = ~mask
        self.assertTrue(torch.all(logits[..., invalid] == torch.finfo(logits.dtype).min))
        prediction = head.predict(logits)
        self.assertEqual(prediction["predictions"].shape, (2, 7))

    def test_passfail_and_error_shapes(self) -> None:
        global_embedding = torch.randn(3, 64)
        temporal = torch.randn(3, 5, 32)
        pass_logits = PassFailHead(global_dim=64, temporal_dim=32, hidden_dim=24)(
            global_embedding,
            temporal,
        )
        self.assertEqual(pass_logits.shape, (3, 2))
        error_head = ErrorHead(global_dim=64, temporal_dim=32, hidden_dim=24)
        valid = ErrorHead.build_valid_error_mask(["knee_valgus"])
        error_logits = error_head(global_embedding, temporal, valid_error_mask=valid)
        self.assertEqual(error_logits.shape, (3, len(ERROR_VOCABULARY)))
        self.assertTrue(torch.all(error_logits[:, ~valid] == torch.finfo(error_logits.dtype).min))


if __name__ == "__main__":
    unittest.main()
