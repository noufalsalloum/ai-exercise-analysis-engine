from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "latest_epoch.bin"


class CheckpointTests(unittest.TestCase):
    def test_known_motionbert_lite_checkpoint_hash(self) -> None:
        self.assertTrue(CHECKPOINT.is_file())
        digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461",
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
    def test_checkpoint_has_model_pos_without_unsafe_fallback(self) -> None:
        import torch

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.assertIn("model_pos", checkpoint)


if __name__ == "__main__":
    unittest.main()
