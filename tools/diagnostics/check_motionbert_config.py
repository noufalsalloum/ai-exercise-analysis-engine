"""Print the tensor shapes stored in the active MotionBERT checkpoint."""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "latest_epoch.bin"


checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
state = checkpoint["model_pos"]

print("=" * 70)
print("MotionBERT Parameters")
print("=" * 70)

for key, tensor in state.items():
    print(key, "->", tuple(tensor.shape))
