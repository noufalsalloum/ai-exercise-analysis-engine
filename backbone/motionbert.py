from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from .DSTformer import DSTformer
except ImportError:
    from backbone.DSTformer import DSTformer


class MotionBERT(nn.Module):
    """Strict MotionBERT-Lite feature wrapper.

    Contract: ``(B,T,17,3)`` x/y/confidence in, ``(B,T,17,512)`` out.
    The MediaPipe z/world branch is intentionally not accepted here.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        num_joints: int = 17,
        dim_in: int = 3,
        dim_out: int = 3,
        dim_feat: int = 256,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        maxlen: int = 243,
    ) -> None:
        super().__init__()
        if (num_joints, dim_in, dim_rep) != (17, 3, 512):
            raise ValueError(
                "This wrapper is fixed to the audited MB-Lite contract: "
                "num_joints=17, dim_in=3, dim_rep=512."
            )
        self.num_joints = num_joints
        self.dim_in = dim_in
        self.dim_rep = dim_rep
        self.maxlen = maxlen
        self.model = DSTformer(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_feat=dim_feat,
            dim_rep=dim_rep,
            depth=depth,
            num_heads=num_heads,
            num_joints=num_joints,
            maxlen=maxlen,
        )
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load the audited pretraining state strictly and without pickle objects."""

        checkpoint: Any = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint["model_pos"] if isinstance(checkpoint, dict) and "model_pos" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("MotionBERT checkpoint does not contain a state dictionary.")
        clean_state = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(clean_state, strict=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[2:] != (self.num_joints, self.dim_in):
            raise ValueError(
                f"Expected MotionBERT input (B,T,17,3), got {tuple(inputs.shape)}."
            )
        if inputs.shape[1] <= 0 or inputs.shape[1] > self.maxlen:
            raise ValueError(f"T must be between 1 and {self.maxlen}.")
        if not torch.is_floating_point(inputs):
            raise TypeError("MotionBERT input must be floating point.")
        return self.model(inputs, return_rep=True)
