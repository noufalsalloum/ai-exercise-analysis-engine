"""Learned per-repetition Squat correctness model for REHAB24-6."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from backbone.motionbert import MotionBERT
from experts.squat_expert import SquatExpert


class SquatCorrectnessHead(nn.Module):
    """Classify a SquatExpert embedding as incorrect (0) or correct (1)."""

    def __init__(self, global_dim: int = 1024, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        if global_embedding.ndim != 2 or global_embedding.shape[-1] != 1024:
            raise ValueError(
                "Expected SquatExpert global embedding (B,1024), got "
                f"{tuple(global_embedding.shape)}."
            )
        return self.network(global_embedding)


class SquatCorrectnessModel(nn.Module):
    """Frozen MotionBERT, trainable SquatExpert, and trainable binary head."""

    def __init__(
        self,
        motionbert_checkpoint: str | Path | None = None,
        *,
        backbone: Optional[nn.Module] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone = backbone or MotionBERT(checkpoint_path=motionbert_checkpoint)
        self.expert = SquatExpert(dropout=dropout)
        self.correctness_head = SquatCorrectnessHead(dropout=dropout)
        self.set_backbone_frozen(True)

    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Set MotionBERT gradient policy; training uses the frozen setting."""

        self.backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "SquatCorrectnessModel":
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        motionbert_input: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full model from normalized x/y/confidence input."""

        if motionbert_input.ndim != 4 or motionbert_input.shape[2:] != (17, 3):
            raise ValueError(
                f"Expected MotionBERT input (B,T,17,3), got {tuple(motionbert_input.shape)}."
            )
        if not torch.is_floating_point(motionbert_input):
            raise TypeError("motionbert_input must be floating point.")
        if temporal_mask is not None and temporal_mask.shape != motionbert_input.shape[:2]:
            raise ValueError("temporal_mask must have shape (B,T).")
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.backbone(motionbert_input)
        else:
            features = self.backbone(motionbert_input)
        return self.forward_features(features, temporal_mask)

    def forward_features(
        self,
        motionbert_features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run only SquatExpert and the head from cached frozen features."""

        if motionbert_features.ndim != 4 or motionbert_features.shape[2:] != (17, 512):
            raise ValueError(
                "Expected frozen MotionBERT features (B,T,17,512), got "
                f"{tuple(motionbert_features.shape)}."
            )
        if temporal_mask is not None and temporal_mask.shape != motionbert_features.shape[:2]:
            raise ValueError("temporal_mask must match feature B,T dimensions.")
        expert = self.expert(motionbert_features, temporal_mask=temporal_mask)
        global_embedding = expert["global_embedding"]
        temporal_embedding = expert["temporal_embedding"]
        if global_embedding is None or temporal_embedding is None:
            raise RuntimeError("SquatExpert did not return required embeddings.")
        logits = self.correctness_head(global_embedding)
        return {
            "logits": logits,
            "correct_probability": torch.softmax(logits, dim=-1)[:, 1],
            "global_embedding": global_embedding,
            "temporal_embedding": temporal_embedding,
            "joint_attention": expert["joint_attention"],
            "temporal_attention": expert["temporal_attention"],
        }

