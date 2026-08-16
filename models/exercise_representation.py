from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from backbone.motionbert import MotionBERT
from experts.base_expert import BaseExpert


class ExerciseRepresentationModel(nn.Module):
    """Frozen MotionBERT plus one shared expert and an exercise classifier.

    The classifier deliberately does not route through exercise-specific experts:
    selecting an expert from the target class would leak the answer.
    """

    def __init__(
        self,
        num_classes: int,
        motionbert_checkpoint: Optional[str | Path] = None,
        *,
        backbone: Optional[nn.Module] = None,
        temporal_dim: int = 512,
        global_dim: int = 1024,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Exercise classification requires at least two classes.")
        self.backbone = backbone or MotionBERT(checkpoint_path=motionbert_checkpoint)
        self.shared_expert = BaseExpert(
            input_dim=512,
            temporal_dim=temporal_dim,
            global_dim=global_dim,
            dropout=dropout,
            exercise_id="shared_exercise_representation",
        )
        hidden_dim = max(128, global_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.backbone_frozen = False
        self.set_backbone_frozen(freeze_backbone)

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Control backbone gradients while keeping the rest trainable."""

        self.backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ExerciseRepresentationModel":
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        motionbert_input: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return logits and learned shared temporal/global embeddings."""

        if motionbert_input.ndim != 4 or motionbert_input.shape[2:] != (17, 3):
            raise ValueError(
                f"Expected input (B,T,17,3), got {tuple(motionbert_input.shape)}."
            )
        if not torch.is_floating_point(motionbert_input):
            raise TypeError("motionbert_input must be floating point.")
        if temporal_mask is not None and temporal_mask.shape != motionbert_input.shape[:2]:
            raise ValueError(
                "temporal_mask must have shape "
                f"{tuple(motionbert_input.shape[:2])}, got {tuple(temporal_mask.shape)}."
            )
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.backbone(motionbert_input)
        else:
            features = self.backbone(motionbert_input)
        if features.ndim != 4 or features.shape[:3] != motionbert_input.shape[:3] or features.shape[-1] != 512:
            raise ValueError(
                "Backbone must return (B,T,17,512), got "
                f"{tuple(features.shape)}."
            )
        return self.forward_features(features, temporal_mask=temporal_mask)

    def forward_features(
        self,
        motionbert_features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the trainable shared expert/head from frozen backbone features."""

        if motionbert_features.ndim != 4 or motionbert_features.shape[2:] != (17, 512):
            raise ValueError(
                "Expected frozen MotionBERT features (B,T,17,512), got "
                f"{tuple(motionbert_features.shape)}."
            )
        if temporal_mask is not None and temporal_mask.shape != motionbert_features.shape[:2]:
            raise ValueError(
                "temporal_mask must match feature B,T dimensions, got "
                f"{tuple(temporal_mask.shape)}."
            )
        expert_outputs = self.shared_expert(
            motionbert_features, temporal_mask=temporal_mask
        )
        global_embedding = expert_outputs["global_embedding"]
        temporal_embedding = expert_outputs["temporal_embedding"]
        if global_embedding is None or temporal_embedding is None:
            raise RuntimeError("Shared expert did not return required embeddings.")
        logits = self.classifier(global_embedding)
        return {
            "logits": logits,
            "global_embedding": global_embedding,
            "temporal_embedding": temporal_embedding,
            "joint_attention": expert_outputs["joint_attention"],
            "temporal_attention": expert_outputs["temporal_attention"],
        }
