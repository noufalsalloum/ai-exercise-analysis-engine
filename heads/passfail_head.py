from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _masked_temporal_mean(
    temporal_embedding: torch.Tensor,
    temporal_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if temporal_mask is None:
        return temporal_embedding.mean(dim=1)
    if temporal_mask.shape != temporal_embedding.shape[:2]:
        raise ValueError("temporal_mask must match the (B, T) dimensions.")
    weights = temporal_mask.to(temporal_embedding.device, temporal_embedding.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (temporal_embedding * weights.unsqueeze(-1)).sum(dim=1) / denominator


class PassFailHead(nn.Module):
    """Trainable binary classifier; logits are meaningful only after training."""

    def __init__(
        self,
        global_dim: int = 1024,
        temporal_dim: int = 512,
        similarity_dim: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.global_dim = global_dim
        self.temporal_dim = temporal_dim
        self.similarity_dim = similarity_dim

        self.global_norm = nn.LayerNorm(global_dim)
        self.temporal_projection = nn.Linear(temporal_dim, hidden_dim)
        self.similarity_projection = nn.Linear(similarity_dim, hidden_dim)
        fused_dim = global_dim + hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        global_embedding: torch.Tensor,
        temporal_embedding: Optional[torch.Tensor] = None,
        similarity_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if global_embedding.ndim != 2 or global_embedding.shape[-1] != self.global_dim:
            raise ValueError(f"global_embedding must be (B, {self.global_dim}).")
        batch_size = global_embedding.shape[0]

        if temporal_embedding is None:
            temporal_features = global_embedding.new_zeros(batch_size, self.temporal_dim)
        else:
            if temporal_embedding.ndim != 3 or temporal_embedding.shape[-1] != self.temporal_dim:
                raise ValueError(
                    f"temporal_embedding must be (B, T, {self.temporal_dim})."
                )
            temporal_features = _masked_temporal_mean(
                temporal_embedding,
                temporal_mask,
            )

        if similarity_features is None:
            similarity_features = global_embedding.new_zeros(
                batch_size,
                self.similarity_dim,
            )
        if similarity_features.shape != (batch_size, self.similarity_dim):
            raise ValueError(
                f"similarity_features must be {(batch_size, self.similarity_dim)}."
            )

        fused = torch.cat(
            [
                self.global_norm(global_embedding),
                self.temporal_projection(temporal_features),
                self.similarity_projection(similarity_features),
            ],
            dim=-1,
        )
        return self.classifier(self.fusion(fused))
