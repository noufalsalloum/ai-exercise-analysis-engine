from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Optional

import torch
import torch.nn as nn


ERROR_VOCABULARY: Final[tuple[str, ...]] = (
    "back_knee_slamming",
    "chin_below_bar",
    "excessive_forward_lean",
    "flaring_elbows",
    "front_knee_past_toe",
    "head_drooping",
    "hips_sagging",
    "hips_too_high",
    "incomplete_rom",
    "insufficient_depth",
    "kipping_momentum",
    "knee_valgus",
    "sagging_hips",
    "torso_excessive_lean",
)
ERROR_TO_INDEX: Final[dict[str, int]] = {
    error: index for index, error in enumerate(ERROR_VOCABULARY)
}


class ErrorHead(nn.Module):
    """Shared multi-label error classifier with exercise-valid masking."""

    def __init__(
        self,
        global_dim: int = 1024,
        temporal_dim: int = 512,
        hidden_dim: int = 512,
        num_errors: int = len(ERROR_VOCABULARY),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.global_dim = global_dim
        self.temporal_dim = temporal_dim
        self.num_errors = num_errors
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, max(64, temporal_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(64, temporal_dim // 4), 1, bias=False),
        )
        fused_dim = global_dim + temporal_dim
        self.network = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_errors),
        )

    @staticmethod
    def build_valid_error_mask(
        valid_errors: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        mask = torch.zeros(len(ERROR_VOCABULARY), dtype=torch.bool, device=device)
        for error in valid_errors:
            normalized = error.lower().strip()
            if normalized not in ERROR_TO_INDEX:
                raise ValueError(
                    f"Unknown error '{error}'. Vocabulary: {ERROR_VOCABULARY}"
                )
            mask[ERROR_TO_INDEX[normalized]] = True
        if not mask.any():
            raise ValueError("At least one valid error is required.")
        return mask

    def _pool_temporal(
        self,
        temporal_embedding: torch.Tensor,
        temporal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = self.temporal_attention(temporal_embedding).squeeze(-1)
        if temporal_mask is not None:
            if temporal_mask.shape != logits.shape:
                raise ValueError("temporal_mask must match (B, T).")
            mask = temporal_mask.to(logits.device, torch.bool)
            if not mask.any(dim=1).all():
                raise ValueError("Every sequence needs a valid frame.")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(temporal_embedding * weights.unsqueeze(-1), dim=1)

    def forward(
        self,
        global_embedding: torch.Tensor,
        temporal_embedding: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        valid_error_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if global_embedding.ndim != 2 or global_embedding.shape[-1] != self.global_dim:
            raise ValueError(f"global_embedding must be (B, {self.global_dim}).")
        if temporal_embedding.ndim != 3 or temporal_embedding.shape[-1] != self.temporal_dim:
            raise ValueError(
                f"temporal_embedding must be (B, T, {self.temporal_dim})."
            )
        if temporal_embedding.shape[0] != global_embedding.shape[0]:
            raise ValueError("Global and temporal batch sizes must match.")

        pooled = self._pool_temporal(temporal_embedding, temporal_mask)
        logits = self.network(torch.cat([global_embedding, pooled], dim=-1))

        if valid_error_mask is not None:
            mask = valid_error_mask.to(logits.device, torch.bool)
            if mask.ndim == 1:
                if mask.shape[0] != self.num_errors:
                    raise ValueError("valid_error_mask has the wrong dimension.")
                mask = mask.unsqueeze(0)
            if mask.shape not in ((1, self.num_errors), logits.shape):
                raise ValueError("valid_error_mask must be (E,) or (B, E).")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits
