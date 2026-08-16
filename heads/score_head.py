from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class ScoreHead(nn.Module):
    """Future supervised scoring interface; disabled until labels exist.

    The module supports regression, pairwise ranking embeddings, or ordinal
    logits, but refuses inference while untrained. Prototype similarity must
    be reported separately as reference similarity.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        mode: Literal["regression", "pairwise", "ordinal"] = "regression",
        ordinal_bins: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mode not in {"regression", "pairwise", "ordinal"}:
            raise ValueError(f"Unsupported score mode: {mode}")
        self.mode = mode
        output_dim = ordinal_bins if mode == "ordinal" else 1
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.register_buffer("trained", torch.tensor(False), persistent=True)

    def forward(
        self,
        global_embedding: torch.Tensor,
        allow_untrained: bool = False,
    ) -> torch.Tensor:
        if not bool(self.trained.item()) and not allow_untrained:
            raise RuntimeError(
                "ScoreHead is untrained. Do not expose its random output as a "
                "quality score."
            )
        if global_embedding.ndim != 2:
            raise ValueError("ScoreHead expects (B, D) global embeddings.")
        return self.network(global_embedding)
