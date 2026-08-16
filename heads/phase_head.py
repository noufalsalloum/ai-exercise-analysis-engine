from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


PHASE_VOCABULARY: Final[tuple[str, ...]] = (
    "REST",
    "SETUP",
    "DESCENDING",
    "BOTTOM",
    "ASCENDING",
    "TOP",
    "HOLD",
    "RETURNING",
    "FINISH",
)
PHASE_TO_INDEX: Final[dict[str, int]] = {
    phase: index for index, phase in enumerate(PHASE_VOCABULARY)
}


class PhaseHead(nn.Module):
    """Shared frame/window phase classifier over temporal expert features."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_phases: int = len(PHASE_VOCABULARY),
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, num_phases, num_layers) <= 0:
            raise ValueError("PhaseHead dimensions must be positive.")

        self.input_dim = input_dim
        self.num_phases = num_phases
        self.input_norm = nn.LayerNorm(input_dim)
        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_phases)

    @staticmethod
    def build_valid_phase_mask(
        valid_phases: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Create a union-vocabulary mask from exercise-valid phase names."""

        mask = torch.zeros(len(PHASE_VOCABULARY), dtype=torch.bool, device=device)
        for phase in valid_phases:
            normalized = phase.upper().strip()
            if normalized not in PHASE_TO_INDEX:
                raise ValueError(
                    f"Unknown phase '{phase}'. Valid union phases: "
                    f"{PHASE_VOCABULARY}"
                )
            mask[PHASE_TO_INDEX[normalized]] = True
        if not mask.any():
            raise ValueError("At least one valid phase is required.")
        return mask

    def _encode(
        self,
        temporal_embedding: torch.Tensor,
        temporal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normalized = self.input_norm(temporal_embedding)
        if temporal_mask is None:
            encoded, _ = self.temporal_encoder(normalized)
            return encoded

        batch_size, frames = temporal_embedding.shape[:2]
        if temporal_mask.shape != (batch_size, frames):
            raise ValueError(
                f"temporal_mask must be {(batch_size, frames)}, got "
                f"{tuple(temporal_mask.shape)}"
            )
        mask = temporal_mask.to(device=temporal_embedding.device, dtype=torch.bool)
        lengths = mask.sum(dim=1)
        if (lengths <= 0).any():
            raise ValueError("Every sequence must contain at least one valid frame.")
        expected = torch.arange(frames, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(mask, expected):
            raise ValueError("temporal_mask must describe contiguous prefix frames.")

        packed = pack_padded_sequence(
            normalized,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_encoded, _ = self.temporal_encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded,
            batch_first=True,
            total_length=frames,
        )
        return encoded

    def forward(
        self,
        temporal_embedding: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        valid_phase_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return unnormalized phase logits with shape ``(B, T, P)``."""

        if temporal_embedding.ndim != 3:
            raise ValueError(
                "PhaseHead expects (B, T, D), got "
                f"{tuple(temporal_embedding.shape)}"
            )
        if temporal_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected temporal dimension {self.input_dim}, got "
                f"{temporal_embedding.shape[-1]}"
            )

        encoded = self._encode(temporal_embedding, temporal_mask)
        logits = self.classifier(self.dropout(self.output_norm(encoded)))

        if valid_phase_mask is not None:
            mask = valid_phase_mask.to(device=logits.device, dtype=torch.bool)
            if mask.ndim == 1:
                if mask.shape[0] != self.num_phases:
                    raise ValueError("valid_phase_mask has the wrong phase dimension.")
                mask = mask.view(1, 1, -1)
            elif mask.ndim == 2:
                if mask.shape != (logits.shape[0], self.num_phases):
                    raise ValueError("Batched valid_phase_mask must be (B, P).")
                mask = mask.unsqueeze(1)
            else:
                raise ValueError("valid_phase_mask must have shape (P,) or (B, P).")
            if not mask.any(dim=-1).all():
                raise ValueError("Every sample needs at least one valid phase.")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        if temporal_mask is not None:
            logits = logits.masked_fill(
                ~temporal_mask.to(logits.device, torch.bool).unsqueeze(-1),
                0.0,
            )
        return logits

    @torch.no_grad()
    def predict(
        self,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convert logits to probabilities and phase indices for inference."""

        if logits.ndim != 3 or logits.shape[-1] != self.num_phases:
            raise ValueError("Expected phase logits with shape (B, T, P).")
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "probabilities": probabilities,
            "predictions": probabilities.argmax(dim=-1),
        }
