"""Exercise-specific temporal boundary model for REHAB24 Ex3 table Push-ups."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.squat_rep_boundary import TemporalResidualBlock


class PushupRepBoundaryModel(nn.Module):
    """Predict active-repetition and GT-boundary proximity per frame."""

    def __init__(self, channels: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(17 * 3), nn.Linear(17 * 3, channels), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal = nn.Sequential(
            TemporalResidualBlock(channels, 1, dropout),
            TemporalResidualBlock(channels, 2, dropout),
            TemporalResidualBlock(channels, 4, dropout),
            TemporalResidualBlock(channels, 8, dropout),
        )
        self.active_head = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(channels, 1, 1))
        self.boundary_head = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 3, padding=1),
            nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels // 2, 1, 1),
        )

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        if values.ndim != 4 or values.shape[2:] != (17, 3):
            raise ValueError(f"Expected preprocessing-v4 input (B,T,17,3), got {tuple(values.shape)}")
        if not torch.is_floating_point(values):
            raise TypeError("Boundary input must be floating point.")
        batch, frames = values.shape[:2]
        encoded = self.frame_encoder(values.reshape(batch, frames, -1)).transpose(1, 2)
        temporal = self.temporal(encoded)
        return {
            "active_logits": self.active_head(temporal).squeeze(1),
            "boundary_logits": self.boundary_head(temporal).squeeze(1),
        }
