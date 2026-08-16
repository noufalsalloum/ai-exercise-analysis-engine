"""Boundary-aware temporal model for Squat AI Experiment 2."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.squat_rep_boundary import TemporalResidualBlock


class SquatRepBoundaryV2Model(nn.Module):
    """Predict active-repetition and true-boundary proximity logits per frame.

    The auxiliary boundary target is derived only from annotated repetition
    start/end frames. It does not introduce synthetic exercise phases.
    """

    def __init__(self, channels: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(17 * 3),
            nn.Linear(17 * 3, channels),
            nn.GELU(),
            nn.Dropout(dropout),
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels // 2, 1, 1),
        )

    def forward(self, motionbert_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if motionbert_input.ndim != 4 or motionbert_input.shape[2:] != (17, 3):
            raise ValueError(
                f"Expected preprocessing-v4 input (B,T,17,3), got {tuple(motionbert_input.shape)}."
            )
        if not torch.is_floating_point(motionbert_input):
            raise TypeError("Boundary input must be floating point.")
        batch, frames = motionbert_input.shape[:2]
        encoded = self.frame_encoder(motionbert_input.reshape(batch, frames, -1))
        temporal = self.temporal(encoded.transpose(1, 2))
        return {
            "active_logits": self.active_head(temporal).squeeze(1),
            "boundary_logits": self.boundary_head(temporal).squeeze(1),
        }

