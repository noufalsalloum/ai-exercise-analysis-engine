"""Small temporal baseline for learned Squat repetition activity boundaries."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values + self.block(values)
        return self.norm(residual.transpose(1, 2)).transpose(1, 2)


class SquatRepBoundaryModel(nn.Module):
    """Predict an ``inside repetition`` logit for every H36M pose frame.

    Input is preprocessing-v4 ``(B,T,17,3)`` x/y/confidence. The labels are
    only OUTSIDE_REP/INSIDE_REP derived directly from real segmentation; no
    synthetic Squat phase labels are introduced.
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
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(channels, 1, 1))

    def forward(self, motionbert_input: torch.Tensor) -> torch.Tensor:
        if motionbert_input.ndim != 4 or motionbert_input.shape[2:] != (17, 3):
            raise ValueError(
                f"Expected preprocessing-v4 input (B,T,17,3), got {tuple(motionbert_input.shape)}."
            )
        if not torch.is_floating_point(motionbert_input):
            raise TypeError("Boundary input must be floating point.")
        batch, frames = motionbert_input.shape[:2]
        encoded = self.frame_encoder(motionbert_input.reshape(batch, frames, -1))
        temporal = self.temporal(encoded.transpose(1, 2))
        return self.classifier(temporal).squeeze(1)

