"""Modular three-class static Squat posture error model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ERROR_CLASSES = ("good", "bad_back", "bad_heel")


class SquatPostureErrorHead(nn.Module):
    """Small residual MLP mapping static posture features to three logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.20) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Error head dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, len(ERROR_CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected static posture features (B,{self.input_dim}), got {tuple(features.shape)}.")
        hidden = torch.nn.functional.gelu(self.input_norm(self.input_projection(features)))
        return self.classifier(self.output_norm(hidden + self.residual(hidden)))


class SquatPostureErrorModel(nn.Module):
    """Train-only normalization plus the modular posture head."""

    def __init__(
        self,
        input_dim: int,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        mean = torch.zeros(input_dim) if feature_mean is None else feature_mean.float()
        scale = torch.ones(input_dim) if feature_scale is None else feature_scale.float()
        if mean.shape != (input_dim,) or scale.shape != (input_dim,):
            raise ValueError("Feature normalization tensors must match input_dim.")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale.clamp_min(1e-6))
        self.head = SquatPostureErrorHead(input_dim, hidden_dim, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(features):
            raise TypeError("Posture features must be floating point.")
        return self.head((features - self.feature_mean) / self.feature_scale)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> dict[str, Any]:
        self.eval(); logits = self(features); probabilities = torch.softmax(logits, dim=-1)
        indices = probabilities.argmax(dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_indices": indices,
            "predicted_errors": [ERROR_CLASSES[int(index)] for index in indices.cpu()],
            "score": None,
        }


def load_squat_posture_error_checkpoint(
    path: str | Path, device: torch.device, *, weights_only: bool = True
) -> tuple[SquatPostureErrorModel, dict[str, Any]]:
    """Create and strictly load the selected neural Error V1 checkpoint."""

    # weights_only defaults to True (unchanged) for every caller — training,
    # squat_error_inference.py, and tests. inference/squat_ai_mvp.py is the
    # sole caller that explicitly passes weights_only=False, scoped there to
    # the one trusted production checkpoint (checkpoints/squat_error_v1/best.pt)
    # — see the comment at that call site for why.
    checkpoint = torch.load(Path(path), map_location=device, weights_only=weights_only)
    if checkpoint.get("model_type") != "small_mlp":
        raise ValueError(f"Unsupported posture error checkpoint type {checkpoint.get('model_type')!r}.")
    model = SquatPostureErrorModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint

