from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _masked_softmax(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Apply softmax while excluding invalid positions."""

    if mask is None:
        return torch.softmax(logits, dim=dim)

    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(
            f"Attention mask shape {tuple(mask.shape)} must match logits "
            f"shape {tuple(logits.shape)}"
        )
    if not mask.any(dim=dim).all():
        raise ValueError("Every attention row must contain at least one valid item.")

    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    return weights.masked_fill(~mask, 0.0)


class ExerciseAdapter(nn.Module):
    """Lightweight trainable exercise conditioning for temporal features.

    Each expert owns an independent token, FiLM transform, gated residual
    bottleneck, and normalization parameters. The shared expert architecture
    therefore stays in :class:`BaseExpert`, while exercise-specific parameters
    materially alter both temporal and global representations.
    """

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int,
        dropout: float,
        gate_init: float = -1.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Adapter dimensions must be positive.")

        self.exercise_token = nn.Parameter(torch.empty(1, 1, feature_dim))
        nn.init.normal_(self.exercise_token, mean=0.0, std=0.02)

        self.film = nn.Linear(feature_dim, feature_dim * 2)
        self.adapter_norm = nn.LayerNorm(feature_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, feature_dim),
        )
        self.adapter_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        if temporal_features.ndim != 3:
            raise ValueError(
                "ExerciseAdapter expects (B, T, D), got "
                f"{tuple(temporal_features.shape)}"
            )

        token = self.exercise_token.expand(temporal_features.shape[0], -1, -1)
        gamma, beta = self.film(token).chunk(2, dim=-1)
        conditioned = temporal_features * (1.0 + torch.tanh(gamma)) + beta

        residual = self.adapter(self.adapter_norm(conditioned))
        gate = torch.sigmoid(self.adapter_gate)
        return self.output_norm(conditioned + gate * residual)


class BaseExpert(nn.Module):
    """Attention-based expert producing temporal and global embeddings.

    Input shape is ``(B, T, J, input_dim)``. Joint attention first produces a
    per-frame representation. A residual temporal projection retains the
    entire time axis, then learnable temporal attention pools the valid frames
    for a global residual MLP.
    """

    def __init__(
        self,
        input_dim: int = 512,
        temporal_dim: int = 512,
        global_dim: int = 1024,
        dropout: float = 0.2,
        adapter_bottleneck_dim: int = 128,
        adapter_gate_init: float = -1.0,
        exercise_id: str = "generic",
    ) -> None:
        super().__init__()
        if min(input_dim, temporal_dim, global_dim) <= 0:
            raise ValueError("Expert dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.global_dim = global_dim
        self.exercise_id = exercise_id

        self.input_norm = nn.LayerNorm(input_dim)
        attention_hidden = max(64, input_dim // 4)
        self.joint_attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1, bias=False),
        )

        self.temporal_residual = (
            nn.Identity()
            if input_dim == temporal_dim
            else nn.Linear(input_dim, temporal_dim, bias=False)
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(input_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, temporal_dim),
            nn.Dropout(dropout),
        )
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.exercise_adapter = ExerciseAdapter(
            feature_dim=temporal_dim,
            bottleneck_dim=adapter_bottleneck_dim,
            dropout=dropout,
            gate_init=adapter_gate_init,
        )

        temporal_attention_hidden = max(64, temporal_dim // 4)
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, temporal_attention_hidden),
            nn.Tanh(),
            nn.Linear(temporal_attention_hidden, 1, bias=False),
        )

        self.global_projection = nn.Linear(temporal_dim, global_dim)
        self.global_norm = nn.LayerNorm(global_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim * 2, global_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(global_dim)

    def _validate_inputs(
        self,
        motionbert_features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor],
        joint_mask: Optional[torch.Tensor],
    ) -> tuple[int, int, int]:
        if motionbert_features.ndim != 4:
            raise ValueError(
                "Expected MotionBERT features (B, T, J, C), got "
                f"{tuple(motionbert_features.shape)}"
            )

        batch_size, frames, joints, channels = motionbert_features.shape
        if frames <= 0 or joints <= 0:
            raise ValueError("Expert input must contain at least one frame and joint.")
        if channels != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {channels}."
            )
        if temporal_mask is not None and temporal_mask.shape != (batch_size, frames):
            raise ValueError(
                "temporal_mask must have shape "
                f"{(batch_size, frames)}, got {tuple(temporal_mask.shape)}"
            )
        if joint_mask is not None and joint_mask.shape != (
            batch_size,
            frames,
            joints,
        ):
            raise ValueError(
                "joint_mask must have shape "
                f"{(batch_size, frames, joints)}, got {tuple(joint_mask.shape)}"
            )
        return batch_size, frames, joints

    def forward(
        self,
        motionbert_features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        joint_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Return temporal/global embeddings and both attention maps."""

        batch_size, frames, joints = self._validate_inputs(
            motionbert_features,
            temporal_mask,
            joint_mask,
        )

        normalized = self.input_norm(motionbert_features)
        joint_logits = self.joint_attention(normalized).squeeze(-1)
        effective_joint_mask = joint_mask
        padded_frames = None
        if temporal_mask is not None:
            padded_frames = ~temporal_mask.to(
                device=motionbert_features.device,
                dtype=torch.bool,
            )
        if effective_joint_mask is not None and padded_frames is not None:
            # Softmax needs one valid item per row. Padded frames are zeroed
            # immediately afterwards and therefore may temporarily expose all
            # joints without letting their values affect either output.
            effective_joint_mask = effective_joint_mask.to(
                device=motionbert_features.device,
                dtype=torch.bool,
            ) | padded_frames.unsqueeze(-1)
        joint_weights = _masked_softmax(
            joint_logits,
            effective_joint_mask,
            dim=-1,
        )
        if padded_frames is not None:
            joint_weights = joint_weights.masked_fill(
                padded_frames.unsqueeze(-1),
                0.0,
            )
        joint_pooled = torch.sum(normalized * joint_weights.unsqueeze(-1), dim=2)

        temporal_embedding = self.temporal_norm(
            self.temporal_residual(joint_pooled)
            + self.temporal_projection(joint_pooled)
        )
        temporal_embedding = self.exercise_adapter(temporal_embedding)
        if temporal_mask is not None:
            temporal_embedding = temporal_embedding.masked_fill(
                ~temporal_mask.to(
                    device=temporal_embedding.device,
                    dtype=torch.bool,
                ).unsqueeze(-1),
                0.0,
            )

        temporal_logits = self.temporal_attention(temporal_embedding).squeeze(-1)
        temporal_weights = _masked_softmax(
            temporal_logits,
            temporal_mask,
            dim=-1,
        )
        pooled = torch.sum(
            temporal_embedding * temporal_weights.unsqueeze(-1),
            dim=1,
        )

        global_seed = self.global_projection(pooled)
        global_embedding = self.output_norm(
            global_seed + self.global_mlp(self.global_norm(global_seed))
        )

        return {
            "temporal_embedding": temporal_embedding,
            "global_embedding": global_embedding,
            "joint_attention": joint_weights,
            "temporal_attention": temporal_weights,
        }
