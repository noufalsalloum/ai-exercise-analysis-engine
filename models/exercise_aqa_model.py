from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    from ..backbone.motionbert import MotionBERT
    from ..experts.registry import ExpertRegistry
    from ..heads.error_head import ERROR_VOCABULARY, ErrorHead
    from ..heads.passfail_head import PassFailHead
    from ..heads.phase_head import PHASE_VOCABULARY, PhaseHead
except ImportError:
    from backbone.motionbert import MotionBERT
    from experts.registry import ExpertRegistry
    from heads.error_head import ERROR_VOCABULARY, ErrorHead
    from heads.passfail_head import PassFailHead
    from heads.phase_head import PHASE_VOCABULARY, PhaseHead


class ExerciseAQAModel(nn.Module):
    """Unified MotionBERT → persistent expert → optional task heads model."""

    def __init__(
        self,
        motionbert_checkpoint: Optional[str | Path] = None,
        temporal_dim: int = 512,
        global_dim: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = MotionBERT(
            checkpoint_path=str(motionbert_checkpoint)
            if motionbert_checkpoint is not None
            else None,
            num_joints=17,
            dim_in=3,
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            maxlen=243,
        )
        self.expert_registry = ExpertRegistry(
            input_dim=512,
            temporal_dim=temporal_dim,
            global_dim=global_dim,
            dropout=dropout,
        )
        self.phase_head = PhaseHead(
            input_dim=temporal_dim,
            num_phases=len(PHASE_VOCABULARY),
            dropout=dropout,
        )
        self.passfail_head = PassFailHead(
            global_dim=global_dim,
            temporal_dim=temporal_dim,
            dropout=dropout,
        )
        self.error_head = ErrorHead(
            global_dim=global_dim,
            temporal_dim=temporal_dim,
            num_errors=len(ERROR_VOCABULARY),
            dropout=dropout,
        )

        self.register_buffer("phase_trained", torch.tensor(False), persistent=True)
        self.register_buffer("passfail_trained", torch.tensor(False), persistent=True)
        self.register_buffer("error_trained", torch.tensor(False), persistent=True)
        self.register_buffer("experts_trained", torch.tensor(False), persistent=True)

    @property
    def head_status(self) -> dict[str, bool]:
        return {
            "experts": bool(self.experts_trained.item()),
            "phase": bool(self.phase_trained.item()),
            "pass_fail": bool(self.passfail_trained.item()),
            "errors": bool(self.error_trained.item()),
        }

    def mark_trained(self, component: str, trained: bool = True) -> None:
        mapping = {
            "experts": self.experts_trained,
            "phase": self.phase_trained,
            "pass_fail": self.passfail_trained,
            "errors": self.error_trained,
        }
        if component not in mapping:
            raise ValueError(f"Unknown trainable component: {component}")
        mapping[component].fill_(trained)

    def forward(
        self,
        motionbert_input: torch.Tensor,
        exercise_id: str,
        temporal_mask: Optional[torch.Tensor] = None,
        joint_mask: Optional[torch.Tensor] = None,
        valid_phase_mask: Optional[torch.Tensor] = None,
        valid_error_mask: Optional[torch.Tensor] = None,
        similarity_features: Optional[torch.Tensor] = None,
        tasks: Optional[set[str]] = None,
    ) -> dict[str, object]:
        """Run the backbone/expert and requested training heads.

        Heads are executed only when explicitly listed in ``tasks``. This
        keeps untrained random logits out of production inference.
        """

        if motionbert_input.ndim != 4:
            raise ValueError("motionbert_input must have shape (B, T, 17, 3).")
        if motionbert_input.shape[2:] != (17, 3):
            raise ValueError(
                "Expected MotionBERT joint/channel dimensions (17, 3), got "
                f"{tuple(motionbert_input.shape[2:])}"
            )
        if motionbert_input.shape[1] > 243:
            raise ValueError("MotionBERT-Lite supports at most 243 frames.")

        motionbert_features = self.backbone(motionbert_input)
        expert_output = self.expert_registry(
            motionbert_features,
            exercise_id=exercise_id,
            temporal_mask=temporal_mask,
            joint_mask=joint_mask,
        )
        result: dict[str, object] = {
            "motionbert_features": motionbert_features,
            **expert_output,
            "head_status": self.head_status,
        }

        requested = tasks or set()
        temporal_embedding = expert_output["temporal_embedding"]
        global_embedding = expert_output["global_embedding"]
        if not isinstance(temporal_embedding, torch.Tensor) or not isinstance(
            global_embedding,
            torch.Tensor,
        ):
            raise RuntimeError("Expert did not return required embeddings.")

        if "phase" in requested:
            result["phase_logits"] = self.phase_head(
                temporal_embedding,
                temporal_mask=temporal_mask,
                valid_phase_mask=valid_phase_mask,
            )
        if "pass_fail" in requested:
            result["passfail_logits"] = self.passfail_head(
                global_embedding,
                temporal_embedding=temporal_embedding,
                similarity_features=similarity_features,
                temporal_mask=temporal_mask,
            )
        if "errors" in requested:
            result["error_logits"] = self.error_head(
                global_embedding,
                temporal_embedding,
                temporal_mask=temporal_mask,
                valid_error_mask=valid_error_mask,
            )
        return result

    def load_aqa_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, object]:
        """Load a complete AQA state dictionary strictly and safely."""

        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError("AQA checkpoint must contain 'model_state'.")
        self.load_state_dict(checkpoint["model_state"], strict=True)
        return checkpoint
