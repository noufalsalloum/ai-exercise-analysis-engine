from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Compute only supervised task losses present in the current batch."""

    def __init__(
        self,
        phase_weight: float = 1.0,
        pass_fail_weight: float = 1.0,
        error_weight: float = 1.0,
        phase_class_weights: Optional[torch.Tensor] = None,
        pass_fail_class_weights: Optional[torch.Tensor] = None,
        error_positive_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.weights = {
            "phase": float(phase_weight),
            "pass_fail": float(pass_fail_weight),
            "errors": float(error_weight),
        }
        self.register_buffer("phase_class_weights", phase_class_weights)
        self.register_buffer("pass_fail_class_weights", pass_fail_class_weights)
        self.register_buffer("error_positive_weights", error_positive_weights)

    def forward(
        self,
        outputs: Mapping[str, object],
        batch: Mapping[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        device = outputs["global_embedding"].device  # type: ignore[union-attr]

        phase_available = batch.get("phase_available")
        if "phase_logits" in outputs and isinstance(phase_available, torch.Tensor) and phase_available.any():
            phase_mask = phase_available.to(device=device, dtype=torch.bool)
            logits = outputs["phase_logits"][phase_mask]
            labels = batch["phase_labels"].to(device)[phase_mask]
            losses["phase"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                weight=self.phase_class_weights,
                ignore_index=-100,
            )

        pass_available = batch.get("pass_fail_available")
        if "passfail_logits" in outputs and isinstance(pass_available, torch.Tensor) and pass_available.any():
            pass_mask = pass_available.to(device=device, dtype=torch.bool)
            losses["pass_fail"] = F.cross_entropy(
                outputs["passfail_logits"][pass_mask],
                batch["pass_fail_labels"].to(device)[pass_mask],
                weight=self.pass_fail_class_weights,
            )

        error_available = batch.get("errors_available")
        if "error_logits" in outputs and isinstance(error_available, torch.Tensor) and error_available.any():
            error_mask = error_available.to(device=device, dtype=torch.bool)
            losses["errors"] = F.binary_cross_entropy_with_logits(
                outputs["error_logits"][error_mask],
                batch["error_labels"].to(device)[error_mask],
                pos_weight=self.error_positive_weights,
            )

        if not losses:
            raise ValueError("This batch contains no labels for any requested task.")
        total = sum(self.weights[name] * loss for name, loss in losses.items())
        losses["total"] = total
        return total, losses
