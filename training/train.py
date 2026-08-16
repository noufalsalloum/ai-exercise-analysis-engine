from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from ..heads.error_head import ErrorHead
    from ..heads.phase_head import PhaseHead
    from ..models.exercise_aqa_model import ExerciseAQAModel
    from .losses import MultiTaskLoss
except ImportError:
    from heads.error_head import ErrorHead
    from heads.phase_head import PhaseHead
    from models.exercise_aqa_model import ExerciseAQAModel
    from training.losses import MultiTaskLoss


@dataclass
class TrainerConfig:
    """Reproducible multi-task optimization settings."""

    epochs: int = 30
    batch_size: int = 8
    backbone_learning_rate: float = 1e-5
    expert_learning_rate: float = 1e-4
    head_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    freeze_backbone: bool = True
    early_stopping_patience: int = 6
    seed: int = 42
    device: str = "auto"


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AQATrainer:
    """Train only tasks for which each batch supplies labels and masks."""

    def __init__(
        self,
        model: ExerciseAQAModel,
        loss_fn: MultiTaskLoss,
        config: TrainerConfig,
        exercise_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        set_reproducibility(config.seed)
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.exercise_configs = dict(exercise_configs or {})
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.set_backbone_frozen(config.freeze_backbone)
        self.optimizer = self._build_optimizer()

    def set_backbone_frozen(self, frozen: bool) -> None:
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad = not frozen

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups = []
        backbone = [item for item in self.model.backbone.parameters() if item.requires_grad]
        experts = [item for item in self.model.expert_registry.parameters() if item.requires_grad]
        heads = [
            item
            for module in (self.model.phase_head, self.model.passfail_head, self.model.error_head)
            for item in module.parameters()
            if item.requires_grad
        ]
        for parameters, rate in (
            (backbone, self.config.backbone_learning_rate),
            (experts, self.config.expert_learning_rate),
            (heads, self.config.head_learning_rate),
        ):
            if parameters:
                groups.append({"params": parameters, "lr": rate})
        return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

    def enable_backbone_finetuning(self) -> None:
        """Unfreeze MotionBERT and rebuild optimizer with its configured LR."""

        self.set_backbone_frozen(False)
        self.optimizer = self._build_optimizer()

    def _task_masks(self, exercise_id: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        config = self.exercise_configs.get(exercise_id, {})
        labels = config.get("labels", {})
        phase_names = labels.get("valid_phases", [])
        error_names = labels.get("valid_errors", [])
        phase_mask = PhaseHead.build_valid_phase_mask(phase_names, self.device) if phase_names else None
        error_mask = ErrorHead.build_valid_error_mask(error_names, self.device) if error_names else None
        return phase_mask, error_mask

    def _move_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _run_batch(self, batch: Mapping[str, Any], training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        moved = self._move_batch(batch)
        tasks = {
            task
            for task in ("phase", "pass_fail", "errors")
            if isinstance(moved.get(f"{task}_available"), torch.Tensor)
            and moved[f"{task}_available"].any()
        }
        if not tasks:
            raise ValueError("Batch has no supervised task labels.")
        exercise_id = str(moved["exercise_id"])
        phase_mask, error_mask = self._task_masks(exercise_id)
        outputs = self.model(
            moved["motionbert_input"],
            exercise_id=exercise_id,
            temporal_mask=moved["temporal_mask"],
            valid_phase_mask=phase_mask,
            valid_error_mask=error_mask,
            tasks=tasks,
        )
        total, losses = self.loss_fn(outputs, moved)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            self.model.mark_trained("experts")
            for task in tasks:
                self.model.mark_trained(task)
        return total.detach(), {name: value.detach() for name, value in losses.items()}

    def run_epoch(self, loader: Iterable[Mapping[str, Any]], training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                _, losses = self._run_batch(batch, training=training)
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.item())
                batches += 1
        if batches == 0:
            raise ValueError("The data loader yielded no batches.")
        return {name: value / batches for name, value in totals.items()}

    def save_checkpoint(self, path: str | Path, epoch: int, validation_loss: float) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "validation_loss": float(validation_loss),
                "trainer_config": asdict(self.config),
            },
            output,
        )
        return output

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Resume model and optimizer state using restricted weight loading."""

        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
            raise ValueError("Trainer checkpoint is missing model/optimizer state.")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        validation_loader: Iterable[Mapping[str, Any]],
        checkpoint_path: str | Path,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        best = float("inf")
        stale_epochs = 0
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.run_epoch(train_loader, training=True)
            validation_metrics = self.run_epoch(validation_loader, training=False)
            validation_loss = validation_metrics["total"]
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
            if validation_loss < best:
                best = validation_loss
                stale_epochs = 0
                self.save_checkpoint(checkpoint_path, epoch, validation_loss)
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    break
        return history
