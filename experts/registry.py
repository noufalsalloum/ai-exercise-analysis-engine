from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from .lunge_expert import LungeExpert
from .plank_expert import PlankExpert
from .pullup_expert import PullupExpert
from .pushup_expert import PushupExpert
from .squat_expert import SquatExpert


EXPERT_CLASSES: dict[str, type[nn.Module]] = {
    "plank": PlankExpert,
    "squat": SquatExpert,
    "pushup": PushupExpert,
    "pullup": PullupExpert,
    "lunge": LungeExpert,
}


def normalize_exercise_id(name: str) -> str:
    """Normalize common dataset spellings to the registry identifier."""

    normalized = name.lower().strip().replace("-", "").replace("_", "")
    normalized = "".join(normalized.split())
    aliases = {
        "pushup": "pushup",
        "pullup": "pullup",
        "plank": "plank",
        "squat": "squat",
        "lunge": "lunge",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown exercise '{name}'. Available: {sorted(EXPERT_CLASSES)}"
        )
    return aliases[normalized]


def create_expert(name: str, **kwargs: object) -> nn.Module:
    """Create an expert during model construction, never during forward."""

    exercise_id = normalize_exercise_id(name)
    return EXPERT_CLASSES[exercise_id](**kwargs)


class ExpertRegistry(nn.Module):
    """Persistent expert collection registered through ``nn.ModuleDict``."""

    def __init__(
        self,
        exercise_ids: Optional[Iterable[str]] = None,
        **expert_kwargs: object,
    ) -> None:
        super().__init__()
        ids = list(exercise_ids or EXPERT_CLASSES.keys())
        normalized_ids = [normalize_exercise_id(item) for item in ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("exercise_ids contains duplicate normalized names.")

        self.experts = nn.ModuleDict(
            {
                exercise_id: create_expert(exercise_id, **expert_kwargs)
                for exercise_id in normalized_ids
            }
        )

    @property
    def available_exercises(self) -> tuple[str, ...]:
        return tuple(self.experts.keys())

    def get(self, exercise_id: str) -> nn.Module:
        normalized = normalize_exercise_id(exercise_id)
        if normalized not in self.experts:
            raise ValueError(
                f"Expert '{normalized}' is not registered. "
                f"Registered: {list(self.experts.keys())}"
            )
        return self.experts[normalized]

    def forward(
        self,
        motionbert_features: torch.Tensor,
        exercise_id: str,
        temporal_mask: Optional[torch.Tensor] = None,
        joint_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        expert = self.get(exercise_id)
        return expert(
            motionbert_features,
            temporal_mask=temporal_mask,
            joint_mask=joint_mask,
        )


def get_expert(name: str, **kwargs: object) -> nn.Module:
    """Backward-compatible construction helper.

    Production forward paths must use :class:`ExpertRegistry`, which keeps
    instances registered and persistent. This helper is intended for setup,
    isolated tests, and migration of older callers.
    """

    return create_expert(name, **kwargs)
