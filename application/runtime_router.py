from __future__ import annotations

import json
from pathlib import Path

from inference.pullup_runtime import PullupRepConfig, PullupRepetitionRuntime
from inference.pushup_runtime import PushupRepConfig, PushupRepetitionRuntime
from inference.squat_runtime import SquatRepConfig, SquatRepetitionRuntime
from inference.plank_runtime import MarchingPlankConfig, MarchingPlankRepetitionRuntime
from inference.lunge_runtime import LungeRepConfig, LungeRepetitionRuntime

from .exercise_registry import ExerciseDefinition, ExerciseRegistry, PROJECT_ROOT


class FamilyRuntimeRouter:
    """Create family runtimes without creating neural Experts during frame updates."""

    def __init__(self, registry: ExerciseRegistry | None = None) -> None:
        self.registry = registry or ExerciseRegistry()

    def create(
        self,
        exercise_id: str,
        input_mode: str,
        camera_view: str | None = None,
    ) -> PushupRepetitionRuntime | PullupRepetitionRuntime | SquatRepetitionRuntime | MarchingPlankRepetitionRuntime | LungeRepetitionRuntime:
        exercise = self.registry.require_runnable(exercise_id, input_mode)
        allowed_variations = {
            "pushup": {"standard", "table_incline"},
            "pullup": {"standard"},
            "squat": {"standard"},
            "plank": {"standard", "marching"},
            "lunge": {"standard"},
        }
        if exercise.variation_id not in allowed_variations.get(exercise.family_id, set()):
            raise RuntimeError(f"No validated runtime is installed for {exercise.display_name}.")
        config_path = PROJECT_ROOT / "configs" / f"{exercise.family_id}.json"
        with config_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        runtime_config = payload.get("runtime", {}).get("repetition_counter", {})
        if exercise.family_id == "pushup":
            return PushupRepetitionRuntime(PushupRepConfig.from_dict(runtime_config))
        if exercise.family_id == "pullup":
            return PullupRepetitionRuntime(PullupRepConfig.from_dict(runtime_config), camera_view=camera_view or exercise.recommended_camera_view)
        if exercise.family_id == "squat":
            return SquatRepetitionRuntime(SquatRepConfig.from_dict(runtime_config), camera_view=camera_view or exercise.recommended_camera_view)
        if exercise.family_id == "lunge":
            return LungeRepetitionRuntime(LungeRepConfig.from_dict(runtime_config), camera_view=camera_view or exercise.recommended_camera_view)
        return MarchingPlankRepetitionRuntime(MarchingPlankConfig.from_dict(runtime_config), camera_view=camera_view or exercise.recommended_camera_view)

    @staticmethod
    def family_expert_class(exercise: ExerciseDefinition) -> type | None:
        """Return the existing family Expert type; variations share this mapping."""

        return exercise.family_expert_class
