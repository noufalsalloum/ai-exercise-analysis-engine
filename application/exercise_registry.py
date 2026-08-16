from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from experts.registry import EXPERT_CLASSES

from .contracts import CapabilityFlags


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_VERSION = "physical_mp33_h36m_root_body_scale_xy_conf_pad_v4"
MOTIONBERT_CHECKPOINT = PROJECT_ROOT / "models" / "latest_epoch.bin"
PILOT_RECOGNITION_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "exercise_representation" / "pilot" / "best.pt"
)

ExerciseStatus = Literal["ready", "partial", "development"]
CameraView = Literal["front", "side", "back", "half_profile"]


@dataclass(frozen=True)
class ExerciseDefinition:
    """One product exercise and its real, currently validated capabilities."""

    exercise_id: str
    display_name: str
    family_id: str
    variation_id: str | None
    status: ExerciseStatus
    supported_input_modes: tuple[str, ...]
    recommended_camera_view: CameraView
    allowed_camera_views: tuple[CameraView, ...]
    positioning_instruction: str
    preparation_seconds: int
    supports_repetitions: bool
    supports_hold_time: bool
    capabilities: CapabilityFlags
    expert_id: str | None
    expert_checkpoint: Path | None = None
    phase_checkpoint: Path | None = None
    assessment_checkpoint: Path | None = None
    error_checkpoint: Path | None = None
    score_checkpoint: Path | None = None
    recognition_checkpoint: Path | None = None
    preprocessing_version: str = PREPROCESSING_VERSION
    limitations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def can_analyze(self) -> bool:
        return self.status != "development" and bool(self.supported_input_modes)

    @property
    def family_expert_class(self) -> type | None:
        return EXPERT_CLASSES.get(self.expert_id) if self.expert_id else None

    def to_dict(self) -> dict[str, object]:
        return {
            "exercise_id": self.exercise_id,
            "display_name": self.display_name,
            "family_id": self.family_id,
            "variation_id": self.variation_id,
            "status": self.status,
            "supported_input_modes": list(self.supported_input_modes),
            "recommended_camera_view": self.recommended_camera_view,
            "allowed_camera_views": list(self.allowed_camera_views),
            "positioning_instruction": self.positioning_instruction,
            "preparation_seconds": self.preparation_seconds,
            "supports_repetitions": self.supports_repetitions,
            "supports_hold_time": self.supports_hold_time,
            "capabilities": self.capabilities.to_dict(),
            "expert_id": self.expert_id,
            "expert_checkpoint": str(self.expert_checkpoint) if self.expert_checkpoint else None,
            "phase_checkpoint": str(self.phase_checkpoint) if self.phase_checkpoint else None,
            "assessment_checkpoint": (
                str(self.assessment_checkpoint) if self.assessment_checkpoint else None
            ),
            "error_checkpoint": str(self.error_checkpoint) if self.error_checkpoint else None,
            "score_checkpoint": str(self.score_checkpoint) if self.score_checkpoint else None,
            "recognition_checkpoint": (
                str(self.recognition_checkpoint) if self.recognition_checkpoint else None
            ),
            "preprocessing_version": self.preprocessing_version,
            "limitations": list(self.limitations),
        }


_DEVELOPMENT_LIMITATIONS = (
    "This exercise has no validated end-to-end runtime yet.",
    "Analysis is disabled; no placeholder inference is produced.",
)


def _development(
    exercise_id: str,
    display_name: str,
    family_id: str,
    expert_id: str | None,
    view: CameraView,
    variation_id: str | None = None,
) -> ExerciseDefinition:
    return ExerciseDefinition(
        exercise_id=exercise_id,
        display_name=display_name,
        family_id=family_id,
        variation_id=variation_id,
        status="development",
        supported_input_modes=(),
        recommended_camera_view=view,
        allowed_camera_views=(view,),
        positioning_instruction="Camera guidance will be finalized with the family runtime.",
        preparation_seconds=5,
        supports_repetitions=False,
        supports_hold_time=False,
        capabilities=CapabilityFlags(),
        expert_id=expert_id,
        limitations=_DEVELOPMENT_LIMITATIONS,
    )


def build_default_exercises() -> tuple[ExerciseDefinition, ...]:
    """Build the truthful MVP registry from validated repository capabilities."""

    recognition_checkpoint = (
        PILOT_RECOGNITION_CHECKPOINT if PILOT_RECOGNITION_CHECKPOINT.is_file() else None
    )
    pushup = ExerciseDefinition(
        exercise_id="pushup",
        display_name="Push-up",
        family_id="pushup",
        variation_id="standard",
        status="partial",
        supported_input_modes=("video", "realtime"),
        recommended_camera_view="side",
        allowed_camera_views=("side", "half_profile"),
        positioning_instruction=(
            "Place the camera at body height from the side and keep shoulders, "
            "wrists, hips, knees, and ankles inside the frame."
        ),
        preparation_seconds=5,
        supports_repetitions=True,
        supports_hold_time=False,
        capabilities=CapabilityFlags(phase=True, repetitions=True),
        expert_id="pushup",
        recognition_checkpoint=recognition_checkpoint,
        limitations=(
            "Phase and repetition outputs are rule-based, not learned assessment.",
            "Correctness, pass/fail, specific errors, and quality score are not available.",
            "The Ex3 learned development model is scoped to table/incline Push-ups and is not applied to this standard variation.",
            "The pilot checkpoint is exercise recognition only and is not used as quality evidence.",
        ),
    )
    table_incline_pushup = ExerciseDefinition(
        exercise_id="table_incline_pushup",
        display_name="Table / Incline Push-up",
        family_id="pushup",
        variation_id="table_incline",
        status="partial",
        supported_input_modes=("video", "realtime"),
        recommended_camera_view="side",
        allowed_camera_views=("side", "half_profile"),
        positioning_instruction=(
            "Use a side or half-profile view and keep the table support, shoulders, "
            "wrists, hips, knees, and ankles visible."
        ),
        preparation_seconds=5,
        supports_repetitions=True,
        supports_hold_time=False,
        capabilities=CapabilityFlags(phase=True, repetitions=True),
        expert_id="pushup",
        assessment_checkpoint=PROJECT_ROOT / "checkpoints" / "pushup_ai_v1" / "correctness" / "final_dev.pt",
        recognition_checkpoint=recognition_checkpoint,
        limitations=(
            "The official repetition count remains rule-based.",
            "Boundary V1 and Correctness V1 are experimental development models scoped only to REHAB24 Ex3 table/incline Push-ups.",
            "Detailed error labels are unavailable; failed assessments use the generic Form Issue explanation.",
        ),
    )
    pullup = ExerciseDefinition(
        exercise_id="pullup",
        display_name="Pull-up",
        family_id="pullup",
        variation_id="standard",
        status="partial",
        supported_input_modes=("video", "realtime"),
        recommended_camera_view="front",
        allowed_camera_views=("front", "side"),
        positioning_instruction=(
            "Use a front view when possible. Keep the head, shoulders, elbows, "
            "wrists, upper torso, and the full vertical travel visible throughout the set."
        ),
        preparation_seconds=5,
        supports_repetitions=True,
        supports_hold_time=False,
        capabilities=CapabilityFlags(phase=True, repetitions=True),
        expert_id="pullup",
        recognition_checkpoint=recognition_checkpoint,
        limitations=(
            "Phase and repetition outputs are rule-based, not learned assessment.",
            "TOP is inferred from elbow flexion and body elevation; the bar is not detected.",
            "Correctness, pass/fail, specific errors, and quality score are not available.",
        ),
    )
    plank = ExerciseDefinition(
        exercise_id="plank",
        display_name="Plank",
        family_id="plank",
        variation_id="marching",
        status="partial",
        supported_input_modes=("video", "realtime"),
        recommended_camera_view="side",
        allowed_camera_views=("side", "half_profile"),
        positioning_instruction=(
            "For the supported Marching Plank variation, keep shoulders, hips, "
            "knees, and ankles visible from the side or half-profile."
        ),
        preparation_seconds=5,
        supports_repetitions=True,
        supports_hold_time=False,
        capabilities=CapabilityFlags(phase=True, repetitions=True),
        expert_id="plank",
        recognition_checkpoint=recognition_checkpoint,
        limitations=(
            "The current product runtime supports Marching Plank leg-lift repetitions, not a static hold assessment.",
            "Learned boundary, correctness, detailed errors, and score are unavailable because local clips have no ground truth.",
            "Cross Knee Plank remains in development.",
        ),
    )
    squat = ExerciseDefinition(
        exercise_id="squat", display_name="Squat", family_id="squat", variation_id="standard",
        status="partial", supported_input_modes=("video", "realtime"), recommended_camera_view="side",
        allowed_camera_views=("side",),
        positioning_instruction="Use a side view. Keep shoulder, hip, knee, and ankle visible throughout each squat.",
        preparation_seconds=5, supports_repetitions=True, supports_hold_time=False,
        capabilities=CapabilityFlags(phase=True, repetitions=True), expert_id="squat",
        assessment_checkpoint=PROJECT_ROOT / "checkpoints" / "squat_ai_v3" / "correctness" / "final_dev.pt",
        error_checkpoint=PROJECT_ROOT / "checkpoints" / "squat_error_v1" / "best.pt",
        recognition_checkpoint=recognition_checkpoint,
        limitations=("The large user-facing repetition count remains rule-based.", "Boundary V2, Correctness V3, and Error V1 outputs are experimental development results.", "Back Squat remains in development.", "Quality score is not available."),
    )
    lunge = ExerciseDefinition(
        exercise_id="lunge", display_name="Lunge", family_id="lunge", variation_id="standard",
        status="partial", supported_input_modes=("video","realtime"), recommended_camera_view="side",
        allowed_camera_views=("side","half_profile"),
        positioning_instruction="Use a side or half-profile view. Keep shoulders, hips, both knees, and ankles visible.",
        preparation_seconds=5, supports_repetitions=True, supports_hold_time=False,
        capabilities=CapabilityFlags(phase=True,repetitions=True), expert_id="lunge",
        assessment_checkpoint=None, recognition_checkpoint=recognition_checkpoint,
        limitations=(
            "The official repetition count is an Ex5-calibrated rule-based baseline.",
            "Boundary V1 is an experimental learned detector.",
            "Correctness V1 failed locked-Test generalization and is not enabled in the product.",
            "Detailed errors and Performance Score are unavailable.",
        ),
    )
    return (
        pushup,
        pullup,
        squat,
        lunge,
        plank,
        table_incline_pushup,
        _development("wall_pushup", "Wall Push-up", "pushup", "pushup", "side", "wall"),
        _development("seal_pushup", "Seal Push-up", "pushup", "pushup", "side", "seal"),
        _development("air_squat", "Air Squat", "squat", "squat", "side", "air"),
        _development("back_squat", "Back Squat", "squat", "squat", "side", "back"),
        _development(
            "bodyweight_lunges", "Bodyweight Lunges", "lunge", "lunge", "side", "bodyweight"
        ),
        _development(
            "reverse_dumbbell_lunges",
            "Reverse Dumbbell Lunges",
            "lunge",
            "lunge",
            "side",
            "reverse_dumbbell",
        ),
        _development(
            "cross_knee_plank", "Cross Knee Plank", "plank", "plank", "side", "cross_knee"
        ),
        _development(
            "marching_plank", "Marching Plank", "plank", "plank", "side", "marching"
        ),
        _development("crunch", "Crunch", "crunch", None, "side"),
        _development("inchworm", "Inchworm", "inchworm", None, "side"),
    )


class ExerciseRegistry:
    """Central product registry; UI and runtimes do not duplicate conditions."""

    def __init__(self, exercises: tuple[ExerciseDefinition, ...] | None = None) -> None:
        values = exercises or build_default_exercises()
        self._items = {item.exercise_id: item for item in values}
        if len(self._items) != len(values):
            raise ValueError("Exercise registry contains duplicate exercise_id values.")
        self._validate()

    def _validate(self) -> None:
        for item in self._items.values():
            if item.preprocessing_version != PREPROCESSING_VERSION:
                raise ValueError(
                    f"{item.exercise_id} preprocessing version is incompatible: "
                    f"{item.preprocessing_version!r}."
                )
            if item.can_analyze and not MOTIONBERT_CHECKPOINT.is_file():
                raise FileNotFoundError(f"MotionBERT checkpoint is missing: {MOTIONBERT_CHECKPOINT}")
            if item.status == "development" and item.can_analyze:
                raise ValueError("Development exercises must not expose runnable input modes.")
            if item.expert_id is not None and item.expert_id not in EXPERT_CLASSES:
                raise ValueError(f"Unknown family Expert id: {item.expert_id}")

    def all(self) -> tuple[ExerciseDefinition, ...]:
        return tuple(self._items.values())

    def main_families(self) -> tuple[ExerciseDefinition, ...]:
        """Return only the five family cards currently exposed by the MVP UI."""

        return tuple(
            self.get(exercise_id)
            for exercise_id in ("pushup", "pullup", "squat", "lunge", "plank")
        )

    def get(self, exercise_id: str) -> ExerciseDefinition:
        try:
            return self._items[exercise_id]
        except KeyError as exc:
            raise KeyError(f"Unknown product exercise: {exercise_id}") from exc

    def require_runnable(self, exercise_id: str, input_mode: str) -> ExerciseDefinition:
        item = self.get(exercise_id)
        if not item.can_analyze:
            raise RuntimeError(f"{item.display_name} is In Development; analysis is disabled.")
        if input_mode not in item.supported_input_modes:
            raise RuntimeError(f"{item.display_name} does not support {input_mode!r} input.")
        return item
