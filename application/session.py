from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .contracts import (
    RepetitionResult,
    SessionResult,
    SessionSummary,
)
from .exercise_registry import ExerciseDefinition, MOTIONBERT_CHECKPOINT


class SessionResultAggregator:
    """Accumulate live rule-based events into the standardized session contract."""

    def __init__(
        self,
        exercise: ExerciseDefinition,
        input_mode: str,
        camera_view: str,
        video_path: str | Path | None = None,
    ) -> None:
        self.exercise = exercise
        self.input_mode = input_mode
        self.camera_view = camera_view
        self.video_path = str(video_path) if video_path is not None else None
        self.reset()

    def reset(self) -> None:
        self.session_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self.repetitions: list[RepetitionResult] = []
        self.phase_counts: Counter[str] = Counter()
        self.incomplete_cycles = 0
        self.hold_duration: float | None = None
        self.experimental_ai: dict | None = None

    def record_phase(self, phase: str | None) -> None:
        if phase and phase not in {"UNKNOWN", "PREPARING"}:
            self.phase_counts[phase] += 1

    def add_repetition(self, repetition: RepetitionResult) -> None:
        self.repetitions.append(repetition)

    def set_incomplete_cycles(self, count: int) -> None:
        self.incomplete_cycles = max(0, int(count))

    def set_hold_duration(self, seconds: float | None) -> None:
        self.hold_duration = None if seconds is None else max(0.0, float(seconds))

    def set_experimental_ai(self, result: dict | None) -> None:
        """Attach a separately labelled MVP result without changing official counts."""

        self.experimental_ai = result

    @staticmethod
    def _mean_available(values: list[float | None]) -> float | None:
        available = [float(value) for value in values if value is not None]
        return sum(available) / len(available) if available else None

    def finalize(self, duration_seconds: float) -> SessionResult:
        ended_at = datetime.now(timezone.utc)
        durations = [item.duration_seconds for item in self.repetitions]
        scores = [item.score for item in self.repetitions if item.score is not None]
        errors = [
            str(error.get("error"))
            for item in self.repetitions
            for error in item.errors
            if error.get("error")
        ]
        most_frequent_error = Counter(errors).most_common(1)[0][0] if errors else None
        summary = SessionSummary(
            total_repetitions=(
                len(self.repetitions) if self.exercise.capabilities.repetitions else None
            ),
            correct_repetitions=None,
            incorrect_repetitions=None,
            average_score=self._mean_available(scores),
            best_score=max(scores) if scores else None,
            lowest_score=min(scores) if scores else None,
            most_frequent_error=most_frequent_error,
            average_repetition_duration=self._mean_available(durations),
            last_repetition_duration=(durations[-1] if durations else None),
            performance_consistency=None,
            hold_duration=self.hold_duration,
            posture_breaks=None,
            completed_cycles=(
                len(self.repetitions) if self.exercise.capabilities.repetitions else None
            ),
            incomplete_cycles=(
                self.incomplete_cycles if self.exercise.capabilities.repetitions else None
            ),
            phase_activity=dict(sorted(self.phase_counts.items())),
        )
        return SessionResult(
            session_id=self.session_id,
            exercise_id=self.exercise.exercise_id,
            family_id=self.exercise.family_id,
            input_mode=self.input_mode,  # type: ignore[arg-type]
            video_path=self.video_path,
            camera_view=self.camera_view,
            started_at=self.started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            duration_seconds=max(0.0, float(duration_seconds)),
            preprocessing_version=self.exercise.preprocessing_version,
            motionbert_checkpoint=str(MOTIONBERT_CHECKPOINT),
            family_checkpoint=(
                str(self.exercise.expert_checkpoint)
                if self.exercise.expert_checkpoint is not None
                else None
            ),
            capabilities=self.exercise.capabilities,
            repetitions=list(self.repetitions),
            summary=summary,
            limitations=list(self.exercise.limitations),
            sources={
                "phase": (
                    "rule_based_repetition_counting"
                    if self.exercise.capabilities.phase
                    else "unavailable"
                ),
                "repetitions": (
                    "rule_based_repetition_counting"
                    if self.exercise.capabilities.repetitions
                    else "unavailable"
                ),
                "pass_fail": "unavailable",
                "errors": "unavailable",
                "score": "unavailable",
                "similarity": "unavailable",
            },
            experimental_ai=self.experimental_ai,
        )
