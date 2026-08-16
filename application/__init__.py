"""Unified application coordination for the exercise-analysis engine."""

from .contracts import CapabilityFlags, RepetitionResult, SessionResult
from .exercise_registry import ExerciseDefinition, ExerciseRegistry
from .session import SessionResultAggregator

__all__ = [
    "CapabilityFlags",
    "ExerciseDefinition",
    "ExerciseRegistry",
    "RepetitionResult",
    "SessionResult",
    "SessionResultAggregator",
]
