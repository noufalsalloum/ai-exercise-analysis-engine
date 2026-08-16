"""Runtime-to-service feedback events with no audio implementation details."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


FeedbackEventType = Literal["first_repetition_completed"]


@dataclass(frozen=True)
class FeedbackEvent:
    """One immutable semantic event emitted by an exercise runtime."""

    event_id: str
    event_type: FeedbackEventType
    session_id: str
    exercise_id: str
    family_id: str
    unit_index: int
    timestamp_seconds: float
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def first_repetition_completed(
        cls,
        *,
        session_id: str,
        exercise_id: str,
        family_id: str,
        timestamp_seconds: float,
        language: str | None = None,
    ) -> "FeedbackEvent":
        return cls(
            event_id=f"{session_id}:first_repetition_completed:1",
            event_type="first_repetition_completed",
            session_id=session_id,
            exercise_id=exercise_id,
            family_id=family_id,
            unit_index=1,
            timestamp_seconds=float(timestamp_seconds),
            language=language,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
