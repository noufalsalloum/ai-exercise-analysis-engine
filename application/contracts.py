from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ResultSource = Literal[
    "learned_model",
    "rule_based_repetition_counting",
    "prototype_similarity",
    "unavailable",
]


@dataclass(frozen=True)
class CapabilityFlags:
    """Truthful availability flags for one exercise/runtime."""

    recognition: bool = False
    repetitions: bool = False
    phase: bool = False
    pass_fail: bool = False
    errors: bool = False
    score: bool = False
    similarity: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass
class RepetitionResult:
    """Standard result for one completed repetition."""

    repetition_index: int
    start_frame: int | None = None
    end_frame: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    duration_seconds: float | None = None
    phase_sequence: list[str] = field(default_factory=list)
    score: float | None = None
    pass_fail: str | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    assessment_source: ResultSource | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSummary:
    """Session aggregates; unavailable learned metrics deliberately stay ``None``."""

    total_repetitions: int | None = None
    correct_repetitions: int | None = None
    incorrect_repetitions: int | None = None
    average_score: float | None = None
    best_score: float | None = None
    lowest_score: float | None = None
    most_frequent_error: str | None = None
    average_repetition_duration: float | None = None
    last_repetition_duration: float | None = None
    performance_consistency: float | None = None
    hold_duration: float | None = None
    posture_breaks: int | None = None
    completed_cycles: int | None = None
    incomplete_cycles: int | None = None
    phase_activity: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionResult:
    """Serializable result contract shared by uploaded and real-time sessions."""

    session_id: str
    exercise_id: str
    family_id: str
    input_mode: Literal["video", "realtime"]
    video_path: str | None
    camera_view: str
    started_at: str
    ended_at: str
    duration_seconds: float
    preprocessing_version: str
    motionbert_checkpoint: str
    family_checkpoint: str | None
    capabilities: CapabilityFlags
    repetitions: list[RepetitionResult]
    summary: SessionSummary
    limitations: list[str]
    sources: dict[str, ResultSource] = field(default_factory=dict)
    experimental_ai: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = self.capabilities.to_dict()
        payload["summary"] = self.summary.to_dict()
        payload["repetitions"] = [item.to_dict() for item in self.repetitions]
        return payload
