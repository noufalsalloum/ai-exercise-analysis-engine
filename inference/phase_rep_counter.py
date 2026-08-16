from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


DEFAULT_PHASE_CYCLES: dict[str, tuple[str, ...]] = {
    "squat": ("REST", "DESCENDING", "BOTTOM", "ASCENDING", "REST"),
    "pushup": ("TOP", "DESCENDING", "BOTTOM", "ASCENDING", "TOP"),
    "pullup": ("REST", "ASCENDING", "TOP", "DESCENDING", "REST"),
    "lunge": ("REST", "DESCENDING", "BOTTOM", "RETURNING", "REST"),
}


@dataclass(frozen=True)
class StablePhase:
    """A debounced phase segment in the original timeline."""

    phase: str
    start_index: int
    end_index: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class PhaseRepCounter:
    """Count validated phase cycles without relying on joint-angle thresholds."""

    def __init__(
        self,
        exercise_id: str,
        phase_vocabulary: Optional[Sequence[str]] = None,
        smoothing_window: int = 3,
        min_phase_frames: int = 2,
        min_phase_duration: float = 0.0,
    ) -> None:
        if smoothing_window < 1 or smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer.")
        if min_phase_frames < 1:
            raise ValueError("min_phase_frames must be positive.")
        if min_phase_duration < 0:
            raise ValueError("min_phase_duration cannot be negative.")
        self.exercise_id = exercise_id.strip().lower().replace("-", "")
        self.phase_vocabulary = (
            tuple(str(item).upper() for item in phase_vocabulary)
            if phase_vocabulary is not None
            else None
        )
        self.smoothing_window = smoothing_window
        self.min_phase_frames = min_phase_frames
        self.min_phase_duration = min_phase_duration

    def _timestamps(self, count: int, timestamps: Optional[Sequence[float]]) -> np.ndarray:
        if timestamps is None:
            return np.arange(count, dtype=np.float64)
        values = np.asarray(timestamps, dtype=np.float64)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError("timestamps must be finite and match the sequence length.")
        if count > 1 and np.any(np.diff(values) <= 0):
            raise ValueError("timestamps must be strictly increasing.")
        return values

    def _from_probabilities(self, probabilities: np.ndarray) -> list[str]:
        values = np.asarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("phase probabilities must have shape (T, P).")
        if not np.isfinite(values).all():
            raise ValueError("phase probabilities contain NaN or infinity.")
        if self.phase_vocabulary is None or values.shape[1] != len(self.phase_vocabulary):
            raise ValueError("phase_vocabulary must match the probability dimension.")

        radius = self.smoothing_window // 2
        smoothed = np.empty_like(values)
        for index in range(len(values)):
            begin = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            smoothed[index] = values[begin:end].mean(axis=0)
        indices = np.argmax(smoothed, axis=1)
        return [self.phase_vocabulary[int(index)] for index in indices]

    def _smooth_labels(self, phases: Sequence[str]) -> list[str]:
        labels = [str(item).upper() for item in phases]
        if not labels:
            return []
        radius = self.smoothing_window // 2
        result: list[str] = []
        for index, original in enumerate(labels):
            begin = max(0, index - radius)
            end = min(len(labels), index + radius + 1)
            counts = Counter(labels[begin:end])
            maximum = max(counts.values())
            candidates = {key for key, value in counts.items() if value == maximum}
            result.append(original if original in candidates else sorted(candidates)[0])
        return result

    def _debounce(self, phases: Sequence[str], timestamps: np.ndarray) -> list[StablePhase]:
        if not phases:
            return []
        segments: list[StablePhase] = []
        start = 0
        for index in range(1, len(phases) + 1):
            if index < len(phases) and phases[index] == phases[start]:
                continue
            frame_count = index - start
            end = index - 1
            duration = float(timestamps[end] - timestamps[start])
            if frame_count >= self.min_phase_frames and duration >= self.min_phase_duration:
                segment = StablePhase(
                    phase=phases[start],
                    start_index=start,
                    end_index=end,
                    start_time=float(timestamps[start]),
                    end_time=float(timestamps[end]),
                )
                if segments and segments[-1].phase == segment.phase:
                    previous = segments[-1]
                    segments[-1] = StablePhase(
                        phase=previous.phase,
                        start_index=previous.start_index,
                        end_index=segment.end_index,
                        start_time=previous.start_time,
                        end_time=segment.end_time,
                    )
                else:
                    segments.append(segment)
            start = index
        return segments

    @staticmethod
    def _count_cycles(segments: Sequence[StablePhase], cycle: Sequence[str]) -> int:
        if len(cycle) < 3 or cycle[0] != cycle[-1]:
            raise ValueError("A repetition cycle must begin and end at the same phase.")
        progress = 0
        repetitions = 0
        for segment in segments:
            phase = segment.phase
            if progress == 0:
                if phase == cycle[0]:
                    progress = 1
                continue
            expected = cycle[progress]
            previous = cycle[progress - 1]
            if phase == previous:
                continue
            if phase == expected:
                progress += 1
                if progress == len(cycle):
                    repetitions += 1
                    progress = 1
                continue
            # An invalid jump cancels the partial cycle. A new start phase can
            # immediately seed the next attempt.
            progress = 1 if phase == cycle[0] else 0
        return repetitions

    def evaluate(
        self,
        phases_or_probabilities: Sequence[str] | np.ndarray,
        timestamps: Optional[Sequence[float]] = None,
    ) -> dict[str, object]:
        """Return validated reps, or hold duration for an isometric plank."""

        if isinstance(phases_or_probabilities, np.ndarray) and phases_or_probabilities.ndim == 2:
            phases = self._from_probabilities(phases_or_probabilities)
        else:
            phases = self._smooth_labels(list(phases_or_probabilities))
        times = self._timestamps(len(phases), timestamps)
        segments = self._debounce(phases, times)

        if self.exercise_id == "plank":
            hold_duration = sum(item.duration for item in segments if item.phase == "HOLD")
            return {
                "available": True,
                "mode": "hold",
                "count": 0,
                "hold_duration": float(hold_duration),
                "stable_phases": [item.phase for item in segments],
            }

        cycle = DEFAULT_PHASE_CYCLES.get(self.exercise_id)
        if cycle is None:
            return {
                "available": False,
                "mode": "repetitions",
                "count": 0,
                "reason": f"No phase cycle configured for '{self.exercise_id}'.",
                "stable_phases": [item.phase for item in segments],
            }
        return {
            "available": True,
            "mode": "repetitions",
            "count": self._count_cycles(segments, cycle),
            "stable_phases": [item.phase for item in segments],
        }
