"""Comparison-only averaged Squat Correctness V3 inference.

The active full-repetition V3 decision and robustness policy are never changed
by this module. Temporal crops are an inference experiment: each crop is
resampled to the original fixed ``(60, 17, 3)`` input contract before the
unchanged frozen model is called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch

from inference.squat_robustness import DEFAULT_CONFIG, SquatRobustnessPolicy
from tools.squat_ai.prepare_rehab24_squat import resample_sequence


SUPPORTED_SAMPLE_COUNTS = (3, 5, 7)


@dataclass(frozen=True)
class AveragedCorrectnessConfig:
    """Fixed comparison settings loaded from the robustness configuration."""

    enabled: bool
    default_sample_count: int
    comparison_sample_counts: tuple[int, ...]
    centers_start: float
    centers_end: float
    crop_fraction: float
    target_frames: int
    raw_threshold: float
    calibrated_threshold: float
    mode: str
    changes_active_assessment: bool

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "AveragedCorrectnessConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        data = payload["averaged_correctness_experiment"]
        config = cls(
            enabled=bool(data["enabled"]),
            default_sample_count=int(data["default_sample_count"]),
            comparison_sample_counts=tuple(
                int(value) for value in data["comparison_sample_counts"]
            ),
            centers_start=float(data["normalized_centers_start"]),
            centers_end=float(data["normalized_centers_end"]),
            crop_fraction=float(data["crop_fraction"]),
            target_frames=int(data["target_frames"]),
            raw_threshold=float(data["raw_threshold"]),
            calibrated_threshold=float(data["calibrated_threshold"]),
            mode=str(data["mode"]),
            changes_active_assessment=bool(data["changes_active_assessment"]),
        )
        config.validate()
        return config

    def with_enabled(self, enabled: bool) -> "AveragedCorrectnessConfig":
        """Return an explicit runtime override without writing the config file."""

        return replace(self, enabled=bool(enabled))

    def validate(self) -> None:
        if self.default_sample_count not in self.comparison_sample_counts:
            raise ValueError("Default sample count must be in comparison_sample_counts.")
        if any(value not in SUPPORTED_SAMPLE_COUNTS for value in self.comparison_sample_counts):
            raise ValueError("Only the predefined 3/5/7 sample comparison is supported.")
        if not 0.0 <= self.centers_start < self.centers_end <= 1.0:
            raise ValueError("Normalized temporal center bounds must be inside [0,1].")
        if not 0.0 < self.crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be inside (0,1].")
        if self.target_frames != 60:
            raise ValueError("Correctness V3 requires the trained 60-frame contract.")
        if not 0.0 < self.raw_threshold < 1.0:
            raise ValueError("Raw threshold must be inside (0,1).")
        if not 0.0 < self.calibrated_threshold < 1.0:
            raise ValueError("Calibrated threshold must be inside (0,1).")
        if self.mode != "comparison_only" or self.changes_active_assessment:
            raise ValueError("Averaged inference is allowed only in comparison-only mode.")


def aggregate_probabilities(probabilities: Sequence[float], threshold: float) -> dict[str, Any]:
    """Compute deterministic mean/median/stability and threshold decisions."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("probabilities must be finite values inside [0,1].")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be inside (0,1).")
    mean = float(values.mean())
    median = float(np.median(values))
    return {
        "probabilities": values.tolist(),
        "mean_probability": mean,
        "median_probability": median,
        "std_probability": float(values.std(ddof=0)),
        "min_probability": float(values.min()),
        "max_probability": float(values.max()),
        "threshold": float(threshold),
        "mean_decision": "PASS" if mean >= threshold else "FAIL",
        "median_decision": "PASS" if median >= threshold else "FAIL",
    }


def provisional_score(comparisons: Sequence[dict[str, Any]]) -> float | None:
    """Return raw mean-decision pass rate; errors never affect this score."""

    decisions = [
        str(row["raw"]["mean_decision"])
        for row in comparisons
        if row.get("raw", {}).get("mean_decision") in {"PASS", "FAIL"}
    ]
    if not decisions:
        return None
    return float(round(100.0 * sum(value == "PASS" for value in decisions) / len(decisions), 1))


class AveragedCorrectnessExperiment:
    """Run batched temporal-crop comparisons through unchanged Correctness V3."""

    def __init__(
        self,
        config: AveragedCorrectnessConfig,
        robustness_policy: SquatRobustnessPolicy | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.robustness_policy = robustness_policy or SquatRobustnessPolicy.load()
        mapped = self.robustness_policy.calibrate(config.raw_threshold)
        if not np.isclose(mapped, config.calibrated_threshold, atol=1e-9):
            raise ValueError("Configured calibrated threshold does not map from raw threshold.")

    def _temporal_views(
        self, sequence: np.ndarray, sample_count: int
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (17, 3):
            raise ValueError(f"Expected normalized sequence (T,17,3), got {values.shape}.")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Normalized repetition must contain at least two finite frames.")
        if sample_count not in self.config.comparison_sample_counts:
            raise ValueError(f"sample_count must be one of {self.config.comparison_sample_counts}.")

        centers = np.linspace(
            self.config.centers_start,
            self.config.centers_end,
            sample_count,
            dtype=np.float64,
        )
        crop_frames = max(2, min(len(values), int(round(len(values) * self.config.crop_fraction))))
        maximum_start = len(values) - crop_frames
        crops: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for center in centers:
            center_index = int(round(float(center) * (len(values) - 1)))
            start = min(max(center_index - crop_frames // 2, 0), maximum_start)
            end = start + crop_frames
            crop = resample_sequence(values[start:end], target=self.config.target_frames)
            crop[..., 2] = np.clip(crop[..., 2], 0.0, 1.0)
            crops.append(crop)
            metadata.append(
                {
                    "normalized_center": float(center),
                    "center_frame": center_index,
                    "crop_start": int(start),
                    "crop_end_exclusive": int(end),
                    "source_crop_frames": int(crop_frames),
                }
            )
        return np.stack(crops).astype(np.float32), metadata

    @torch.no_grad()
    def compare(
        self,
        model: torch.nn.Module,
        normalized_segment: np.ndarray,
        device: torch.device | str,
        *,
        sample_count: int | None = None,
    ) -> dict[str, Any] | None:
        """Return A/B probabilities without mutating the active rep assessment."""

        if not self.config.enabled:
            return None
        count = int(sample_count or self.config.default_sample_count)
        total_started = perf_counter()
        crops, crop_metadata = self._temporal_views(normalized_segment, count)
        masks = crops[..., 2].mean(axis=2) > 0.01
        if not bool(masks.any(axis=1).all()):
            raise ValueError("Every temporal crop must contain a valid confidence frame.")
        model_started = perf_counter()
        output = model(
            torch.from_numpy(crops).to(device),
            torch.from_numpy(masks).to(device),
        )
        model_latency_ms = (perf_counter() - model_started) * 1000.0
        raw_values = output["correct_probability"].detach().cpu().numpy().astype(np.float64)
        calibrated_values = np.asarray(
            [self.robustness_policy.calibrate(value) for value in raw_values],
            dtype=np.float64,
        )
        return {
            "experimental": True,
            "comparison_only": True,
            "changes_active_assessment": False,
            "sample_count": count,
            "target_frames": self.config.target_frames,
            "crop_fraction": self.config.crop_fraction,
            "crop_metadata": crop_metadata,
            "raw": aggregate_probabilities(raw_values, self.config.raw_threshold),
            "calibrated": aggregate_probabilities(
                calibrated_values, self.config.calibrated_threshold
            ),
            "inference_latency_ms": float(model_latency_ms),
            "total_latency_ms": float((perf_counter() - total_started) * 1000.0),
            "latency_per_window_ms": float(model_latency_ms / count),
        }
