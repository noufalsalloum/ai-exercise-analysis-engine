from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


ROOT_JOINT: Final[int] = 0
BODY_SCALE_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (11, 14),  # shoulder width
    (1, 4),    # hip width
    (0, 8),    # pelvis-to-neck torso length
)


@dataclass(frozen=True)
class CoordinateNormalizationDiagnostics:
    """Per-segment evidence emitted by H36M coordinate normalization."""

    sequence_scale: float
    frame_scales: np.ndarray
    jump_scores: np.ndarray
    near_zero_scale_mask: np.ndarray
    jump_outlier_mask: np.ndarray
    clipped_mask: np.ndarray
    outlier_mask: np.ndarray


class H36MCoordinateNormalizer:
    """Root-center and robustly scale H36M XY without dataset-wide statistics.

    A single scale is calculated per contiguous sequence as the median of valid
    per-frame body scales. Each frame uses the largest of shoulder width, hip
    width, and pelvis-to-neck torso length so foreshortened side-view widths do
    not collapse the scale. The temporal median rejects isolated measurement
    spikes without using dataset-level statistics. MediaPipe ``z`` never
    influences MotionBERT ``x/y``.
    """

    ROOT_JOINT = ROOT_JOINT
    SCALE_METHOD = "sequence_median_of_frame_max(shoulder_width,hip_width,torso_length)"
    CLIPPING_METHOD = "fixed_normalized_xy_clip[-4,4]+robust_temporal_jump_flag"
    CONFIDENCE_POLICY = (
        "observed=1.0;short_gap_interpolated=0.5;"
        "outlier=min(source,0.25);invalid_or_near_zero_scale=0.0;"
        "derived_joint_confidence=min(source_confidences)"
    )

    def __init__(
        self,
        *,
        minimum_scale: float = 1e-6,
        coordinate_clip: float = 4.0,
        minimum_jump_threshold: float = 1.5,
        outlier_confidence: float = 0.25,
    ) -> None:
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive.")
        if coordinate_clip <= 0 or minimum_jump_threshold <= 0:
            raise ValueError("Clipping and jump thresholds must be positive.")
        if not 0.0 <= outlier_confidence <= 1.0:
            raise ValueError("outlier_confidence must be within [0,1].")
        self.minimum_scale = float(minimum_scale)
        self.coordinate_clip = float(coordinate_clip)
        self.minimum_jump_threshold = float(minimum_jump_threshold)
        self.outlier_confidence = float(outlier_confidence)

    def normalize(
        self, h36m_xy_confidence: np.ndarray
    ) -> tuple[np.ndarray, CoordinateNormalizationDiagnostics]:
        """Normalize one contiguous ``(T,17,3)`` sequence and return diagnostics."""

        values = np.asarray(h36m_xy_confidence, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (17, 3):
            raise ValueError(f"Expected (T,17,3), got {tuple(values.shape)}.")
        if len(values) == 0:
            raise ValueError("Cannot normalize an empty H36M sequence.")
        if not np.isfinite(values).all():
            raise ValueError("H36M coordinates and confidence must be finite.")
        confidence = values[..., 2]
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("Confidence must be within [0,1].")

        xy = values[..., :2]
        centered = xy - xy[:, ROOT_JOINT : ROOT_JOINT + 1, :]
        scale_candidates = np.stack(
            [np.linalg.norm(centered[:, left] - centered[:, right], axis=-1) for left, right in BODY_SCALE_PAIRS],
            axis=1,
        )
        valid_candidates = np.isfinite(scale_candidates) & (scale_candidates > self.minimum_scale)
        frame_scales = np.asarray(
            [
                np.max(row[valid]) if np.any(valid) else 0.0
                for row, valid in zip(scale_candidates, valid_candidates)
            ],
            dtype=np.float32,
        )
        near_zero = frame_scales <= self.minimum_scale
        valid_frame_scales = frame_scales[~near_zero]
        if len(valid_frame_scales) == 0:
            sequence_scale = 1.0
        else:
            sequence_scale = float(np.median(valid_frame_scales))
        if not np.isfinite(sequence_scale) or sequence_scale <= self.minimum_scale:
            sequence_scale = 1.0
            near_zero[:] = True

        normalized_xy = centered / np.float32(sequence_scale)
        jump_scores = np.zeros(len(values), dtype=np.float32)
        if len(values) > 1:
            displacement = np.linalg.norm(np.diff(normalized_xy, axis=0), axis=-1)
            jump_scores[1:] = np.median(displacement, axis=1)
        finite_scores = jump_scores[np.isfinite(jump_scores)]
        score_median = float(np.median(finite_scores)) if len(finite_scores) else 0.0
        score_mad = float(np.median(np.abs(finite_scores - score_median))) if len(finite_scores) else 0.0
        robust_threshold = max(
            self.minimum_jump_threshold,
            score_median + 6.0 * 1.4826 * score_mad,
        )
        jump_outlier = jump_scores > robust_threshold
        clipped = np.any(np.abs(normalized_xy) > self.coordinate_clip, axis=(1, 2))
        normalized_xy = np.clip(
            normalized_xy, -self.coordinate_clip, self.coordinate_clip
        ).astype(np.float32)
        outlier = jump_outlier | clipped

        output = values.copy()
        output[..., :2] = normalized_xy
        output[near_zero, :, 2] = 0.0
        if np.any(outlier):
            output[outlier, :, 2] = np.minimum(
                output[outlier, :, 2], self.outlier_confidence
            )
        if not np.isfinite(output).all():
            raise FloatingPointError("Normalization produced non-finite values.")
        if np.any((output[..., 2] < 0.0) | (output[..., 2] > 1.0)):
            raise FloatingPointError("Normalization produced confidence outside [0,1].")

        diagnostics = CoordinateNormalizationDiagnostics(
            sequence_scale=sequence_scale,
            frame_scales=frame_scales,
            jump_scores=jump_scores,
            near_zero_scale_mask=near_zero,
            jump_outlier_mask=jump_outlier,
            clipped_mask=clipped,
            outlier_mask=outlier,
        )
        return output, diagnostics
