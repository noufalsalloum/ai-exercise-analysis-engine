"""Observational Squat Boundary V2 inference that never changes product results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from training.squat_rep_boundary_v2 import v2_segments


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SquatShadowConfig:
    """Configuration for the non-user-facing Boundary V2 observer."""

    enabled: bool
    checkpoint: Path
    output_dir: Path
    device: str
    minimum_frames: int
    contract: str
    user_facing_count_source: str

    @classmethod
    def load(cls, path: Path | None = None) -> "SquatShadowConfig":
        config_path = path or ROOT / "configs" / "squat_shadow.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else ROOT / candidate

        return cls(
            enabled=bool(data["enabled"]),
            checkpoint=resolve(str(data["checkpoint"])),
            output_dir=resolve(str(data["output_dir"])),
            device=str(data.get("device", "auto")),
            minimum_frames=int(data.get("minimum_frames", 38)),
            contract=str(data["contract"]),
            user_facing_count_source=str(data["user_facing_count_source"]),
        )


class SquatBoundaryShadow:
    """Collect active-session landmarks and compare AI boundaries after a session.

    The class has no reference to UI widgets, session aggregators, or the visible
    repetition counter. Its only side effect is a JSON comparison log.
    """

    def __init__(
        self,
        config: SquatShadowConfig,
        *,
        model: SquatRepBoundaryV2Model | None = None,
        postprocessing: dict[str, Any] | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("Cannot instantiate a disabled Squat shadow observer.")
        self.config = config
        self.device = torch.device(
            "cuda"
            if config.device == "auto" and torch.cuda.is_available()
            else ("cpu" if config.device == "auto" else config.device)
        )
        self.model = model
        self.checkpoint_sha256: str | None = None
        if self.model is None:
            checkpoint = torch.load(
                config.checkpoint, map_location=self.device, weights_only=True
            )
            experiment = checkpoint.get("experiment", {})
            if experiment.get("architecture") != "boundary_aux_tcn":
                raise ValueError("Squat shadow requires the Boundary V2 dual-head checkpoint.")
            self.model = SquatRepBoundaryV2Model()
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            postprocessing = dict(checkpoint["postprocessing"])
            self.checkpoint_sha256 = _sha256(config.checkpoint)
        if postprocessing is None:
            raise ValueError("Boundary postprocessing configuration is required.")
        self.postprocessing = dict(postprocessing)
        self.model.to(self.device).eval()
        self.selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}})
        self.normalizer = H36MCoordinateNormalizer()
        self.reset()

    @classmethod
    def from_default_config(cls) -> "SquatBoundaryShadow | None":
        config = SquatShadowConfig.load()
        return cls(config) if config.enabled else None

    def reset(self) -> None:
        """Discard all data from a previous session."""

        self._landmarks: list[np.ndarray] = []
        self._frame_indices: list[int] = []
        self._timestamps: list[float] = []
        self._rule_cycles: list[dict[str, Any]] = []
        self._last_valid: np.ndarray | None = None

    def record_frame(
        self,
        landmarks_33: np.ndarray | None,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Record one active-analysis frame without affecting runtime state."""

        if landmarks_33 is None:
            if self._last_valid is None:
                value = np.zeros((33, 4), dtype=np.float32)
            else:
                value = self._last_valid.copy()
            value[:, 3] = 0.0
        else:
            value = np.asarray(landmarks_33, dtype=np.float32)
            if value.shape != (33, 4) or not np.isfinite(value).all():
                raise ValueError("Shadow landmarks must be finite MediaPipe (33,4).")
            value = value.copy()
            value[:, 3] = np.clip(value[:, 3], 0.0, 1.0)
            self._last_valid = value.copy()
        self._landmarks.append(value)
        self._frame_indices.append(int(frame_index))
        self._timestamps.append(float(timestamp_seconds))

    def record_rule_cycle(self, cycle: Any) -> None:
        """Copy rule-based boundaries for later comparison only."""

        self._rule_cycles.append(
            {
                "start_frame": getattr(cycle, "start_frame", None),
                "end_frame": getattr(cycle, "end_frame", None),
                "start_time": getattr(cycle, "start_time", None),
                "end_time": getattr(cycle, "end_time", None),
                "confidence": getattr(cycle, "confidence", None),
            }
        )

    @torch.no_grad()
    def analyze(self) -> dict[str, Any]:
        """Run preprocessing-v4 and Boundary V2 over the recorded sequence."""

        if len(self._landmarks) < self.config.minimum_frames:
            return {
                "available": False,
                "reason": "insufficient_frames",
                "ai_total_reps": 0,
                "ai_segments": [],
                "recorded_frames": len(self._landmarks),
            }
        media_pipe = np.stack(self._landmarks).astype(np.float32, copy=False)
        h36m = self.selector.to_h36m_17(media_pipe)
        normalized, diagnostics = self.normalizer.normalize(h36m)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        assert self.model is not None
        output = self.model(tensor)
        active = torch.sigmoid(output["active_logits"][0]).cpu().numpy()
        boundary = torch.sigmoid(output["boundary_logits"][0]).cpu().numpy()
        segments = v2_segments(active, boundary, self.postprocessing)
        rows: list[dict[str, Any]] = []
        for start, end in segments:
            rows.append(
                {
                    "start_sequence_index": int(start),
                    "end_sequence_index": int(end),
                    "start_frame": self._frame_indices[start],
                    "end_frame": self._frame_indices[end],
                    "start_time": self._timestamps[start],
                    "end_time": self._timestamps[end],
                    "confidence": float(np.mean(active[start : end + 1])),
                    "boundary_confidence": float(
                        max(float(boundary[start]), float(boundary[end]))
                    ),
                }
            )
        return {
            "available": True,
            "reason": None,
            "ai_total_reps": len(rows),
            "ai_segments": rows,
            "recorded_frames": len(self._landmarks),
            "normalization": {
                "sequence_scale": diagnostics.sequence_scale,
                "outlier_frames": int(diagnostics.outlier_mask.sum()),
                "near_zero_scale_frames": int(
                    diagnostics.near_zero_scale_mask.sum()
                ),
            },
        }

    def finalize_and_write(
        self,
        *,
        session_id: str,
        rule_based_total_reps: int,
        camera_view: str,
        input_mode: str,
        duration_seconds: float,
        video_path: str | None,
        cancelled: bool,
    ) -> Path:
        """Write an observational comparison; neither count is treated as truth."""

        analysis = self.analyze()
        ai_total = int(analysis["ai_total_reps"])
        payload = {
            "schema_version": "squat_boundary_shadow_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "exercise_family": "squat",
            "mode": "shadow_observation_only",
            "user_facing_count_source": self.config.user_facing_count_source,
            "rule_based_total_reps": int(rule_based_total_reps),
            "ai_total_reps": ai_total,
            "difference_ai_minus_rule": ai_total - int(rule_based_total_reps),
            "rule_based_cycle_boundaries": self._rule_cycles,
            "ai_segments": analysis["ai_segments"],
            "ai_available": analysis["available"],
            "ai_unavailable_reason": analysis["reason"],
            "camera_view": camera_view,
            "input_mode": input_mode,
            "duration_seconds": float(duration_seconds),
            "video_path": video_path,
            "cancelled": bool(cancelled),
            "recorded_frames": analysis["recorded_frames"],
            "normalization": analysis.get("normalization"),
            "input_contract": self.config.contract,
            "boundary_checkpoint": str(self.config.checkpoint),
            "boundary_checkpoint_sha256": self.checkpoint_sha256,
            "postprocessing": self.postprocessing,
            "interpretation": (
                "Observational comparison only. Without ground truth, neither "
                "the rule-based nor AI count is declared correct."
            ),
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.config.output_dir / f"{session_id}.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

