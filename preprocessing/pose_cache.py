from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class PoseCache:
    """Safe numeric NPZ cache for MediaPipe ``(T, 33, 4)`` landmarks."""

    VERSION = "mediapipe_pose_cache_v1"

    @staticmethod
    def _validate(landmarks: np.ndarray) -> np.ndarray:
        values = np.asarray(landmarks, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (33, 4):
            raise ValueError(
                f"Expected pose landmarks (T, 33, 4), got {values.shape}."
            )
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Pose cache landmarks must be non-empty and finite.")
        return values

    @classmethod
    def save(
        cls,
        path: str | Path,
        landmarks: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        values = cls._validate(landmarks)
        payload = {"cache_version": cls.VERSION, **dict(metadata or {})}
        np.savez_compressed(
            output,
            landmarks=values,
            metadata_json=np.asarray(json.dumps(payload, ensure_ascii=False)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            if "landmarks" not in archive or "metadata_json" not in archive:
                raise ValueError(f"Invalid pose cache: {source}")
            values = cls._validate(archive["landmarks"])
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("cache_version") != cls.VERSION:
            raise ValueError("Unsupported pose cache version.")
        return values, metadata
