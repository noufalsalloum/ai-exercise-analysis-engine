from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np


PrototypeStrategy = Literal["mean", "medoid", "trimmed_mean"]


def _as_matrix(embeddings: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D) with N,D > 0.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return safe L2-normalized float32 vectors."""

    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot L2-normalize a zero embedding.")
    return (array / norms).astype(np.float32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PrototypeArtifact:
    """In-memory prototype plus safe JSON-serializable metadata."""

    prototype: np.ndarray
    reference_embeddings: np.ndarray
    reference_similarities: np.ndarray
    metadata: dict[str, Any]


class PrototypeBuilder:
    """Aggregate multiple reference videos into one exercise prototype."""

    def __init__(
        self,
        strategy: PrototypeStrategy = "mean",
        min_reference_videos: int = 3,
        trim_fraction: float = 0.1,
        normalize_embeddings: bool = True,
        center_vector: Optional[np.ndarray] = None,
        reject_outliers: bool = True,
        outlier_mad_threshold: float = 3.5,
    ) -> None:
        if strategy not in {"mean", "medoid", "trimmed_mean"}:
            raise ValueError(f"Unsupported prototype strategy: {strategy}")
        if min_reference_videos < 2:
            raise ValueError("min_reference_videos must be at least 2.")
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5).")
        if outlier_mad_threshold <= 0:
            raise ValueError("outlier_mad_threshold must be positive.")

        self.strategy = strategy
        self.min_reference_videos = min_reference_videos
        self.trim_fraction = trim_fraction
        self.normalize_embeddings = normalize_embeddings
        self.center_vector = (
            np.asarray(center_vector, dtype=np.float32)
            if center_vector is not None
            else None
        )
        self.reject_outliers = reject_outliers
        self.outlier_mad_threshold = outlier_mad_threshold

    def _aggregate_windows(self, windows: np.ndarray, video_id: str) -> np.ndarray:
        matrix = _as_matrix(windows, f"embeddings for video '{video_id}'")
        if self.normalize_embeddings:
            matrix = l2_normalize(matrix)
        video_embedding = matrix.mean(axis=0)
        if self.center_vector is not None:
            if self.center_vector.shape != video_embedding.shape:
                raise ValueError("center_vector dimension does not match embeddings.")
            video_embedding = video_embedding - self.center_vector
        if self.normalize_embeddings:
            return l2_normalize(video_embedding).reshape(-1)
        if np.linalg.norm(video_embedding) <= 1e-12:
            raise ValueError("Video aggregation produced a zero embedding.")
        return video_embedding.astype(np.float32).reshape(-1)

    def _reject_outliers(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.reject_outliers or len(embeddings) < 4:
            return embeddings, np.ones(len(embeddings), dtype=bool)

        normalized = l2_normalize(embeddings)
        center = l2_normalize(normalized.mean(axis=0)).reshape(-1)
        distances = 1.0 - normalized @ center
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        if mad <= 1e-12:
            keep = distances <= max(median, 1e-6)
        else:
            robust_z = 0.6745 * (distances - median) / mad
            keep = robust_z <= self.outlier_mad_threshold

        if int(keep.sum()) < self.min_reference_videos:
            raise ValueError(
                "Outlier rejection left fewer than the configured minimum "
                "reference videos. Review the references instead of silently "
                "building an unstable prototype."
            )
        return embeddings[keep], keep

    def _aggregate_reference_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.strategy == "mean":
            prototype = embeddings.mean(axis=0)
        elif self.strategy == "medoid":
            normalized = l2_normalize(embeddings)
            cosine_distances = 1.0 - normalized @ normalized.T
            prototype = embeddings[np.argmin(cosine_distances.sum(axis=1))]
        else:
            trim_count = int(np.floor(len(embeddings) * self.trim_fraction))
            sorted_values = np.sort(embeddings, axis=0)
            trimmed = (
                sorted_values[trim_count : len(embeddings) - trim_count]
                if trim_count > 0
                else sorted_values
            )
            prototype = trimmed.mean(axis=0)
        if np.linalg.norm(prototype) <= 1e-12:
            raise ValueError("Reference aggregation produced a zero prototype.")
        if self.normalize_embeddings:
            return l2_normalize(prototype).reshape(-1)
        return np.asarray(prototype, dtype=np.float32).reshape(-1)

    def build(
        self,
        exercise_id: str,
        video_window_embeddings: Mapping[str, np.ndarray],
        model_checkpoint_path: Optional[str | Path] = None,
        preprocessing_version: str = "unknown",
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> PrototypeArtifact:
        """Build a prototype from window embeddings grouped by source video."""

        if len(video_window_embeddings) < self.min_reference_videos:
            raise ValueError(
                f"Need at least {self.min_reference_videos} reference videos; "
                f"received {len(video_window_embeddings)}."
            )

        video_ids = list(video_window_embeddings.keys())
        matrices = {
            item: _as_matrix(video_window_embeddings[item], item) for item in video_ids
        }
        window_counts = [len(matrices[item]) for item in video_ids]
        video_embeddings = np.stack(
            [self._aggregate_windows(matrices[item], item) for item in video_ids],
            axis=0,
        ).astype(np.float32)

        kept_embeddings, keep_mask = self._reject_outliers(video_embeddings)
        prototype = self._aggregate_reference_embeddings(kept_embeddings)
        similarities = (
            l2_normalize(kept_embeddings) @ l2_normalize(prototype).reshape(-1)
        ).astype(np.float32)

        metadata: dict[str, Any] = {
            "exercise_id": exercise_id,
            "number_of_videos": int(len(kept_embeddings)),
            "number_of_input_videos": int(len(video_embeddings)),
            "number_of_windows": int(
                sum(count for count, keep in zip(window_counts, keep_mask) if keep)
            ),
            "embedding_dim": int(prototype.shape[0]),
            "model_checkpoint_hash": (
                sha256_file(model_checkpoint_path)
                if model_checkpoint_path is not None
                else None
            ),
            "preprocessing_version": preprocessing_version,
            "prototype_strategy": self.strategy,
            "l2_normalized": self.normalize_embeddings,
            "mean_centered": self.center_vector is not None,
            "outlier_rejection": self.reject_outliers,
            "rejected_video_ids": [
                video_id for video_id, keep in zip(video_ids, keep_mask) if not keep
            ],
            "reference_similarity_mean": float(similarities.mean()),
            "reference_similarity_std": float(similarities.std()),
            "reference_similarity_min": float(similarities.min()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            metadata.update(dict(extra_metadata))

        return PrototypeArtifact(
            prototype=prototype.astype(np.float32),
            reference_embeddings=kept_embeddings.astype(np.float32),
            reference_similarities=similarities,
            metadata=metadata,
        )

    def build_from_reference_videos(
        self,
        exercise_id: str,
        video_paths: Sequence[str | Path],
        embed_video: Callable[[Path, str], np.ndarray],
        **metadata_kwargs: Any,
    ) -> PrototypeArtifact:
        """Run a supplied real video-to-window-embedding pipeline then build."""

        embeddings: dict[str, np.ndarray] = {}
        for video_path in video_paths:
            path = Path(video_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            embeddings[str(path)] = embed_video(path, exercise_id)
        return self.build(exercise_id, embeddings, **metadata_kwargs)
