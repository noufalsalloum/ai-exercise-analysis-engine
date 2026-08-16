from __future__ import annotations

from typing import Optional

import numpy as np

from .builder import PrototypeArtifact, l2_normalize


class SimilarityEvaluator:
    """Evaluate reference similarity without calling it a quality score."""

    def __init__(self, outlier_percentile: float = 5.0) -> None:
        if not 0.0 <= outlier_percentile <= 50.0:
            raise ValueError("outlier_percentile must be in [0, 50].")
        self.outlier_percentile = outlier_percentile

    def evaluate(
        self,
        user_global_embedding: np.ndarray,
        artifact: PrototypeArtifact,
        include_euclidean: bool = True,
    ) -> dict[str, Optional[float] | bool]:
        user = np.asarray(user_global_embedding, dtype=np.float32)
        if user.ndim == 2:
            if user.shape[0] == 0:
                raise ValueError("No user embeddings were provided.")
            user = l2_normalize(user).mean(axis=0)
        if user.ndim != 1:
            raise ValueError("User embedding must have shape (D,) or (N, D).")
        prototype = np.asarray(artifact.prototype, dtype=np.float32)
        if user.shape != prototype.shape:
            raise ValueError("User and prototype embedding dimensions differ.")
        if not np.isfinite(user).all():
            raise ValueError("User embedding contains non-finite values.")

        user = l2_normalize(user).reshape(-1)
        prototype = l2_normalize(prototype).reshape(-1)
        similarity = float(np.clip(np.dot(user, prototype), -1.0, 1.0))
        cosine_distance = float(1.0 - similarity)
        euclidean_distance = (
            float(np.linalg.norm(user - prototype)) if include_euclidean else None
        )

        reference_similarities = np.asarray(
            artifact.reference_similarities,
            dtype=np.float32,
        )
        if reference_similarities.size:
            percentile = float(100.0 * np.mean(reference_similarities <= similarity))
            threshold = float(
                np.percentile(reference_similarities, self.outlier_percentile)
            )
            confidence = percentile / 100.0
            is_outlier = similarity < threshold
        else:
            percentile = None
            confidence = None
            is_outlier = False

        return {
            "similarity": similarity,
            "cosine_distance": cosine_distance,
            "euclidean_distance": euclidean_distance,
            "similarity_percentile": percentile,
            "prototype_confidence": confidence,
            "is_outlier": bool(is_outlier),
        }
