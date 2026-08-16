from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .builder import PrototypeArtifact


class PrototypeStore:
    """Safe NPZ persistence without object arrays or pickle."""

    SCHEMA_VERSION = "1.0.0"

    @classmethod
    def save(cls, path: str | Path, artifact: PrototypeArtifact) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(artifact.metadata)
        metadata["store_schema_version"] = cls.SCHEMA_VERSION
        np.savez_compressed(
            destination,
            prototype=np.asarray(artifact.prototype, dtype=np.float32),
            reference_embeddings=np.asarray(
                artifact.reference_embeddings,
                dtype=np.float32,
            ),
            reference_similarities=np.asarray(
                artifact.reference_similarities,
                dtype=np.float32,
            ),
            metadata_json=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> PrototypeArtifact:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as data:
            required = {
                "prototype",
                "reference_embeddings",
                "reference_similarities",
                "metadata_json",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Prototype file is missing fields: {sorted(missing)}")
            prototype = data["prototype"].astype(np.float32)
            references = data["reference_embeddings"].astype(np.float32)
            similarities = data["reference_similarities"].astype(np.float32)
            metadata: dict[str, Any] = json.loads(str(data["metadata_json"].item()))

        if metadata.get("store_schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported prototype store schema version.")
        if prototype.ndim != 1 or references.ndim != 2:
            raise ValueError("Invalid prototype tensor shapes.")
        if references.shape[1] != prototype.shape[0]:
            raise ValueError("Prototype and reference dimensions do not match.")
        if similarities.shape != (references.shape[0],):
            raise ValueError("Reference similarities have an invalid shape.")
        if not all(
            np.isfinite(item).all()
            for item in (prototype, references, similarities)
        ):
            raise ValueError("Prototype artifact contains non-finite values.")
        return PrototypeArtifact(prototype, references, similarities, metadata)
