"""Image-level inference for the independent Squat Error V1 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models.squat_posture_error import load_squat_posture_error_checkpoint
from preprocessing.squat_posture_features import (
    MediaPipeImagePoseExtractor,
    SquatPostureFeatureExtractor,
)


class SquatErrorImageInference:
    """Run static-pose error inference without invoking temporal Squat models."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        pose_model_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model, self.metadata = load_squat_posture_error_checkpoint(
            checkpoint_path, self.device
        )
        self.feature_extractor = SquatPostureFeatureExtractor()
        if self.metadata.get("feature_version") != self.feature_extractor.VERSION:
            raise ValueError(
                "Squat posture feature contract mismatch: "
                f"checkpoint={self.metadata.get('feature_version')!r}, "
                f"runtime={self.feature_extractor.VERSION!r}."
            )
        if int(self.metadata["input_dim"]) != self.feature_extractor.feature_dim:
            raise ValueError("Squat posture feature dimension does not match checkpoint.")
        self.pose_extractor = MediaPipeImagePoseExtractor(pose_model_path)
        self.closed = False

    @staticmethod
    def _decode(path: Path) -> np.ndarray | None:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        """Return a truthful three-class prediction for one readable detected pose."""

        if self.closed:
            raise RuntimeError("Squat Error V1 inference is closed.")
        path = Path(image_path)
        image = self._decode(path)
        if image is None:
            return self._unavailable(path, "image_unreadable")
        landmarks = self.pose_extractor.process(image)
        if landmarks is None:
            return self._unavailable(path, "pose_not_detected")
        features = self.feature_extractor.extract(landmarks)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        result = self.model.predict(tensor)
        probabilities = result["probabilities"][0].cpu().tolist()
        class_names = [self.metadata["class_vocabulary"][str(i)] for i in range(3)]
        return {
            "available": True,
            "trained": True,
            "model_stage": "development",
            "scope": "static_frame_only",
            "image_path": str(path.resolve()),
            "predicted_error": result["predicted_errors"][0],
            "probabilities": {
                name: float(probabilities[index]) for index, name in enumerate(class_names)
            },
            "pose_detected": True,
            "score": None,
            "reason": None,
        }

    @staticmethod
    def _unavailable(path: Path, reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "trained": True,
            "model_stage": "development",
            "scope": "static_frame_only",
            "image_path": str(path.resolve()),
            "predicted_error": None,
            "probabilities": None,
            "pose_detected": False,
            "score": None,
            "reason": reason,
        }

    def close(self) -> None:
        if not self.closed:
            self.pose_extractor.close()
            self.closed = True

    def __enter__(self) -> "SquatErrorImageInference":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

