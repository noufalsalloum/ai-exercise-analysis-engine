from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

try:
    from ..heads.error_head import ERROR_VOCABULARY, ErrorHead
    from ..heads.phase_head import PHASE_VOCABULARY, PhaseHead
    from ..experts.registry import normalize_exercise_id
    from ..models.exercise_aqa_model import ExerciseAQAModel
    from ..preprocessing.preprocessor import PreprocessorController
    from ..prototypes.similarity import SimilarityEvaluator
    from ..prototypes.store import PrototypeStore
    from ..prototypes.builder import sha256_file
    from .phase_rep_counter import PhaseRepCounter
except ImportError:
    from heads.error_head import ERROR_VOCABULARY, ErrorHead
    from heads.phase_head import PHASE_VOCABULARY, PhaseHead
    from experts.registry import normalize_exercise_id
    from models.exercise_aqa_model import ExerciseAQAModel
    from preprocessing.preprocessor import PreprocessorController
    from prototypes.similarity import SimilarityEvaluator
    from prototypes.store import PrototypeStore
    from prototypes.builder import sha256_file
    from inference.phase_rep_counter import PhaseRepCounter


class ExercisePredictor:
    """End-to-end video/landmark inference with explicit trained-head status."""

    def __init__(
        self,
        motionbert_checkpoint: str | Path,
        pose_model_path: str | Path,
        configs_dir: str | Path,
        aqa_checkpoint: str | Path | None = None,
        prototypes_dir: str | Path | None = None,
        device: str = "auto",
        window_size: int = 30,
        step_size: int = 5,
        batch_size: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.configs_dir = Path(configs_dir)
        self.pose_model_path = Path(pose_model_path)
        self.prototypes_dir = Path(prototypes_dir) if prototypes_dir is not None else None
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.model = ExerciseAQAModel(motionbert_checkpoint=motionbert_checkpoint)
        if aqa_checkpoint is not None:
            self.model.load_aqa_checkpoint(aqa_checkpoint, map_location=self.device)
        self.embedding_checkpoint_hash = (
            sha256_file(aqa_checkpoint) if aqa_checkpoint is not None else None
        )
        self.model.to(self.device).eval()
        self.similarity_evaluator = SimilarityEvaluator()

    def _load_config(self, exercise_id: str) -> dict[str, Any]:
        path = self.configs_dir / f"{exercise_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Exercise config not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _preprocessor(
        self,
        exercise_id: str,
        with_pose_extractor: bool = False,
    ) -> PreprocessorController:
        return PreprocessorController(
            spec_path=str(self.configs_dir / f"{exercise_id}.json"),
            window_size=self.window_size,
            step_size=self.step_size,
            pose_model_path=str(self.pose_model_path) if with_pose_extractor else None,
        )

    @staticmethod
    def _aggregate_temporal(
        temporal_windows: torch.Tensor,
        windows: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_frames = max(int(item["window_end"]) for item in windows)
        feature_dim = temporal_windows.shape[-1]
        summed = torch.zeros(total_frames, feature_dim, device=temporal_windows.device)
        counts = torch.zeros(total_frames, 1, device=temporal_windows.device)
        for embedding, window in zip(temporal_windows, windows):
            start = int(window["window_start"])
            end = int(window["window_end"])
            valid = end - start
            summed[start:end] += embedding[:valid]
            counts[start:end] += 1
        if (counts == 0).any():
            raise RuntimeError("Window aggregation left uncovered frames.")
        return summed / counts, torch.ones(1, total_frames, dtype=torch.bool, device=summed.device)

    @torch.no_grad()
    def embed_windows(
        self,
        windows: Sequence[dict[str, Any]],
        exercise_id: str,
    ) -> dict[str, torch.Tensor]:
        """Return real MotionBERT/expert embeddings for preprocessed windows."""

        if not windows:
            raise ValueError("Preprocessing produced no windows.")
        inputs = np.stack([item["motionbert_input"] for item in windows]).astype(np.float32)
        if inputs.ndim != 4 or inputs.shape[2:] != (17, 3) or not np.isfinite(inputs).all():
            raise ValueError(f"Expected finite windows (N,T,17,3), got {inputs.shape}.")
        tensor = torch.from_numpy(inputs).to(self.device)
        global_parts: list[torch.Tensor] = []
        temporal_parts: list[torch.Tensor] = []
        for start in range(0, len(tensor), self.batch_size):
            batch = tensor[start:start + self.batch_size]
            mask = torch.ones(batch.shape[:2], dtype=torch.bool, device=self.device)
            output = self.model(batch, exercise_id=exercise_id, temporal_mask=mask)
            global_parts.append(output["global_embedding"])
            temporal_parts.append(output["temporal_embedding"])
        return {
            "global_windows": torch.cat(global_parts, dim=0),
            "temporal_windows": torch.cat(temporal_parts, dim=0),
        }

    def _prototype_result(
        self,
        exercise_id: str,
        global_windows: torch.Tensor,
    ) -> dict[str, Any]:
        if not self.model.head_status["experts"]:
            return {
                "available": False,
                "similarity": None,
                "distance": None,
                "is_outlier": False,
                "reason": (
                    "Reference similarity requires a trained expert checkpoint; "
                    "the current exercise adapters are untrained."
                ),
            }
        path = self.prototypes_dir / f"{exercise_id}.npz" if self.prototypes_dir else None
        if path is None or not path.is_file():
            return {
                "available": False,
                "similarity": None,
                "distance": None,
                "is_outlier": False,
                "reason": "No compatible reference prototype is installed.",
            }
        artifact = PrototypeStore.load(path)
        artifact_hash = artifact.metadata.get("model_checkpoint_hash")
        if artifact_hash and artifact_hash != self.embedding_checkpoint_hash:
            return {
                "available": False,
                "similarity": None,
                "distance": None,
                "is_outlier": False,
                "reason": "Prototype and AQA embedding checkpoint hashes do not match.",
            }
        evaluated = self.similarity_evaluator.evaluate(
            global_windows.detach().cpu().numpy(),
            artifact,
        )
        return {
            "available": True,
            "similarity": evaluated["similarity"],
            "distance": evaluated["cosine_distance"],
            "euclidean_distance": evaluated["euclidean_distance"],
            "prototype_confidence": evaluated["prototype_confidence"],
            "similarity_percentile": evaluated["similarity_percentile"],
            "is_outlier": evaluated["is_outlier"],
            "name": "Reference Similarity",
        }

    @torch.no_grad()
    def _optional_heads(
        self,
        exercise_id: str,
        config: dict[str, Any],
        global_windows: torch.Tensor,
        temporal_video: torch.Tensor,
        temporal_mask: torch.Tensor,
        prototype_result: dict[str, Any],
        fps: Optional[float] = None,
    ) -> dict[str, Any]:
        status = self.model.head_status
        labels = config.get("labels", {})
        phase_result: dict[str, Any] = {
            "available": False,
            "trained": status["phase"],
            "predictions": [],
            "reason": "Phase head has no trained checkpoint." if not status["phase"] else None,
        }
        repetitions: dict[str, Any] = {
            "available": False,
            "count": 0,
            "reason": "Repetition counting requires trained phase predictions.",
        }
        pooled_global = global_windows.mean(dim=0, keepdim=True)
        temporal_batch = temporal_video.unsqueeze(0)

        if status["phase"]:
            valid_phase_mask = PhaseHead.build_valid_phase_mask(labels["valid_phases"], self.device)
            logits = self.model.phase_head(temporal_batch, temporal_mask, valid_phase_mask)
            prediction = self.model.phase_head.predict(logits)
            probabilities = prediction["probabilities"][0].cpu().numpy()
            phase_names = [PHASE_VOCABULARY[int(index)] for index in prediction["predictions"][0].cpu()]
            phase_result = {
                "available": True,
                "trained": True,
                "predictions": phase_names,
                "probabilities": probabilities.tolist(),
            }
            repetitions = PhaseRepCounter(
                exercise_id,
                phase_vocabulary=PHASE_VOCABULARY,
            ).evaluate(
                probabilities,
                timestamps=(
                    np.arange(len(probabilities), dtype=np.float64) / fps
                    if fps is not None and fps > 0
                    else None
                ),
            )

        pass_result: dict[str, Any] = {
            "available": False,
            "trained": status["pass_fail"],
            "prediction": None,
            "reason": "Pass/fail head has no trained checkpoint." if not status["pass_fail"] else None,
        }
        if status["pass_fail"]:
            similarity_features = None
            if prototype_result["available"]:
                similarity_features = torch.tensor(
                    [[
                        prototype_result["similarity"],
                        prototype_result["distance"],
                        prototype_result.get("prototype_confidence") or 0.0,
                    ]],
                    dtype=pooled_global.dtype,
                    device=self.device,
                )
            logits = self.model.passfail_head(
                pooled_global,
                temporal_batch,
                similarity_features,
                temporal_mask,
            )
            probabilities = torch.softmax(logits, dim=-1)[0]
            pass_result = {
                "available": True,
                "trained": True,
                "prediction": "PASS" if int(probabilities.argmax()) == 0 else "FAIL",
                "probabilities": probabilities.cpu().tolist(),
            }

        error_result: dict[str, Any] = {
            "available": False,
            "trained": status["errors"],
            "predictions": [],
            "reason": "Error head has no trained checkpoint." if not status["errors"] else None,
        }
        if status["errors"]:
            valid_error_mask = ErrorHead.build_valid_error_mask(labels["valid_errors"], self.device)
            logits = self.model.error_head(
                pooled_global,
                temporal_batch,
                temporal_mask,
                valid_error_mask,
            )
            probabilities = torch.sigmoid(logits)[0]
            predictions = [
                {"error": ERROR_VOCABULARY[index], "probability": float(value)}
                for index, value in enumerate(probabilities.cpu())
                if valid_error_mask[index] and value >= 0.5
            ]
            error_result = {"available": True, "trained": True, "predictions": predictions}
        return {
            "phase": phase_result,
            "repetitions": repetitions,
            "pass_fail": pass_result,
            "errors": error_result,
        }

    def predict_landmarks(
        self,
        raw_landmarks_33: np.ndarray,
        exercise_id: str,
        video_path: str = "<landmark-stream>",
        fps: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run cached/streamed MediaPipe landmarks without invoking MediaPipe."""

        exercise_id = normalize_exercise_id(exercise_id)
        config = self._load_config(exercise_id)
        windows = self._preprocessor(exercise_id).process_landmarks(raw_landmarks_33)
        embeddings = self.embed_windows(windows, exercise_id)
        temporal_video, temporal_mask = self._aggregate_temporal(embeddings["temporal_windows"], windows)
        prototype = self._prototype_result(exercise_id, embeddings["global_windows"])
        optional = self._optional_heads(
            exercise_id,
            config,
            embeddings["global_windows"],
            temporal_video,
            temporal_mask,
            prototype,
            fps=fps,
        )
        return {
            "exercise_id": exercise_id,
            "video_path": video_path,
            "num_windows": len(windows),
            "prototype_similarity": prototype,
            **optional,
        }

    def predict_video(
        self,
        video_path: str | Path,
        exercise_id: str,
        max_frames: Optional[int] = None,
        pose_cache_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run Video → MediaPipe → MotionBERT → expert → optional trained tasks."""

        exercise_id = normalize_exercise_id(exercise_id)
        video = Path(video_path)
        preprocessor = self._preprocessor(exercise_id, with_pose_extractor=True)
        windows = preprocessor.process_video(
            video,
            max_frames=max_frames,
            cache_path=pose_cache_path,
        )
        embeddings = self.embed_windows(windows, exercise_id)
        config = self._load_config(exercise_id)
        temporal_video, temporal_mask = self._aggregate_temporal(embeddings["temporal_windows"], windows)
        prototype = self._prototype_result(exercise_id, embeddings["global_windows"])
        optional = self._optional_heads(
            exercise_id,
            config,
            embeddings["global_windows"],
            temporal_video,
            temporal_mask,
            prototype,
            fps=preprocessor.last_video_fps,
        )
        return {
            "exercise_id": exercise_id,
            "video_path": str(video),
            "num_windows": len(windows),
            "prototype_similarity": prototype,
            **optional,
        }
