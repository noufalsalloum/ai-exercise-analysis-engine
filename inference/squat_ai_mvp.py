"""Experimental end-to-end Squat AI orchestration; no model training occurs here."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from models.squat_correctness import SquatCorrectnessModel
from models.squat_posture_error import (
    ERROR_CLASSES,
    load_squat_posture_error_checkpoint,
)
from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector, MEDIAPIPE_LANDMARKS
from preprocessing.squat_posture_features import SquatPostureFeatureExtractor
from tools.squat_ai.prepare_rehab24_squat import resample_sequence
from training.squat_correctness import load_checkpoint_strict
from training.squat_rep_boundary_v2 import v2_segments
from inference.squat_averaged_correctness import (
    AveragedCorrectnessConfig,
    AveragedCorrectnessExperiment,
)
from inference.squat_robustness import SquatRobustnessPolicy


ROOT = Path(__file__).resolve().parents[1]
MODEL_STATUSES = {
    "boundary": "Development / Shadow Candidate",
    "correctness": "Development Final — External Validation Pending",
    "error": "Development Static Error Model",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SquatAIMVPConfig:
    """Paths and fixed inference policy for the Experimental Squat MVP."""

    enabled: bool
    device: str
    boundary_checkpoint: Path
    correctness_checkpoint: Path
    error_checkpoint: Path
    motionbert_checkpoint: Path
    correctness_threshold: float
    minimum_frames: int
    completion_lag_frames: int
    live_poll_interval_frames: int
    representative_radius: int
    minimum_representative_confidence: float
    output_dir: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "SquatAIMVPConfig":
        config_path = path or ROOT / "configs" / "squat_ai_experimental.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else ROOT / candidate

        threshold = float(data["correctness_threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError("Correctness threshold must be inside (0,1).")
        return cls(
            enabled=bool(data["enabled"]),
            device=str(data.get("device", "auto")),
            boundary_checkpoint=resolve(str(data["boundary_checkpoint"])),
            correctness_checkpoint=resolve(str(data["correctness_checkpoint"])),
            error_checkpoint=resolve(str(data["error_checkpoint"])),
            motionbert_checkpoint=resolve(str(data["motionbert_checkpoint"])),
            correctness_threshold=threshold,
            minimum_frames=int(data["minimum_frames"]),
            completion_lag_frames=int(data["completion_lag_frames"]),
            live_poll_interval_frames=int(data["live_poll_interval_frames"]),
            representative_radius=int(data["representative_radius"]),
            minimum_representative_confidence=float(
                data["minimum_representative_confidence"]
            ),
            output_dir=resolve(str(data["output_dir"])),
        )


def apply_decision_policy(
    *,
    rep_index: int,
    start_frame: int,
    end_frame: int,
    correct_probability: float | None,
    threshold: float,
    raw_error_class: str | None,
    error_confidence: float | None,
    representative_frame: int | None,
    representative_frames: list[int] | None = None,
    correctness_latency_ms: float | None = None,
    error_latency_ms: float | None = None,
) -> dict[str, Any]:
    """Apply the fixed MVP policy; Error V1 can explain FAIL but never create it."""

    correctness: str | None = None
    pass_fail: str | None = None
    displayed_error: str | None = None
    conflict = False
    if correct_probability is not None:
        correctness = "correct" if correct_probability >= threshold else "incorrect"
        pass_fail = "PASS" if correctness == "correct" else "FAIL"
        if correctness == "incorrect":
            displayed_error = (
                raw_error_class
                if raw_error_class in {"bad_back", "bad_heel"}
                else "form_issue"
            )
        else:
            conflict = raw_error_class in {"bad_back", "bad_heel"}
    return {
        "rep_index": int(rep_index),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "correct_probability": (
            None if correct_probability is None else float(correct_probability)
        ),
        "correctness": correctness,
        "pass_fail": pass_fail,
        "error_class": displayed_error,
        "raw_error_class": raw_error_class,
        "error_confidence": (
            None if error_confidence is None else float(error_confidence)
        ),
        "representative_frame": representative_frame,
        "representative_frames": representative_frames or [],
        "ai_conflict": conflict,
        "correctness_latency_ms": correctness_latency_ms,
        "error_latency_ms": error_latency_ms,
        "total_ai_latency_ms": (
            None
            if correctness_latency_ms is None and error_latency_ms is None
            else float((correctness_latency_ms or 0.0) + (error_latency_ms or 0.0))
        ),
        "score": None,
        "experimental": True,
    }


def aggregate_ai_results(
    results: list[dict[str, Any]],
    *,
    detected_count: int | None = None,
    failures: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Aggregate only available AI results; never call Pass Rate accuracy."""

    ordered = sorted(results, key=lambda item: int(item["rep_index"]))
    assessments = [item.get("assessment", item.get("pass_fail")) for item in ordered]
    correct = sum(value == "PASS" for value in assessments)
    incorrect = sum(value == "FAIL" for value in assessments)
    needs_review = sum(value == "NEEDS_REVIEW" for value in assessments)
    classified = correct + incorrect
    errors = {
        name: sum(
            item.get("assessment", item.get("pass_fail")) == "FAIL"
            and item.get("error_class") == name
            for item in ordered
        )
        for name in ("bad_back", "bad_heel", "form_issue")
    }
    score = SquatRobustnessPolicy.load().performance_score(ordered)
    return {
        "experimental": True,
        "status": "Experimental",
        "available": bool(detected_count is not None),
        "ai_detected_reps": int(len(ordered) if detected_count is None else detected_count),
        "ai_correct_reps": int(correct),
        "ai_incorrect_reps": int(incorrect),
        "pass_count": int(correct),
        "fail_count": int(incorrect),
        "needs_review_count": int(needs_review),
        "ai_pass_rate": (None if classified == 0 else float(correct / classified)),
        "error_counts": errors,
        "conflicts": int(sum(bool(item.get("ai_conflict")) for item in ordered)),
        "per_rep_results": ordered,
        "score": score.score,
        "performance_score": score.score,
        "assessed_reps": score.assessed_reps,
        "assessment_coverage": score.coverage,
        "score_unavailable_reason": score.reason,
        "failures": list(failures or []),
    }


class SquatAIExperimentalOrchestrator:
    """Boundary V2 → Correctness V3 → representative-frame Error V1."""

    def __init__(
        self,
        config: SquatAIMVPConfig,
        *,
        allow_partial: bool = True,
        camera_view: str | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("Experimental Squat AI is disabled.")
        self.config = config
        self.camera_view = camera_view
        self.robustness_policy = SquatRobustnessPolicy.load()
        self.averaged_correctness_experiment = AveragedCorrectnessExperiment(
            AveragedCorrectnessConfig.load(), self.robustness_policy
        )
        self.device = torch.device(
            "cuda"
            if config.device == "auto" and torch.cuda.is_available()
            else ("cpu" if config.device == "auto" else config.device)
        )
        self.selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}})
        self.normalizer = H36MCoordinateNormalizer()
        self.feature_extractor = SquatPostureFeatureExtractor()
        self.boundary_model: SquatRepBoundaryV2Model | None = None
        self.correctness_model: SquatCorrectnessModel | None = None
        self.error_model: torch.nn.Module | None = None
        self.postprocessing: dict[str, Any] | None = None
        self.model_failures: list[dict[str, str]] = []
        self.checkpoint_hashes: dict[str, str | None] = {
            "boundary_v2": None,
            "correctness_v3": None,
            "error_v1": None,
        }
        self._load_models(allow_partial=allow_partial)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="squat-ai-mvp")
        self._future: Future[list[dict[str, Any]]] | None = None
        self._closed = False
        self.reset()

    @classmethod
    def from_default_config(
        cls, *, camera_view: str | None = None
    ) -> "SquatAIExperimentalOrchestrator | None":
        config = SquatAIMVPConfig.load()
        return cls(config, camera_view=camera_view) if config.enabled else None

    def _failure(self, component: str, exc: Exception) -> None:
        self.model_failures.append(
            {"component": component, "reason": f"{type(exc).__name__}: {exc}"}
        )

    def _load_models(self, *, allow_partial: bool) -> None:
        try:
            checkpoint = torch.load(
                self.config.boundary_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
            if checkpoint.get("experiment", {}).get("architecture") != "boundary_aux_tcn":
                raise ValueError("Boundary V2 checkpoint architecture mismatch.")
            model = SquatRepBoundaryV2Model().to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.boundary_model = model.eval()
            self.postprocessing = dict(checkpoint["postprocessing"])
            self.checkpoint_hashes["boundary_v2"] = sha256(
                self.config.boundary_checkpoint
            )
        except Exception as exc:
            self._failure("boundary_v2", exc)
            if not allow_partial:
                raise
        try:
            model = SquatCorrectnessModel(self.config.motionbert_checkpoint).to(self.device)
            checkpoint = load_checkpoint_strict(
                self.config.correctness_checkpoint, model, self.device
            )
            if not bool(checkpoint.get("motionbert_frozen", False)):
                raise ValueError("Correctness V3 does not attest frozen MotionBERT.")
            checkpoint_threshold = float(checkpoint["decision_threshold"])
            if abs(checkpoint_threshold - self.config.correctness_threshold) > 1e-12:
                raise ValueError("Correctness V3 threshold differs from configured 0.61.")
            self.correctness_model = model.eval()
            self.checkpoint_hashes["correctness_v3"] = sha256(
                self.config.correctness_checkpoint
            )
        except Exception as exc:
            self._failure("correctness_v3", exc)
            if not allow_partial:
                raise
        try:
            model, checkpoint = load_squat_posture_error_checkpoint(
                self.config.error_checkpoint, self.device
            )
            if checkpoint.get("feature_version") != self.feature_extractor.VERSION:
                raise ValueError("Error V1 feature contract mismatch.")
            self.error_model = model.eval()
            self.checkpoint_hashes["error_v1"] = sha256(self.config.error_checkpoint)
        except Exception as exc:
            self._failure("error_v1", exc)
            if not allow_partial:
                raise

    @property
    def boundary_available(self) -> bool:
        return self.boundary_model is not None and self.postprocessing is not None

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._landmarks: list[np.ndarray] = []
            self._frame_indices: list[int] = []
            self._timestamps: list[float] = []
            self._last_valid: np.ndarray | None = None
            self._results: list[dict[str, Any]] = []
            self._last_requested_length = 0

    def record_frame(
        self,
        landmarks_33: np.ndarray | None,
        frame_index: int,
        timestamp_seconds: float,
    ) -> None:
        """Record landmarks only; countdown frames must never call this method."""

        if landmarks_33 is None:
            value = (
                np.zeros((33, 4), dtype=np.float32)
                if self._last_valid is None
                else self._last_valid.copy()
            )
            value[:, 3] = 0.0
        else:
            value = np.asarray(landmarks_33, dtype=np.float32)
            if value.shape != (33, 4) or not np.isfinite(value).all():
                raise ValueError("Squat AI landmarks must be finite MediaPipe (33,4).")
            value = value.copy()
            value[:, 3] = np.clip(value[:, 3], 0.0, 1.0)
            self._last_valid = value.copy()
        with self._lock:
            self._landmarks.append(value)
            self._frame_indices.append(int(frame_index))
            self._timestamps.append(float(timestamp_seconds))

    def _snapshot(self) -> tuple[np.ndarray, list[int], list[float]]:
        with self._lock:
            values = (
                np.stack(self._landmarks).astype(np.float32, copy=False)
                if self._landmarks
                else np.empty((0, 33, 4), dtype=np.float32)
            )
            return values, list(self._frame_indices), list(self._timestamps)

    @staticmethod
    def _range_matches(
        start: int, end: int, known: list[dict[str, Any]], threshold: float = 0.5
    ) -> bool:
        for item in known:
            left = max(start, int(item["start_sequence_index"]))
            right = min(end, int(item["end_sequence_index"]))
            intersection = max(0, right - left + 1)
            union = max(end, int(item["end_sequence_index"])) - min(
                start, int(item["start_sequence_index"])
            ) + 1
            if intersection / max(union, 1) >= threshold:
                return True
        return False

    @torch.no_grad()
    def _detect_segments(self, landmarks: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], float]:
        if not self.boundary_available or len(landmarks) < self.config.minimum_frames:
            return np.empty((0, 17, 3), np.float32), [], 0.0
        started = perf_counter()
        h36m = self.selector.to_h36m_17(landmarks)
        normalized, _ = self.normalizer.normalize(h36m)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        assert self.boundary_model is not None and self.postprocessing is not None
        output = self.boundary_model(tensor)
        active = torch.sigmoid(output["active_logits"][0]).cpu().numpy()
        boundary = torch.sigmoid(output["boundary_logits"][0]).cpu().numpy()
        return normalized, v2_segments(active, boundary, self.postprocessing), (perf_counter() - started) * 1000.0

    @torch.no_grad()
    def _correctness(self, normalized_segment: np.ndarray) -> tuple[float | None, float | None]:
        if self.correctness_model is None:
            return None, None
        started = perf_counter()
        resampled = resample_sequence(normalized_segment)
        mask = resampled[..., 2].mean(axis=1) > 0.01
        if not bool(mask.any()):
            return None, None
        output = self.correctness_model(
            torch.from_numpy(resampled).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
        )
        return float(output["correct_probability"].item()), (perf_counter() - started) * 1000.0

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        first = a - b
        second = c - b
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-8:
            return float("inf")
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _representative_indices(self, segment: np.ndarray) -> tuple[int | None, list[int]]:
        candidate_rows: list[tuple[float, int]] = []
        for index, frame in enumerate(segment):
            angles: list[float] = []
            for side in ("LEFT", "RIGHT"):
                ids = [
                    MEDIAPIPE_LANDMARKS[f"{side}_HIP"],
                    MEDIAPIPE_LANDMARKS[f"{side}_KNEE"],
                    MEDIAPIPE_LANDMARKS[f"{side}_ANKLE"],
                ]
                if float(frame[ids, 3].min()) >= self.config.minimum_representative_confidence:
                    angles.append(self._angle(frame[ids[0], :2], frame[ids[1], :2], frame[ids[2], :2]))
            if angles and np.isfinite(angles).all():
                candidate_rows.append((float(np.mean(angles)), index))
        if not candidate_rows:
            return None, []
        center = min(candidate_rows)[1]
        radius = self.config.representative_radius
        valid = {index for _, index in candidate_rows}
        selected = [
            index
            for index in range(max(0, center - radius), min(len(segment), center + radius + 1))
            if index in valid
        ]
        return center, selected

    @torch.no_grad()
    def _error(
        self, segment: np.ndarray
    ) -> tuple[str | None, float | None, int | None, list[int], float | None]:
        if self.error_model is None:
            return None, None, None, [], None
        center, selected = self._representative_indices(segment)
        if center is None or not selected:
            return None, None, None, [], None
        started = perf_counter()
        features = np.stack(
            [self.feature_extractor.extract(segment[index]) for index in selected]
        )
        output = self.error_model.predict(torch.from_numpy(features).to(self.device))
        probabilities = output["probabilities"].mean(dim=0).cpu().numpy()
        class_index = int(np.argmax(probabilities))
        return (
            ERROR_CLASSES[class_index],
            float(probabilities[class_index]),
            center,
            selected,
            (perf_counter() - started) * 1000.0,
        )

    def _analyze_snapshot(
        self,
        landmarks: np.ndarray,
        frame_indices: list[int],
        timestamps: list[float],
        known: list[dict[str, Any]],
        final: bool,
    ) -> list[dict[str, Any]]:
        normalized, segments, boundary_latency = self._detect_segments(landmarks)
        outputs: list[dict[str, Any]] = []
        for start, end in segments:
            if not final and end > len(landmarks) - 1 - self.config.completion_lag_frames:
                continue
            if self._range_matches(start, end, known + outputs):
                continue
            correct_probability: float | None = None
            correctness_latency: float | None = None
            averaged_comparison: dict[str, Any] | None = None
            raw_error: str | None = None
            error_confidence: float | None = None
            representative: int | None = None
            selected: list[int] = []
            error_latency: float | None = None
            try:
                correct_probability, correctness_latency = self._correctness(
                    normalized[start : end + 1]
                )
            except Exception as exc:
                self._failure("correctness_inference", exc)
            if self.correctness_model is not None:
                try:
                    averaged_comparison = self.averaged_correctness_experiment.compare(
                        self.correctness_model,
                        normalized[start : end + 1],
                        self.device,
                    )
                except Exception as exc:
                    self._failure("averaged_correctness_experiment", exc)
            try:
                raw_error, error_confidence, representative, selected, error_latency = self._error(
                    landmarks[start : end + 1]
                )
            except Exception as exc:
                self._failure("error_inference", exc)
            item = apply_decision_policy(
                rep_index=len(known) + len(outputs) + 1,
                start_frame=frame_indices[start],
                end_frame=frame_indices[end],
                correct_probability=correct_probability,
                threshold=self.config.correctness_threshold,
                raw_error_class=raw_error,
                error_confidence=error_confidence,
                representative_frame=(
                    None if representative is None else frame_indices[start + representative]
                ),
                representative_frames=[frame_indices[start + index] for index in selected],
                correctness_latency_ms=correctness_latency,
                error_latency_ms=error_latency,
            )
            item = self.robustness_policy.assess_rep(item, self.camera_view)
            item.update(
                {
                    "start_sequence_index": int(start),
                    "end_sequence_index": int(end),
                    "start_time": timestamps[start],
                    "end_time": timestamps[end],
                    "boundary_latency_ms": boundary_latency,
                }
            )
            if averaged_comparison is not None:
                item["averaged_correctness_experiment"] = averaged_comparison
            outputs.append(item)
        return outputs

    def _collect_future(self, *, wait: bool = False) -> list[dict[str, Any]]:
        future = self._future
        if future is None or (not wait and not future.done()):
            return []
        try:
            rows = future.result()
        except Exception as exc:
            self._failure("live_ai_inference", exc)
            rows = []
        self._future = None
        for item in rows:
            if not self._range_matches(
                int(item["start_sequence_index"]),
                int(item["end_sequence_index"]),
                self._results,
            ):
                item["rep_index"] = len(self._results) + 1
                self._results.append(item)
        return rows

    def request_live_analysis(self) -> None:
        """Submit a non-blocking snapshot; camera/video processing never waits here."""

        self._collect_future()
        if self._future is not None or not self.boundary_available:
            return
        landmarks, frame_indices, timestamps = self._snapshot()
        if (
            len(landmarks) < self.config.minimum_frames
            or len(landmarks) - self._last_requested_length
            < self.config.live_poll_interval_frames
        ):
            return
        self._last_requested_length = len(landmarks)
        known = list(self._results)
        self._future = self._executor.submit(
            self._analyze_snapshot,
            landmarks,
            frame_indices,
            timestamps,
            known,
            False,
        )

    def live_status(self) -> dict[str, Any]:
        self._collect_future()
        last = self._results[-1] if self._results else None
        return {
            "experimental": True,
            "ai_detected_reps": len(self._results),
            "last_ai_rep": last,
            "analyzing": self._future is not None,
            "analyzing_rep_index": (
                len(self._results) + 1 if self._future is not None else None
            ),
            "boundary_available": self.boundary_available,
            "correctness_available": self.correctness_model is not None,
            "error_available": self.error_model is not None,
        }

    def finalize(self) -> dict[str, Any]:
        """Wait only after capture ends, then reconcile every final AI segment."""

        self._collect_future(wait=True)
        landmarks, frame_indices, timestamps = self._snapshot()
        if self.boundary_available and len(landmarks) >= self.config.minimum_frames:
            try:
                rows = self._analyze_snapshot(
                    landmarks,
                    frame_indices,
                    timestamps,
                    list(self._results),
                    True,
                )
                for item in rows:
                    if not self._range_matches(
                        int(item["start_sequence_index"]),
                        int(item["end_sequence_index"]),
                        self._results,
                    ):
                        item["rep_index"] = len(self._results) + 1
                        self._results.append(item)
            except Exception as exc:
                self._failure("final_ai_inference", exc)
        aggregate = aggregate_ai_results(
            self._results,
            detected_count=(len(self._results) if self.boundary_available else None),
            failures=self.model_failures,
        )
        aggregate.update(
            {
                "schema_version": "squat_ai_experimental_mvp_v1",
                "model_statuses": dict(MODEL_STATUSES),
                "checkpoint_hashes": dict(self.checkpoint_hashes),
                "correctness_threshold": self.config.correctness_threshold,
                "representative_frame_policy": "minimum mean bilateral knee angle; mean probabilities over frame ±2",
                "latency": self._latency_summary(self._results),
                "boundary_available": self.boundary_available,
                "correctness_available": self.correctness_model is not None,
                "error_available": self.error_model is not None,
                "robustness_policy": "squat_ai_robustness_v1",
            }
        )
        return aggregate

    @staticmethod
    def _latency_summary(results: list[dict[str, Any]]) -> dict[str, float | None]:
        def mean(name: str) -> float | None:
            values = [float(item[name]) for item in results if item.get(name) is not None]
            return None if not values else float(np.mean(values))

        return {
            "average_correctness_inference_ms": mean("correctness_latency_ms"),
            "average_error_inference_ms": mean("error_latency_ms"),
            "average_total_per_rep_ms": mean("total_ai_latency_ms"),
        }

    def finalize_and_write(
        self,
        *,
        session_id: str,
        rule_based_count: int,
        camera_view: str,
        input_mode: str,
        duration_seconds: float,
        cancelled: bool,
    ) -> tuple[dict[str, Any], Path]:
        result = self.finalize()
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "exercise_family": "squat",
            "rule_based_count": int(rule_based_count),
            "ai_detected_count": result["ai_detected_reps"],
            "correct_count": result["ai_correct_reps"],
            "incorrect_count": result["ai_incorrect_reps"],
            "needs_review_count": result["needs_review_count"],
            "bad_back_count": result["error_counts"]["bad_back"],
            "bad_heel_count": result["error_counts"]["bad_heel"],
            "form_issue_count": result["error_counts"]["form_issue"],
            "conflicts": result["conflicts"],
            "per_rep_results": result["per_rep_results"],
            "model_versions": result["model_statuses"],
            "checkpoint_hashes": result["checkpoint_hashes"],
            "correctness_threshold": result["correctness_threshold"],
            "latency": result["latency"],
            "performance_score": result["performance_score"],
            "assessment_coverage": result["assessment_coverage"],
            "assessed_reps": result["assessed_reps"],
            "score_unavailable_reason": result["score_unavailable_reason"],
            "score": result["performance_score"],
            "camera_view": camera_view,
            "input_mode": input_mode,
            "duration_seconds": float(duration_seconds),
            "cancelled": bool(cancelled),
            "failures": result["failures"],
            "experimental": True,
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.config.output_dir / f"{session_id}_ai.json"
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result["session_ai_log"] = str(destination.resolve())
        return result, destination

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
