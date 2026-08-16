"""Table/incline Push-up AI V1 inference; never applied to floor Push-ups."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import numpy as np
import torch

from inference.family_ai_status import deterministic_rep_score
from models.pushup_correctness import PushupCorrectnessModel
from models.pushup_rep_boundary import PushupRepBoundaryModel
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from tools.squat_ai.prepare_rehab24_squat import resample_sequence
from training.squat_rep_boundary_v2 import v2_segments


class PushupTableAIOrchestrator:
    """Strictly loaded development inference for REHAB24 Ex3 scope."""

    SCOPE = "REHAB24 Ex3 table/incline Push-up only"
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def __init__(self, boundary_checkpoint: Path, correctness_checkpoint: Path, motionbert_checkpoint: Path, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boundary = torch.load(boundary_checkpoint, map_location=self.device, weights_only=True)
        self.boundary = PushupRepBoundaryModel().to(self.device); self.boundary.load_state_dict(boundary["model_state_dict"], strict=True); self.boundary.eval()
        self.postprocessing = dict(boundary["postprocessing"])
        correctness = torch.load(correctness_checkpoint, map_location=self.device, weights_only=True)
        self.correctness = PushupCorrectnessModel(motionbert_checkpoint).to(self.device)
        self.correctness.expert.load_state_dict(correctness["expert_state_dict"], strict=True)
        self.correctness.correctness_head.load_state_dict(correctness["correctness_head_state_dict"], strict=True); self.correctness.eval()
        self.threshold = float(correctness["decision_threshold"])
        self.selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}}); self.normalizer = H36MCoordinateNormalizer()
        self._frames: list[np.ndarray] = []
        self._frame_indices: list[int] = []
        self._timestamps: list[float] = []
        self._results: list[dict[str, Any]] = []
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pushup-ai")
        self._future: Future[list[dict[str, Any]]] | None = None
        self._last_requested_length = 0
        self._closed = False

    @classmethod
    def from_default_config(cls) -> "PushupTableAIOrchestrator":
        root = cls.PROJECT_ROOT
        return cls(
            root / "checkpoints" / "pushup_ai_v1" / "boundary" / "best.pt",
            root / "checkpoints" / "pushup_ai_v1" / "correctness" / "final_dev.pt",
            root / "models" / "latest_epoch.bin",
        )

    @torch.no_grad()
    def detect(self, full_motionbert_input: np.ndarray) -> list[tuple[int, int]]:
        values = torch.from_numpy(np.asarray(full_motionbert_input, np.float32)).unsqueeze(0).to(self.device)
        output = self.boundary(values)
        active = torch.sigmoid(output["active_logits"]).squeeze(0).cpu().numpy(); boundary = torch.sigmoid(output["boundary_logits"]).squeeze(0).cpu().numpy()
        return v2_segments(active, boundary, self.postprocessing)

    @torch.no_grad()
    def assess(self, raw_mp33_segment: np.ndarray, rep_index: int, start: int, end: int) -> dict[str, Any]:
        h36m = self.selector.to_h36m_17(np.asarray(raw_mp33_segment, np.float32)); normalized, _ = self.normalizer.normalize(h36m); values = resample_sequence(normalized)
        values[..., 2] = np.clip(values[..., 2], 0.0, 1.0); mask = values[..., 2].mean(axis=1) > 0.01
        output = self.correctness(torch.from_numpy(values).unsqueeze(0).to(self.device), torch.from_numpy(mask).unsqueeze(0).to(self.device))
        probability = float(output["correct_probability"].item()); passed = probability >= self.threshold
        return {"rep_index": int(rep_index), "start_frame": int(start + 1), "end_frame": int(end + 1), "assessment": "PASS" if passed else "FAIL", "correctness_probability": probability, "confidence": probability if passed else 1.0 - probability, "error_class": None if passed else "form_issue", "error_confidence": None, "experimental": True, "scope": self.SCOPE}

    def analyze(self, raw_mp33: np.ndarray, full_motionbert_input: np.ndarray) -> dict[str, Any]:
        segments = self.detect(full_motionbert_input); reps = [self.assess(raw_mp33[start : end + 1], index, start, end) for index, (start, end) in enumerate(segments, 1)]
        pass_count = sum(value["assessment"] == "PASS" for value in reps); fail_count = len(reps) - pass_count
        return {"exercise": "pushup", "scope": self.SCOPE, "experimental": True, "ai_detected_count": len(reps), "per_rep_results": reps, "pass_count": pass_count, "fail_count": fail_count, "needs_review_count": 0, "performance_score": deterministic_rep_score(pass_count, fail_count), "assessment_coverage": 1.0 if reps else 0.0, "model_status": {"detection": "Boundary V1 Development", "correctness": "Correctness V1 Development", "errors": "Generic Form Issue only; no detailed error labels"}}

    def record_frame(self, landmarks_33: np.ndarray | None, frame_index: int, timestamp_seconds: float) -> None:
        """Buffer post-countdown pose data without blocking the capture loop."""
        if landmarks_33 is None:
            value = self._frames[-1].copy() if self._frames else np.zeros((33, 4), np.float32)
            value[:, 3] = 0.0
        else:
            value = np.asarray(landmarks_33, dtype=np.float32)
            if value.shape != (33, 4) or not np.isfinite(value).all():
                raise ValueError("Push-up AI landmarks must be finite MediaPipe (33,4).")
        with self._lock:
            self._frames.append(value.copy())
            self._frame_indices.append(int(frame_index))
            self._timestamps.append(float(timestamp_seconds))

    def _analyze_frames(self, frames: np.ndarray, stable_only: bool) -> list[dict[str, Any]]:
        if len(frames) < 24:
            return []
        h36m = self.selector.to_h36m_17(frames)
        normalized, _ = self.normalizer.normalize(h36m)
        segments = self.detect(normalized)
        if stable_only:
            segments = [(start, end) for start, end in segments if end <= len(frames) - 10]
        return [
            self.assess(frames[start : end + 1], index, start, end)
            for index, (start, end) in enumerate(segments, 1)
        ]

    def _collect_future(self) -> None:
        if self._future is not None and self._future.done():
            self._results = self._future.result()
            self._future = None

    def request_live_analysis(self) -> None:
        self._collect_future()
        with self._lock:
            length = len(self._frames)
            if self._future is not None or length < 32 or length - self._last_requested_length < 15:
                return
            snapshot = np.asarray(self._frames, dtype=np.float32)
            self._last_requested_length = length
        self._future = self._executor.submit(self._analyze_frames, snapshot, True)

    def live_status(self) -> dict[str, Any]:
        self._collect_future()
        last = self._results[-1] if self._results else None
        pass_count = sum(rep["assessment"] == "PASS" for rep in self._results)
        fail_count = sum(rep["assessment"] == "FAIL" for rep in self._results)
        assessed = pass_count + fail_count
        return {
            "exercise": "pushup",
            "scope": self.SCOPE,
            "experimental": True,
            "available": True,
            "ai_detected_count": len(self._results),
            "ai_detected_reps": len(self._results),
            "last_ai_rep": last,
            "per_rep_results": list(self._results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "needs_review_count": 0,
            "performance_score": deterministic_rep_score(pass_count, fail_count),
            "assessment_coverage": (
                assessed / len(self._results) if self._results else 0.0
            ),
            "analyzing": self._future is not None,
            "boundary_available": True,
            "correctness_available": True,
            "error_available": False,
        }

    def finalize(self) -> dict[str, Any]:
        if self._future is not None:
            self._future.result()
            self._collect_future()
        with self._lock:
            frames = np.asarray(self._frames, dtype=np.float32)
        reps = self._analyze_frames(frames, False)
        pass_count = sum(rep["assessment"] == "PASS" for rep in reps)
        fail_count = sum(rep["assessment"] == "FAIL" for rep in reps)
        assessed = pass_count + fail_count
        return {
            "exercise": "pushup", "scope": self.SCOPE, "experimental": True,
            "available": True, "official_count": None,
            "ai_detected_count": len(reps), "ai_detected_reps": len(reps),
            "per_rep_results": reps, "pass_count": pass_count, "fail_count": fail_count,
            "needs_review_count": 0,
            "performance_score": deterministic_rep_score(pass_count, fail_count),
            "assessment_coverage": (assessed / len(reps)) if reps else 0.0,
            "model_status": {
                "detection": "Boundary V1 — Development",
                "correctness": "Correctness V1 — Development Final",
                "errors": "Not Available — generic Form Issue only",
                "variation": "Table/Incline Push-up",
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
