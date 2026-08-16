"""Development-only diagnostics for Squat AI calibration and domain shift.

This tool never trains or overwrites an active model. It reads the existing
V3/Error V1 artifacts, traces one user video, and emits a small evidence set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors

from inference.squat_ai_mvp import SquatAIExperimentalOrchestrator, SquatAIMVPConfig
from input_sources.frame_sources import VideoFrameSource
from input_sources.pose_stream import PoseStreamProcessor
from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS
from tools.squat_ai.prepare_rehab24_squat import resample_sequence


ROOT = Path(__file__).resolve().parents[2]
POSE_MODEL = Path(r"C:\MediaPipe\pose_landmarker_full.task")
OOF_BACKUP = Path(
    r"C:\Users\JoudA\OneDrive\سطح المكتب\AI_ENGINE_RECOVERY_TEMP"
) / "code2_20260812_105158" / "ai_engine" / "results" / "squat_ai_v3" / "correctness" / "selected_oof_predictions.csv"
SIDE_VIDEO = ROOT / "datasets" / "air_squat" / "user" / "side.mp4"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (
            probabilities <= right if right == 1.0 else probabilities < right
        )
        if mask.any():
            value += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value)


def _classification(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | float,
    assessed: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = np.ones(len(labels), dtype=bool) if assessed is None else assessed.astype(bool)
    if not mask.any():
        return {"coverage": 0.0, "samples": 0}
    threshold_array = np.broadcast_to(thresholds, probabilities.shape)
    prediction = (probabilities >= threshold_array).astype(np.int64)
    y = labels[mask]
    pred = prediction[mask]
    return {
        "coverage": float(mask.mean()),
        "samples": int(mask.sum()),
        "accuracy": float((y == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "incorrect_recall": float(recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)[0]),
        "correct_recall": float(recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)[1]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


class _Calibrator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model: Any = None

    @staticmethod
    def _logit(probability: np.ndarray | float) -> np.ndarray:
        value = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        return np.log(value / (1.0 - value))

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "_Calibrator":
        logits = self._logit(probabilities)
        if self.name == "identity":
            self.model = None
        elif self.name == "temperature":
            def loss(log_temperature: float) -> float:
                temperature = float(np.exp(log_temperature))
                calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
                calibrated = np.clip(calibrated, 1e-7, 1.0 - 1e-7)
                return float(-np.mean(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)))

            result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
            self.model = float(np.exp(result.x))
        elif self.name == "platt":
            model = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
            model.fit(logits.reshape(-1, 1), labels)
            self.model = model
        elif self.name == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(probabilities, labels)
            self.model = model
        else:
            raise ValueError(self.name)
        return self

    def transform(self, probabilities: np.ndarray | float) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if self.name == "identity":
            return values
        if self.name == "temperature":
            return 1.0 / (1.0 + np.exp(-self._logit(values) / float(self.model)))
        if self.name == "platt":
            return self.model.predict_proba(self._logit(values).reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(values), dtype=np.float64)

    def parameters(self) -> dict[str, Any]:
        if self.name == "identity":
            return {}
        if self.name == "temperature":
            return {"temperature": float(self.model)}
        if self.name == "platt":
            return {
                "coefficient": float(self.model.coef_[0, 0]),
                "intercept": float(self.model.intercept_[0]),
            }
        return {
            "x_thresholds": self.model.X_thresholds_.tolist(),
            "y_thresholds": self.model.y_thresholds_.tolist(),
        }


def calibration_analysis(oof_path: Path, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(oof_path)
    labels = frame["correctness"].to_numpy(np.int64)
    raw = frame["correct_probability"].to_numpy(np.float64)
    subjects = frame["subject_id"].astype(str).to_numpy()
    names = ("identity", "temperature", "platt", "isotonic")
    rows: list[dict[str, Any]] = []
    cross_fitted: dict[str, np.ndarray] = {}
    mapped_thresholds: dict[str, np.ndarray] = {}
    for name in names:
        calibrated = np.zeros_like(raw)
        thresholds = np.zeros_like(raw)
        for subject in sorted(set(subjects)):
            validation = subjects == subject
            training = ~validation
            model = _Calibrator(name).fit(raw[training], labels[training])
            calibrated[validation] = model.transform(raw[validation])
            thresholds[validation] = float(model.transform(np.asarray([0.61]))[0])
        cross_fitted[name] = calibrated
        mapped_thresholds[name] = thresholds
        metrics = _classification(labels, calibrated, thresholds)
        rows.append(
            {
                "method": name,
                "brier_score": float(brier_score_loss(labels, calibrated)),
                "ece_10_bins": _ece(labels, calibrated),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["brier_score", "ece_10_bins"])
    comparison.to_csv(output_dir / "calibration_candidates.csv", index=False)
    eligible = comparison[comparison["macro_f1"] >= comparison.loc[comparison["method"] == "identity", "macro_f1"].iloc[0] - 1e-12]
    selected_name = str(eligible.iloc[0]["method"])

    # Cross-fitted data-driven abstention: on every held subject, choose the
    # margin that best separates errors from correct decisions using other subjects.
    selected_probability = cross_fitted[selected_name]
    selected_threshold = mapped_thresholds[selected_name]
    assessed = np.ones(len(frame), dtype=bool)
    fold_margins: dict[str, float] = {}
    for subject in sorted(set(subjects)):
        validation = subjects == subject
        training = ~validation
        margins = np.abs(selected_probability[training] - selected_threshold[training])
        predictions = selected_probability[training] >= selected_threshold[training]
        mistakes = predictions.astype(np.int64) != labels[training]
        candidates = np.unique(margins)
        best = (float("-inf"), 0.0)
        for margin in candidates:
            uncertain = margins < margin
            tpr = float(uncertain[mistakes].mean()) if mistakes.any() else 0.0
            fpr = float(uncertain[~mistakes].mean()) if (~mistakes).any() else 0.0
            score = tpr - fpr
            candidate = (score, -float(margin))
            if candidate > (best[0], -best[1]):
                best = (score, float(margin))
        fold_margins[subject] = best[1]
        assessed[validation] = (
            np.abs(selected_probability[validation] - selected_threshold[validation]) >= best[1]
        )
    abstention_metrics = _classification(labels, selected_probability, selected_threshold, assessed)

    final_calibrator = _Calibrator(selected_name).fit(raw, labels)
    final_threshold = float(final_calibrator.transform(np.asarray([0.61]))[0])
    margins = np.abs(final_calibrator.transform(raw) - final_threshold)
    predictions = final_calibrator.transform(raw) >= final_threshold
    mistakes = predictions.astype(np.int64) != labels
    best = (float("-inf"), 0.0)
    for margin in np.unique(margins):
        uncertain = margins < margin
        tpr = float(uncertain[mistakes].mean()) if mistakes.any() else 0.0
        fpr = float(uncertain[~mistakes].mean()) if (~mistakes).any() else 0.0
        score = tpr - fpr
        if (score, -float(margin)) > (best[0], -best[1]):
            best = (score, float(margin))

    payload = {
        "source": str(oof_path),
        "samples": len(frame),
        "subjects": sorted(set(subjects)),
        "historical_test_subjects_used": False,
        "raw_threshold": 0.61,
        "selected_calibration": selected_name,
        "selection_rule": "lowest cross-fitted Brier score; classification boundary preserved by mapped threshold",
        "final_parameters": final_calibrator.parameters(),
        "calibrated_threshold": final_threshold,
        "abstention_margin": best[1],
        "abstention_margin_selection": "Youden J for detecting misclassified OOF samples from absolute calibrated decision margin",
        "cross_fitted_fold_margins": fold_margins,
        "cross_fitted_abstention_metrics": abstention_metrics,
        "candidates": rows,
    }
    write_json(output_dir / "calibration_analysis.json", payload)
    return payload


def video_inventory(output_path: Path) -> pd.DataFrame:
    paths = []
    for path in (ROOT / "datasets").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            text = str(path).lower()
            if "squat" in text or f"{Path('external/Ex6')}".lower().replace("/", str(Path('/'))) in text:
                paths.append(path)
    segmentation = pd.read_csv(ROOT / "datasets" / "external" / "Segmentation.csv", sep=";")
    segmentation = segmentation[segmentation["exercise_id"] == 6]
    rows = []
    for path in sorted(paths):
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        readable = bool(capture.isOpened() and frames > 0)
        capture.release()
        source = "REHAB24-6" if path.parent.name == "Ex6" else ("local_user" if "user" in path.parts else "local_reference_or_raw")
        name_lower = path.name.lower()
        approximate_view = "front" if "front" in name_lower else ("side" if "side" in name_lower else None)
        match = segmentation[segmentation["video_id"].astype(str).map(lambda value: value in path.name)] if source == "REHAB24-6" else segmentation.iloc[0:0]
        if source == "REHAB24-6" and len(match):
            unique_views = sorted(match["cam17_orientation"].dropna().astype(str).unique())
            approximate_view = ",".join(unique_views) if "Camera17" in path.name else "camera18_paired_view_not_directly_labeled"
        rows.append(
            {
                "path": str(path.resolve()),
                "source": source,
                "approximate_view": approximate_view,
                "view_evidence": "filename" if approximate_view in {"front", "side"} else ("REHAB24 metadata" if source == "REHAB24-6" else "unknown"),
                "fps": fps if fps > 0 else None,
                "frames": frames if frames > 0 else None,
                "duration_seconds": frames / fps if fps > 0 and frames > 0 else None,
                "manual_rep_count": None,
                "annotation_rep_count": int(len(match)) if len(match) else None,
                "correctness_gt": bool(len(match)),
                "error_gt": False,
                "readable": readable,
                "historical_split_subject": (str(match.iloc[0]["person_id"]) if len(match) else None),
            }
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def _required_visibility(segment: np.ndarray) -> dict[str, float]:
    names = (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        "HEEL_LEFT", "HEEL_RIGHT", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    )
    return {
        name.lower(): float(segment[:, MEDIAPIPE_LANDMARKS[name], 3].mean())
        for name in names
    }


@torch.no_grad()
def trace_video(video_path: Path, pose_model: Path, output_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    orchestrator = SquatAIExperimentalOrchestrator(
        SquatAIMVPConfig.load(), allow_partial=False
    )
    source = VideoFrameSource(video_path)
    pose = PoseStreamProcessor(pose_model)
    pose_success = 0
    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            landmarks = pose.process(packet.frame, packet.timestamp_seconds)
            pose_success += int(landmarks is not None)
            orchestrator.record_frame(landmarks, packet.frame_index, packet.timestamp_seconds)
    finally:
        pose.close()
        source.close()
    landmarks, frame_indices, timestamps = orchestrator._snapshot()
    normalized, segments, boundary_latency = orchestrator._detect_segments(landmarks)
    rep_rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    representative_features: list[np.ndarray] = []
    for rep_index, (start, end) in enumerate(segments, 1):
        normalized_segment = normalized[start : end + 1]
        resampled = resample_sequence(normalized_segment)
        mask = resampled[..., 2].mean(axis=1) > 0.01
        output = orchestrator.correctness_model(
            torch.from_numpy(resampled).unsqueeze(0).to(orchestrator.device),
            torch.from_numpy(mask).unsqueeze(0).to(orchestrator.device),
        )
        logits = output["logits"][0].detach().cpu().numpy()
        embedding = output["global_embedding"][0].detach().cpu().numpy()
        embeddings.append(embedding)
        center, selected = orchestrator._representative_indices(landmarks[start : end + 1])
        features = np.stack(
            [orchestrator.feature_extractor.extract(landmarks[start + index]) for index in selected]
        )
        representative_features.append(features.mean(axis=0))
        error_output = orchestrator.error_model.predict(torch.from_numpy(features).to(orchestrator.device))
        frame_probabilities = error_output["probabilities"].cpu().numpy()
        mean_probability = frame_probabilities.mean(axis=0)
        median_probability = np.median(frame_probabilities, axis=0)
        majority = np.bincount(frame_probabilities.argmax(axis=1), minlength=3)
        confidence_weights = features[:, 58].clip(1e-6)
        weighted_probability = np.average(frame_probabilities, axis=0, weights=confidence_weights)
        segment_landmarks = landmarks[start : end + 1]
        rep_rows.append(
            {
                "rep_index": rep_index,
                "start_frame": frame_indices[start],
                "end_frame": frame_indices[end],
                "duration_seconds": timestamps[end] - timestamps[start],
                "segment_frames": end - start + 1,
                "representative_frame": None if center is None else frame_indices[start + center],
                "representative_frames": [frame_indices[start + index] for index in selected],
                "mean_mediapipe_confidence": float(segment_landmarks[..., 3].mean()),
                "minimum_required_confidence": float(features[:, 59].min()),
                "key_joint_visibility": _required_visibility(segment_landmarks),
                "motionbert_input_mean": float(resampled[..., :2].mean()),
                "motionbert_input_std": float(resampled[..., :2].std()),
                "motionbert_confidence_mean": float(resampled[..., 2].mean()),
                "global_embedding_norm": float(np.linalg.norm(embedding)),
                "global_embedding_std": float(embedding.std()),
                "temporal_embedding_std": float(output["temporal_embedding"].detach().cpu().numpy().std()),
                "correctness_logits": logits.tolist(),
                "correct_probability": float(output["correct_probability"].item()),
                "threshold": orchestrator.config.correctness_threshold,
                "distance_from_threshold": float(output["correct_probability"].item() - orchestrator.config.correctness_threshold),
                "decision": "correct" if float(output["correct_probability"].item()) >= orchestrator.config.correctness_threshold else "incorrect",
                "error_frame_probabilities": frame_probabilities.tolist(),
                "error_mean_probabilities": mean_probability.tolist(),
                "error_median_probabilities": median_probability.tolist(),
                "error_majority_votes": majority.tolist(),
                "error_confidence_weighted_probabilities": weighted_probability.tolist(),
                "error_class": ("good", "bad_back", "bad_heel")[int(mean_probability.argmax())],
                "error_confidence": float(mean_probability.max()),
            }
        )
    trace = {
        "video": str(video_path.resolve()),
        "fps": source.fps,
        "frames": len(landmarks),
        "pose_success_frames": pose_success,
        "pose_success_rate": pose_success / max(len(landmarks), 1),
        "boundary_latency_ms": boundary_latency,
        "segments": rep_rows,
    }
    write_json(output_dir / "side_trace.json", trace)
    pd.DataFrame(
        [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rep_rows]
    ).to_csv(output_dir / "side_rep_trace.csv", index=False)
    orchestrator.close()
    return trace, np.stack(embeddings), np.stack(representative_features)


@torch.no_grad()
def correctness_ood_reference(user_embeddings: np.ndarray, output_dir: Path) -> dict[str, Any]:
    manifest = pd.read_csv(ROOT / "results" / "squat_ai_v3" / "correctness" / "development_manifest.csv")
    orchestrator = SquatAIExperimentalOrchestrator(
        SquatAIMVPConfig.load(), allow_partial=False
    )
    embeddings = []
    sample_ids = []
    for _, row in manifest.iterrows():
        cache_path = (
            ROOT
            / "datasets"
            / "window_cache"
            / "rehab24_squat_v1"
            / "repetitions"
            / f"{row['sample_id']}.npz"
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            values = np.asarray(archive["motionbert_input"], dtype=np.float32)
            mask = np.asarray(archive["temporal_mask"], dtype=bool)
        output = orchestrator.correctness_model(
            torch.from_numpy(values).unsqueeze(0).to(orchestrator.device),
            torch.from_numpy(mask).unsqueeze(0).to(orchestrator.device),
        )
        embeddings.append(output["global_embedding"][0].cpu().numpy())
        sample_ids.append(str(row["sample_id"]))
    orchestrator.close()
    reference = np.stack(embeddings).astype(np.float32)
    normalized = reference / np.linalg.norm(reference, axis=1, keepdims=True).clip(1e-8)
    neighbors = NearestNeighbors(n_neighbors=6, metric="cosine").fit(normalized)
    distances, _ = neighbors.kneighbors(normalized)
    leave_one_out = distances[:, 5]
    threshold = float(np.percentile(leave_one_out, 95))
    user_normalized = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True).clip(1e-8)
    user_distances, _ = neighbors.kneighbors(user_normalized, n_neighbors=5)
    user_scores = user_distances[:, 4]
    artifact = output_dir / "correctness_ood_reference.npz"
    np.savez_compressed(
        artifact,
        normalized_embeddings=normalized,
        sample_ids=np.asarray(sample_ids),
        kth_neighbor=5,
        distance_threshold=np.asarray(threshold, dtype=np.float32),
    )
    payload = {
        "method": "cosine distance to 5th nearest Development embedding",
        "development_samples": len(reference),
        "historical_test_subjects_used": False,
        "threshold_source": "95th percentile leave-one-out Development distance",
        "threshold": threshold,
        "development_distance_percentiles": {
            str(percentile): float(np.percentile(leave_one_out, percentile))
            for percentile in (50, 90, 95, 99)
        },
        "user_rep_distances": user_scores.tolist(),
        "user_rep_out_of_domain": (user_scores > threshold).tolist(),
        "artifact": str(artifact.resolve()),
    }
    write_json(output_dir / "correctness_ood_analysis.json", payload)
    return payload


@torch.no_grad()
def error_domain_analysis(user_features: np.ndarray, output_dir: Path) -> dict[str, Any]:
    data_dir = ROOT / "results" / "squat_error_v1" / "data"
    manifest = pd.read_csv(data_dir / "development_split_manifest.csv")
    features = np.load(data_dir / "pose_features.npy", mmap_mode="r")
    names = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    train_rows = manifest[(manifest["source_split"] == "train") & (manifest["development_split"] == "train")]
    validation_rows = manifest[(manifest["source_split"] == "train") & (manifest["development_split"] == "validation")]
    train = np.asarray(features[train_rows["feature_index"].to_numpy(np.int64)], dtype=np.float32)
    validation = np.asarray(features[validation_rows["feature_index"].to_numpy(np.int64)], dtype=np.float32)
    mean = train.mean(axis=0); std = train.std(axis=0).clip(1e-6)
    lower = np.percentile(train, 1, axis=0); upper = np.percentile(train, 99, axis=0)

    def ood_score(values: np.ndarray) -> np.ndarray:
        return ((values < lower) | (values > upper)).mean(axis=1)

    validation_scores = ood_score(validation)
    threshold = float(np.percentile(validation_scores, 95))
    user_scores = ood_score(user_features)
    shift = (user_features.mean(axis=0) - mean) / std
    out_frequency = ((user_features < lower) | (user_features > upper)).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": names,
            "training_mean": mean,
            "training_std": std,
            "training_p01": lower,
            "training_p99": upper,
            "user_mean": user_features.mean(axis=0),
            "standardized_shift": shift,
            "user_out_of_range_frequency": out_frequency,
        }
    ).sort_values("standardized_shift", key=lambda values: values.abs(), ascending=False)
    table.to_csv(output_dir / "feature_shift.csv", index=False)

    orchestrator = SquatAIExperimentalOrchestrator(
        SquatAIMVPConfig.load(), allow_partial=False
    )
    model = orchestrator.error_model
    device = next(model.parameters()).device
    probabilities = model.predict(torch.from_numpy(validation).to(device))["probabilities"].cpu().numpy()
    targets = validation_rows["label_index"].to_numpy(np.int64)
    predictions = probabilities.argmax(axis=1)
    correct_confidence = probabilities.max(axis=1)[predictions == targets]
    confidence_threshold = float(np.percentile(correct_confidence, 5))
    orchestrator.close()

    payload = {
        "reference": "SquatDataset provided Train / development-train only",
        "training_samples": len(train),
        "validation_samples": len(validation),
        "feature_dim": len(names),
        "pose_success_index": names.index("pose_success"),
        "confidence_feature_indices": [index for index, name in enumerate(names) if "confidence" in name],
        "ood_score": "fraction of 64 features outside training p01-p99",
        "ood_threshold": threshold,
        "ood_threshold_source": "95th percentile of untouched development-validation OOD score",
        "user_rep_ood_scores": user_scores.tolist(),
        "user_rep_out_of_domain": (user_scores > threshold).tolist(),
        "error_confidence_threshold": confidence_threshold,
        "error_confidence_threshold_source": "5th percentile confidence among correctly classified development-validation samples",
        "top_shifted_features": table.head(15).to_dict("records"),
    }
    write_json(output_dir / "error_domain_analysis.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=SIDE_VIDEO)
    parser.add_argument("--pose-model", type=Path, default=POSE_MODEL)
    parser.add_argument("--oof", type=Path, default=OOF_BACKUP)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "squat_ai_v4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    diagnostics = args.output_dir / "diagnostics"
    calibration = args.output_dir / "calibration"
    error_domain = args.output_dir / "error_domain"
    for path in (diagnostics, calibration, error_domain):
        path.mkdir(parents=True, exist_ok=True)
    inventory = video_inventory(diagnostics / "user_video_inventory.csv")
    trace, embeddings, features = trace_video(args.video, args.pose_model, diagnostics)
    calibration_result = calibration_analysis(args.oof, calibration)
    correctness_ood = correctness_ood_reference(embeddings, calibration)
    error_domain_result = error_domain_analysis(features, error_domain)
    summary = {
        "inventory_videos": len(inventory),
        "trace_repetitions": len(trace["segments"]),
        "calibration": calibration_result,
        "correctness_ood": correctness_ood,
        "error_domain": error_domain_result,
    }
    write_json(args.output_dir / "diagnostic_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
