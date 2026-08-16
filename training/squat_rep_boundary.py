"""Learned OUTSIDE_REP/INSIDE_REP baseline and segment evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from models.squat_rep_boundary import SquatRepBoundaryModel


@dataclass
class FullVideoRecord:
    key: str
    video_id: str
    subject_id: str
    camera_id: str
    orientation_raw: str
    split: str
    poses: np.ndarray
    labels: np.ndarray
    segments: list[tuple[int, int]]


def load_full_video_records(manifest: pd.DataFrame, full_cache_dir: Path) -> list[FullVideoRecord]:
    """Load each full cached camera stream once and derive binary GT labels."""

    records: list[FullVideoRecord] = []
    for (video_id, camera_id), rows in manifest.groupby(["video_id", "camera_id"], sort=True):
        source_stem = Path(str(rows.iloc[0]["video_path"])).stem
        cache_stem = source_stem.replace("-30fps-transposed", "_cam18").replace("-30fps", "_cam17")
        path = full_cache_dir / f"{cache_stem}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            poses = np.asarray(archive["motionbert_input"], dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (17, 3) or not np.isfinite(poses).all():
            raise ValueError(f"Invalid full-video cache {path}: {poses.shape}")
        segments = sorted(
            (int(row.start_frame) - 1, min(int(row.end_frame), len(poses)) - 1)
            for row in rows.itertuples(index=False)
        )
        labels = np.zeros(len(poses), dtype=np.float32)
        for start, end in segments:
            labels[start : end + 1] = 1.0
        orientations = sorted(str(value) for value in rows["orientation_raw"].dropna().unique())
        orientation_summary = orientations[0] if len(orientations) == 1 else "mixed:" + "|".join(orientations)
        records.append(
            FullVideoRecord(
                key=f"{video_id}_{camera_id}", video_id=str(video_id),
                subject_id=str(rows.iloc[0]["subject_id"]), camera_id=str(camera_id),
                orientation_raw=orientation_summary, split=str(rows.iloc[0]["split"]),
                poses=poses, labels=labels, segments=segments,
            )
        )
    split_subjects = {
        split: {record.subject_id for record in records if record.split == split}
        for split in ("train", "validation", "test")
    }
    if any(split_subjects[a] & split_subjects[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Subject leakage in full-video records.")
    return records


class BoundaryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Overlapping full-video windows; padded positions are masked from loss."""

    def __init__(self, records: Sequence[FullVideoRecord], window_size: int = 256, stride: int = 128) -> None:
        self.records = list(records); self.window_size = int(window_size)
        self.indices: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            starts = list(range(0, max(len(record.poses) - window_size + 1, 1), stride))
            final = max(0, len(record.poses) - window_size)
            if not starts or starts[-1] != final: starts.append(final)
            self.indices.extend((record_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index, start = self.indices[index]; record = self.records[record_index]
        stop = min(start + self.window_size, len(record.poses)); valid = stop - start
        poses = np.zeros((self.window_size, 17, 3), np.float32)
        labels = np.zeros(self.window_size, np.float32); mask = np.zeros(self.window_size, bool)
        poses[:valid] = record.poses[start:stop]; labels[:valid] = record.labels[start:stop]; mask[:valid] = True
        return torch.from_numpy(poses), torch.from_numpy(labels), torch.from_numpy(mask)


def smooth_probabilities(probabilities: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1: return probabilities.copy()
    return np.convolve(probabilities, np.ones(kernel, dtype=np.float64) / kernel, mode="same")


def probabilities_to_segments(
    probabilities: np.ndarray,
    threshold: float,
    smoothing_kernel: int,
    min_length: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    active = smooth_probabilities(probabilities, smoothing_kernel) >= threshold
    transitions = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1); ends = np.flatnonzero(transitions == -1) - 1
    segments = [(int(start), int(end)) for start, end in zip(starts, ends)]
    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if merged and start - merged[-1][1] - 1 <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start + 1 >= min_length]


def temporal_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]) + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return intersection / union if union else 0.0


def match_segments(
    predicted: Sequence[tuple[int, int]], ground_truth: Sequence[tuple[int, int]], iou_threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        ((p, g, temporal_iou(pred, gt)) for p, pred in enumerate(predicted) for g, gt in enumerate(ground_truth)),
        key=lambda item: item[2], reverse=True,
    )
    used_pred: set[int] = set(); used_gt: set[int] = set(); matches: list[tuple[int, int, float]] = []
    for pred_index, gt_index, iou in candidates:
        if iou < iou_threshold: break
        if pred_index not in used_pred and gt_index not in used_gt:
            used_pred.add(pred_index); used_gt.add(gt_index); matches.append((pred_index, gt_index, iou))
    return matches


@torch.no_grad()
def predict_full_video(model: SquatRepBoundaryModel, record: FullVideoRecord, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(record.poses).unsqueeze(0).to(device)).squeeze(0)
    return torch.sigmoid(logits).cpu().numpy()


@torch.no_grad()
def evaluate_records(
    model: SquatRepBoundaryModel,
    records: Sequence[FullVideoRecord],
    device: torch.device,
    postprocess: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    video_rows: list[dict[str, Any]] = []; segment_rows: list[dict[str, Any]] = []
    all_targets: list[np.ndarray] = []; all_predictions: list[np.ndarray] = []
    total_matches = 0; total_predicted = 0; total_gt = 0; ious: list[float] = []; start_errors: list[int] = []; end_errors: list[int] = []
    for record in records:
        probabilities = predict_full_video(model, record, device)
        predicted = probabilities_to_segments(
            probabilities,
            threshold=float(postprocess["threshold"]),
            smoothing_kernel=int(postprocess["smoothing_kernel"]),
            min_length=int(postprocess["min_length"]),
            merge_gap=int(postprocess["merge_gap"]),
        )
        matches = match_segments(predicted, record.segments, float(postprocess.get("iou_threshold", 0.5)))
        total_matches += len(matches); total_predicted += len(predicted); total_gt += len(record.segments)
        matched_pred = {item[0]: item for item in matches}
        for pred_index, (start, end) in enumerate(predicted):
            match = matched_pred.get(pred_index)
            gt_index = match[1] if match else None; iou = match[2] if match else None
            segment_rows.append({"video_key": record.key, "subject_id": record.subject_id, "camera_id": record.camera_id, "predicted_segment_index": pred_index + 1, "start_frame": start + 1, "end_frame": end + 1, "matched_gt_index": None if gt_index is None else gt_index + 1, "temporal_iou": iou})
            if match:
                gt = record.segments[gt_index]; ious.append(float(iou)); start_errors.append(abs(start - gt[0])); end_errors.append(abs(end - gt[1]))
        binary = (probabilities >= float(postprocess["threshold"])).astype(np.int64)
        all_targets.append(record.labels.astype(np.int64)); all_predictions.append(binary)
        error = len(predicted) - len(record.segments)
        video_rows.append({"video_key": record.key, "video_id": record.video_id, "subject_id": record.subject_id, "camera_id": record.camera_id, "orientation_raw": record.orientation_raw, "ground_truth_repetitions": len(record.segments), "predicted_repetitions": len(predicted), "absolute_count_error": abs(error), "over_count": max(error, 0), "under_count": max(-error, 0), "exact_count": int(error == 0)})
    precision = total_matches / total_predicted if total_predicted else 0.0
    recall = total_matches / total_gt if total_gt else 0.0
    segment_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    frame_precision, frame_recall, frame_f1, _ = precision_recall_fscore_support(
        np.concatenate(all_targets), np.concatenate(all_predictions), average="binary", zero_division=0
    )
    videos = pd.DataFrame(video_rows)
    metrics = {
        "videos": len(records), "ground_truth_repetitions": total_gt, "predicted_repetitions": total_predicted,
        "mean_absolute_count_error": float(videos["absolute_count_error"].mean()),
        "exact_count_accuracy": float(videos["exact_count"].mean()),
        "over_count_total": int(videos["over_count"].sum()), "under_count_total": int(videos["under_count"].sum()),
        "segment_precision": precision, "segment_recall": recall, "segment_f1": segment_f1,
        "matched_segments": total_matches, "mean_temporal_iou": float(np.mean(ious)) if ious else None,
        "mean_start_frame_error": float(np.mean(start_errors)) if start_errors else None,
        "mean_end_frame_error": float(np.mean(end_errors)) if end_errors else None,
        "frame_precision": float(frame_precision), "frame_recall": float(frame_recall), "frame_f1": float(frame_f1),
        "postprocessing": postprocess,
    }
    return metrics, videos, pd.DataFrame(segment_rows)


def train_boundary(
    records: Sequence[FullVideoRecord], output_dir: Path, checkpoint_dir: Path,
    device: torch.device, config: dict[str, Any], metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    test_records = [record for record in records if record.split == "test"]
    train_lengths = [end - start + 1 for record in train_records for start, end in record.segments]
    postprocess = {"threshold": 0.5, "smoothing_kernel": 9, "min_length": max(15, int(np.percentile(train_lengths, 5) * 0.45)), "merge_gap": 12, "iou_threshold": 0.5}
    dataset = BoundaryWindowDataset(train_records, int(config["window_size"]), int(config["stride"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0)
    positives = sum(float(record.labels.sum()) for record in train_records); negatives = sum(len(record.labels) - float(record.labels.sum()) for record in train_records)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(negatives / positives, device=device))
    model = SquatRepBoundaryModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    best_f1 = -1.0; best_count_mae = float("inf"); best_epoch = 0; stale = 0; history: list[dict[str, Any]] = []; started = perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); losses: list[float] = []
        for poses, labels, mask in loader:
            optimizer.zero_grad(set_to_none=True); logits = model(poses.to(device)); raw = loss_fn(logits, labels.to(device))
            valid = mask.to(device); loss = raw[valid].mean()
            if not torch.isfinite(loss): raise FloatingPointError("Boundary loss is non-finite.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])); optimizer.step(); losses.append(float(loss.item()))
        validation_metrics, _, _ = evaluate_records(model, validation_records, device, postprocess)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_segment_f1": validation_metrics["segment_f1"], "validation_count_mae": validation_metrics["mean_absolute_count_error"], "validation_exact_count_accuracy": validation_metrics["exact_count_accuracy"]}
        history.append(row); print(json.dumps(row), flush=True)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "config": config, "postprocessing": postprocess, "validation_metrics": validation_metrics, **metadata}
        torch.save(state, checkpoint_dir / "last.pt")
        improved = row["validation_segment_f1"] > best_f1 + 1e-9 or (abs(row["validation_segment_f1"] - best_f1) < 1e-9 and row["validation_count_mae"] < best_count_mae)
        if improved:
            best_f1 = row["validation_segment_f1"]; best_count_mae = row["validation_count_mae"]; best_epoch = epoch; stale = 0; torch.save(state, checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(config["patience"]): break
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Calibrate only the probability threshold on validation after model selection.
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.30, 0.70, 9):
        candidate = {**postprocess, "threshold": float(threshold)}
        metrics, _, _ = evaluate_records(model, validation_records, device, candidate)
        candidates.append((metrics["segment_f1"], -metrics["mean_absolute_count_error"], float(threshold)))
    postprocess["threshold"] = max(candidates)[2]
    checkpoint["postprocessing"] = postprocess
    checkpoint["threshold_selection"] = "validation-only grid 0.30..0.70; segment F1 then count MAE"
    torch.save(checkpoint, checkpoint_dir / "best.pt")
    validation_metrics, validation_videos, validation_segments = evaluate_records(model, validation_records, device, postprocess)
    test_metrics, test_videos, test_segments = evaluate_records(model, test_records, device, postprocess)
    (output_dir / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    pd.concat([validation_videos.assign(split="validation"), test_videos.assign(split="test")]).to_csv(output_dir / "video_count_results.csv", index=False)
    pd.concat([validation_segments.assign(split="validation"), test_segments.assign(split="test")]).to_csv(output_dir / "predicted_segments.csv", index=False)
    result = {"best_epoch": best_epoch, "epochs_completed": len(history), "training_seconds": perf_counter() - started, "strict_checkpoint_loaded": True, "model_selection": "validation segment F1; tie-break lower count MAE", "validation": validation_metrics, "test": test_metrics}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
