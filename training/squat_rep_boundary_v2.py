"""Development-only boundary experiments for Squat AI Experiment 2."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.squat_rep_boundary import SquatRepBoundaryModel
from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from training.squat_rep_boundary import FullVideoRecord, match_segments, smooth_probabilities, temporal_iou


@dataclass(frozen=True)
class BoundaryExperiment:
    experiment_id: str
    architecture: str
    loss: str
    active_dilation: int = 0
    boundary_radius: int = 0
    boundary_loss_weight: float = 0.0


class BoundaryV2WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Train windows with original-boundary-derived active and boundary targets."""

    def __init__(
        self,
        records: Sequence[FullVideoRecord],
        window_size: int,
        stride: int,
        active_dilation: int,
        boundary_radius: int,
    ) -> None:
        self.records = list(records); self.window_size = int(window_size)
        self.active_targets: list[np.ndarray] = []; self.boundary_targets: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            active = record.labels.copy(); boundary = np.zeros(len(record.labels), np.float32)
            for start, end in record.segments:
                if active_dilation:
                    active[max(0, start - active_dilation) : min(len(active), end + active_dilation + 1)] = 1.0
                if boundary_radius:
                    boundary[max(0, start - boundary_radius) : min(len(boundary), start + boundary_radius + 1)] = 1.0
                    boundary[max(0, end - boundary_radius) : min(len(boundary), end + boundary_radius + 1)] = 1.0
            self.active_targets.append(active); self.boundary_targets.append(boundary)
            starts = list(range(0, max(len(record.poses) - window_size + 1, 1), stride))
            final = max(0, len(record.poses) - window_size)
            if not starts or starts[-1] != final: starts.append(final)
            self.indices.extend((record_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index, start = self.indices[index]; record = self.records[record_index]
        stop = min(start + self.window_size, len(record.poses)); valid = stop - start
        poses = np.zeros((self.window_size, 17, 3), np.float32)
        active = np.zeros(self.window_size, np.float32); boundary = np.zeros(self.window_size, np.float32); mask = np.zeros(self.window_size, bool)
        poses[:valid] = record.poses[start:stop]
        active[:valid] = self.active_targets[record_index][start:stop]
        boundary[:valid] = self.boundary_targets[record_index][start:stop]
        mask[:valid] = True
        return torch.from_numpy(poses), torch.from_numpy(active), torch.from_numpy(boundary), torch.from_numpy(mask)


class BinaryFocalLoss(nn.Module):
    """Numerically stable binary focal loss with train-derived alpha."""

    def __init__(self, alpha_positive: float, gamma: float = 2.0) -> None:
        super().__init__(); self.alpha_positive = float(alpha_positive); self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha = self.alpha_positive * targets + (1.0 - self.alpha_positive) * (1.0 - targets)
        return alpha * (1.0 - pt).pow(self.gamma) * bce


def make_model(architecture: str) -> nn.Module:
    if architecture == "active_tcn": return SquatRepBoundaryModel()
    if architecture == "boundary_aux_tcn": return SquatRepBoundaryV2Model()
    raise ValueError(f"Unknown architecture {architecture!r}.")


def model_logits(model: nn.Module, poses: torch.Tensor, architecture: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(poses)
    if architecture == "active_tcn": return output, None
    return output["active_logits"], output["boundary_logits"]


@torch.no_grad()
def predict_records(
    model: nn.Module,
    records: Sequence[FullVideoRecord],
    architecture: str,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray | None]]:
    model.eval(); output: dict[str, dict[str, np.ndarray | None]] = {}
    for record in records:
        poses = torch.from_numpy(record.poses).unsqueeze(0).to(device)
        active_logits, boundary_logits = model_logits(model, poses, architecture)
        output[record.key] = {
            "active": torch.sigmoid(active_logits.squeeze(0)).cpu().numpy(),
            "boundary": None if boundary_logits is None else torch.sigmoid(boundary_logits.squeeze(0)).cpu().numpy(),
        }
    return output


def _hysteresis_runs(values: np.ndarray, enter: float, exit_value: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []; active = False; start = 0
    for index, probability in enumerate(values):
        if not active and probability >= enter:
            active = True; start = index
        elif active and probability < exit_value:
            runs.append((start, index - 1)); active = False
    if active: runs.append((start, len(values) - 1))
    return runs


def _cluster_centers(mask: np.ndarray, probabilities: np.ndarray, cluster_gap: int) -> list[int]:
    indices = np.flatnonzero(mask)
    if not len(indices): return []
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        if int(value) - groups[-1][-1] <= cluster_gap:
            groups[-1].append(int(value))
        else:
            groups.append([int(value)])
    return [max(group, key=lambda index: float(probabilities[index])) for group in groups]


def v2_segments(
    active_probability: np.ndarray,
    boundary_probability: np.ndarray | None,
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    """Hysteresis activity with boundary/valley splitting for merged repetitions."""

    active = smooth_probabilities(active_probability, int(config["smoothing_kernel"]))
    runs = _hysteresis_runs(active, float(config["enter_threshold"]), float(config["exit_threshold"]))
    # Small gaps may be sensor dropouts; merge before looking for learned boundaries.
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= int(config["merge_gap"]):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    boundary = None if boundary_probability is None else smooth_probabilities(boundary_probability, int(config["boundary_smoothing_kernel"]))
    minimum = int(config["min_duration"]); maximum = config.get("max_duration")
    segments: list[tuple[int, int]] = []
    for run_start, run_end in merged:
        run_segments: list[tuple[int, int]] = []
        split_points: list[int] = []
        if boundary is not None:
            mask = np.zeros(len(boundary), bool)
            mask[run_start : run_end + 1] = boundary[run_start : run_end + 1] >= float(config["boundary_threshold"])
            centers = _cluster_centers(mask, boundary, int(config["boundary_cluster_gap"]))
            split_points = [center for center in centers if center - run_start >= minimum and run_end - center >= minimum]
        # If a plateau is still implausibly long, split at its deepest active valleys.
        current = run_start
        for point in sorted(split_points):
            if point - current >= minimum:
                run_segments.append((current, point - 1)); current = point
        run_segments.append((current, run_end))
        if maximum is not None:
            revised: list[tuple[int, int]] = []
            for start, end in run_segments:
                pending = [(start, end)]
                while pending:
                    left, right = pending.pop(0)
                    if right - left + 1 <= int(maximum) or right - left + 1 < 2 * minimum:
                        revised.append((left, right)); continue
                    low = left + minimum; high = right - minimum + 1
                    valley = low + int(np.argmin(active[low:high]))
                    pending.insert(0, (valley, right)); pending.insert(0, (left, valley - 1))
            run_segments = revised
        segments.extend(run_segments)
    return [(start, end) for start, end in segments if end - start + 1 >= minimum]


def evaluate_prediction_cache(
    records: Sequence[FullVideoRecord],
    predictions: dict[str, dict[str, np.ndarray | None]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    video_rows: list[dict[str, Any]] = []; segment_rows: list[dict[str, Any]] = []
    total_matches = total_predicted = total_gt = 0; ious: list[float] = []; start_errors: list[int] = []; end_errors: list[int] = []
    for record in records:
        cached = predictions[record.key]
        predicted = v2_segments(cached["active"], cached["boundary"], config)
        matches = match_segments(predicted, record.segments, float(config["iou_threshold"]))
        total_matches += len(matches); total_predicted += len(predicted); total_gt += len(record.segments)
        matched = {pred_index: (gt_index, iou) for pred_index, gt_index, iou in matches}
        for index, (start, end) in enumerate(predicted):
            match = matched.get(index); gt_index = match[0] if match else None
            segment_rows.append({"video_key": record.key, "subject_id": record.subject_id, "camera_id": record.camera_id, "segment_index": index + 1, "start_frame": start + 1, "end_frame": end + 1, "matched_gt_index": None if gt_index is None else gt_index + 1, "temporal_iou": None if match is None else match[1]})
            if match:
                gt = record.segments[gt_index]; ious.append(float(match[1])); start_errors.append(abs(start - gt[0])); end_errors.append(abs(end - gt[1]))
        error = len(predicted) - len(record.segments)
        video_rows.append({"video_key": record.key, "subject_id": record.subject_id, "camera_id": record.camera_id, "ground_truth_repetitions": len(record.segments), "predicted_repetitions": len(predicted), "absolute_count_error": abs(error), "over_count": max(error, 0), "under_count": max(-error, 0), "exact_count": int(error == 0)})
    videos = pd.DataFrame(video_rows)
    precision = total_matches / total_predicted if total_predicted else 0.0; recall = total_matches / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"videos": len(records), "ground_truth_repetitions": total_gt, "predicted_repetitions": total_predicted, "mean_absolute_count_error": float(videos["absolute_count_error"].mean()), "exact_count_accuracy": float(videos["exact_count"].mean()), "over_count_total": int(videos["over_count"].sum()), "under_count_total": int(videos["under_count"].sum()), "segment_precision": precision, "segment_recall": recall, "segment_f1": f1, "matched_segments": total_matches, "mean_temporal_iou": float(np.mean(ious)) if ious else None, "mean_start_frame_error": float(np.mean(start_errors)) if start_errors else None, "mean_end_frame_error": float(np.mean(end_errors)) if end_errors else None, "postprocessing": config}
    return metrics, videos, pd.DataFrame(segment_rows)


def objective(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (-float(metrics["mean_absolute_count_error"]), float(metrics["segment_f1"]), -float(metrics["under_count_total"]), -float(metrics["over_count_total"]))


def postprocess_grid(train_records: Sequence[FullVideoRecord], has_boundary: bool) -> Iterable[dict[str, Any]]:
    lengths = np.asarray([end - start + 1 for record in train_records for start, end in record.segments])
    min_values = sorted({max(20, int(np.percentile(lengths, 5) * factor)) for factor in (0.35, 0.50)})
    max_values: list[int | None] = [int(math.ceil(np.percentile(lengths, 95) * 1.05)), None]
    boundary_thresholds = (0.35, 0.50, 0.65) if has_boundary else (1.1,)
    for enter in (0.35, 0.45, 0.55):
        for exit_value in (0.20, 0.30, 0.40):
            if exit_value >= enter: continue
            for minimum in min_values:
                for maximum in max_values:
                    for merge_gap in (0, 5):
                        for boundary_threshold in boundary_thresholds:
                            yield {"enter_threshold": enter, "exit_threshold": exit_value, "smoothing_kernel": 9, "min_duration": minimum, "max_duration": maximum, "merge_gap": merge_gap, "minimum_rest_gap": 8, "boundary_threshold": boundary_threshold, "boundary_smoothing_kernel": 7, "boundary_cluster_gap": 10, "iou_threshold": 0.5}


def calibrate(
    train_records: Sequence[FullVideoRecord],
    validation_records: Sequence[FullVideoRecord],
    predictions: dict[str, dict[str, np.ndarray | None]],
    has_boundary: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_config: dict[str, Any] | None = None; best_metrics: dict[str, Any] | None = None
    for config in postprocess_grid(train_records, has_boundary):
        metrics, _, _ = evaluate_prediction_cache(validation_records, predictions, config)
        if best_metrics is None or objective(metrics) > objective(best_metrics):
            best_config = config; best_metrics = metrics
    assert best_config is not None and best_metrics is not None
    return best_config, best_metrics


def _loss_values(
    experiment: BoundaryExperiment,
    active_logits: torch.Tensor,
    boundary_logits: torch.Tensor | None,
    active_targets: torch.Tensor,
    boundary_targets: torch.Tensor,
    active_weight: torch.Tensor,
    boundary_weight: torch.Tensor,
    focal_alpha: float,
) -> torch.Tensor:
    if experiment.loss == "weighted_bce":
        active_loss = nn.functional.binary_cross_entropy_with_logits(active_logits, active_targets, reduction="none", pos_weight=active_weight)
    elif experiment.loss == "focal":
        active_loss = BinaryFocalLoss(focal_alpha)(active_logits, active_targets)
    else:
        raise ValueError(experiment.loss)
    if boundary_logits is None: return active_loss
    boundary_loss = nn.functional.binary_cross_entropy_with_logits(boundary_logits, boundary_targets, reduction="none", pos_weight=boundary_weight)
    return active_loss + experiment.boundary_loss_weight * boundary_loss


def train_experiment(
    experiment: BoundaryExperiment,
    train_records: Sequence[FullVideoRecord],
    validation_records: Sequence[FullVideoRecord],
    checkpoint_dir: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    random.seed(int(config["seed"])); np.random.seed(int(config["seed"])); torch.manual_seed(int(config["seed"]))
    dataset = BoundaryV2WindowDataset(train_records, int(config["window_size"]), int(config["stride"]), experiment.active_dilation, experiment.boundary_radius)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(int(config["seed"])), num_workers=0)
    all_active = np.concatenate(dataset.active_targets); all_boundary = np.concatenate(dataset.boundary_targets)
    active_positive = float(all_active.sum()); active_negative = len(all_active) - active_positive
    boundary_positive = float(all_boundary.sum()); boundary_negative = len(all_boundary) - boundary_positive
    active_weight = torch.tensor(active_negative / active_positive, device=device)
    boundary_weight = torch.tensor(min(boundary_negative / max(boundary_positive, 1.0), 20.0), device=device)
    focal_alpha = active_negative / len(all_active)
    model = make_model(experiment.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True); best_metrics: dict[str, Any] | None = None; best_epoch = 0; stale = 0; history: list[dict[str, Any]] = []; started = perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); losses: list[float] = []
        for poses, active_targets, boundary_targets, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            active_logits, boundary_logits = model_logits(model, poses.to(device), experiment.architecture)
            raw = _loss_values(experiment, active_logits, boundary_logits, active_targets.to(device), boundary_targets.to(device), active_weight, boundary_weight, focal_alpha)
            loss = raw[mask.to(device)].mean()
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss in {experiment.experiment_id}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"])); optimizer.step(); losses.append(float(loss.item()))
        prediction_cache = predict_records(model, validation_records, experiment.architecture, device)
        calibrated, metrics = calibrate(train_records, validation_records, prediction_cache, experiment.architecture == "boundary_aux_tcn")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_count_mae": metrics["mean_absolute_count_error"], "val_segment_f1": metrics["segment_f1"], "val_under_count": metrics["under_count_total"], "val_over_count": metrics["over_count_total"]}
        history.append(row); print(json.dumps({"experiment": experiment.experiment_id, **row}), flush=True)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "experiment": experiment.__dict__, "training_config": config, "postprocessing": calibrated, "validation_metrics": metrics, "test_locked": True, "active_pos_weight": float(active_weight), "boundary_pos_weight": float(boundary_weight), "focal_alpha_positive": focal_alpha}
        torch.save(state, checkpoint_dir / "last.pt")
        if best_metrics is None or objective(metrics) > objective(best_metrics):
            best_metrics = metrics; best_epoch = epoch; stale = 0; torch.save(state, checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(config["patience"]): break
    pd.DataFrame(history).to_csv(checkpoint_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True)
    strict_model = make_model(experiment.architecture).to(device); strict_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {"experiment_id": experiment.experiment_id, "loss": experiment.loss, "architecture": experiment.architecture, "class_weights": f"active_pos_weight={float(active_weight):.6f};boundary_pos_weight={float(boundary_weight):.6f}", "threshold": checkpoint["postprocessing"]["enter_threshold"], "hysteresis": f"enter={checkpoint['postprocessing']['enter_threshold']};exit={checkpoint['postprocessing']['exit_threshold']}", "min_duration": checkpoint["postprocessing"]["min_duration"], "max_duration": checkpoint["postprocessing"]["max_duration"], "merge_gap": checkpoint["postprocessing"]["merge_gap"], "boundary_threshold": checkpoint["postprocessing"]["boundary_threshold"], "val_segment_f1": checkpoint["validation_metrics"]["segment_f1"], "val_count_mae": checkpoint["validation_metrics"]["mean_absolute_count_error"], "val_exact_count_accuracy": checkpoint["validation_metrics"]["exact_count_accuracy"], "val_over_count": checkpoint["validation_metrics"]["over_count_total"], "val_under_count": checkpoint["validation_metrics"]["under_count_total"], "best_epoch": int(checkpoint["epoch"]), "epochs_completed": len(history), "training_seconds": perf_counter() - started, "checkpoint_path": str((checkpoint_dir / "best.pt").resolve())}


def mini_overfit_boundary_v2(
    train_records: Sequence[FullVideoRecord], device: torch.device, seed: int = 42
) -> dict[str, Any]:
    """Prove finite gradients and memorization for the new dual-head model."""

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset = BoundaryV2WindowDataset(train_records[:2], 256, 128, active_dilation=2, boundary_radius=5)
    batch = [dataset[index] for index in range(min(4, len(dataset)))]
    poses = torch.stack([item[0] for item in batch]).to(device)
    active = torch.stack([item[1] for item in batch]).to(device)
    boundary = torch.stack([item[2] for item in batch]).to(device)
    mask = torch.stack([item[3] for item in batch]).to(device)
    model = SquatRepBoundaryV2Model(channels=48, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    initial = final = 0.0
    for step in range(120):
        model.train(); optimizer.zero_grad(set_to_none=True); output = model(poses)
        active_loss = nn.functional.binary_cross_entropy_with_logits(output["active_logits"], active, reduction="none")
        boundary_loss = nn.functional.binary_cross_entropy_with_logits(output["boundary_logits"], boundary, reduction="none")
        loss = (active_loss[mask].mean() + 0.35 * boundary_loss[mask].mean())
        if not torch.isfinite(loss): raise FloatingPointError("Boundary V2 mini-overfit is non-finite.")
        loss.backward(); optimizer.step()
        if step == 0: initial = float(loss.item())
        final = float(loss.item())
        if final < initial * 0.20: break
    gradient_flow = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    return {"passed": bool(final < initial * 0.35 and gradient_flow), "steps": step + 1, "initial_loss": initial, "final_loss": final, "gradient_flow": gradient_flow, "finite": True}


def run_boundary_experiments(
    records: Sequence[FullVideoRecord],
    v1_checkpoint: Path,
    output_dir: Path,
    checkpoint_root: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    if any(record.split == "test" for record in records): raise RuntimeError("Test lock violation: development runner received Test records.")
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    overfit = mini_overfit_boundary_v2(train_records, device, int(config["seed"]))
    (output_dir / "mini_overfit.json").write_text(json.dumps(overfit, indent=2), encoding="utf-8")
    if not overfit["passed"]: raise RuntimeError(f"Boundary V2 mini-overfit failed: {overfit}")
    rows: list[dict[str, Any]] = []
    # B0 diagnoses how far calibrated separation alone can take the locked V1 weights.
    old = torch.load(v1_checkpoint, map_location=device, weights_only=True); v1_model = SquatRepBoundaryModel().to(device); v1_model.load_state_dict(old["model_state_dict"], strict=True)
    v1_predictions = predict_records(v1_model, validation_records, "active_tcn", device)
    v1_post, v1_metrics = calibrate(train_records, validation_records, v1_predictions, False)
    v1_v2_checkpoint = checkpoint_root / "B0_v1_enhanced_postprocess" / "best.pt"; v1_v2_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**old, "postprocessing": v1_post, "validation_metrics": v1_metrics, "experiment": {"experiment_id": "B0_v1_enhanced_postprocess", "architecture": "active_tcn", "loss": "existing_weighted_bce"}, "test_locked": True}, v1_v2_checkpoint)
    rows.append({"experiment_id": "B0_v1_enhanced_postprocess", "loss": "existing_weighted_bce", "architecture": "active_tcn", "class_weights": f"active_pos_weight={old.get('training_config', old.get('config', {}))}", "threshold": v1_post["enter_threshold"], "hysteresis": f"enter={v1_post['enter_threshold']};exit={v1_post['exit_threshold']}", "min_duration": v1_post["min_duration"], "max_duration": v1_post["max_duration"], "merge_gap": v1_post["merge_gap"], "boundary_threshold": v1_post["boundary_threshold"], "val_segment_f1": v1_metrics["segment_f1"], "val_count_mae": v1_metrics["mean_absolute_count_error"], "val_exact_count_accuracy": v1_metrics["exact_count_accuracy"], "val_over_count": v1_metrics["over_count_total"], "val_under_count": v1_metrics["under_count_total"], "best_epoch": int(old["epoch"]), "epochs_completed": 0, "training_seconds": 0.0, "checkpoint_path": str(v1_v2_checkpoint.resolve())})
    experiments = [
        BoundaryExperiment("B1_weighted_bce", "active_tcn", "weighted_bce"),
        BoundaryExperiment("B2_focal", "active_tcn", "focal"),
        BoundaryExperiment("B3_boundary_aux", "boundary_aux_tcn", "weighted_bce", active_dilation=2, boundary_radius=5, boundary_loss_weight=0.35),
    ]
    for experiment in experiments:
        rows.append(train_experiment(experiment, train_records, validation_records, checkpoint_root / experiment.experiment_id, device, config))
    experiments_frame = pd.DataFrame(rows); experiments_frame.to_csv(output_dir / "experiments.csv", index=False)
    best_index = max(range(len(rows)), key=lambda index: (-float(rows[index]["val_count_mae"]), float(rows[index]["val_segment_f1"]), -float(rows[index]["val_under_count"]), -float(rows[index]["val_over_count"])))
    selected = rows[best_index]; source = Path(str(selected["checkpoint_path"])); destination = checkpoint_root / "best.pt"; shutil.copy2(source, destination)
    frozen = {"test_lock": True, "mini_overfit": overfit, "selection_priority": ["minimum validation count MAE", "maximum validation segment F1", "minimum under-count", "minimum over-count"], "selected_experiment": selected, "final_checkpoint": str(destination.resolve()), "all_experiments": rows}
    (output_dir / "selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen
