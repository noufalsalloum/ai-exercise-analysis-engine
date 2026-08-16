"""Test-locked Squat correctness experiments for Experiment 2."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.squat_correctness import SquatCorrectnessModel
from training.squat_correctness import (
    SquatFeatureDataset,
    classification_metrics,
    grouped_metrics,
    load_checkpoint_strict,
    mini_overfit,
    save_checkpoint,
    seed_everything,
)


@dataclass(frozen=True)
class CorrectnessExperiment:
    experiment_id: str
    loss: str
    sampling: str


class MulticlassFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__(); self.register_buffer("alpha", alpha); self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probability = nn.functional.log_softmax(logits, dim=-1)
        probability = log_probability.exp(); indices = torch.arange(len(targets), device=targets.device)
        target_log = log_probability[indices, targets]; target_probability = probability[indices, targets]
        return (-self.alpha[targets] * (1.0 - target_probability).pow(self.gamma) * target_log).mean()


@torch.no_grad()
def prediction_frame(
    model: SquatCorrectnessModel,
    dataset: SquatFeatureDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0); model.eval()
    probabilities: list[float] = []; row_indices: list[int] = []; embeddings: list[np.ndarray] = []
    for features, masks, _, rows in loader:
        output = model.forward_features(features.to(device), masks.to(device))
        probabilities.extend(output["correct_probability"].cpu().tolist()); row_indices.extend(rows.tolist()); embeddings.append(output["global_embedding"].cpu().numpy())
    frame = dataset.manifest.iloc[row_indices].copy().reset_index(drop=True); frame["correct_probability"] = probabilities
    return frame, np.concatenate(embeddings)


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    predictions = (frame["correct_probability"].to_numpy() >= threshold).astype(np.int64)
    return classification_metrics(frame["correctness"].to_numpy(), predictions, frame["correct_probability"].to_numpy())


def calibrate_threshold(
    frame: pd.DataFrame, minimum_incorrect_recall: float = 0.60
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(0.25, 0.85, 61):
        metrics = threshold_metrics(frame, float(threshold)); incorrect = metrics["classes"]["incorrect"]
        rows.append({"threshold": float(threshold), "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"], "incorrect_precision": incorrect["precision"], "incorrect_recall": incorrect["recall"], "incorrect_f1": incorrect["f1"], "correct_recall": metrics["classes"]["correct"]["recall"], "roc_auc": metrics["roc_auc"], "constraint_satisfied": incorrect["recall"] >= minimum_incorrect_recall})
    table = pd.DataFrame(rows); feasible = table[table["constraint_satisfied"]]
    pool = feasible if len(feasible) else table
    selected = pool.sort_values(["macro_f1", "incorrect_recall", "balanced_accuracy", "threshold"], ascending=[False, False, False, True]).iloc[0]
    threshold = float(selected["threshold"])
    return threshold, threshold_metrics(frame, threshold), table


def ranking(metrics: dict[str, Any]) -> tuple[int, float, float, float]:
    incorrect_recall = float(metrics["classes"]["incorrect"]["recall"])
    return (int(incorrect_recall >= 0.60), float(metrics["macro_f1"]), incorrect_recall, float(metrics["balanced_accuracy"]))


def make_loader(
    dataset: SquatFeatureDataset,
    experiment: CorrectnessExperiment,
    batch_size: int,
    seed: int,
) -> DataLoader:
    if experiment.sampling == "balanced":
        labels = dataset.manifest.iloc[dataset.row_indices]["correctness"].to_numpy(np.int64)
        counts = np.bincount(labels, minlength=2); weights = np.asarray([1.0 / counts[label] for label in labels], np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)


def train_variant(
    experiment: CorrectnessExperiment,
    train_dataset: SquatFeatureDataset,
    validation_dataset: SquatFeatureDataset,
    motionbert_checkpoint: Path,
    checkpoint_dir: Path,
    device: torch.device,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(int(config["seed"])); model = SquatCorrectnessModel(motionbert_checkpoint).to(device)
    labels = train_dataset.manifest.iloc[train_dataset.row_indices]["correctness"].to_numpy(np.int64); counts = np.bincount(labels, minlength=2)
    inverse = len(labels) / (2.0 * counts); alpha = torch.tensor((1.0 / counts) / (1.0 / counts).sum(), dtype=torch.float32, device=device)
    if experiment.loss == "weighted_ce": loss_fn: nn.Module = nn.CrossEntropyLoss(weight=torch.tensor(inverse, dtype=torch.float32, device=device))
    elif experiment.loss == "unweighted_ce": loss_fn = nn.CrossEntropyLoss()
    elif experiment.loss == "focal": loss_fn = MulticlassFocalLoss(alpha)
    else: raise ValueError(experiment.loss)
    loader = make_loader(train_dataset, experiment, int(config["batch_size"]), int(config["seed"]))
    optimizer = torch.optim.AdamW(list(model.expert.parameters()) + list(model.correctness_head.parameters()), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True); best_metrics: dict[str, Any] | None = None; best_epoch = 0; stale = 0; history: list[dict[str, Any]] = []; started = perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); losses: list[float] = []; gradient_norms: list[float] = []
        for features, masks, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True); output = model.forward_features(features.to(device), masks.to(device)); loss = loss_fn(output["logits"], targets.to(device))
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite correctness loss in {experiment.experiment_id}")
            loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(list(model.expert.parameters()) + list(model.correctness_head.parameters()), float(config["gradient_clip"])); optimizer.step(); losses.append(float(loss.item())); gradient_norms.append(float(gradient))
        if any(parameter.grad is not None for parameter in model.backbone.parameters()): raise RuntimeError("Frozen MotionBERT received gradients.")
        validation_frame, embeddings = prediction_frame(model, validation_dataset, device, int(config["batch_size"]))
        threshold, metrics, _ = calibrate_threshold(validation_frame)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "decision_threshold": threshold, "val_macro_f1": metrics["macro_f1"], "val_balanced_accuracy": metrics["balanced_accuracy"], "val_incorrect_recall": metrics["classes"]["incorrect"]["recall"], "val_correct_recall": metrics["classes"]["correct"]["recall"], "gradient_norm": float(np.mean(gradient_norms)), "embedding_std": float(embeddings.std())}
        history.append(row); print(json.dumps({"experiment": experiment.experiment_id, **row}), flush=True)
        state_metadata = {**metadata, "experiment": experiment.__dict__, "decision_threshold": threshold, "test_locked": True, "selection_constraint": "incorrect_recall>=0.60", "train_class_counts": counts.tolist(), "train_inverse_frequency_weights": inverse.tolist(), "focal_alpha": alpha.cpu().tolist()}
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, metrics, state_metadata)
        if best_metrics is None or ranking(metrics) > ranking(best_metrics):
            best_metrics = metrics; best_epoch = epoch; stale = 0; save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, config, metrics, state_metadata)
        else:
            stale += 1
            if stale >= int(config["patience"]): break
    pd.DataFrame(history).to_csv(checkpoint_dir / "training_history.csv", index=False)
    strict_model = SquatCorrectnessModel(motionbert_checkpoint).to(device); checkpoint = load_checkpoint_strict(checkpoint_dir / "best.pt", strict_model, device)
    validation_frame, embeddings = prediction_frame(strict_model, validation_dataset, device, int(config["batch_size"]))
    threshold, metrics, table = calibrate_threshold(validation_frame); table.to_csv(checkpoint_dir / "threshold_table.csv", index=False)
    if abs(threshold - float(checkpoint["decision_threshold"])) > 1e-12: raise RuntimeError("Recalibrated threshold differs from selected checkpoint metadata.")
    validation_frame["predicted_correctness"] = (validation_frame["correct_probability"] >= threshold).astype(int)
    validation_frame.to_csv(checkpoint_dir / "validation_predictions.csv", index=False)
    return {"experiment_id": experiment.experiment_id, "loss": experiment.loss, "sampling": experiment.sampling, "class_weights": inverse.tolist() if experiment.loss == "weighted_ce" else None, "focal_alpha": alpha.cpu().tolist() if experiment.loss == "focal" else None, "decision_threshold": threshold, "val_accuracy": metrics["accuracy"], "val_balanced_accuracy": metrics["balanced_accuracy"], "val_macro_f1": metrics["macro_f1"], "val_correct_recall": metrics["classes"]["correct"]["recall"], "val_incorrect_recall": metrics["classes"]["incorrect"]["recall"], "val_incorrect_precision": metrics["classes"]["incorrect"]["precision"], "val_roc_auc": metrics["roc_auc"], "best_epoch": int(checkpoint["epoch"]), "epochs_completed": len(history), "embedding_std": float(embeddings.std()), "training_seconds": perf_counter() - started, "checkpoint_path": str((checkpoint_dir / "best.pt").resolve())}


def fusion_metrics(frame: pd.DataFrame, threshold: float, confidence_weighted: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    source = frame.copy()
    if confidence_weighted:
        weighted = source.assign(weighted_probability=source["correct_probability"] * source["signal_confidence"])
        fused = weighted.groupby("pair_id", as_index=False).agg(correctness=("correctness", "first"), weighted_sum=("weighted_probability", "sum"), confidence_sum=("signal_confidence", "sum"))
        fused["correct_probability"] = fused["weighted_sum"] / fused["confidence_sum"].clip(lower=1e-8)
    else:
        fused = source.groupby("pair_id", as_index=False).agg(correctness=("correctness", "first"), correct_probability=("correct_probability", "mean"))
    fused["predicted_correctness"] = (fused["correct_probability"] >= threshold).astype(int)
    return classification_metrics(fused["correctness"], fused["predicted_correctness"], fused["correct_probability"]), fused


def add_signal_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy(); confidences: list[float] = []
    for path in output["repetition_cache_path"]:
        with np.load(Path(str(path)), allow_pickle=False) as archive:
            confidences.append(float(np.asarray(archive["motionbert_input"])[..., 2].mean()))
    output["signal_confidence"] = confidences
    return output


def save_curves(frame: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    targets = frame["correctness"].to_numpy(); probability = frame["correct_probability"].to_numpy()
    precision, recall, _ = precision_recall_curve(targets, probability); fpr, tpr, _ = roc_curve(targets, probability)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(recall, precision, color="#0f62fe"); axes[0].set(xlabel="Correct recall", ylabel="Correct precision", title="Validation precision-recall curve")
    axes[1].plot(fpr, tpr, color="#d62728"); axes[1].plot([0, 1], [0, 1], "--", color="#888888"); axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Validation ROC curve")
    figure.suptitle(f"Selected threshold={threshold:.2f} (Test locked)"); figure.tight_layout(); figure.savefig(output_dir / "validation_pr_roc_curves.png", dpi=160); plt.close(figure)


def run_correctness_experiments(
    train_dataset: SquatFeatureDataset,
    validation_dataset: SquatFeatureDataset,
    motionbert_checkpoint: Path,
    v1_checkpoint: Path,
    output_dir: Path,
    checkpoint_root: Path,
    device: torch.device,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if set(train_dataset.manifest.iloc[train_dataset.row_indices]["split"]) != {"train"} or set(validation_dataset.manifest.iloc[validation_dataset.row_indices]["split"]) != {"validation"}: raise RuntimeError("Test lock violation in correctness datasets.")
    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    overfit_model = SquatCorrectnessModel(motionbert_checkpoint).to(device); overfit = mini_overfit(overfit_model, train_dataset, device)
    (output_dir / "mini_overfit.json").write_text(json.dumps(overfit, indent=2), encoding="utf-8")
    if not overfit["passed"]: raise RuntimeError(f"Correctness mini-overfit failed: {overfit}")
    del overfit_model
    rows: list[dict[str, Any]] = []
    # Validation-only V1 reference at its original fixed 0.5 threshold.
    v1_model = SquatCorrectnessModel(motionbert_checkpoint).to(device); load_checkpoint_strict(v1_checkpoint, v1_model, device)
    v1_frame, v1_embeddings = prediction_frame(v1_model, validation_dataset, device, int(config["batch_size"])); v1_metrics = threshold_metrics(v1_frame, 0.5)
    rows.append({"experiment_id": "C0_v1_fixed_threshold", "loss": "weighted_ce", "sampling": "shuffle", "class_weights": "V1 train-only inverse frequency", "focal_alpha": None, "decision_threshold": 0.5, "val_accuracy": v1_metrics["accuracy"], "val_balanced_accuracy": v1_metrics["balanced_accuracy"], "val_macro_f1": v1_metrics["macro_f1"], "val_correct_recall": v1_metrics["classes"]["correct"]["recall"], "val_incorrect_recall": v1_metrics["classes"]["incorrect"]["recall"], "val_incorrect_precision": v1_metrics["classes"]["incorrect"]["precision"], "val_roc_auc": v1_metrics["roc_auc"], "best_epoch": 7, "epochs_completed": 0, "embedding_std": float(v1_embeddings.std()), "training_seconds": 0.0, "checkpoint_path": str(v1_checkpoint.resolve())})
    experiments = [CorrectnessExperiment("C1_weighted_ce", "weighted_ce", "shuffle"), CorrectnessExperiment("C2_balanced_sampler", "unweighted_ce", "balanced"), CorrectnessExperiment("C3_focal", "focal", "shuffle")]
    for experiment in experiments:
        rows.append(train_variant(experiment, train_dataset, validation_dataset, motionbert_checkpoint, checkpoint_root / experiment.experiment_id, device, config, metadata))
    frame = pd.DataFrame(rows); frame.to_csv(output_dir / "experiments.csv", index=False)
    candidates = rows[1:]; selected = max(candidates, key=lambda row: (int(float(row["val_incorrect_recall"]) >= 0.60), float(row["val_macro_f1"]), float(row["val_incorrect_recall"]), float(row["val_balanced_accuracy"])))
    source = Path(str(selected["checkpoint_path"])); destination = checkpoint_root / "best.pt"; shutil.copy2(source, destination)
    selected_model = SquatCorrectnessModel(motionbert_checkpoint).to(device); selected_checkpoint = load_checkpoint_strict(destination, selected_model, device)
    selected_frame, embeddings = prediction_frame(selected_model, validation_dataset, device, int(config["batch_size"])); selected_frame = add_signal_confidence(selected_frame)
    threshold = float(selected["decision_threshold"]); threshold_value, selected_metrics, table = calibrate_threshold(selected_frame)
    if abs(threshold - threshold_value) > 1e-12: raise RuntimeError("Frozen threshold mismatch.")
    table.to_csv(output_dir / "threshold_table.csv", index=False); selected_frame["predicted_correctness"] = (selected_frame["correct_probability"] >= threshold).astype(int); selected_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    save_curves(selected_frame, output_dir, threshold)
    mean_metrics, mean_frame = fusion_metrics(selected_frame, threshold, False); weighted_metrics, weighted_frame = fusion_metrics(selected_frame, threshold, True)
    fusion = {"single_view": selected_metrics, "mean_two_view": mean_metrics, "confidence_weighted_two_view": weighted_metrics}
    (output_dir / "validation_fusion.json").write_text(json.dumps(fusion, indent=2), encoding="utf-8"); mean_frame.to_csv(output_dir / "validation_mean_fusion.csv", index=False); weighted_frame.to_csv(output_dir / "validation_confidence_weighted_fusion.csv", index=False)
    grouped_metrics(selected_frame.assign(predicted_correctness=selected_frame["predicted_correctness"]), "camera_id").to_csv(output_dir / "validation_per_camera.csv", index=False)
    grouped_metrics(selected_frame.assign(predicted_correctness=selected_frame["predicted_correctness"]), "orientation_raw").to_csv(output_dir / "validation_per_orientation.csv", index=False)
    frozen = {"test_lock": True, "selection_objective": "maximize validation Macro F1 subject to incorrect recall >= 0.60", "selected_experiment": selected, "final_checkpoint": str(destination.resolve()), "decision_threshold": threshold, "validation_metrics": selected_metrics, "validation_fusion": fusion, "embedding_std": float(embeddings.std()), "strict_checkpoint_epoch": int(selected_checkpoint["epoch"]), "all_experiments": rows, "mini_overfit": overfit}
    (output_dir / "selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen

