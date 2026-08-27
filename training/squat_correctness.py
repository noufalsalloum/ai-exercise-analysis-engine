"""Training utilities for REHAB24-6 per-repetition Squat correctness."""

from __future__ import annotations

import csv
import hashlib
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from models.squat_correctness import SquatCorrectnessModel


FEATURE_CACHE_VERSION = "rehab24_squat_motionbert_lite_rep60_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class FeatureCache:
    path: Path
    metadata_path: Path
    sample_ids: tuple[str, ...]


class SquatFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """Memory-mapped frozen MotionBERT features plus repetition metadata."""

    def __init__(self, manifest: pd.DataFrame, feature_cache: FeatureCache, split: str) -> None:
        self.manifest = manifest.reset_index(drop=True)
        index_by_id = {sample_id: index for index, sample_id in enumerate(feature_cache.sample_ids)}
        rows = self.manifest[self.manifest["split"] == split]
        if rows.empty:
            raise ValueError(f"No samples for split {split!r}.")
        self.row_indices = rows.index.to_numpy(np.int64)
        self.feature_indices = np.asarray(
            [index_by_id[str(self.manifest.iloc[index]["sample_id"])] for index in self.row_indices],
            dtype=np.int64,
        )
        self.features = np.load(feature_cache.path, mmap_mode="r")
        if self.features.shape != (len(feature_cache.sample_ids), 60, 17, 512):
            raise ValueError(f"Invalid frozen feature cache shape {self.features.shape}.")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row_index = int(self.row_indices[index])
        row = self.manifest.iloc[row_index]
        feature = np.asarray(self.features[int(self.feature_indices[index])], dtype=np.float32)
        rep_cache = Path(str(row["repetition_cache_path"]))
        with np.load(rep_cache, allow_pickle=False) as archive:
            mask = np.asarray(archive["temporal_mask"], dtype=bool)
        if feature.shape != (60, 17, 512) or mask.shape != (60,):
            raise ValueError(f"Invalid sample contract for {row['sample_id']}.")
        if not np.isfinite(feature).all() or not mask.any():
            raise FloatingPointError(f"Invalid feature values for {row['sample_id']}.")
        return (
            torch.from_numpy(feature.copy()),
            torch.from_numpy(mask.copy()),
            torch.tensor(int(row["correctness"]), dtype=torch.long),
            row_index,
        )


def load_manifest(manifest_path: Path, repetition_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, dtype={"subject_id": str})
    required = {"sample_id", "pair_id", "subject_id", "camera_id", "orientation_raw", "correctness", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame["repetition_cache_path"] = frame["sample_id"].map(
        lambda value: str((repetition_dir / f"{value}.npz").resolve())
    )
    missing_files = [path for path in frame["repetition_cache_path"] if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} repetition caches; first={missing_files[0]}")
    split_subjects = {
        split: set(frame.loc[frame["split"] == split, "subject_id"])
        for split in ("train", "validation", "test")
    }
    if any(split_subjects[left] & split_subjects[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Subject leakage detected in repetition manifest.")
    return frame


@torch.no_grad()
def build_frozen_feature_cache(
    model: SquatCorrectnessModel,
    manifest: pd.DataFrame,
    output_dir: Path,
    motionbert_checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> FeatureCache:
    """Evaluate frozen MotionBERT once and store float16 features in a memmap."""

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    metadata_path = output_dir / "metadata.json"
    sample_ids = tuple(str(value) for value in manifest["sample_id"])
    checkpoint_hash = sha256(motionbert_checkpoint)
    if feature_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("feature_cache_version") == FEATURE_CACHE_VERSION
            and tuple(metadata.get("sample_ids", ())) == sample_ids
            and metadata.get("motionbert_sha256") == checkpoint_hash
        ):
            array = np.load(feature_path, mmap_mode="r")
            if array.shape == (len(sample_ids), 60, 17, 512) and np.isfinite(array).all():
                return FeatureCache(feature_path, metadata_path, sample_ids)
        raise ValueError("Existing frozen feature cache metadata is incompatible.")

    inputs: list[np.ndarray] = []
    output = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16, shape=(len(sample_ids), 60, 17, 512)
    )
    model.backbone.eval()
    offset = 0
    for row in manifest.itertuples(index=False):
        with np.load(Path(row.repetition_cache_path), allow_pickle=False) as archive:
            value = np.asarray(archive["motionbert_input"], dtype=np.float32)
        if value.shape != (60, 17, 3) or not np.isfinite(value).all():
            raise ValueError(f"Invalid MotionBERT input for {row.sample_id}: {value.shape}")
        inputs.append(value)
        if len(inputs) == batch_size or offset + len(inputs) == len(sample_ids):
            tensor = torch.from_numpy(np.stack(inputs)).to(device)
            features = model.backbone(tensor).cpu().numpy().astype(np.float16)
            output[offset : offset + len(inputs)] = features
            offset += len(inputs)
            inputs.clear()
            if offset % max(batch_size * 5, 1) == 0 or offset == len(sample_ids):
                print(f"frozen MotionBERT features {offset}/{len(sample_ids)}", flush=True)
    output.flush()
    metadata = {
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "shape": list(output.shape),
        "dtype": "float16",
        "sample_ids": list(sample_ids),
        "motionbert_sha256": checkpoint_hash,
        "input_contract": "preprocessing_v4_(T,17,3)_x_y_confidence",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return FeatureCache(feature_path, metadata_path, sample_ids)


def classification_metrics(targets: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.int64)
    prediction = np.asarray(predictions, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=[0, 1], zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
        "classes": {
            "incorrect": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "correct": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
    }
    try:
        result["roc_auc"] = float(roc_auc_score(target, probability))
    except ValueError:
        result["roc_auc"] = None
    return result


@torch.no_grad()
def evaluate(
    model: SquatCorrectnessModel,
    dataset: SquatFeatureDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    row_indices: list[int] = []
    embeddings: list[np.ndarray] = []
    for features, masks, labels, rows in loader:
        output = model.forward_features(features.to(device), masks.to(device))
        probability = output["correct_probability"].cpu().numpy()
        prediction = (probability >= 0.5).astype(np.int64)
        targets.extend(labels.numpy().tolist())
        predictions.extend(prediction.tolist())
        probabilities.extend(probability.tolist())
        row_indices.extend(rows.numpy().tolist())
        embeddings.append(output["global_embedding"].cpu().numpy())
    rows = dataset.manifest.iloc[row_indices].copy().reset_index(drop=True)
    rows["correct_probability"] = probabilities
    rows["predicted_correctness"] = predictions
    rows["pass_fail"] = ["PASS" if value else "FAIL" for value in predictions]
    rows["score"] = None
    return classification_metrics(targets, predictions, probabilities), rows, np.concatenate(embeddings)


def grouped_metrics(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in predictions.groupby(column, dropna=False):
        metrics = classification_metrics(group["correctness"], group["predicted_correctness"], group["correct_probability"])
        rows.append({column: value, "samples": len(group), "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"], "incorrect_recall": metrics["classes"]["incorrect"]["recall"], "correct_recall": metrics["classes"]["correct"]["recall"], "roc_auc": metrics["roc_auc"]})
    return pd.DataFrame(rows)


def fused_camera_metrics(predictions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    fused = predictions.groupby("pair_id", as_index=False).agg(
        correctness=("correctness", "first"),
        correct_probability=("correct_probability", "mean"),
        subject_id=("subject_id", "first"),
        video_id=("video_id", "first"),
        repetition_number=("repetition_number", "first"),
    )
    fused["predicted_correctness"] = (fused["correct_probability"] >= 0.5).astype(int)
    metrics = classification_metrics(fused["correctness"], fused["predicted_correctness"], fused["correct_probability"])
    return metrics, fused


def save_checkpoint(
    path: Path,
    model: SquatCorrectnessModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expert_state_dict": model.expert.state_dict(),
            "correctness_head_state_dict": model.correctness_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            **metadata,
        },
        path,
    )


def load_checkpoint_strict(
    path: Path, model: SquatCorrectnessModel, device: torch.device, *, weights_only: bool = True
) -> dict[str, Any]:
    # weights_only defaults to True (unchanged) for every caller — training
    # scripts, evaluation, tests, and archived-checkpoint comparisons.
    # inference/squat_ai_mvp.py is the sole caller that explicitly passes
    # weights_only=False, scoped there to the one trusted production
    # checkpoint (checkpoints/squat_ai_v3/correctness/final_dev.pt) — see
    # the comment at that call site for why.
    checkpoint = torch.load(path, map_location=device, weights_only=weights_only)
    model.expert.load_state_dict(checkpoint["expert_state_dict"], strict=True)
    model.correctness_head.load_state_dict(checkpoint["correctness_head_state_dict"], strict=True)
    return checkpoint


def mini_overfit(
    model: SquatCorrectnessModel,
    dataset: SquatFeatureDataset,
    device: torch.device,
    steps: int = 160,
) -> dict[str, float | int | bool]:
    """Prove gradient/data wiring on a tiny balanced train-only subset."""

    labels = dataset.manifest.iloc[dataset.row_indices]["correctness"].to_numpy()
    local = np.arange(len(dataset))
    selected = np.concatenate([local[labels == 0][:6], local[labels == 1][:6]])
    batch = [dataset[int(index)] for index in selected]
    features = torch.stack([item[0] for item in batch]).to(device)
    masks = torch.stack([item[1] for item in batch]).to(device)
    targets = torch.stack([item[2] for item in batch]).to(device)
    optimizer = torch.optim.AdamW(
        list(model.expert.parameters()) + list(model.correctness_head.parameters()), lr=1e-3
    )
    loss_fn = nn.CrossEntropyLoss()
    initial_loss = final_loss = 0.0
    final_accuracy = 0.0
    for step in range(steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model.forward_features(features, masks)
        loss = loss_fn(output["logits"], targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Mini-overfit loss is not finite.")
        loss.backward(); optimizer.step()
        if step == 0:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())
        final_accuracy = float((output["logits"].argmax(-1) == targets).float().mean().item())
        if final_accuracy == 1.0 and final_loss < 0.03:
            break
    passed = final_accuracy >= 0.95 and final_loss < initial_loss * 0.25
    return {"passed": passed, "steps": step + 1, "initial_loss": initial_loss, "final_loss": final_loss, "accuracy": final_accuracy}


def train_correctness(
    model: SquatCorrectnessModel,
    datasets: dict[str, SquatFeatureDataset],
    output_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Train on train subjects, select by validation Macro F1, then open test once."""

    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_labels = datasets["train"].manifest.iloc[datasets["train"].row_indices]["correctness"].to_numpy(np.int64)
    counts = np.bincount(train_labels, minlength=2)
    class_weights = len(train_labels) / (2.0 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    train_loader = DataLoader(datasets["train"], batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        list(model.expert.parameters()) + list(model.correctness_head.parameters()),
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    history: list[dict[str, Any]] = []; best_macro = -1.0; best_epoch = 0; stale = 0
    started = perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); losses: list[float] = []; gradient_norms: list[float] = []
        for features, masks, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_features(features.to(device), masks.to(device))
            loss = loss_fn(output["logits"], labels.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("Training loss is not finite.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(model.expert.parameters()) + list(model.correctness_head.parameters()),
                float(config["gradient_clip"]),
            )
            optimizer.step(); losses.append(float(loss.item())); gradient_norms.append(float(gradient_norm))
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise RuntimeError("Frozen MotionBERT unexpectedly received gradients.")
        val_metrics, val_predictions, embeddings = evaluate(model, datasets["validation"], device, int(config["batch_size"]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": val_metrics["macro_f1"], "validation_accuracy": val_metrics["accuracy"], "validation_incorrect_recall": val_metrics["classes"]["incorrect"]["recall"], "gradient_norm": float(np.mean(gradient_norms)), "embedding_mean": float(embeddings.mean()), "embedding_std": float(embeddings.std())}
        history.append(row); print(json.dumps(row), flush=True)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, val_metrics, metadata)
        if val_metrics["macro_f1"] > best_macro + 1e-9:
            best_macro = val_metrics["macro_f1"]; best_epoch = epoch; stale = 0
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, config, val_metrics, metadata)
            val_predictions.to_csv(output_dir / "best_validation_predictions.csv", index=False)
        else:
            stale += 1
            if stale >= int(config["patience"]): break
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = load_checkpoint_strict(checkpoint_dir / "best.pt", model, device)
    validation_metrics, validation_predictions, validation_embeddings = evaluate(model, datasets["validation"], device, int(config["batch_size"]))
    test_metrics, test_predictions, test_embeddings = evaluate(model, datasets["test"], device, int(config["batch_size"]))
    validation_fused, _ = fused_camera_metrics(validation_predictions)
    test_fused, fused_predictions = fused_camera_metrics(test_predictions)
    validation_metrics["camera_fused"] = validation_fused; test_metrics["camera_fused"] = test_fused
    for name, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_predictions.to_csv(output_dir / "predictions.csv", index=False)
    fused_predictions.to_csv(output_dir / "fused_test_predictions.csv", index=False)
    grouped_metrics(test_predictions, "subject_id").to_csv(output_dir / "per_subject_metrics.csv", index=False)
    grouped_metrics(test_predictions, "camera_id").to_csv(output_dir / "per_camera_metrics.csv", index=False)
    grouped_metrics(test_predictions, "orientation_raw").to_csv(output_dir / "per_orientation_metrics.csv", index=False)
    np.savetxt(output_dir / "confusion_matrix.csv", np.asarray(test_metrics["confusion_matrix"], np.int64), fmt="%d", delimiter=",")
    diagnostics = {"validation_embedding_mean": float(validation_embeddings.mean()), "validation_embedding_std": float(validation_embeddings.std()), "test_embedding_mean": float(test_embeddings.mean()), "test_embedding_std": float(test_embeddings.std())}
    (output_dir / "embedding_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    result = {"best_epoch": best_epoch, "best_validation_macro_f1": best_macro, "epochs_completed": len(history), "training_seconds": perf_counter() - started, "class_weights_train_only": class_weights.tolist(), "strict_checkpoint_epoch": int(checkpoint["epoch"]), "validation": validation_metrics, "test": test_metrics, "embedding_diagnostics": diagnostics}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

