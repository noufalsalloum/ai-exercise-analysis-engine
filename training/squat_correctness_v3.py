"""Subject-wise LOSO development for Squat Correctness V3.

Historical Test subjects 4 and 7 are prohibited by construction. The module
operates only on the explicitly enumerated development subjects and never
creates a Dataset for any other subject.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.squat_correctness import SquatCorrectnessModel
from training.squat_correctness import (
    FeatureCache,
    SquatFeatureDataset,
    classification_metrics,
    grouped_metrics,
    load_checkpoint_strict,
    save_checkpoint,
    seed_everything,
    sha256,
)
from training.squat_correctness_v2 import (
    MulticlassFocalLoss,
    calibrate_threshold,
    fusion_metrics,
    prediction_frame,
    ranking,
)


DEVELOPMENT_SUBJECTS: tuple[str, ...] = ("1", "2", "3", "5", "6", "8", "9")
HISTORICAL_TEST_SUBJECTS: frozenset[str] = frozenset({"4", "7"})


@dataclass(frozen=True)
class V3Strategy:
    """One development-only loss/sampling experiment."""

    strategy_id: str
    loss: str
    sampling: str


STRATEGY_A = V3Strategy("V3_A_weighted_ce", "weighted_ce", "shuffle")
STRATEGY_B = V3Strategy(
    "V3_B_subject_class_balanced", "unweighted_ce", "subject_class_balanced"
)
STRATEGY_C = V3Strategy("V3_C_focal", "focal", "shuffle")


@dataclass(frozen=True)
class TemporalSeedCache:
    """Fold-independent output of the frozen shared part of SquatExpert."""

    path: Path
    sample_ids: tuple[str, ...]


class TemporalSeedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """Load `(T,512)` seeds while preserving the original dataset interface."""

    def __init__(
        self, manifest: pd.DataFrame, cache: TemporalSeedCache, split: str
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        rows = self.manifest[self.manifest["split"] == split]
        if rows.empty:
            raise ValueError(f"No samples for split {split!r}.")
        self.row_indices = rows.index.to_numpy(np.int64)
        index_by_id = {value: index for index, value in enumerate(cache.sample_ids)}
        self.feature_indices = np.asarray(
            [index_by_id[str(self.manifest.iloc[index]["sample_id"])] for index in self.row_indices],
            dtype=np.int64,
        )
        self.seeds = np.load(cache.path, mmap_mode="r")
        if self.seeds.shape != (len(cache.sample_ids), 60, 512):
            raise ValueError(f"Invalid V3 temporal seed cache shape {self.seeds.shape}.")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row_index = int(self.row_indices[index]); row = self.manifest.iloc[row_index]
        seed = np.asarray(self.seeds[int(self.feature_indices[index])], dtype=np.float32)
        with np.load(Path(str(row["repetition_cache_path"])), allow_pickle=False) as archive:
            mask = np.asarray(archive["temporal_mask"], dtype=bool)
        if not np.isfinite(seed).all() or mask.shape != (60,) or not mask.any():
            raise FloatingPointError(f"Invalid V3 temporal seed for {row['sample_id']}.")
        return (
            torch.from_numpy(seed.copy()),
            torch.from_numpy(mask.copy()),
            torch.tensor(int(row["correctness"]), dtype=torch.long),
            row_index,
        )


def development_only_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the locked development cohort and validate its identity."""

    required = {
        "sample_id",
        "pair_id",
        "subject_id",
        "camera_id",
        "orientation_raw",
        "correctness",
        "repetition_cache_path",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"V3 manifest missing columns: {sorted(missing)}")
    source = frame.copy()
    source["subject_id"] = source["subject_id"].astype(str)
    development = source[source["subject_id"].isin(DEVELOPMENT_SUBJECTS)].copy()
    observed = set(development["subject_id"])
    if observed != set(DEVELOPMENT_SUBJECTS):
        raise ValueError(
            f"Development cohort mismatch: expected {DEVELOPMENT_SUBJECTS}, got {sorted(observed)}"
        )
    if set(development["subject_id"]) & HISTORICAL_TEST_SUBJECTS:
        raise RuntimeError("Historical Test subjects entered V3 development data.")
    if development["correctness"].isna().any():
        raise ValueError("Correctness labels are missing in development data.")
    return development.reset_index(drop=True)


def build_loso_folds(development: pd.DataFrame) -> list[pd.DataFrame]:
    """Build seven folds with complete subject/camera pairs kept together."""

    development = development_only_manifest(development)
    folds: list[pd.DataFrame] = []
    held_counts: dict[str, int] = {}
    for held_subject in DEVELOPMENT_SUBJECTS:
        fold = development.copy()
        fold["split"] = np.where(
            fold["subject_id"] == held_subject, "validation", "train"
        )
        train_subjects = set(fold.loc[fold["split"] == "train", "subject_id"])
        validation_subjects = set(
            fold.loc[fold["split"] == "validation", "subject_id"]
        )
        if train_subjects & validation_subjects:
            raise RuntimeError("Subject leakage in LOSO fold.")
        if validation_subjects != {held_subject}:
            raise RuntimeError("A LOSO fold must hold exactly one subject.")
        if fold.groupby("pair_id")["split"].nunique().max() != 1:
            raise RuntimeError("A camera pair crossed LOSO split boundaries.")
        held_counts[held_subject] = held_counts.get(held_subject, 0) + 1
        fold.attrs["held_subject"] = held_subject
        folds.append(fold)
    if held_counts != {subject: 1 for subject in DEVELOPMENT_SUBJECTS}:
        raise RuntimeError("Each development subject must be held out exactly once.")
    return folds


def subject_class_sample_weights(rows: pd.DataFrame) -> np.ndarray:
    """Equalize subjects first, then their available correctness classes."""

    if rows.empty:
        raise ValueError("Cannot sample an empty training fold.")
    counts = rows.groupby(["subject_id", "correctness"]).size().to_dict()
    classes_per_subject = rows.groupby("subject_id")["correctness"].nunique().to_dict()
    subjects = rows["subject_id"].nunique()
    weights = np.asarray(
        [
            1.0
            / (
                subjects
                * int(classes_per_subject[str(row.subject_id)])
                * int(counts[(str(row.subject_id), int(row.correctness))])
            )
            for row in rows.itertuples()
        ],
        dtype=np.float64,
    )
    return weights / weights.sum()


def initialize_from_external_representation(
    model: SquatCorrectnessModel, checkpoint_path: Path
) -> dict[str, Any]:
    """Load every shape-compatible Shared Expert tensor with explicit auditing.

    The source model was trained on PhysicalExerciseRecognition, not REHAB24.
    Squat's wider residual bottleneck is intentionally left family-specific.
    No ``strict=False`` load is used.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = checkpoint["model_state_dict"]
    target = model.expert.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    unexpected: list[str] = []
    for source_key, value in source.items():
        if not source_key.startswith("shared_expert."):
            continue
        key = source_key.removeprefix("shared_expert.")
        if key not in target:
            unexpected.append(key); continue
        if target[key].shape != value.shape:
            skipped_shape.append(key); continue
        mapped[key] = value
    allowed_shape_mismatch = {
        "exercise_adapter.adapter.0.weight",
        "exercise_adapter.adapter.0.bias",
        "exercise_adapter.adapter.3.weight",
    }
    if set(skipped_shape) != allowed_shape_mismatch or unexpected:
        raise RuntimeError(
            "Unexpected representation initialization mismatch: "
            f"shape={skipped_shape}, unexpected={unexpected}"
        )
    merged = {key: mapped.get(key, value) for key, value in target.items()}
    model.expert.load_state_dict(merged, strict=True)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "loaded_tensor_count": len(mapped),
        "family_specific_random_tensors": sorted(set(target) - set(mapped)),
        "source_dataset": checkpoint.get("source_dataset"),
    }


def set_v3_trainable_scope(model: SquatCorrectnessModel) -> list[nn.Parameter]:
    """Freeze the externally pretrained shared trunk; tune Squat-specific layers."""

    for parameter in model.expert.parameters():
        parameter.requires_grad_(False)
    trainable_modules = (
        model.expert.exercise_adapter,
        model.expert.temporal_attention,
        model.expert.global_projection,
        model.expert.global_norm,
        model.expert.global_mlp,
        model.expert.output_norm,
        model.correctness_head,
    )
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.no_grad()
def build_temporal_seed_cache(
    development: pd.DataFrame,
    feature_cache: FeatureCache,
    motionbert_checkpoint: Path,
    representation_checkpoint: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> TemporalSeedCache:
    """Cache the frozen shared-expert trunk once for all valid LOSO folds."""

    sample_ids = tuple(str(value) for value in development["sample_id"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "features.npy"; metadata_path = cache_dir / "metadata.json"
    expected_metadata = {
        "version": "squat_correctness_v3_external_shared_trunk_seed_v1",
        "sample_ids": list(sample_ids),
        "shape": [len(sample_ids), 60, 512],
        "representation_checkpoint_sha256": sha256(representation_checkpoint),
        "motionbert_feature_cache_metadata_sha256": sha256(feature_cache.metadata_path),
        "historical_test_subjects_included": False,
    }
    if path.is_file() and metadata_path.is_file():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed == expected_metadata:
            return TemporalSeedCache(path, sample_ids)

    manifest = development.copy(); manifest["split"] = "train"
    source_dataset = SquatFeatureDataset(manifest, feature_cache, "train")
    loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = SquatCorrectnessModel(motionbert_checkpoint).to(device)
    initialize_from_external_representation(model, representation_checkpoint)
    model.eval(); captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(args[0].detach().cpu())

    handle = model.expert.exercise_adapter.register_forward_pre_hook(capture)
    try:
        for features, masks, _, _ in loader:
            model.forward_features(features.to(device), masks.to(device))
    finally:
        handle.remove()
    seeds = torch.cat(captured).numpy()
    if seeds.shape != (len(sample_ids), 60, 512) or not np.isfinite(seeds).all():
        raise FloatingPointError(f"Invalid V3 temporal seed cache {seeds.shape}.")
    np.save(path, seeds.astype(np.float16))
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    return TemporalSeedCache(path, sample_ids)


def forward_from_temporal_seed(
    model: SquatCorrectnessModel,
    temporal_seed: torch.Tensor,
    temporal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run exactly the trainable tail of SquatExpert from a frozen trunk seed."""

    if temporal_seed.ndim != 3 or temporal_seed.shape[-1] != 512:
        raise ValueError("Expected V3 temporal seed (B,T,512).")
    mask = temporal_mask.to(device=temporal_seed.device, dtype=torch.bool)
    temporal = model.expert.exercise_adapter(temporal_seed)
    temporal = temporal.masked_fill(~mask.unsqueeze(-1), 0.0)
    logits = model.expert.temporal_attention(temporal).squeeze(-1)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    pooled = torch.sum(temporal * weights.unsqueeze(-1), dim=1)
    global_seed = model.expert.global_projection(pooled)
    global_embedding = model.expert.output_norm(
        global_seed + model.expert.global_mlp(model.expert.global_norm(global_seed))
    )
    classification_logits = model.correctness_head(global_embedding)
    return {
        "logits": classification_logits,
        "correct_probability": torch.softmax(classification_logits, dim=-1)[:, 1],
        "global_embedding": global_embedding,
    }


@torch.no_grad()
def prediction_from_temporal_seed(
    model: SquatCorrectnessModel,
    dataset: TemporalSeedDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); probabilities = []; row_indices = []; embeddings = []
    for seeds, masks, _, rows in loader:
        output = forward_from_temporal_seed(model, seeds.to(device), masks.to(device))
        probabilities.extend(output["correct_probability"].cpu().tolist())
        row_indices.extend(rows.tolist()); embeddings.append(output["global_embedding"].cpu().numpy())
    frame = dataset.manifest.iloc[row_indices].copy().reset_index(drop=True)
    frame["correct_probability"] = probabilities
    return frame, np.concatenate(embeddings)


def _loader(
    dataset: TemporalSeedDataset,
    strategy: V3Strategy,
    batch_size: int,
    seed: int,
) -> DataLoader:
    rows = dataset.manifest.iloc[dataset.row_indices].copy()
    if strategy.sampling == "subject_class_balanced":
        weights = subject_class_sample_weights(rows)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _loss(
    strategy: V3Strategy, labels: np.ndarray, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both correctness classes are required in every training fold.")
    inverse = len(labels) / (2.0 * counts)
    alpha_values = (1.0 / counts) / (1.0 / counts).sum()
    alpha = torch.tensor(alpha_values, dtype=torch.float32, device=device)
    if strategy.loss == "weighted_ce":
        return (
            nn.CrossEntropyLoss(
                weight=torch.tensor(inverse, dtype=torch.float32, device=device)
            ),
            {"class_weights": inverse.tolist(), "focal_alpha": None},
        )
    if strategy.loss == "unweighted_ce":
        return nn.CrossEntropyLoss(), {"class_weights": None, "focal_alpha": None}
    if strategy.loss == "focal":
        return MulticlassFocalLoss(alpha), {
            "class_weights": None,
            "focal_alpha": alpha_values.tolist(),
        }
    raise ValueError(f"Unsupported V3 loss {strategy.loss!r}.")


def _model_parameters(model: SquatCorrectnessModel) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _fit_fold(
    strategy: V3Strategy,
    fold: pd.DataFrame,
    temporal_seed_cache: TemporalSeedCache,
    motionbert_checkpoint: Path,
    representation_checkpoint: Path,
    result_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    held_subject = str(fold.attrs["held_subject"])
    if held_subject in HISTORICAL_TEST_SUBJECTS:
        raise RuntimeError("Historical Test subject cannot be a V3 fold.")
    train_dataset = TemporalSeedDataset(fold, temporal_seed_cache, "train")
    validation_dataset = TemporalSeedDataset(fold, temporal_seed_cache, "validation")
    seed = int(config["seed"]) + int(held_subject)
    seed_everything(seed)
    model = SquatCorrectnessModel(motionbert_checkpoint).to(device)
    initialization = initialize_from_external_representation(
        model, representation_checkpoint
    )
    set_v3_trainable_scope(model)
    train_rows = fold.loc[fold["split"] == "train"]
    labels = train_rows["correctness"].to_numpy(np.int64)
    loss_fn, loss_metadata = _loss(strategy, labels, device)
    loader = _loader(train_dataset, strategy, int(config["batch_size"]), seed)
    optimizer = torch.optim.AdamW(
        _model_parameters(model),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metrics: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for features, masks, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            output = forward_from_temporal_seed(
                model, features.to(device), masks.to(device)
            )
            loss_value = loss_fn(output["logits"], targets.to(device))
            if not torch.isfinite(loss_value):
                raise FloatingPointError("Non-finite V3 correctness loss.")
            loss_value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                _model_parameters(model), float(config["gradient_clip"])
            )
            optimizer.step()
            losses.append(float(loss_value.item()))
            gradient_norms.append(float(gradient))
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise RuntimeError("Frozen MotionBERT received a gradient in V3.")
        prediction, embeddings = prediction_from_temporal_seed(
            model, validation_dataset, device, int(config["batch_size"])
        )
        threshold, metrics, _ = calibrate_threshold(
            prediction, float(config["minimum_incorrect_recall"])
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "decision_threshold": threshold,
            "validation_macro_f1": metrics["macro_f1"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_incorrect_recall": metrics["classes"]["incorrect"]["recall"],
            "validation_correct_recall": metrics["classes"]["correct"]["recall"],
            "gradient_norm": float(np.mean(gradient_norms)),
            "embedding_std": float(embeddings.std()),
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "strategy": strategy.strategy_id,
                    "held_subject": held_subject,
                    **row,
                }
            ),
            flush=True,
        )
        metadata = {
            "training_stage": "development_loso_cv",
            "held_subject": held_subject,
            "train_subjects": sorted(set(train_rows["subject_id"]), key=int),
            "historical_test_subjects_excluded": sorted(HISTORICAL_TEST_SUBJECTS),
            "strategy": strategy.__dict__,
            "decision_threshold": threshold,
            "motionbert_frozen": True,
            "shared_trunk_frozen": True,
            "representation_initialization": initialization,
            "test_locked": True,
            **loss_metadata,
        }
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            metrics,
            metadata,
        )
        if best_metrics is None or ranking(metrics) > ranking(best_metrics):
            best_metrics = metrics
            stale = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                metrics,
                metadata,
            )
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break

    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
    strict_model = SquatCorrectnessModel(motionbert_checkpoint).to(device)
    checkpoint = load_checkpoint_strict(
        checkpoint_dir / "best.pt", strict_model, device
    )
    prediction, embeddings = prediction_from_temporal_seed(
        strict_model, validation_dataset, device, int(config["batch_size"])
    )
    threshold, metrics, threshold_table = calibrate_threshold(
        prediction, float(config["minimum_incorrect_recall"])
    )
    if not math.isclose(threshold, float(checkpoint["decision_threshold"]), abs_tol=1e-12):
        raise RuntimeError("Fold threshold changed after strict checkpoint reload.")
    prediction["predicted_correctness"] = (
        prediction["correct_probability"] >= threshold
    ).astype(int)
    prediction.to_csv(result_dir / "validation_predictions.csv", index=False)
    threshold_table.to_csv(result_dir / "threshold_table.csv", index=False)
    grouped_metrics(prediction, "camera_id").to_csv(
        result_dir / "per_camera_metrics.csv", index=False
    )
    grouped_metrics(prediction, "orientation_raw").to_csv(
        result_dir / "per_orientation_metrics.csv", index=False
    )
    interaction = prediction.assign(
        camera_orientation=prediction["camera_id"].astype(str)
        + "|"
        + prediction["orientation_raw"].astype(str)
    )
    grouped_metrics(interaction, "camera_orientation").to_csv(
        result_dir / "camera_orientation_metrics.csv", index=False
    )
    fused_metrics, fused = fusion_metrics(prediction, threshold)
    fused.to_csv(result_dir / "multi_view_mean_predictions.csv", index=False)
    fold_result = {
        "strategy_id": strategy.strategy_id,
        "held_subject": held_subject,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_subjects": sorted(set(train_rows["subject_id"]), key=int),
        "validation_class_counts": {
            str(key): int(value)
            for key, value in prediction["correctness"].value_counts().sort_index().items()
        },
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_completed": len(history),
        "decision_threshold": threshold,
        "metrics": metrics,
        "multi_view_mean_metrics": fused_metrics,
        "embedding_std": float(embeddings.std()),
        "training_seconds": perf_counter() - started,
        "checkpoint": str((checkpoint_dir / "best.pt").resolve()),
        "strict_reload": True,
        "motionbert_frozen": True,
        "shared_trunk_frozen": True,
    }
    (result_dir / "fold_result.json").write_text(
        json.dumps(fold_result, indent=2), encoding="utf-8"
    )
    return fold_result


def aggregate_strategy(
    strategy: V3Strategy, fold_results: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Aggregate subject-level metrics without weighting large subjects more."""

    rows = []
    for result in fold_results:
        metrics = result["metrics"]
        rows.append(
            {
                "strategy_id": strategy.strategy_id,
                "held_subject": result["held_subject"],
                "validation_samples": result["validation_samples"],
                "best_epoch": result["best_epoch"],
                "threshold": result["decision_threshold"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "incorrect_recall": metrics["classes"]["incorrect"]["recall"],
                "correct_recall": metrics["classes"]["correct"]["recall"],
                "roc_auc": metrics["roc_auc"],
                "multi_view_macro_f1": result["multi_view_mean_metrics"]["macro_f1"],
            }
        )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "incorrect_recall",
        "correct_recall",
        "roc_auc",
        "multi_view_macro_f1",
    ]
    aggregate: dict[str, Any] = {
        "strategy_id": strategy.strategy_id,
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "folds": len(frame),
        "median_threshold": float(frame["threshold"].median()),
        "median_best_epoch": int(round(float(frame["best_epoch"].median()))),
        "metrics": {},
    }
    for column in metric_columns:
        values = frame[column].dropna().astype(float)
        aggregate["metrics"][column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    aggregate["selection_constraint_satisfied"] = bool(
        aggregate["metrics"]["incorrect_recall"]["mean"] >= 0.60
    )
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate


def orientation_audit(development: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Quantify only the raw orientation/camera metadata present in REHAB24."""

    rows = (
        development.groupby(["orientation_raw", "camera_id", "correctness"])
        .size()
        .rename("samples")
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "orientation_camera_label_counts.csv", index=False)
    orientation_counts = development["orientation_raw"].value_counts()
    ratio = float(orientation_counts.min() / orientation_counts.max())
    severe = bool(ratio < 0.50)
    summary = {
        "orientations_raw": {
            str(key): int(value) for key, value in orientation_counts.items()
        },
        "cameras": {
            str(key): int(value)
            for key, value in development["camera_id"].value_counts().items()
        },
        "minimum_to_maximum_orientation_count_ratio": ratio,
        "severe_orientation_count_imbalance": severe,
        "orientation_aware_sampling_decision": (
            "eligible" if severe else "not_run_no_severe_count_imbalance"
        ),
        "note": "Raw dataset labels only; no view labels were inferred.",
    }
    (output_dir / "orientation_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _strategy_score(aggregate: dict[str, Any]) -> tuple[int, float, float, float]:
    metrics = aggregate["metrics"]
    mean_incorrect = float(metrics["incorrect_recall"]["mean"])
    return (
        int(mean_incorrect >= 0.60),
        float(metrics["macro_f1"]["mean"]),
        mean_incorrect,
        float(metrics["incorrect_recall"]["min"]),
    )


def _plot_strategy_comparison(rows: pd.DataFrame, destination: Path) -> None:
    metrics = ["mean_macro_f1", "mean_incorrect_recall", "min_incorrect_recall"]
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows)); width = 0.24
    for offset, column in enumerate(metrics):
        axis.bar(x + (offset - 1) * width, rows[column], width, label=column)
    axis.set_xticks(x, rows["strategy_id"], rotation=15, ha="right")
    axis.set_ylim(0.0, 1.0); axis.legend(); axis.set_title("Development LOSO comparison")
    figure.tight_layout(); figure.savefig(destination, dpi=160); plt.close(figure)


def _train_final_development_model(
    strategy: V3Strategy,
    development: pd.DataFrame,
    temporal_seed_cache: TemporalSeedCache,
    motionbert_checkpoint: Path,
    representation_checkpoint: Path,
    checkpoint_path: Path,
    device: torch.device,
    config: dict[str, Any],
    epochs: int,
    threshold: float,
    cv_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Fit all seven development subjects with CV-selected hyperparameters."""

    final_manifest = development.copy()
    final_manifest["split"] = "train"
    if set(final_manifest["subject_id"]) & HISTORICAL_TEST_SUBJECTS:
        raise RuntimeError("Historical Test subjects entered final development fit.")
    dataset = TemporalSeedDataset(final_manifest, temporal_seed_cache, "train")
    seed_everything(int(config["seed"]))
    model = SquatCorrectnessModel(motionbert_checkpoint).to(device)
    initialization = initialize_from_external_representation(
        model, representation_checkpoint
    )
    set_v3_trainable_scope(model)
    labels = final_manifest["correctness"].to_numpy(np.int64)
    loss_fn, loss_metadata = _loss(strategy, labels, device)
    loader = _loader(dataset, strategy, int(config["batch_size"]), int(config["seed"]))
    optimizer = torch.optim.AdamW(
        _model_parameters(model),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; gradients = []
        for features, masks, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            output = forward_from_temporal_seed(
                model, features.to(device), masks.to(device)
            )
            value = loss_fn(output["logits"], targets.to(device))
            if not torch.isfinite(value):
                raise FloatingPointError("Non-finite final development loss.")
            value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                _model_parameters(model), float(config["gradient_clip"])
            )
            optimizer.step(); losses.append(float(value.detach())); gradients.append(float(gradient))
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise RuntimeError("Frozen MotionBERT received gradients in final fit.")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "gradient_norm": float(np.mean(gradients)),
            }
        )
    metadata = {
        "training_stage": "development_final_model",
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "historical_test_subjects_excluded": sorted(HISTORICAL_TEST_SUBJECTS),
        "historical_test_subjects_re_evaluated": False,
        "strategy": strategy.__dict__,
        "decision_threshold": float(threshold),
        "threshold_source": "median_of_development_LOSO_fold_thresholds",
        "epoch_source": "median_of_selected_strategy_LOSO_best_epochs",
        "motionbert_frozen": True,
        "shared_trunk_frozen": True,
        "representation_initialization": initialization,
        "test_locked": True,
        "cv_aggregate": cv_aggregate,
        **loss_metadata,
    }
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epochs,
        config,
        cv_aggregate,
        metadata,
    )
    strict = SquatCorrectnessModel(motionbert_checkpoint).to(device)
    loaded = load_checkpoint_strict(checkpoint_path, strict, device)
    history_path = checkpoint_path.parent / "final_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        "name": "Squat Correctness V3 Development Final Model",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "strict_reload": True,
        "epochs": epochs,
        "decision_threshold": float(threshold),
        "strategy": strategy.__dict__,
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "historical_test_subjects_re_evaluated": False,
        "motionbert_frozen": bool(loaded["motionbert_frozen"]),
        "training_history": str(history_path.resolve()),
        "cv_performance_reference": cv_aggregate,
    }


def save_selected_oof_analysis(
    strategy: V3Strategy,
    result_root: Path,
) -> dict[str, Any]:
    """Combine each held subject's predictions exactly once for diagnostics."""

    frames = []
    for subject in DEVELOPMENT_SUBJECTS:
        path = (
            result_root
            / "folds"
            / strategy.strategy_id
            / f"subject_{subject}"
            / "validation_predictions.csv"
        )
        frame = pd.read_csv(path, dtype={"subject_id": str})
        if set(frame["subject_id"]) != {subject}:
            raise RuntimeError("OOF prediction file contains the wrong subject.")
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if len(oof) != 310 or oof["sample_id"].duplicated().any():
        raise RuntimeError("Selected OOF predictions must contain 310 unique camera samples.")
    oof.to_csv(result_root / "selected_oof_predictions.csv", index=False)
    per_subject = grouped_metrics(oof, "subject_id")
    per_camera = grouped_metrics(oof, "camera_id")
    per_orientation = grouped_metrics(oof, "orientation_raw")
    interaction = oof.assign(
        camera_orientation=oof["camera_id"].astype(str)
        + "|"
        + oof["orientation_raw"].astype(str)
    )
    per_interaction = grouped_metrics(interaction, "camera_orientation")
    per_subject.to_csv(result_root / "selected_oof_per_subject.csv", index=False)
    per_camera.to_csv(result_root / "selected_oof_per_camera.csv", index=False)
    per_orientation.to_csv(
        result_root / "selected_oof_per_orientation.csv", index=False
    )
    per_interaction.to_csv(
        result_root / "selected_oof_camera_orientation.csv", index=False
    )
    summary = {
        "samples": len(oof),
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "fold_specific_thresholds_used": True,
        "per_subject": per_subject.to_dict(orient="records"),
        "per_camera": per_camera.to_dict(orient="records"),
        "per_orientation_raw": per_orientation.to_dict(orient="records"),
        "camera_orientation_interaction": per_interaction.to_dict(orient="records"),
        "note": "Out-of-fold development diagnostics; not a Test benchmark.",
    }
    (result_root / "selected_oof_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_v3_loso(
    manifest: pd.DataFrame,
    feature_cache: FeatureCache,
    motionbert_checkpoint: Path,
    representation_checkpoint: Path,
    result_root: Path,
    checkpoint_root: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run A/B and conditionally C; select and fit a final dev-only model."""

    development = development_only_manifest(manifest)
    folds = build_loso_folds(development)
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    audit = orientation_audit(development, result_root)
    temporal_seed_cache = build_temporal_seed_cache(
        development,
        feature_cache,
        motionbert_checkpoint,
        representation_checkpoint,
        feature_cache.path.parent.parent / "squat_v3_temporal_seed",
        device,
        int(config["batch_size"]),
    )
    strategies: list[V3Strategy] = [STRATEGY_A, STRATEGY_B]
    aggregates: list[dict[str, Any]] = []
    strategy_results: dict[str, list[dict[str, Any]]] = {}

    def run_strategy(strategy: V3Strategy) -> dict[str, Any]:
        outputs = []
        for fold in folds:
            subject = str(fold.attrs["held_subject"])
            outputs.append(
                _fit_fold(
                    strategy,
                    fold,
                    temporal_seed_cache,
                    motionbert_checkpoint,
                    representation_checkpoint,
                    result_root / "folds" / strategy.strategy_id / f"subject_{subject}",
                    checkpoint_root / "folds" / strategy.strategy_id / f"subject_{subject}",
                    device,
                    config,
                )
            )
        strategy_results[strategy.strategy_id] = outputs
        return aggregate_strategy(
            strategy, outputs, result_root / "strategies" / strategy.strategy_id
        )

    aggregate_a = run_strategy(STRATEGY_A); aggregates.append(aggregate_a)
    strong_bias = bool(
        aggregate_a["metrics"]["incorrect_recall"]["mean"] < 0.60
        or aggregate_a["metrics"]["incorrect_recall"]["min"] < 0.40
    )
    aggregate_b = run_strategy(STRATEGY_B); aggregates.append(aggregate_b)
    if strong_bias:
        strategies.append(STRATEGY_C)
        aggregates.append(run_strategy(STRATEGY_C))

    comparison_rows = []
    for aggregate in aggregates:
        comparison_rows.append(
            {
                "strategy_id": aggregate["strategy_id"],
                "mean_macro_f1": aggregate["metrics"]["macro_f1"]["mean"],
                "std_macro_f1": aggregate["metrics"]["macro_f1"]["std"],
                "mean_incorrect_recall": aggregate["metrics"]["incorrect_recall"]["mean"],
                "min_incorrect_recall": aggregate["metrics"]["incorrect_recall"]["min"],
                "mean_correct_recall": aggregate["metrics"]["correct_recall"]["mean"],
                "constraint_satisfied": aggregate["selection_constraint_satisfied"],
                "median_threshold": aggregate["median_threshold"],
                "median_best_epoch": aggregate["median_best_epoch"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(result_root / "strategy_comparison.csv", index=False)
    _plot_strategy_comparison(comparison, result_root / "strategy_comparison.png")
    selected_aggregate = max(aggregates, key=_strategy_score)
    selected_strategy = next(
        strategy
        for strategy in strategies
        if strategy.strategy_id == selected_aggregate["strategy_id"]
    )
    final_config = {
        "selected_strategy": selected_strategy.__dict__,
        "selection_objective": (
            "mean subject-level CV Macro F1 with mean incorrect recall >= 0.60; "
            "minimum subject incorrect recall monitored"
        ),
        "selected_cv_aggregate": selected_aggregate,
        "fold_thresholds": [
            result["decision_threshold"]
            for result in strategy_results[selected_strategy.strategy_id]
        ],
        "final_threshold": selected_aggregate["median_threshold"],
        "final_epochs": selected_aggregate["median_best_epoch"],
        "focal_run_reason": (
            "V3-A showed strong incorrect-class bias"
            if strong_bias
            else "not_run_V3-A_bias_trigger_not_met"
        ),
        "orientation_aware_sampling": audit[
            "orientation_aware_sampling_decision"
        ],
        "historical_test_subjects_re_evaluated": False,
    }
    (result_root / "selected_config.json").write_text(
        json.dumps(final_config, indent=2), encoding="utf-8"
    )
    oof_analysis = save_selected_oof_analysis(selected_strategy, result_root)
    final_summary = _train_final_development_model(
        selected_strategy,
        development,
        temporal_seed_cache,
        motionbert_checkpoint,
        representation_checkpoint,
        checkpoint_root / "final_dev.pt",
        device,
        config,
        max(1, int(selected_aggregate["median_best_epoch"])),
        float(selected_aggregate["median_threshold"]),
        selected_aggregate,
    )
    (result_root / "development_final_model_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    result = {
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "historical_test_subjects": sorted(HISTORICAL_TEST_SUBJECTS),
        "historical_test_subjects_re_evaluated": False,
        "fold_count": len(folds),
        "strategies_run": [item["strategy_id"] for item in aggregates],
        "selected": final_config,
        "selected_oof_analysis": oof_analysis,
        "final_development_model": final_summary,
    }
    (result_root / "v3_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
