"""Development and one-time benchmark utilities for static Squat Error V1."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, TensorDataset

from models.squat_posture_error import (
    ERROR_CLASSES,
    SquatPostureErrorModel,
    load_squat_posture_error_checkpoint,
)


CLASS_TO_INDEX = {name: index for index, name in enumerate(ERROR_CLASSES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_development_split(manifest: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratify only the provided Train folder; official Test remains locked."""

    frame = manifest.copy()
    frame["development_split"] = "test_locked"
    train_rows = frame.index[frame["source_split"] == "train"].to_numpy()
    labels = frame.loc[train_rows, "label_index"].to_numpy(np.int64)
    train_indices, validation_indices = train_test_split(
        train_rows,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    frame.loc[train_indices, "development_split"] = "train"
    frame.loc[validation_indices, "development_split"] = "validation"
    if set(frame.loc[frame["source_split"] == "test", "development_split"]) != {"test_locked"}:
        raise RuntimeError("Official Test entered development selection.")
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Train/validation overlap detected.")
    return frame


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], probabilities: np.ndarray
) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.int64); prediction = np.asarray(predictions, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=[0, 1, 2], zero_division=0
    )
    matrix = confusion_matrix(target, prediction, labels=[0, 1, 2])
    error_mask = target != CLASS_TO_INDEX["good"]
    false_good = int(np.sum(error_mask & (prediction == CLASS_TO_INDEX["good"])))
    try:
        binary = label_binarize(target, classes=[0, 1, 2])
        per_class_auc = roc_auc_score(binary, probability, average=None)
        macro_auc = float(roc_auc_score(binary, probability, average="macro"))
    except ValueError:
        per_class_auc = np.asarray([np.nan, np.nan, np.nan]); macro_auc = None
    classes = {}
    for index, name in enumerate(ERROR_CLASSES):
        classes[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "roc_auc_ovr": None if not np.isfinite(per_class_auc[index]) else float(per_class_auc[index]),
        }
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_roc_auc_ovr": macro_auc,
        "false_good_count": false_good,
        "actual_error_count": int(error_mask.sum()),
        "false_good_rate": false_good / max(int(error_mask.sum()), 1),
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["macro_f1"]),
        min(float(metrics["classes"]["bad_back"]["recall"]), float(metrics["classes"]["bad_heel"]["recall"])),
        -float(metrics["false_good_rate"]),
        float(metrics["balanced_accuracy"]),
    )


def _prediction_frame(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    output = rows.copy().reset_index(drop=True)
    output["evaluation_split"] = split_name
    output["predicted_index"] = predictions.astype(int)
    output["predicted_error"] = [ERROR_CLASSES[int(value)] for value in predictions]
    for index, name in enumerate(ERROR_CLASSES):
        output[f"probability_{name}"] = probabilities[:, index]
    output["score"] = None
    return output


@dataclass
class MLPDevelopmentResult:
    model: SquatPostureErrorModel
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    history: list[dict[str, Any]]
    best_epoch: int
    hidden_dim: int
    dropout: float
    learning_rate: float


def train_mlp_development(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    validation_rows: pd.DataFrame,
    device: torch.device,
    seed: int,
) -> MLPDevelopmentResult:
    """Train one compact residual head with validation-only early stopping."""

    seed_everything(seed)
    mean = torch.from_numpy(train_features.mean(axis=0).astype(np.float32))
    scale = torch.from_numpy(train_features.std(axis=0).astype(np.float32)).clamp_min(1e-6)
    hidden_dim = 128; dropout = 0.20; learning_rate = 1e-3
    model = SquatPostureErrorModel(train_features.shape[1], mean, scale, hidden_dim, dropout).to(device)
    counts = np.bincount(train_targets, minlength=3)
    weights = len(train_targets) / (3.0 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    dataset = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_targets))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x_val = torch.from_numpy(validation_features).to(device)
    best_state: dict[str, torch.Tensor] | None = None; best_metrics: dict[str, Any] | None = None
    best_epoch = 0; stale = 0; history = []
    for epoch in range(1, 101):
        model.train(); losses = []
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Squat Error MLP loss.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(x_val), dim=-1).cpu().numpy()
        prediction = probability.argmax(axis=1)
        metrics = classification_metrics(validation_targets, prediction, probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": metrics["macro_f1"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_bad_back_recall": metrics["classes"]["bad_back"]["recall"],
                "validation_bad_heel_recall": metrics["classes"]["bad_heel"]["recall"],
                "validation_false_good_rate": metrics["false_good_rate"],
            }
        )
        if best_metrics is None or selection_key(metrics) > selection_key(best_metrics):
            best_metrics = metrics; best_epoch = epoch; stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 15:
                break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state, strict=True); model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(x_val), dim=-1).cpu().numpy()
    prediction = probability.argmax(axis=1)
    return MLPDevelopmentResult(
        model=model,
        metrics=classification_metrics(validation_targets, prediction, probability),
        predictions=_prediction_frame(validation_rows, prediction, probability, "validation"),
        history=history,
        best_epoch=best_epoch,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
    )


def fit_final_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
) -> tuple[SquatPostureErrorModel, list[dict[str, Any]]]:
    """Fit the frozen selected architecture on all provided Train images."""

    seed_everything(seed)
    mean = torch.from_numpy(features.mean(axis=0).astype(np.float32))
    scale = torch.from_numpy(features.std(axis=0).astype(np.float32)).clamp_min(1e-6)
    model = SquatPostureErrorModel(features.shape[1], mean, scale, hidden_dim, dropout).to(device)
    counts = np.bincount(targets, minlength=3); weights = len(targets) / (3.0 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(targets)),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(batch.to(device)), target.to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return model, history


def _evaluate_neural(
    model: SquatPostureErrorModel,
    features: np.ndarray,
    targets: np.ndarray,
    rows: pd.DataFrame,
    split_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(features).to(device)), dim=-1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    return classification_metrics(targets, predictions, probabilities), _prediction_frame(rows, predictions, probabilities, split_name)


def _detected_only_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    rows = predictions[predictions["pose_success"].astype(bool)]
    probabilities = rows[[f"probability_{name}" for name in ERROR_CLASSES]].to_numpy()
    return classification_metrics(rows["label_index"], rows["predicted_index"], probabilities)


def _save_confusion(metrics: dict[str, Any], output_dir: Path, prefix: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    pd.DataFrame(matrix, index=ERROR_CLASSES, columns=ERROR_CLASSES).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(3), ERROR_CLASSES, rotation=20); axis.set_yticks(range(3), ERROR_CLASSES)
    axis.set(xlabel="Predicted", ylabel="Actual", title=f"{prefix.title()} confusion matrix")
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160); plt.close(figure)


def run_experiment(
    data_dir: Path,
    result_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    *,
    seed: int = 42,
    development_only: bool = False,
) -> dict[str, Any]:
    """Select on Validation only, then optionally open official Test once."""

    manifest = pd.read_csv(data_dir / "feature_manifest.csv")
    features = np.load(data_dir / "pose_features.npy", mmap_mode="r")
    feature_metadata = json.loads((data_dir / "feature_names.json").read_text(encoding="utf-8"))
    if features.shape != (len(manifest), int(feature_metadata["feature_dim"])) or not np.isfinite(features).all():
        raise ValueError("Invalid Squat Error feature cache.")
    split = make_development_split(manifest, seed)
    result_root.mkdir(parents=True, exist_ok=True); (result_root / "data").mkdir(exist_ok=True)
    split.to_csv(result_root / "data" / "development_split_manifest.csv", index=False)
    train_rows = split[split["development_split"] == "train"]
    validation_rows = split[split["development_split"] == "validation"]
    train_indices = train_rows["feature_index"].to_numpy(np.int64)
    validation_indices = validation_rows["feature_index"].to_numpy(np.int64)
    x_train = np.asarray(features[train_indices], dtype=np.float32); y_train = train_rows["label_index"].to_numpy(np.int64)
    x_validation = np.asarray(features[validation_indices], dtype=np.float32); y_validation = validation_rows["label_index"].to_numpy(np.int64)

    experiments = []; prediction_cache: dict[str, pd.DataFrame] = {}; metric_cache: dict[str, dict[str, Any]] = {}
    logistic = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed))]
    )
    started = perf_counter(); logistic.fit(x_train, y_train)
    probability = logistic.predict_proba(x_validation); prediction = probability.argmax(axis=1)
    metric_cache["logistic_regression"] = classification_metrics(y_validation, prediction, probability)
    prediction_cache["logistic_regression"] = _prediction_frame(validation_rows, prediction, probability, "validation")
    experiments.append({"model": "logistic_regression", "training_seconds": perf_counter() - started, **_flat_metrics(metric_cache["logistic_regression"])})

    forest = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
    )
    started = perf_counter(); forest.fit(x_train, y_train)
    probability = forest.predict_proba(x_validation); prediction = probability.argmax(axis=1)
    metric_cache["random_forest"] = classification_metrics(y_validation, prediction, probability)
    prediction_cache["random_forest"] = _prediction_frame(validation_rows, prediction, probability, "validation")
    experiments.append({"model": "random_forest", "training_seconds": perf_counter() - started, **_flat_metrics(metric_cache["random_forest"])})

    started = perf_counter()
    mlp = train_mlp_development(x_train, y_train, x_validation, y_validation, validation_rows, device, seed)
    metric_cache["small_mlp"] = mlp.metrics; prediction_cache["small_mlp"] = mlp.predictions
    experiments.append({"model": "small_mlp", "training_seconds": perf_counter() - started, "best_epoch": mlp.best_epoch, **_flat_metrics(mlp.metrics)})
    baseline_dir = result_root / "baselines"; baseline_dir.mkdir(exist_ok=True)
    experiment_frame = pd.DataFrame(experiments); experiment_frame.to_csv(baseline_dir / "experiment_results.csv", index=False)
    # The requested modular neural head is the deterministic tie-breaker only.
    # Validation metrics remain the complete primary/secondary selection criteria.
    model_tie_breaker = {
        "logistic_regression": 0,
        "random_forest": 1,
        "small_mlp": 2,
    }
    selected = max(
        metric_cache,
        key=lambda name: (*selection_key(metric_cache[name]), model_tie_breaker[name]),
    )
    selection = {
        "primary": "validation_macro_f1",
        "monitored": ["bad_back_recall", "bad_heel_recall", "false_good_rate"],
        "tie_breaker": "prefer_modular_small_mlp_only_when_all_monitored_validation_metrics_tie",
        "official_test_used": False,
        "selected_model": selected,
        "validation_metrics": metric_cache[selected],
        "all_models": experiments,
    }
    (baseline_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if development_only:
        return selection
    if selected != "small_mlp":
        raise RuntimeError(
            f"Validation selected {selected}; official Test remains locked until a safe modular checkpoint is implemented."
        )

    # Architecture and epoch count are now frozen; fit all provided Train images.
    all_train_rows = split[split["source_split"] == "train"]
    all_train_indices = all_train_rows["feature_index"].to_numpy(np.int64)
    final_model, final_history = fit_final_mlp(
        np.asarray(features[all_train_indices], dtype=np.float32),
        all_train_rows["label_index"].to_numpy(np.int64),
        mlp.best_epoch, device, seed, mlp.hidden_dim, mlp.dropout, mlp.learning_rate,
    )
    selected_dir = result_root / "selected_model"; selected_dir.mkdir(exist_ok=True)
    metadata = {
        "model_type": "small_mlp",
        "model_state_dict": final_model.state_dict(),
        "input_dim": int(features.shape[1]),
        "hidden_dim": mlp.hidden_dim,
        "dropout": mlp.dropout,
        "class_vocabulary": {str(index): name for index, name in enumerate(ERROR_CLASSES)},
        "feature_version": feature_metadata["feature_version"],
        "feature_names": feature_metadata["feature_names"],
        "training_stage": "provided_dataset_development_error_model",
        "training_epochs": mlp.best_epoch,
        "selection_split": "stratified validation from provided Train; not subject-wise",
        "test_used_for_selection": False,
        "subject_generalization_claim": False,
        "score": None,
        "seed": seed,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True); torch.save(metadata, checkpoint_path)
    strict_model, strict_checkpoint = load_squat_posture_error_checkpoint(checkpoint_path, device)
    pd.DataFrame(final_history).to_csv(selected_dir / "training_history.csv", index=False)
    config = {key: value for key, value in metadata.items() if key != "model_state_dict"}
    config["checkpoint"] = str(checkpoint_path.resolve()); config["checkpoint_sha256"] = sha256(checkpoint_path)
    config["trainable_parameters"] = sum(parameter.numel() for parameter in strict_model.parameters() if parameter.requires_grad)
    (selected_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (selected_dir / "validation_metrics.json").write_text(json.dumps(mlp.metrics, indent=2), encoding="utf-8")
    mlp.predictions.to_csv(selected_dir / "validation_predictions.csv", index=False)

    marker_path = result_root / "official_test_opened_once.json"
    if marker_path.exists():
        raise RuntimeError("Official SquatDataset Test was already evaluated; refusing to open it again.")
    test_rows = split[split["source_split"] == "test"]
    test_indices = test_rows["feature_index"].to_numpy(np.int64)
    test_features = np.asarray(features[test_indices], dtype=np.float32)
    test_targets = test_rows["label_index"].to_numpy(np.int64)
    test_metrics, test_predictions = _evaluate_neural(
        strict_model, test_features, test_targets, test_rows, "official_test", device
    )
    test_metrics["pose_success_only"] = _detected_only_metrics(test_predictions)
    (selected_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    combined_predictions = pd.concat([mlp.predictions, test_predictions], ignore_index=True)
    combined_predictions.to_csv(selected_dir / "predictions.csv", index=False)
    per_class_rows = []
    for split_name, metrics in (("validation", mlp.metrics), ("official_test", test_metrics)):
        for name, values in metrics["classes"].items():
            per_class_rows.append({"split": split_name, "class": name, **values})
    pd.DataFrame(per_class_rows).to_csv(selected_dir / "per_class_metrics.csv", index=False)
    _save_confusion(mlp.metrics, selected_dir, "validation")
    _save_confusion(test_metrics, selected_dir, "test")

    analysis_dir = result_root / "analysis"; analysis_dir.mkdir(exist_ok=True)
    matrix = np.asarray(test_metrics["confusion_matrix"])
    false_rows = []
    for actual in ("bad_back", "bad_heel"):
        actual_index = CLASS_TO_INDEX[actual]; count = int(matrix[actual_index, 0]); support = int(matrix[actual_index].sum())
        false_rows.append({"actual_class": actual, "predicted_class": "good", "count": count, "support": support, "false_good_rate": count / max(support, 1)})
    false_rows.append({"actual_class": "all_errors", "predicted_class": "good", "count": test_metrics["false_good_count"], "support": test_metrics["actual_error_count"], "false_good_rate": test_metrics["false_good_rate"]})
    pd.DataFrame(false_rows).to_csv(analysis_dir / "false_good_analysis.csv", index=False)
    failures = test_predictions[test_predictions["label_index"] != test_predictions["predicted_index"]].copy()
    failures["actual_error"] = failures["canonical_label"]
    failures.to_csv(analysis_dir / "failure_cases.csv", index=False)
    marker = {
        "opened_once": True,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(test_rows),
        "selected_model": selected,
        "checkpoint_sha256": sha256(checkpoint_path),
        "test_used_for_selection": False,
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return {
        "selected_model": selected,
        "validation_metrics": mlp.metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "strict_reload": strict_checkpoint["model_type"] == "small_mlp",
        "test_opened_once": True,
    }


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "validation_balanced_accuracy": metrics["balanced_accuracy"],
        "validation_good_recall": metrics["classes"]["good"]["recall"],
        "validation_bad_back_recall": metrics["classes"]["bad_back"]["recall"],
        "validation_bad_heel_recall": metrics["classes"]["bad_heel"]["recall"],
        "validation_false_good_rate": metrics["false_good_rate"],
        "validation_macro_roc_auc_ovr": metrics["macro_roc_auc_ovr"],
    }
