from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from datasets.adapters.physical_exercise_recognition_adapter import (
    PhysicalExerciseRecognitionAdapter,
)
from models.exercise_representation import ExerciseRepresentationModel


@dataclass(frozen=True)
class WindowRecord:
    cache_path: Path
    window_index: int
    label_index: int
    video_id: str


class PilotWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, int, torch.Tensor]]):
    """Full split window dataset with video identity and padding validity masks."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        self.data_dir = Path(data_dir).resolve()
        frame = pd.read_csv(
            self.data_dir / "cache_manifest.csv", dtype={"video_id": str}
        )
        frame = frame[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No windows for split {split!r}.")
        self.records: list[WindowRecord] = []
        self.video_to_indices: dict[str, list[int]] = defaultdict(list)
        self.video_labels: dict[str, int] = {}
        for row in frame.sort_values(["video_id"]).itertuples(index=False):
            video_id = str(row.video_id)
            label_index = int(row.label_index)
            existing = self.video_labels.setdefault(video_id, label_index)
            if existing != label_index:
                raise ValueError(f"Conflicting labels for video {video_id}.")
            for window_index in range(int(row.num_windows)):
                record_index = len(self.records)
                self.records.append(
                    WindowRecord(Path(row.cache_path), window_index, label_index, video_id)
                )
                self.video_to_indices[video_id].append(record_index)
        self.video_ids = tuple(sorted(self.video_to_indices, key=lambda value: int(value)))

    def __len__(self) -> int:
        return len(self.records)

    @lru_cache(maxsize=4)
    def _load_video(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            motion = np.asarray(archive["motionbert_input"], dtype=np.float32)
            if "padding_mask" not in archive:
                raise ValueError(f"Pilot cache lacks padding_mask: {path}")
            padding = np.asarray(archive["padding_mask"], dtype=bool)
        return motion, padding

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, int, torch.Tensor]:
        record = self.records[index]
        motion, padding = self._load_video(record.cache_path)
        window = motion[record.window_index]
        valid_mask = ~padding[record.window_index]
        if window.shape != (30, 17, 3) or valid_mask.shape != (30,):
            raise ValueError(f"Invalid cached pilot contract in {record.cache_path}")
        if not np.isfinite(window).all():
            raise ValueError(f"Non-finite cached values in {record.cache_path}")
        return (
            torch.from_numpy(window.copy()),
            torch.tensor(record.label_index, dtype=torch.long),
            record.video_id,
            record.window_index,
            torch.from_numpy(valid_mask.copy()),
        )


class FrozenFeatureDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str, int, torch.Tensor]]
):
    """In-memory float16 MotionBERT features converted to float32 per batch."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Sequence[int],
        video_ids: Sequence[str],
        window_indices: Sequence[int],
        valid_masks: np.ndarray,
    ) -> None:
        if features.ndim != 4 or features.shape[1:] != (30, 17, 512):
            raise ValueError(f"Expected (N,30,17,512), got {features.shape}.")
        if len({len(features), len(labels), len(video_ids), len(window_indices), len(valid_masks)}) != 1:
            raise ValueError("Frozen feature metadata lengths do not match.")
        self.features = features
        self.labels = np.asarray(labels, dtype=np.int64)
        self.video_ids = tuple(str(value) for value in video_ids)
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.valid_masks = np.asarray(valid_masks, dtype=bool)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, int, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index], dtype=torch.long),
            self.video_ids[index],
            int(self.window_indices[index]),
            torch.from_numpy(self.valid_masks[index]),
        )


def representative_window_indices(
    dataset: PilotWindowDataset, windows_per_video: int
) -> list[int]:
    """Select evenly spaced deterministic windows with an equal per-video cap."""

    selected: list[int] = []
    for video_id in dataset.video_ids:
        candidates = dataset.video_to_indices[video_id]
        if len(candidates) >= windows_per_video:
            positions = np.linspace(0, len(candidates) - 1, windows_per_video, dtype=int)
            selected.extend(candidates[int(position)] for position in positions)
        else:
            selected.extend(
                candidates[index % len(candidates)] for index in range(windows_per_video)
            )
    return selected


@torch.no_grad()
def precompute_frozen_features(
    backbone: nn.Module,
    source: PilotWindowDataset,
    record_indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    split_name: str,
) -> FrozenFeatureDataset:
    """Evaluate frozen MotionBERT once and retain compact float16 features."""

    subset = torch.utils.data.Subset(source, list(record_indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = np.empty((len(record_indices), 30, 17, 512), dtype=np.float16)
    labels: list[int] = []
    video_ids: list[str] = []
    window_indices: list[int] = []
    valid_masks: list[np.ndarray] = []
    offset = 0
    backbone.eval()
    for batch_index, (inputs, batch_labels, batch_video_ids, batch_windows, masks) in enumerate(loader, start=1):
        representation = backbone(inputs.to(device=device, dtype=torch.float32))
        batch_features = representation.detach().cpu().numpy().astype(np.float16)
        stop = offset + len(batch_features)
        features[offset:stop] = batch_features
        offset = stop
        labels.extend(int(value) for value in batch_labels.tolist())
        video_ids.extend(str(value) for value in batch_video_ids)
        window_indices.extend(int(value) for value in batch_windows.tolist())
        valid_masks.extend(masks.numpy().astype(bool))
        if batch_index % 20 == 0 or stop == len(record_indices):
            print(
                f"precompute split={split_name} windows={stop}/{len(record_indices)}",
                flush=True,
            )
    return FrozenFeatureDataset(
        features,
        labels,
        video_ids,
        window_indices,
        np.stack(valid_masks),
    )


class VideoBalancedSampler(Sampler[int]):
    """Give every video the same capped contribution and rotate windows by epoch."""

    def __init__(
        self,
        dataset: PilotWindowDataset,
        windows_per_video: int = 4,
        seed: int = 42,
    ) -> None:
        if windows_per_video <= 0:
            raise ValueError("windows_per_video must be positive.")
        self.dataset = dataset
        self.windows_per_video = int(windows_per_video)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        videos = list(self.dataset.video_ids)
        rng.shuffle(videos)
        selected: list[int] = []
        for video_id in videos:
            candidates = list(self.dataset.video_to_indices[video_id])
            rng.shuffle(candidates)
            if len(candidates) >= self.windows_per_video:
                chosen = candidates[: self.windows_per_video]
            else:
                chosen = [candidates[index % len(candidates)] for index in range(self.windows_per_video)]
            selected.extend(chosen)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.dataset.video_ids) * self.windows_per_video


class PilotSplitGuard:
    """Enforce that test evaluation cannot occur before model selection ends."""

    def __init__(self) -> None:
        self.training_complete = False

    def mark_training_complete(self) -> None:
        self.training_complete = True

    def assert_evaluation_allowed(self, split: str) -> None:
        if split == "test" and not self.training_complete:
            raise RuntimeError("Test split evaluation is prohibited during pilot training/tuning.")


def _confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def _classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], classes: Sequence[str]
) -> dict[str, Any]:
    matrix = _confusion_matrix(targets, predictions, len(classes))
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(classes):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index].sum() - true_positive)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_values.append(f1)
        per_class[name] = {
            "support": int(matrix[index].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def aggregate_video_logits(
    logits: np.ndarray,
    labels: Sequence[int],
    video_ids: Sequence[str],
) -> tuple[np.ndarray, list[int], list[str]]:
    """Aggregate window logits with arithmetic mean for each ground-truth video."""

    grouped_logits: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for logit, label, video_id in zip(logits, labels, video_ids):
        existing = grouped_labels.setdefault(str(video_id), int(label))
        if existing != int(label):
            raise ValueError(f"Video {video_id} has inconsistent labels.")
        grouped_logits[str(video_id)].append(logit)
    ordered_ids = sorted(grouped_logits, key=lambda value: int(value))
    means = np.stack(
        [np.mean(grouped_logits[video_id], axis=0) for video_id in ordered_ids]
    )
    return means, [grouped_labels[video_id] for video_id in ordered_ids], ordered_ids


def _diagnostics(
    embeddings: np.ndarray, logits: np.ndarray, predictions: Sequence[int], classes: Sequence[str]
) -> dict[str, Any]:
    counts = Counter(classes[int(index)] for index in predictions)
    return {
        "embedding_mean": float(embeddings.mean()),
        "embedding_std": float(embeddings.std()),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "prediction_class_distribution": {name: int(counts.get(name, 0)) for name in classes},
    }


def _run_train_epoch(
    model: ExerciseRepresentationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, Any]:
    model.train()
    losses: list[float] = []
    gradients: list[float] = []
    embeddings: list[np.ndarray] = []
    logits_values: list[np.ndarray] = []
    predictions: list[int] = []
    targets: list[int] = []
    for inputs, labels, _, _, valid_mask in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        if inputs.shape[-1] == 512:
            outputs = model.forward_features(
                inputs.to(device=device, dtype=torch.float32), temporal_mask=valid_mask
            )
        else:
            outputs = model(inputs.to(device), temporal_mask=valid_mask)
        loss = criterion(outputs["logits"], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite pilot loss: {loss.item()}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=gradient_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradient norm.")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradients.append(float(gradient_norm.detach().cpu()))
        batch_logits = outputs["logits"].detach().cpu().numpy()
        logits_values.append(batch_logits)
        embeddings.append(outputs["global_embedding"].detach().cpu().numpy())
        predictions.extend(batch_logits.argmax(axis=1).tolist())
        targets.extend(labels.detach().cpu().tolist())
    logits = np.concatenate(logits_values)
    embedding_array = np.concatenate(embeddings)
    result = {
        "loss": float(np.mean(losses)),
        "gradient_norm_mean": float(np.mean(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
        "motionbert_all_gradients_none": all(
            parameter.grad is None for parameter in model.backbone.parameters()
        ),
    }
    return result | {
        "embedding_mean": float(embedding_array.mean()),
        "embedding_std": float(embedding_array.std()),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "prediction_indices": predictions,
        "target_indices": targets,
    }


@torch.no_grad()
def _evaluate(
    model: ExerciseRepresentationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    logits_values: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    labels_all: list[int] = []
    video_ids_all: list[str] = []
    window_indices_all: list[int] = []
    for inputs, labels, video_ids, window_indices, valid_mask in loader:
        mask = valid_mask.to(device=device, dtype=torch.bool)
        if inputs.shape[-1] == 512:
            outputs = model.forward_features(
                inputs.to(device=device, dtype=torch.float32), temporal_mask=mask
            )
        else:
            outputs = model(inputs.to(device), temporal_mask=mask)
        loss = criterion(outputs["logits"], labels.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss.")
        losses.append(float(loss.cpu()))
        logits_values.append(outputs["logits"].cpu().numpy())
        embeddings.append(outputs["global_embedding"].cpu().numpy())
        labels_all.extend(labels.tolist())
        video_ids_all.extend(str(value) for value in video_ids)
        window_indices_all.extend(int(value) for value in window_indices.tolist())
    logits = np.concatenate(logits_values)
    embedding_array = np.concatenate(embeddings)
    window_predictions = logits.argmax(axis=1).tolist()
    window_metrics = _classification_metrics(labels_all, window_predictions, classes)
    video_logits, video_labels, ordered_video_ids = aggregate_video_logits(
        logits, labels_all, video_ids_all
    )
    video_predictions = video_logits.argmax(axis=1).tolist()
    video_metrics = _classification_metrics(video_labels, video_predictions, classes)
    diagnostics = _diagnostics(embedding_array, logits, window_predictions, classes)
    return {
        "loss": float(np.mean(losses)),
        "window_metrics": window_metrics,
        "video_metrics": video_metrics,
        "diagnostics": diagnostics,
        "logits": logits,
        "labels": labels_all,
        "predictions": window_predictions,
        "video_ids": video_ids_all,
        "window_indices": window_indices_all,
        "video_logits": video_logits,
        "video_labels": video_labels,
        "video_predictions": video_predictions,
        "ordered_video_ids": ordered_video_ids,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_confusion(path: Path, metrics: dict[str, Any], classes: Sequence[str]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *classes])
        for name, row in zip(classes, matrix):
            writer.writerow([name, *row.tolist()])


def _write_predictions(
    path: Path,
    *,
    logits: np.ndarray,
    labels: Sequence[int],
    predictions: Sequence[int],
    video_ids: Sequence[str],
    classes: Sequence[str],
    window_indices: Sequence[int] | None = None,
) -> None:
    fields = ["video_id"]
    if window_indices is not None:
        fields.append("window_index")
    fields.extend(["actual_index", "actual_label", "predicted_index", "predicted_label"])
    fields.extend(f"logit_{name}" for name in classes)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, (logit, label, prediction, video_id) in enumerate(
            zip(logits, labels, predictions, video_ids)
        ):
            row: dict[str, Any] = {
                "video_id": video_id,
                "actual_index": int(label),
                "actual_label": classes[int(label)],
                "predicted_index": int(prediction),
                "predicted_label": classes[int(prediction)],
            }
            if window_indices is not None:
                row["window_index"] = int(window_indices[row_index])
            row.update({f"logit_{name}": float(logit[index]) for index, name in enumerate(classes)})
            writer.writerow(row)


def _checkpoint_payload(
    model: ExerciseRepresentationModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    epochs_requested: int,
    classes: Sequence[str],
    best_epoch: int,
    best_video_macro_f1: float,
    best_window_macro_f1: float,
    cache_statistics: dict[str, Any],
    sampling_strategy: str,
    split_video_ids: dict[str, list[str]],
    motionbert_hash: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "training_stage": "pilot",
        "epochs_requested": epochs_requested,
        "best_epoch": best_epoch,
        "best_video_macro_f1": best_video_macro_f1,
        "best_window_macro_f1": best_window_macro_f1,
        "class_vocabulary": list(classes),
        "source_dataset": "physical_exercise_recognition",
        "motionbert_sha256": motionbert_hash,
        "preprocessing_version": cache_statistics["preprocessing_version"],
        "h36m_mapping_version": cache_statistics["h36m_mapping_version"],
        "window_size": cache_statistics["window_size"],
        "stride": cache_statistics["stride"],
        "seed": seed,
        "class_weights": torch.ones(len(classes), dtype=torch.float32),
        "sampling_strategy": sampling_strategy,
        "video_aggregation_method": "mean_logits",
        "train_video_ids": split_video_ids["train"],
        "validation_video_ids": split_video_ids["validation"],
        "test_video_ids": split_video_ids["test"],
    }


def run_pilot_training(args: Any, project_root: Path) -> None:
    """Run the approved eight-epoch full-video pilot and one post-selection test."""

    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    checkpoint_path = (
        args.motionbert_checkpoint
        if args.motionbert_checkpoint.is_absolute()
        else project_root / args.motionbert_checkpoint
    )
    if (output_dir / "smoke_training_result.json").exists():
        raise RuntimeError("Pilot output must not overwrite the smoke checkpoint directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_statistics = json.loads(
        (data_dir / "cache_statistics.json").read_text(encoding="utf-8")
    )
    expected_version = PhysicalExerciseRecognitionAdapter.PREPROCESSING_VERSION
    if cache_statistics.get("preprocessing_version") != expected_version:
        raise RuntimeError(
            "Preprocessing version mismatch: "
            f"cache={cache_statistics.get('preprocessing_version')!r}, expected={expected_version!r}."
        )
    sanity_path = project_root / "results" / "external_skeleton_sanity" / "numeric_diagnostics.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    if sanity.get("preprocessing_version") != expected_version or not sanity.get("sanity_pass"):
        raise RuntimeError("Coordinate sanity report is missing, failed, or belongs to another cache version.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    vocabulary = json.loads((data_dir / "class_vocabulary.json").read_text(encoding="utf-8"))
    classes = [str(value) for value in vocabulary["classes"]]
    manifest = pd.read_csv(data_dir / "cache_manifest.csv", dtype={"video_id": str})
    split_video_ids = {
        split: sorted(manifest.loc[manifest["split"] == split, "video_id"].tolist(), key=int)
        for split in ("train", "validation", "test")
    }
    if any(
        set(split_video_ids[left]) & set(split_video_ids[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("Video leakage detected before pilot training.")

    train_source_dataset = PilotWindowDataset(data_dir, "train")
    validation_source_dataset = PilotWindowDataset(data_dir, "validation")
    split_guard = PilotSplitGuard()
    model = ExerciseRepresentationModel(
        len(classes), motionbert_checkpoint=checkpoint_path, freeze_backbone=True
    ).to(device)
    train_record_indices = representative_window_indices(
        train_source_dataset, args.windows_per_video
    )
    train_dataset = precompute_frozen_features(
        model.backbone, train_source_dataset, train_record_indices,
        device, args.batch_size, "train",
    )
    validation_dataset = precompute_frozen_features(
        model.backbone, validation_source_dataset,
        list(range(len(validation_source_dataset))),
        device, args.batch_size, "validation",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    sampling_strategy = (
        f"video_balanced_fixed_cap_{args.windows_per_video}_representative_windows_per_video;"
        "frozen_motionbert_features_precomputed_float16_in_memory;no_class_reweighting"
    )
    windows_per_video = np.asarray(
        [len(indices) for indices in train_source_dataset.video_to_indices.values()], dtype=np.int64
    )
    sampling_report = {
        "strategy": sampling_strategy,
        "full_train_videos": len(train_source_dataset.video_ids),
        "available_windows": len(train_source_dataset),
        "sampled_windows_per_epoch": len(train_dataset),
        "windows_per_video_available": {
            "min": int(windows_per_video.min()),
            "p50": float(np.percentile(windows_per_video, 50)),
            "p90": float(np.percentile(windows_per_video, 90)),
            "p95": float(np.percentile(windows_per_video, 95)),
            "max": int(windows_per_video.max()),
            "mean": float(windows_per_video.mean()),
        },
        "maximum_single_video_contribution_percentage": 100.0 / len(train_source_dataset.video_ids),
        "sampled_windows_per_class_per_epoch": {
            classes[label]: args.windows_per_video * sum(
                value == label for value in train_source_dataset.video_labels.values()
            )
            for label in range(len(classes))
        },
    }

    best_epoch = 0
    best_video_macro_f1 = float("-inf")
    best_window_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    motionbert_hash = _sha256(checkpoint_path)
    best_path = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_result = _run_train_epoch(
            model, train_loader, optimizer, criterion, device, args.gradient_clip
        )
        # Validation is the only evaluation used for checkpoint selection.
        split_guard.assert_evaluation_allowed("validation")
        validation_result = _evaluate(
            model, validation_loader, criterion, device, classes
        )
        video_macro_f1 = float(validation_result["video_metrics"]["macro_f1"])
        window_macro_f1 = float(validation_result["window_metrics"]["macro_f1"])
        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "validation_window_macro_f1": window_macro_f1,
            "validation_video_macro_f1": video_macro_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_mean": train_result["gradient_norm_mean"],
            "gradient_norm_max": train_result["gradient_norm_max"],
            "embedding_mean": validation_result["diagnostics"]["embedding_mean"],
            "embedding_std": validation_result["diagnostics"]["embedding_std"],
            "logit_mean": validation_result["diagnostics"]["logit_mean"],
            "logit_std": validation_result["diagnostics"]["logit_std"],
            "prediction_class_distribution": validation_result["diagnostics"]["prediction_class_distribution"],
            "motionbert_all_gradients_none": train_result["motionbert_all_gradients_none"],
        }
        history.append(record)
        if video_macro_f1 > best_video_macro_f1:
            best_epoch = epoch
            best_video_macro_f1 = video_macro_f1
            best_window_macro_f1 = window_macro_f1
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model, optimizer, epoch=epoch, epochs_requested=args.epochs,
                    classes=classes, best_epoch=best_epoch,
                    best_video_macro_f1=best_video_macro_f1,
                    best_window_macro_f1=best_window_macro_f1,
                    cache_statistics=cache_statistics,
                    sampling_strategy=sampling_strategy,
                    split_video_ids=split_video_ids,
                    motionbert_hash=motionbert_hash, seed=args.seed,
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_result['loss']:.6f} "
            f"val_loss={validation_result['loss']:.6f} "
            f"window_macro_f1={window_macro_f1:.4f} "
            f"video_macro_f1={video_macro_f1:.4f}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} patience={args.early_stopping_patience}")
            break

    last_epoch = int(history[-1]["epoch"])
    torch.save(
        _checkpoint_payload(
            model, optimizer, epoch=last_epoch, epochs_requested=args.epochs,
            classes=classes, best_epoch=best_epoch,
            best_video_macro_f1=best_video_macro_f1,
            best_window_macro_f1=best_window_macro_f1,
            cache_statistics=cache_statistics,
            sampling_strategy=sampling_strategy, split_video_ids=split_video_ids,
            motionbert_hash=motionbert_hash, seed=args.seed,
        ),
        output_dir / "last.pt",
    )

    best_payload = torch.load(best_path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(best_payload["model_state_dict"], strict=True)
    strict_checkpoint_load = not incompatible.missing_keys and not incompatible.unexpected_keys
    best_validation = _evaluate(model, validation_loader, criterion, device, classes)

    # Test objects and metrics are created only after training and best-model selection.
    split_guard.mark_training_complete()
    split_guard.assert_evaluation_allowed("test")
    del train_loader, train_dataset, train_source_dataset
    del validation_loader, validation_dataset, validation_source_dataset
    gc.collect()
    test_source_dataset = PilotWindowDataset(data_dir, "test")
    test_dataset = precompute_frozen_features(
        model.backbone,
        test_source_dataset,
        list(range(len(test_source_dataset))),
        device,
        args.batch_size,
        "test_after_model_selection",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_result = _evaluate(model, test_loader, criterion, device, classes)
    if len(test_result["ordered_video_ids"]) != 67:
        raise RuntimeError(
            f"Expected all 67 test videos, got {len(test_result['ordered_video_ids'])}."
        )

    _write_confusion(output_dir / "validation_window_confusion_matrix.csv", best_validation["window_metrics"], classes)
    _write_confusion(output_dir / "validation_video_confusion_matrix.csv", best_validation["video_metrics"], classes)
    _write_confusion(output_dir / "test_window_confusion_matrix.csv", test_result["window_metrics"], classes)
    _write_confusion(output_dir / "test_video_confusion_matrix.csv", test_result["video_metrics"], classes)
    _write_predictions(
        output_dir / "test_window_predictions.csv",
        logits=test_result["logits"], labels=test_result["labels"],
        predictions=test_result["predictions"], video_ids=test_result["video_ids"],
        window_indices=test_result["window_indices"], classes=classes,
    )
    _write_predictions(
        output_dir / "test_video_predictions.csv",
        logits=test_result["video_logits"], labels=test_result["video_labels"],
        predictions=test_result["video_predictions"],
        video_ids=test_result["ordered_video_ids"], classes=classes,
    )

    validation_predictions = best_validation["diagnostics"]["prediction_class_distribution"]
    max_validation_fraction = max(validation_predictions.values()) / max(1, sum(validation_predictions.values()))
    overfitting_signal = bool(
        best_epoch < last_epoch
        and history[-1]["train_loss"] < history[best_epoch - 1]["train_loss"]
        and history[-1]["validation_loss"] > history[best_epoch - 1]["validation_loss"]
    )
    embedding_collapse = any(float(record["embedding_std"]) <= 1e-6 for record in history)
    prediction_collapse = max_validation_fraction >= 0.9
    missing_validation_prediction_classes = [
        name for name, count in validation_predictions.items() if int(count) == 0
    ]
    class_prediction_collapse = bool(missing_validation_prediction_classes)
    embedding_diagnostics = {
        "per_epoch": [
            {
                key: record[key]
                for key in (
                    "epoch", "embedding_mean", "embedding_std", "logit_mean", "logit_std",
                    "prediction_class_distribution", "gradient_norm_mean", "gradient_norm_max",
                )
            }
            for record in history
        ],
        "best_validation": best_validation["diagnostics"],
        "pilot_test": test_result["diagnostics"],
        "embedding_collapse": embedding_collapse,
        "prediction_collapse": prediction_collapse,
        "class_prediction_collapse": class_prediction_collapse,
        "missing_validation_prediction_classes": missing_validation_prediction_classes,
        "overfitting_signal": overfitting_signal,
    }
    (output_dir / "embedding_diagnostics.json").write_text(
        json.dumps(embedding_diagnostics, indent=2), encoding="utf-8"
    )
    result = {
        "training_stage": "pilot",
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": last_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "best_epoch": best_epoch,
        "best_video_macro_f1": best_video_macro_f1,
        "best_window_macro_f1": best_window_macro_f1,
        "preprocessing_version": cache_statistics["preprocessing_version"],
        "sampling": sampling_report,
        "video_aggregation_method": "mean_logits",
        "validation_window_metrics": best_validation["window_metrics"],
        "validation_video_metrics": best_validation["video_metrics"],
        "pilot_test_window_metrics": test_result["window_metrics"],
        "pilot_test_video_metrics": test_result["video_metrics"],
        "validation_videos": len(best_validation["ordered_video_ids"]),
        "test_videos": len(test_result["ordered_video_ids"]),
        "test_windows": len(test_dataset),
        "test_created_and_evaluated_after_training": True,
        "strict_checkpoint_load": strict_checkpoint_load,
        "motionbert_frozen": all(not parameter.requires_grad for parameter in model.backbone.parameters()),
        "motionbert_all_gradients_none_every_epoch": all(
            bool(record["motionbert_all_gradients_none"]) for record in history
        ),
        "embedding_collapse": embedding_collapse,
        "prediction_collapse": prediction_collapse,
        "class_prediction_collapse": class_prediction_collapse,
        "missing_validation_prediction_classes": missing_validation_prediction_classes,
        "overfitting_signal": overfitting_signal,
        "limitations": "Pilot Test Evaluation is a one-time pipeline diagnostic, not a final benchmark.",
    }
    (output_dir / "training_history.json").write_text(
        json.dumps({"epochs": history, "sampling": sampling_report}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pilot_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "class_vocabulary.json").write_text(
        json.dumps({"classes": classes}, indent=2), encoding="utf-8"
    )
    shutil.copyfile(data_dir / "cache_manifest.csv", output_dir / "split_manifest.csv")
    print(json.dumps(result, indent=2))
