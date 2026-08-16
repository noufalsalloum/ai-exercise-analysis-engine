from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.exercise_representation import ExerciseRepresentationModel


SOURCE_DATASET = "physical_exercise_recognition"
SEED = 42


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible smoke runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    """Resolve auto/cpu/cuda and reject unavailable CUDA explicitly."""

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CachedExerciseWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy reader for safe ``.npz`` PhysicalExerciseRecognition windows."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        *,
        smoke_test: bool = False,
        videos_per_class: int = 2,
        windows_per_video: int = 2,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        manifest_path = self.data_dir / "cache_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
        frame = pd.read_csv(manifest_path, dtype={"video_id": str})
        required = {
            "video_id", "exercise_label", "label_index", "split",
            "num_windows", "cache_path",
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"Cache manifest lacks columns: {sorted(required - set(frame.columns))}")
        frame = frame[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No cached videos for split {split!r}.")
        if smoke_test:
            selected = []
            for _, group in frame.groupby("exercise_label", sort=True):
                selected.append(group.sort_values("video_id").head(videos_per_class))
            frame = pd.concat(selected, ignore_index=True)
        self.video_ids = tuple(frame["video_id"].astype(str))
        self.samples: list[tuple[Path, int, int]] = []
        for row in frame.itertuples(index=False):
            count = int(row.num_windows)
            indices = list(range(count))
            if smoke_test:
                indices = indices[:windows_per_video]
            for window_index in indices:
                self.samples.append((Path(row.cache_path), window_index, int(row.label_index)))
        if not self.samples:
            raise ValueError(f"No windows selected for split {split!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, window_index, label_index = self.samples[index]
        with np.load(path, allow_pickle=False) as archive:
            window = np.asarray(archive["motionbert_input"][window_index], dtype=np.float32)
        if window.shape != (30, 17, 3):
            raise ValueError(f"Invalid cached window {window.shape} in {path}")
        if not np.isfinite(window).all():
            raise ValueError(f"Non-finite cached window in {path}")
        return torch.from_numpy(window), torch.tensor(label_index, dtype=torch.long)


def confusion_matrix(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> dict[str, object]:
    """Compute dependency-free multiclass metrics from a confusion matrix."""

    matrix = confusion_matrix(targets, predictions, len(class_names))
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[name] = {
            "support": int(matrix[index].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total = int(matrix.sum())
    target_counts = Counter(int(value) for value in targets)
    return {
        "accuracy": float(np.trace(matrix) / max(1, total)),
        "macro_f1": float(np.mean(f1_scores)),
        "balanced_accuracy": float(np.mean(recalls)),
        "majority_class_baseline": max(target_counts.values(), default=0) / max(1, total),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def run_epoch(
    model: ExerciseRepresentationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, object]:
    """Run one train or evaluation epoch and collect embeddings and labels."""

    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    embeddings: list[np.ndarray] = []
    first_gradient_report: dict[str, object] | None = None
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs["logits"], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        if optimizer is not None:
            loss.backward()
            if first_gradient_report is None:
                expert_grads = [
                    parameter.grad for parameter in model.shared_expert.parameters()
                    if parameter.requires_grad
                ]
                head_grads = [
                    parameter.grad for parameter in model.classifier.parameters()
                    if parameter.requires_grad
                ]
                backbone_grads = [parameter.grad for parameter in model.backbone.parameters()]
                first_gradient_report = {
                    "shared_expert_has_finite_nonzero_gradient": any(
                        grad is not None and torch.isfinite(grad).all() and bool(torch.any(grad != 0))
                        for grad in expert_grads
                    ),
                    "classification_head_has_finite_nonzero_gradient": any(
                        grad is not None and torch.isfinite(grad).all() and bool(torch.any(grad != 0))
                        for grad in head_grads
                    ),
                    "motionbert_all_gradients_none": all(grad is None for grad in backbone_grads),
                }
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=gradient_clip,
            )
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(outputs["logits"].argmax(dim=-1).detach().cpu().tolist())
        embeddings.append(outputs["global_embedding"].detach().cpu().numpy())
    all_embeddings = np.concatenate(embeddings, axis=0)
    return {
        "loss": float(np.mean(losses)),
        "targets": targets,
        "predictions": predictions,
        "embedding_std": float(all_embeddings.std(axis=0).mean()),
        "gradient_report": first_gradient_report,
    }


def read_class_vocabulary(data_dir: Path) -> list[str]:
    payload = json.loads((data_dir / "class_vocabulary.json").read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("Invalid class_vocabulary.json")
    return [str(value) for value in classes]


def full_video_ids_by_split(data_dir: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(data_dir / "cache_manifest.csv", dtype={"video_id": str})
    result: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        result[split] = sorted(frame.loc[frame["split"] == split, "video_id"].astype(str).tolist())
    sets = [set(values) for values in result.values()]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Video leakage detected between splits.")
    return result


def checkpoint_payload(
    model: ExerciseRepresentationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    classes: list[str],
    motionbert_hash: str,
    class_weights: torch.Tensor,
    split_ids: dict[str, list[str]],
    best_macro_f1: float,
    seed: int,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "class_vocabulary": classes,
        "source_dataset": SOURCE_DATASET,
        "motionbert_sha256": motionbert_hash,
        "preprocessing_version": "physical_mp33_h36m_xy_conf_v1",
        "h36m_mapping_version": "mediapipe33_to_h36m17_official_v1",
        "window_size": 30,
        "stride": 10,
        "seed": seed,
        "class_weights": class_weights.detach().cpu(),
        "train_video_ids": split_ids["train"],
        "validation_video_ids": split_ids["validation"],
        "test_video_ids": split_ids["test"],
        "best_validation_macro_f1": best_macro_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhysicalExerciseRecognition representations.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motionbert-checkpoint", type=Path, default=Path("models/latest_epoch.bin"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--training-stage", choices=["smoke", "pilot"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--windows-per-video", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_stage = args.training_stage or ("smoke" if args.smoke_test else None)
    project_root = Path(__file__).resolve().parents[2]
    if training_stage == "pilot":
        from training.exercise_representation_pilot import run_pilot_training

        run_pilot_training(args, project_root)
        return
    if training_stage != "smoke":
        raise SystemExit(
            "Choose --smoke-test or --training-stage pilot. Full training remains disabled."
        )
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    checkpoint_path = (
        args.motionbert_checkpoint
        if args.motionbert_checkpoint.is_absolute()
        else project_root / args.motionbert_checkpoint
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed)
    device = resolve_device(args.device)
    classes = read_class_vocabulary(data_dir)
    split_ids = full_video_ids_by_split(data_dir)
    train_dataset = CachedExerciseWindowDataset(data_dir, "train", smoke_test=True)
    validation_dataset = CachedExerciseWindowDataset(
        data_dir, "validation", smoke_test=True, videos_per_class=1, windows_per_video=1
    )
    test_dataset = CachedExerciseWindowDataset(
        data_dir, "test", smoke_test=True, videos_per_class=1, windows_per_video=1
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ExerciseRepresentationModel(
        num_classes=len(classes), motionbert_checkpoint=checkpoint_path, freeze_backbone=True
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    training_label_counts = Counter(label for _, _, label in train_dataset.samples)
    weights = torch.tensor(
        [len(train_dataset) / max(1, len(classes) * training_label_counts[index]) for index in range(len(classes))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict[str, object]] = []
    best_macro_f1 = float("-inf")
    best_path = output_dir / "best.pt"
    first_gradient_report: dict[str, object] | None = None
    initial_loss: float | None = None
    last_loss: float | None = None
    motionbert_hash = sha256_file(checkpoint_path)
    for epoch in range(1, 3):
        train_result = run_epoch(
            model, train_loader, criterion, device, optimizer, args.gradient_clip
        )
        validation_result = run_epoch(model, validation_loader, criterion, device)
        train_metrics = classification_metrics(
            train_result["targets"], train_result["predictions"], classes
        )
        validation_metrics = classification_metrics(
            validation_result["targets"], validation_result["predictions"], classes
        )
        if first_gradient_report is None:
            first_gradient_report = train_result["gradient_report"]
        if initial_loss is None:
            initial_loss = float(train_result["loss"])
        last_loss = float(train_result["loss"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "validation_embedding_std": validation_result["embedding_std"],
        }
        history.append(epoch_record)
        macro_f1 = float(validation_metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                checkpoint_payload(
                    model, optimizer, epoch, classes, motionbert_hash, weights,
                    split_ids, best_macro_f1, args.seed,
                ),
                best_path,
            )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.6f} "
            f"val_loss={validation_result['loss']:.6f} val_macro_f1={macro_f1:.4f}"
        )

    last_path = output_dir / "last.pt"
    torch.save(
        checkpoint_payload(
            model, optimizer, 2, classes, motionbert_hash, weights, split_ids,
            best_macro_f1, args.seed,
        ),
        last_path,
    )
    test_result = run_epoch(model, test_loader, criterion, device)
    test_metrics = classification_metrics(test_result["targets"], test_result["predictions"], classes)
    matrix = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row.tolist()])
    (output_dir / "class_vocabulary.json").write_text(
        json.dumps({"classes": classes}, indent=2), encoding="utf-8"
    )
    shutil.copyfile(data_dir / "cache_manifest.csv", output_dir / "split_manifest.csv")

    loaded = torch.load(last_path, map_location="cpu", weights_only=True)
    roundtrip_model = deepcopy(model).cpu()
    incompatible = roundtrip_model.load_state_dict(loaded["model_state_dict"], strict=True)
    strict_roundtrip = not incompatible.missing_keys and not incompatible.unexpected_keys
    roundtrip_optimizer = torch.optim.AdamW(
        [parameter for parameter in roundtrip_model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    roundtrip_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    result = {
        "smoke_test": True,
        "device": str(device),
        "epochs": 2,
        "batch_size": args.batch_size,
        "dataset_sizes": {
            "train_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "test_windows": len(test_dataset),
        },
        "batch_contract": [args.batch_size, 30, 17, 3],
        "loss_finite": bool(initial_loss is not None and last_loss is not None and np.isfinite([initial_loss, last_loss]).all()),
        "initial_train_loss": initial_loss,
        "final_train_loss": last_loss,
        "loss_changed": bool(initial_loss is not None and last_loss is not None and initial_loss != last_loss),
        "gradient_report": first_gradient_report,
        "motionbert_frozen": all(not parameter.requires_grad for parameter in model.backbone.parameters()),
        "no_video_leakage": True,
        "strict_checkpoint_roundtrip": strict_roundtrip,
        "best_validation_macro_f1": best_macro_f1,
        "test_metrics": test_metrics,
        "test_embedding_std": test_result["embedding_std"],
        "embedding_noncollapsed_smoke_check": bool(float(test_result["embedding_std"]) > 1e-8),
        "limitations": "Tiny smoke subset validates execution only; metrics do not estimate final model quality.",
    }
    (output_dir / "training_history.json").write_text(
        json.dumps({"epochs": history, "smoke_result": result}, indent=2), encoding="utf-8"
    )
    (output_dir / "smoke_training_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
