"""Subject-safe training for the REHAB24 Ex3 table Push-up MVP."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.pushup_correctness import PushupCorrectnessModel
from models.pushup_rep_boundary import PushupRepBoundaryModel
from training.squat_correctness import (
    SquatFeatureDataset,
    build_frozen_feature_cache,
    classification_metrics,
    load_manifest,
)
from training.squat_rep_boundary import FullVideoRecord, load_full_video_records
from training.squat_rep_boundary_v2 import (
    BoundaryExperiment,
    BoundaryV2WindowDataset,
    calibrate,
    evaluate_prediction_cache,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def _boundary_predictions(model: PushupRepBoundaryModel, records: Sequence[FullVideoRecord], device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    model.eval(); output: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        values = model(torch.from_numpy(record.poses).unsqueeze(0).to(device))
        output[record.key] = {
            "active": torch.sigmoid(values["active_logits"]).squeeze(0).cpu().numpy(),
            "boundary": torch.sigmoid(values["boundary_logits"]).squeeze(0).cpu().numpy(),
        }
    return output


def train_boundary(
    records: Sequence[FullVideoRecord], checkpoint: Path, result_dir: Path, device: torch.device,
    *, seed: int = 42, epochs: int = 8, batch_size: int = 8,
) -> dict[str, Any]:
    """Train on seven subjects, select on one, and open two-subject Test once."""
    train = [value for value in records if value.split == "train"]
    validation = [value for value in records if value.split == "validation"]
    test = [value for value in records if value.split == "test"]
    if ({value.subject_id for value in train} & {value.subject_id for value in validation + test}) or ({value.subject_id for value in validation} & {value.subject_id for value in test}):
        raise RuntimeError("Boundary subject leakage.")
    _seed(seed)
    experiment = BoundaryExperiment("pushup_boundary_v1", "boundary_aux_tcn", "weighted_bce", active_dilation=2, boundary_radius=5, boundary_loss_weight=0.35)
    dataset = BoundaryV2WindowDataset(train, 256, 128, 2, 5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    active = np.concatenate(dataset.active_targets); boundary = np.concatenate(dataset.boundary_targets)
    active_weight = torch.tensor((len(active) - active.sum()) / active.sum(), device=device)
    boundary_weight = torch.tensor(min((len(boundary) - boundary.sum()) / max(boundary.sum(), 1.0), 20.0), device=device)
    model = PushupRepBoundaryModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None; best_config: dict[str, Any] | None = None; best_metrics: dict[str, Any] | None = None; best_epoch = 0; history = []; started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for poses, active_target, boundary_target, mask in loader:
            optimizer.zero_grad(set_to_none=True); outputs = model(poses.to(device))
            active_loss = nn.functional.binary_cross_entropy_with_logits(outputs["active_logits"], active_target.to(device), reduction="none", pos_weight=active_weight)
            boundary_loss = nn.functional.binary_cross_entropy_with_logits(outputs["boundary_logits"], boundary_target.to(device), reduction="none", pos_weight=boundary_weight)
            loss = (active_loss + 0.35 * boundary_loss)[mask.to(device)].mean()
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite Push-up boundary loss.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.item()))
        predictions = _boundary_predictions(model, validation, device)
        postprocess, metrics = calibrate(train, validation, predictions, True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_segment_f1": metrics["segment_f1"], "validation_count_mae": metrics["mean_absolute_count_error"]}
        history.append(row); print(json.dumps({"pushup_boundary": row}), flush=True)
        rank = (-float(metrics["mean_absolute_count_error"]), float(metrics["segment_f1"]), -float(metrics["under_count_total"]), -float(metrics["over_count_total"]))
        previous = None if best_metrics is None else (-float(best_metrics["mean_absolute_count_error"]), float(best_metrics["segment_f1"]), -float(best_metrics["under_count_total"]), -float(best_metrics["over_count_total"]))
        if previous is None or rank > previous:
            best_state = copy.deepcopy(model.state_dict()); best_config = postprocess; best_metrics = metrics; best_epoch = epoch
    assert best_state is not None and best_config is not None
    model.load_state_dict(best_state, strict=True)
    validation_metrics, validation_videos, validation_segments = evaluate_prediction_cache(validation, _boundary_predictions(model, validation, device), best_config)
    test_metrics, test_videos, test_segments = evaluate_prediction_cache(test, _boundary_predictions(model, test, device), best_config)
    checkpoint.parent.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "epoch": best_epoch, "postprocessing": best_config, "validation_metrics": validation_metrics, "exercise_scope": "REHAB24 Ex3 table/incline Push-up only", "test_locked_during_selection": True, "preprocessing_version": "physical_mp33_h36m_root_body_scale_xy_conf_pad_v4"}, checkpoint)
    strict = PushupRepBoundaryModel().to(device); strict.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"], strict=True)
    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
    pd.concat([validation_videos.assign(split="validation"), test_videos.assign(split="test")]).to_csv(result_dir / "video_metrics.csv", index=False)
    pd.concat([validation_segments.assign(split="validation"), test_segments.assign(split="test")]).to_csv(result_dir / "segments.csv", index=False)
    result = {"architecture": "preprocessing-v4 -> exercise-specific dual-head dilated TCN", "best_epoch": best_epoch, "validation": validation_metrics, "test": test_metrics, "strict_reload": True, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "training_seconds": perf_counter() - started}
    (result_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def _predict_correctness(model: PushupCorrectnessModel, dataset: SquatFeatureDataset, device: torch.device, batch_size: int = 16) -> pd.DataFrame:
    model.eval(); probabilities = []; labels = []; indices = []
    for features, masks, target, index in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        probability = model.forward_features(features.to(device), masks.to(device))["correct_probability"]
        probabilities.extend(probability.cpu().tolist()); labels.extend(target.tolist()); indices.extend(index.tolist())
    rows = dataset.manifest.iloc[indices].copy().reset_index(drop=True); rows["correct_probability"] = probabilities; rows["correctness"] = labels
    return rows


def _best_threshold(rows: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.linspace(0.10, 0.90, 81):
        prediction = (rows["correct_probability"].to_numpy() >= threshold).astype(int)
        metrics = classification_metrics(rows["correctness"], prediction, rows["correct_probability"])
        candidates.append((metrics["macro_f1"], metrics["classes"]["incorrect"]["recall"], metrics["classes"]["correct"]["recall"], -abs(threshold - 0.5), float(threshold), metrics))
    selected = max(candidates, key=lambda item: item[:4])
    return selected[4], selected[5]


def _fit_correctness_fold(train_data: SquatFeatureDataset, validation_data: SquatFeatureDataset, motionbert: Path, device: torch.device, seed: int, epochs: int) -> tuple[dict[str, torch.Tensor], int, pd.DataFrame]:
    _seed(seed); model = PushupCorrectnessModel(motionbert).to(device)
    labels = train_data.manifest.iloc[train_data.row_indices]["correctness"].to_numpy(int); counts = np.bincount(labels, minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(list(model.expert.parameters()) + list(model.correctness_head.parameters()), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    best_state = copy.deepcopy({**model.expert.state_dict(), **{f"head.{key}": value for key, value in model.correctness_head.state_dict().items()}}); best_epoch = 1; best_macro = -1.0; best_predictions = pd.DataFrame()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, masks, target, _ in loader:
            optimizer.zero_grad(set_to_none=True); output = model.forward_features(features.to(device), masks.to(device)); loss = loss_fn(output["logits"], target.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite Push-up correctness loss.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.expert.parameters()) + list(model.correctness_head.parameters()), 1.0); optimizer.step()
        predictions = _predict_correctness(model, validation_data, device); _, metrics = _best_threshold(predictions)
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]; best_epoch = epoch; best_predictions = predictions
            best_state = {f"expert.{key}": value.detach().cpu().clone() for key, value in model.expert.state_dict().items()}
            best_state.update({f"head.{key}": value.detach().cpu().clone() for key, value in model.correctness_head.state_dict().items()})
    return best_state, best_epoch, best_predictions


def _load_tail(model: PushupCorrectnessModel, state: dict[str, torch.Tensor]) -> None:
    model.expert.load_state_dict({key.removeprefix("expert."): value for key, value in state.items() if key.startswith("expert.")}, strict=True)
    model.correctness_head.load_state_dict({key.removeprefix("head."): value for key, value in state.items() if key.startswith("head.")}, strict=True)


def train_correctness_loso(manifest: pd.DataFrame, repetition_dir: Path, feature_dir: Path, motionbert: Path, checkpoint: Path, result_dir: Path, device: torch.device, *, seed: int = 42, epochs: int = 6) -> dict[str, Any]:
    """Run development-only LOSO, then fit once and evaluate locked Test once."""
    template = PushupCorrectnessModel(motionbert).to(device)
    feature_cache = build_frozen_feature_cache(template, manifest, feature_dir, motionbert, device, 8)
    development = manifest[manifest["split"] != "test"].copy(); test_subjects = set(manifest.loc[manifest["split"] == "test", "subject_id"])
    if set(development["subject_id"]) & test_subjects: raise RuntimeError("Correctness Test leakage.")
    folds = []; oof = []
    for fold_index, subject in enumerate(sorted(development["subject_id"].unique(), key=int), 1):
        frame = manifest.copy(); frame["split"] = np.where(frame["subject_id"].isin(test_subjects), "test", np.where(frame["subject_id"] == subject, "validation", "train"))
        train_data = SquatFeatureDataset(frame, feature_cache, "train"); validation_data = SquatFeatureDataset(frame, feature_cache, "validation")
        _, best_epoch, predictions = _fit_correctness_fold(train_data, validation_data, motionbert, device, seed + fold_index, epochs)
        threshold, metrics = _best_threshold(predictions); predictions["fold_threshold"] = threshold; predictions["held_subject"] = subject; oof.append(predictions)
        folds.append({"subject_id": subject, "best_epoch": best_epoch, "threshold": threshold, **metrics})
        print(json.dumps({"pushup_correctness_fold": subject, "macro_f1": metrics["macro_f1"], "threshold": threshold}), flush=True)
    oof_frame = pd.concat(oof, ignore_index=True); threshold = float(np.median([row["threshold"] for row in folds])); final_epochs = max(1, int(round(np.median([row["best_epoch"] for row in folds]))))
    final_frame = manifest.copy(); final_frame.loc[final_frame["split"] != "test", "split"] = "train"
    train_data = SquatFeatureDataset(final_frame, feature_cache, "train")
    _seed(seed + 100); model = PushupCorrectnessModel(motionbert).to(device)
    final_labels = train_data.manifest.iloc[train_data.row_indices]["correctness"].to_numpy(int); counts = np.bincount(final_labels, minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(final_labels) / (2.0 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(list(model.expert.parameters()) + list(model.correctness_head.parameters()), lr=3e-4, weight_decay=1e-4)
    final_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed + 100), num_workers=0)
    for _ in range(final_epochs):
        model.train()
        for features, masks, target, _ in final_loader:
            optimizer.zero_grad(set_to_none=True); output = model.forward_features(features.to(device), masks.to(device)); loss = loss_fn(output["logits"], target.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite final Push-up correctness loss.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.expert.parameters()) + list(model.correctness_head.parameters()), 1.0); optimizer.step()
    test_data = SquatFeatureDataset(final_frame, feature_cache, "test"); test_predictions = _predict_correctness(model, test_data, device)
    test_binary = (test_predictions["correct_probability"] >= threshold).astype(int); test_metrics = classification_metrics(test_predictions["correctness"], test_binary, test_predictions["correct_probability"])
    test_predictions["prediction"] = test_binary; test_predictions["assessment"] = np.where(test_binary == 1, "PASS", "FAIL")
    checkpoint.parent.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"expert_state_dict": model.expert.state_dict(), "correctness_head_state_dict": model.correctness_head.state_dict(), "decision_threshold": threshold, "exercise_scope": "REHAB24 Ex3 table/incline Push-up only", "motionbert_frozen": True, "development_subjects": sorted(development["subject_id"].unique(), key=int), "test_subjects": sorted(test_subjects, key=int), "test_locked_during_selection": True, "loso_folds": folds}, checkpoint)
    strict = PushupCorrectnessModel(motionbert).to(device); saved = torch.load(checkpoint, map_location=device, weights_only=True); strict.expert.load_state_dict(saved["expert_state_dict"], strict=True); strict.correctness_head.load_state_dict(saved["correctness_head_state_dict"], strict=True)
    oof_frame.to_csv(result_dir / "oof_predictions.csv", index=False); pd.DataFrame(folds).to_json(result_dir / "loso_folds.json", orient="records", indent=2); test_predictions.to_csv(result_dir / "test_predictions.csv", index=False)
    aggregate = {"folds": len(folds), "mean_macro_f1": float(np.mean([row["macro_f1"] for row in folds])), "std_macro_f1": float(np.std([row["macro_f1"] for row in folds])), "min_macro_f1": float(np.min([row["macro_f1"] for row in folds])), "mean_correct_recall": float(np.mean([row["classes"]["correct"]["recall"] for row in folds])), "mean_incorrect_recall": float(np.mean([row["classes"]["incorrect"]["recall"] for row in folds])), "fold_thresholds": [row["threshold"] for row in folds], "selected_threshold": threshold, "test": test_metrics, "strict_reload": True, "motionbert_frozen": True, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint)}
    (result_dir / "result.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def run(project: Path, device: torch.device) -> dict[str, Any]:
    manifest_path = project / "results/pushup_ai_v1/data_summary/repetition_manifest.csv"; cache = project / "datasets/window_cache/rehab24_pushup_v1"
    manifest = load_manifest(manifest_path, cache / "repetitions")
    records = load_full_video_records(manifest, cache / "full_videos")
    boundary = train_boundary(records, project / "checkpoints/pushup_ai_v1/boundary/best.pt", project / "results/pushup_ai_v1/cv_evaluation/boundary", device)
    correctness = train_correctness_loso(manifest, cache / "repetitions", cache / "motionbert_features", project / "models/latest_epoch.bin", project / "checkpoints/pushup_ai_v1/correctness/final_dev.pt", project / "results/pushup_ai_v1/cv_evaluation/correctness", device)
    result = {"scope": "table/incline Push-up only", "boundary": boundary, "correctness": correctness, "detailed_error": {"supported": False, "fallback": "Form Issue after correctness FAIL only"}}
    (project / "results/pushup_ai_v1/end_to_end").mkdir(parents=True, exist_ok=True); (project / "results/pushup_ai_v1/end_to_end/result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
