"""Open the frozen Squat Experiment-2 Test split exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.squat_correctness import SquatCorrectnessModel
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from tools.squat_ai.prepare_rehab24_squat import resample_sequence
from training.squat_correctness import FeatureCache, SquatFeatureDataset, classification_metrics, grouped_metrics, load_checkpoint_strict, sha256
from training.squat_correctness_v2 import prediction_frame
from training.squat_rep_boundary import load_full_video_records
from training.squat_rep_boundary_v2 import evaluate_prediction_cache, make_model, predict_records


ROOT = Path(__file__).resolve().parents[1]


def full_cache_path(full_dir: Path, source_video_path: str) -> Path:
    stem = Path(source_video_path).stem.replace("-30fps-transposed", "_cam18").replace("-30fps", "_cam17")
    return full_dir / f"{stem}.npz"


@torch.no_grad()
def classify_detected_segments(
    model: SquatCorrectnessModel,
    landmarks: np.ndarray,
    segments: list[tuple[int, int]],
    threshold: float,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}}); normalizer = H36MCoordinateNormalizer()
    inputs: list[np.ndarray] = []; masks: list[np.ndarray] = []
    for start, end in segments:
        clip = landmarks[start - 1 : end]
        h36m = selector.to_h36m_17(clip); normalized, _ = normalizer.normalize(h36m); resampled = resample_sequence(normalized)
        inputs.append(resampled); masks.append(resampled[..., 2].mean(axis=1) > 0.01)
    results: list[dict[str, Any]] = []; model.eval()
    for offset in range(0, len(inputs), batch_size):
        values = torch.from_numpy(np.stack(inputs[offset : offset + batch_size])).to(device); mask = torch.from_numpy(np.stack(masks[offset : offset + batch_size])).to(device)
        probabilities = model(values, mask)["correct_probability"].cpu().numpy()
        for probability in probabilities:
            predicted = int(probability >= threshold); results.append({"correct_probability": float(probability), "predicted_correctness": predicted, "pass_fail": "PASS" if predicted else "FAIL", "score": None})
    return results


def comparison_rows(v1: dict[str, Any], v2: dict[str, Any], task: str) -> list[dict[str, Any]]:
    if task == "correctness":
        pairs = {
            "accuracy": (v1["accuracy"], v2["accuracy"]), "macro_f1": (v1["macro_f1"], v2["macro_f1"]),
            "balanced_accuracy": (v1["balanced_accuracy"], v2["balanced_accuracy"]),
            "correct_recall": (v1["classes"]["correct"]["recall"], v2["classes"]["correct"]["recall"]),
            "incorrect_recall": (v1["classes"]["incorrect"]["recall"], v2["classes"]["incorrect"]["recall"]),
            "roc_auc": (v1["roc_auc"], v2["roc_auc"]),
        }
    else:
        pairs = {"segment_f1": (v1["segment_f1"], v2["segment_f1"]), "count_mae": (v1["mean_absolute_count_error"], v2["mean_absolute_count_error"]), "exact_count_accuracy": (v1["exact_count_accuracy"], v2["exact_count_accuracy"]), "over_count": (v1["over_count_total"], v2["over_count_total"]), "under_count": (v1["under_count_total"], v2["under_count_total"])}
    return [{"task": task, "metric": metric, "v1": old, "v2": new, "change": new - old} for metric, (old, new) in pairs.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/squat_ai/data/repetition_manifest.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/window_cache/rehab24_squat_v1"))
    parser.add_argument("--motionbert-checkpoint", type=Path, default=Path("models/latest_epoch.bin"))
    parser.add_argument("--boundary-checkpoint", type=Path, default=Path("checkpoints/squat_ai_v2/rep_boundary/best.pt"))
    parser.add_argument("--correctness-checkpoint", type=Path, default=Path("archive/checkpoints/squat_ai_v2/correctness/best.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("results/squat_ai_v2"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); resolve = lambda path: path if path.is_absolute() else ROOT / path
    output_root = resolve(args.output_root); lock_path = output_root / "test_opened_once.json"
    if lock_path.exists(): raise RuntimeError(f"Experiment-2 Test already opened: {lock_path}")
    boundary_selection = json.loads((output_root / "rep_boundary/selection.json").read_text(encoding="utf-8")); correctness_selection = json.loads((output_root / "correctness/selection.json").read_text(encoding="utf-8"))
    if not boundary_selection.get("test_lock") or not correctness_selection.get("test_lock"): raise RuntimeError("Frozen development selection evidence is missing.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)); started = perf_counter()
    manifest_path = resolve(args.manifest); full_manifest = pd.read_csv(manifest_path, dtype={"subject_id": str}); test_manifest = full_manifest[full_manifest["split"] == "test"].copy().reset_index(drop=True)
    if set(test_manifest["subject_id"].unique()) != {"4", "7"}: raise RuntimeError("Unexpected locked Test subjects.")
    cache_dir = resolve(args.cache_dir); all_records = load_full_video_records(full_manifest, cache_dir / "full_videos"); test_records = [record for record in all_records if record.split == "test"]

    boundary_checkpoint_path = resolve(args.boundary_checkpoint); boundary_checkpoint = torch.load(boundary_checkpoint_path, map_location=device, weights_only=True)
    architecture = str(boundary_checkpoint["experiment"]["architecture"]); boundary_model = make_model(architecture).to(device); boundary_model.load_state_dict(boundary_checkpoint["model_state_dict"], strict=True)
    boundary_predictions = predict_records(boundary_model, test_records, architecture, device)
    boundary_metrics, boundary_videos, boundary_segments = evaluate_prediction_cache(test_records, boundary_predictions, dict(boundary_checkpoint["postprocessing"]))
    boundary_dir = output_root / "rep_boundary"; boundary_dir.mkdir(parents=True, exist_ok=True)
    (boundary_dir / "test_metrics.json").write_text(json.dumps(boundary_metrics, indent=2), encoding="utf-8"); boundary_videos.to_csv(boundary_dir / "test_video_counts.csv", index=False); boundary_segments.to_csv(boundary_dir / "test_predicted_segments.csv", index=False)

    feature_dir = cache_dir / "motionbert_features"; feature_metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8")); feature_cache = FeatureCache(feature_dir / "features.npy", feature_dir / "metadata.json", tuple(feature_metadata["sample_ids"]))
    test_manifest["repetition_cache_path"] = test_manifest["sample_id"].map(lambda value: str((cache_dir / "repetitions" / f"{value}.npz").resolve()))
    test_dataset = SquatFeatureDataset(test_manifest, feature_cache, "test")
    correctness_checkpoint_path = resolve(args.correctness_checkpoint); correctness_model = SquatCorrectnessModel(resolve(args.motionbert_checkpoint)).to(device); correctness_checkpoint = load_checkpoint_strict(correctness_checkpoint_path, correctness_model, device)
    decision_threshold = float(correctness_checkpoint["decision_threshold"]); correctness_frame, embeddings = prediction_frame(correctness_model, test_dataset, device, args.batch_size)
    correctness_frame["predicted_correctness"] = (correctness_frame["correct_probability"] >= decision_threshold).astype(int); correctness_frame["pass_fail"] = correctness_frame["predicted_correctness"].map({0: "FAIL", 1: "PASS"}); correctness_frame["score"] = None
    correctness_metrics = classification_metrics(correctness_frame["correctness"], correctness_frame["predicted_correctness"], correctness_frame["correct_probability"]); correctness_metrics["decision_threshold"] = decision_threshold; correctness_metrics["embedding_std"] = float(embeddings.std())
    correctness_dir = output_root / "correctness"; correctness_dir.mkdir(parents=True, exist_ok=True)
    (correctness_dir / "test_metrics.json").write_text(json.dumps(correctness_metrics, indent=2), encoding="utf-8"); correctness_frame.to_csv(correctness_dir / "test_predictions.csv", index=False)
    grouped_metrics(correctness_frame, "subject_id").to_csv(correctness_dir / "test_per_subject.csv", index=False); grouped_metrics(correctness_frame, "camera_id").to_csv(correctness_dir / "test_per_camera.csv", index=False); grouped_metrics(correctness_frame, "orientation_raw").to_csv(correctness_dir / "test_per_orientation.csv", index=False)

    # Locked end-to-end Test evaluation.
    end_rows: list[dict[str, Any]] = []; classified_rows: list[dict[str, Any]] = []
    for (video_id, camera_id), ground_truth in test_manifest.groupby(["video_id", "camera_id"], sort=True):
        video_key = f"{video_id}_{camera_id}"; detected = boundary_segments[boundary_segments["video_key"] == video_key].sort_values("start_frame"); segments = [(int(row.start_frame), int(row.end_frame)) for row in detected.itertuples(index=False)]
        path = full_cache_path(cache_dir / "full_videos", str(ground_truth.iloc[0]["video_path"]));
        with np.load(path, allow_pickle=False) as archive: landmarks = np.asarray(archive["landmarks"], np.float32)
        classified = classify_detected_segments(correctness_model, landmarks, segments, decision_threshold, device, args.batch_size) if segments else []
        for index, ((start, end), result) in enumerate(zip(segments, classified), 1): classified_rows.append({"video_key": video_key, "predicted_rep_index": index, "start_frame": start, "end_frame": end, **result})
        gt_total = len(ground_truth); gt_correct = int(ground_truth["correctness"].sum()); gt_incorrect = gt_total - gt_correct; pred_total = len(classified); pred_correct = sum(item["predicted_correctness"] for item in classified); pred_incorrect = pred_total - pred_correct
        end_rows.append({"video_key": video_key, "video_id": video_id, "subject_id": str(ground_truth.iloc[0]["subject_id"]), "camera_id": camera_id, "gt_total_reps": gt_total, "predicted_total_reps": pred_total, "total_count_error": abs(pred_total - gt_total), "gt_correct_reps": gt_correct, "predicted_correct_reps": pred_correct, "correct_count_error": abs(pred_correct - gt_correct), "gt_incorrect_reps": gt_incorrect, "predicted_incorrect_reps": pred_incorrect, "incorrect_count_error": abs(pred_incorrect - gt_incorrect)})
    end_frame = pd.DataFrame(end_rows); end_dir = output_root / "end_to_end"; end_dir.mkdir(parents=True, exist_ok=True); end_frame.to_csv(end_dir / "video_results.csv", index=False); pd.DataFrame(classified_rows).to_csv(end_dir / "segment_classifications.csv", index=False)
    end_summary = {"videos": len(end_frame), "mean_total_count_error": float(end_frame["total_count_error"].mean()), "mean_correct_count_error": float(end_frame["correct_count_error"].mean()), "mean_incorrect_count_error": float(end_frame["incorrect_count_error"].mean()), "exact_total_count_accuracy": float((end_frame["total_count_error"] == 0).mean()), "pass_fail_contract": "derived from learned correctness", "score": None}
    (end_dir / "summary.json").write_text(json.dumps(end_summary, indent=2), encoding="utf-8")

    v1_correctness = json.loads((ROOT / "results/squat_ai/correctness/test_metrics.json").read_text(encoding="utf-8")); v1_boundary = json.loads((ROOT / "results/squat_ai/rep_boundary/test_metrics.json").read_text(encoding="utf-8")); v1_end = json.loads((ROOT / "results/squat_ai/end_to_end/summary.json").read_text(encoding="utf-8"))
    comparisons = comparison_rows(v1_correctness, correctness_metrics, "correctness") + comparison_rows(v1_boundary, boundary_metrics, "rep_boundary"); pd.DataFrame(comparisons).to_csv(output_root / "v1_vs_v2_test_comparison.csv", index=False)
    end_comparison = {metric: {"v1": v1_end[metric], "v2": end_summary[metric], "change": end_summary[metric] - v1_end[metric]} for metric in ("mean_total_count_error", "mean_correct_count_error", "mean_incorrect_count_error", "exact_total_count_accuracy")}; (end_dir / "v1_vs_v2.json").write_text(json.dumps(end_comparison, indent=2), encoding="utf-8")
    lock = {"opened_once": True, "elapsed_seconds": perf_counter() - started, "test_subjects": ["4", "7"], "boundary_checkpoint_sha256": sha256(boundary_checkpoint_path), "correctness_checkpoint_sha256": sha256(correctness_checkpoint_path), "frozen_boundary_experiment": boundary_checkpoint["experiment"], "frozen_boundary_postprocessing": boundary_checkpoint["postprocessing"], "frozen_correctness_experiment": correctness_checkpoint["experiment"], "frozen_correctness_threshold": decision_threshold, "boundary_test": boundary_metrics, "correctness_test": correctness_metrics, "end_to_end_test": end_summary}
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2), flush=True)


if __name__ == "__main__":
    main()
