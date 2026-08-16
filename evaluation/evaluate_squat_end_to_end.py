"""Evaluate detected Squat segments followed by learned correctness classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.squat_correctness import SquatCorrectnessModel
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from tools.squat_ai.prepare_rehab24_squat import resample_sequence
from training.squat_correctness import load_checkpoint_strict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def full_cache_path(full_dir: Path, source_video_path: str) -> Path:
    stem = Path(source_video_path).stem.replace("-30fps-transposed", "_cam18").replace("-30fps", "_cam17")
    return full_dir / f"{stem}.npz"


def classify_segments(
    model: SquatCorrectnessModel,
    landmarks: np.ndarray,
    segments: list[tuple[int, int]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}})
    normalizer = H36MCoordinateNormalizer()
    inputs: list[np.ndarray] = []; masks: list[np.ndarray] = []
    for start, end in segments:
        clip = landmarks[start - 1 : end]
        if len(clip) == 0: raise ValueError(f"Empty predicted segment {start}:{end}")
        h36m = selector.to_h36m_17(clip); normalized, _ = normalizer.normalize(h36m)
        resampled = resample_sequence(normalized)
        inputs.append(resampled); masks.append(resampled[..., 2].mean(axis=1) > 0.01)
    outputs: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(inputs), batch_size):
            tensor = torch.from_numpy(np.stack(inputs[offset : offset + batch_size])).to(device)
            mask = torch.from_numpy(np.stack(masks[offset : offset + batch_size])).to(device)
            result = model(tensor, mask); probabilities = result["correct_probability"].cpu().numpy()
            for probability in probabilities:
                predicted = int(probability >= 0.5)
                outputs.append({"correct_probability": float(probability), "predicted_correctness": predicted, "pass_fail": "PASS" if predicted else "FAIL", "score": None})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/window_cache/rehab24_squat_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("results/squat_ai/data/repetition_manifest.csv"))
    parser.add_argument("--predicted-segments", type=Path, default=Path("results/squat_ai/rep_boundary/predicted_segments.csv"))
    parser.add_argument("--correctness-checkpoint", type=Path, default=Path("archive/checkpoints/squat_ai_v1/correctness/best.pt"))
    parser.add_argument("--motionbert-checkpoint", type=Path, default=Path("models/latest_epoch.bin"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/squat_ai/end_to_end"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(); resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    cache_dir = resolve(args.cache_dir); manifest = pd.read_csv(resolve(args.manifest), dtype={"subject_id": str})
    predicted = pd.read_csv(resolve(args.predicted_segments)); predicted = predicted[predicted["split"] == "test"].copy()
    model = SquatCorrectnessModel(resolve(args.motionbert_checkpoint)).to(device)
    checkpoint = load_checkpoint_strict(resolve(args.correctness_checkpoint), model, device)
    if not bool(checkpoint.get("motionbert_frozen", False)):
        raise ValueError("Correctness checkpoint does not attest frozen MotionBERT.")
    video_rows: list[dict[str, Any]] = []; segment_rows: list[dict[str, Any]] = []
    test_manifest = manifest[manifest["split"] == "test"]
    for (video_id, camera_id), ground_truth in test_manifest.groupby(["video_id", "camera_id"], sort=True):
        video_key = f"{video_id}_{camera_id}"; detected = predicted[predicted["video_key"] == video_key].sort_values("start_frame")
        segments = [(int(row.start_frame), int(row.end_frame)) for row in detected.itertuples(index=False)]
        cache_path = full_cache_path(cache_dir / "full_videos", str(ground_truth.iloc[0]["video_path"]))
        with np.load(cache_path, allow_pickle=False) as archive:
            landmarks = np.asarray(archive["landmarks"], dtype=np.float32)
        classifications = classify_segments(model, landmarks, segments, device, args.batch_size) if segments else []
        for index, ((start, end), classification) in enumerate(zip(segments, classifications), 1):
            segment_rows.append({"video_key": video_key, "predicted_rep_index": index, "start_frame": start, "end_frame": end, **classification})
        gt_correct = int(ground_truth["correctness"].sum()); gt_total = len(ground_truth); gt_incorrect = gt_total - gt_correct
        pred_correct = sum(item["predicted_correctness"] for item in classifications); pred_total = len(classifications); pred_incorrect = pred_total - pred_correct
        video_rows.append({"video_key": video_key, "video_id": video_id, "subject_id": str(ground_truth.iloc[0]["subject_id"]), "camera_id": camera_id, "gt_total_reps": gt_total, "predicted_total_reps": pred_total, "total_count_error": abs(pred_total - gt_total), "gt_correct_reps": gt_correct, "predicted_correct_reps": pred_correct, "correct_count_error": abs(pred_correct - gt_correct), "gt_incorrect_reps": gt_incorrect, "predicted_incorrect_reps": pred_incorrect, "incorrect_count_error": abs(pred_incorrect - gt_incorrect)})
    videos = pd.DataFrame(video_rows); output_dir = resolve(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    videos.to_csv(output_dir / "video_results.csv", index=False); pd.DataFrame(segment_rows).to_csv(output_dir / "segment_classifications.csv", index=False)
    summary = {"videos": len(videos), "mean_total_count_error": float(videos["total_count_error"].mean()), "mean_correct_count_error": float(videos["correct_count_error"].mean()), "mean_incorrect_count_error": float(videos["incorrect_count_error"].mean()), "exact_total_count_accuracy": float((videos["total_count_error"] == 0).mean()), "correctness_checkpoint_epoch": int(checkpoint["epoch"]), "pass_fail_contract": "derived from correctness; no separately trained head", "score": None}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
