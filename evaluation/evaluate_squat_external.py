"""Evaluate future external Squat videos with locked Boundary V2/Correctness V3.

This entry point is intentionally separate from the historical REHAB24 Test
subjects. It reports boundary counts and per-segment correctness probabilities;
it does not emit a quality score or detailed error labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.squat_correctness import SquatCorrectnessModel
from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from tools.squat_ai.prepare_rehab24_squat import (
    extract_video,
    fill_missing_pose,
    resample_sequence,
)
from training.squat_correctness import classification_metrics, load_checkpoint_strict
from training.squat_rep_boundary import match_segments
from training.squat_rep_boundary_v2 import v2_segments


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED_HISTORICAL_TEST_SUBJECTS = {"4", "7"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_ground_truth(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Ground truth JSON must be a list or contain a segments list.")
    output = []
    for row in rows:
        start = int(row["start_frame"]); end = int(row["end_frame"])
        if start < 0 or end < start:
            raise ValueError("Invalid external ground-truth boundary.")
        correctness = row.get("correctness")
        if correctness is not None and int(correctness) not in (0, 1):
            raise ValueError("Correctness must be 0, 1, or missing.")
        output.append(
            {
                "start_frame": start,
                "end_frame": end,
                "correctness": None if correctness is None else int(correctness),
            }
        )
    return output


@torch.no_grad()
def evaluate_video(
    video_path: Path,
    pose_model: Path,
    boundary_checkpoint_path: Path,
    correctness_checkpoint_path: Path,
    motionbert_checkpoint_path: Path,
    device: torch.device,
    *,
    subject_id: str | None = None,
    orientation: str | None = None,
    ground_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate one never-before-used subject/video with locked artifacts."""

    if subject_id in PROHIBITED_HISTORICAL_TEST_SUBJECTS:
        raise ValueError(
            "Subjects 4 and 7 are historical locked Test subjects and cannot be re-evaluated."
        )
    raw, detected, fps = extract_video(video_path, pose_model, None)
    filled, gap_stats = fill_missing_pose(raw, detected)
    selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}})
    normalizer = H36MCoordinateNormalizer()
    normalized, diagnostics = normalizer.normalize(selector.to_h36m_17(filled))

    boundary_checkpoint = torch.load(
        boundary_checkpoint_path, map_location=device, weights_only=True
    )
    if boundary_checkpoint["experiment"]["architecture"] != "boundary_aux_tcn":
        raise ValueError("Expected the locked Boundary V2 dual-head checkpoint.")
    boundary_model = SquatRepBoundaryV2Model().to(device)
    boundary_model.load_state_dict(boundary_checkpoint["model_state_dict"], strict=True)
    boundary_model.eval()
    output = boundary_model(torch.from_numpy(normalized).unsqueeze(0).to(device))
    active_probability = torch.sigmoid(output["active_logits"][0]).cpu().numpy()
    boundary_probability = torch.sigmoid(output["boundary_logits"][0]).cpu().numpy()
    segments = v2_segments(
        active_probability,
        boundary_probability,
        boundary_checkpoint["postprocessing"],
    )

    correctness_model = SquatCorrectnessModel(motionbert_checkpoint_path).to(device)
    correctness_checkpoint = load_checkpoint_strict(
        correctness_checkpoint_path, correctness_model, device
    )
    if correctness_checkpoint.get("training_stage") != "development_final_model":
        raise ValueError("Expected the Correctness V3 Development Final Model.")
    threshold = float(correctness_checkpoint["decision_threshold"])
    correctness_model.eval()
    segment_rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(segments, start=1):
        repetition = resample_sequence(normalized[start : end + 1])
        temporal_mask = torch.ones((1, len(repetition)), dtype=torch.bool, device=device)
        prediction = correctness_model(
            torch.from_numpy(repetition).unsqueeze(0).to(device), temporal_mask
        )
        probability = float(prediction["correct_probability"][0].cpu())
        segment_rows.append(
            {
                "repetition_index": index,
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_seconds": float((end - start + 1) / max(fps, 1e-8)),
                "boundary_confidence": float(
                    np.mean(active_probability[start : end + 1])
                ),
                "correct_probability": probability,
                "correctness_prediction": int(probability >= threshold),
                "correctness_label": (
                    "correct" if probability >= threshold else "incorrect"
                ),
                "pass_fail": "PASS" if probability >= threshold else "FAIL",
            }
        )

    result: dict[str, Any] = {
        "evaluation_stage": "future_external_subject_evaluation",
        "video_path": str(video_path),
        "subject_id": subject_id,
        "orientation_raw": orientation,
        "fps": fps,
        "frame_count": len(normalized),
        "detected_frame_percentage": float(100.0 * detected.mean()),
        "gap_handling": gap_stats,
        "normalization": {
            "sequence_scale": diagnostics.sequence_scale,
            "outlier_frames": int(diagnostics.outlier_mask.sum()),
            "near_zero_scale_frames": int(diagnostics.near_zero_scale_mask.sum()),
        },
        "predicted_repetition_count": len(segment_rows),
        "segments": segment_rows,
        "correctness_threshold": threshold,
        "quality_score": None,
        "detailed_errors": None,
        "historical_test_subjects_re_evaluated": False,
        "ground_truth_evaluation": None,
    }
    if ground_truth is not None:
        gt_segments = [
            (int(row["start_frame"]), int(row["end_frame"])) for row in ground_truth
        ]
        matches = match_segments(segments, gt_segments, 0.5)
        evaluation: dict[str, Any] = {
            "ground_truth_repetition_count": len(gt_segments),
            "count_absolute_error": abs(len(segments) - len(gt_segments)),
            "matched_segments_iou_0_5": len(matches),
        }
        labelled = [row for row in ground_truth if row["correctness"] is not None]
        matched_predictions = []
        matched_targets = []
        matched_probabilities = []
        for predicted_index, gt_index, _ in matches:
            target = ground_truth[gt_index]["correctness"]
            if target is None:
                continue
            matched_targets.append(int(target))
            matched_predictions.append(
                int(segment_rows[predicted_index]["correctness_prediction"])
            )
            matched_probabilities.append(
                float(segment_rows[predicted_index]["correct_probability"])
            )
        evaluation["labelled_ground_truth_segments"] = len(labelled)
        evaluation["correctness_metrics"] = (
            classification_metrics(
                matched_targets, matched_predictions, matched_probabilities
            )
            if matched_targets
            else None
        )
        result["ground_truth_evaluation"] = evaluation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--pose-model", type=Path, default=Path(r"C:\MediaPipe\pose_landmarker_full.task")
    )
    parser.add_argument("--subject-id")
    parser.add_argument("--orientation")
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument(
        "--boundary-checkpoint",
        type=Path,
        default=Path("checkpoints/squat_ai_v2/rep_boundary/best.pt"),
    )
    parser.add_argument(
        "--correctness-checkpoint",
        type=Path,
        default=Path("checkpoints/squat_ai_v3/correctness/final_dev.pt"),
    )
    parser.add_argument(
        "--motionbert-checkpoint", type=Path, default=Path("models/latest_epoch.bin")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/squat_ai_v3/external_evaluation.json"))
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    result = evaluate_video(
        _resolve(args.video),
        _resolve(args.pose_model),
        _resolve(args.boundary_checkpoint),
        _resolve(args.correctness_checkpoint),
        _resolve(args.motionbert_checkpoint),
        device,
        subject_id=args.subject_id,
        orientation=args.orientation,
        ground_truth=_load_ground_truth(
            None if args.ground_truth is None else _resolve(args.ground_truth)
        ),
    )
    output = _resolve(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
