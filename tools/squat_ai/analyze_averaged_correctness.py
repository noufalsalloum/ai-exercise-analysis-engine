"""Run comparison-only Squat V3 averaged temporal inference on local videos.

No labels are inferred, no thresholds are tuned, and no checkpoint is written.
The active robustness result is retained beside raw/calibrated 3/5/7-window
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from inference.squat_ai_mvp import (
    SquatAIExperimentalOrchestrator,
    SquatAIMVPConfig,
    apply_decision_policy,
)
from inference.squat_averaged_correctness import (
    AveragedCorrectnessConfig,
    AveragedCorrectnessExperiment,
    provisional_score,
)
from input_sources.frame_sources import VideoFrameSource
from input_sources.pose_stream import PoseStreamProcessor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSE_MODEL = Path(r"C:\MediaPipe\pose_landmarker_full.task")
DEFAULT_VIDEOS = (
    ROOT / "datasets" / "air_squat" / "user" / "side.mp4",
    ROOT / "datasets" / "air_squat" / "reference" / "side.mp4",
    ROOT / "datasets" / "air_squat" / "user" / "front (3).mp4",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inferred_view(path: Path) -> str | None:
    """Return only a filename-derived approximate view for comparison policy."""

    name = path.name.lower()
    if "side" in name:
        return "side"
    if "front" in name:
        return "front"
    return None


def collect_video(
    path: Path,
    pose_model: Path,
    orchestrator: SquatAIExperimentalOrchestrator,
) -> tuple[np.ndarray, list[int], list[float], dict[str, Any]]:
    orchestrator.reset()
    source = VideoFrameSource(path)
    pose = PoseStreamProcessor(pose_model)
    successes = 0
    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            landmarks = pose.process(packet.frame, packet.timestamp_seconds)
            successes += int(landmarks is not None)
            orchestrator.record_frame(
                landmarks, packet.frame_index, packet.timestamp_seconds
            )
        landmarks, frame_indices, timestamps = orchestrator._snapshot()
        metadata = {
            "fps": float(source.fps),
            "frames": int(len(landmarks)),
            "pose_success_frames": int(successes),
            "pose_success_rate": float(successes / max(len(landmarks), 1)),
        }
        return landmarks, frame_indices, timestamps, metadata
    finally:
        pose.close()
        source.close()


def analyze_video(
    path: Path,
    pose_model: Path,
    orchestrator: SquatAIExperimentalOrchestrator,
    experiment: AveragedCorrectnessExperiment,
) -> dict[str, Any]:
    landmarks, frame_indices, timestamps, metadata = collect_video(
        path, pose_model, orchestrator
    )
    normalized, segments, boundary_latency = orchestrator._detect_segments(landmarks)
    view = inferred_view(path)
    repetitions: list[dict[str, Any]] = []
    for rep_index, (start, end) in enumerate(segments, 1):
        segment = normalized[start : end + 1]
        current_probability, current_latency = orchestrator._correctness(segment)
        current_rep = apply_decision_policy(
            rep_index=rep_index,
            start_frame=frame_indices[start],
            end_frame=frame_indices[end],
            correct_probability=current_probability,
            threshold=orchestrator.config.correctness_threshold,
            raw_error_class=None,
            error_confidence=None,
            representative_frame=None,
        )
        safe = orchestrator.robustness_policy.assess_rep(current_rep, view)
        comparisons: dict[str, Any] = {}
        assert orchestrator.correctness_model is not None
        for sample_count in experiment.config.comparison_sample_counts:
            result = experiment.compare(
                orchestrator.correctness_model,
                segment,
                orchestrator.device,
                sample_count=sample_count,
            )
            if result is None:
                raise RuntimeError("Explicit experiment override unexpectedly remained disabled.")
            comparisons[str(sample_count)] = result
        repetitions.append(
            {
                "rep_index": rep_index,
                "start_frame": int(frame_indices[start]),
                "end_frame": int(frame_indices[end]),
                "segment_frames": int(end - start + 1),
                "duration_seconds": float(timestamps[end] - timestamps[start]),
                "current_v3_raw_probability": current_probability,
                "current_v3_calibrated_probability": (
                    None
                    if current_probability is None
                    else orchestrator.robustness_policy.calibrate(current_probability)
                ),
                "current_v3_latency_ms": current_latency,
                "current_raw_decision": safe.get("raw_model_decision"),
                "current_safe_status": safe.get("assessment"),
                "comparisons": comparisons,
            }
        )

    summaries: dict[str, Any] = {}
    for sample_count in experiment.config.comparison_sample_counts:
        key = str(sample_count)
        rows = [rep["comparisons"][key] for rep in repetitions]
        raw_rows = [{"raw": row["raw"]} for row in rows]
        summaries[key] = {
            "sample_count": sample_count,
            "raw_mean_pass": sum(row["raw"]["mean_decision"] == "PASS" for row in rows),
            "raw_mean_fail": sum(row["raw"]["mean_decision"] == "FAIL" for row in rows),
            "raw_median_pass": sum(row["raw"]["median_decision"] == "PASS" for row in rows),
            "raw_median_fail": sum(row["raw"]["median_decision"] == "FAIL" for row in rows),
            "calibrated_mean_pass": sum(
                row["calibrated"]["mean_decision"] == "PASS" for row in rows
            ),
            "calibrated_mean_fail": sum(
                row["calibrated"]["mean_decision"] == "FAIL" for row in rows
            ),
            "provisional_score": provisional_score(raw_rows),
            "mean_probability_std": (
                None if not rows else float(mean(row["raw"]["std_probability"] for row in rows))
            ),
            "mean_total_latency_ms": (
                None if not rows else float(mean(row["total_latency_ms"] for row in rows))
            ),
            "mean_inference_latency_ms": (
                None if not rows else float(mean(row["inference_latency_ms"] for row in rows))
            ),
        }
    current_statuses = [row["current_safe_status"] for row in repetitions]
    return {
        "video": str(path.resolve()),
        "source": "local_unlabeled_video",
        "approximate_view": view,
        "view_evidence": "filename_only",
        "correctness_ground_truth": False,
        "error_ground_truth": False,
        **metadata,
        "boundary_latency_ms": float(boundary_latency),
        "detected_repetitions": len(repetitions),
        "current_safe_mode": {
            "pass": sum(value == "PASS" for value in current_statuses),
            "fail": sum(value == "FAIL" for value in current_statuses),
            "needs_review": sum(value == "NEEDS_REVIEW" for value in current_statuses),
            "score": None,
        },
        "averaged_summaries": summaries,
        "repetitions": repetitions,
    }


def write_flat_csv(path: Path, videos: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for video in videos:
        for rep in video["repetitions"]:
            for count, comparison in rep["comparisons"].items():
                rows.append(
                    {
                        "video": video["video"],
                        "approximate_view": video["approximate_view"],
                        "rep_index": rep["rep_index"],
                        "start_frame": rep["start_frame"],
                        "end_frame": rep["end_frame"],
                        "current_v3_raw_probability": rep["current_v3_raw_probability"],
                        "current_v3_latency_ms": rep["current_v3_latency_ms"],
                        "current_safe_status": rep["current_safe_status"],
                        "sample_count": count,
                        "window_probabilities": json.dumps(comparison["raw"]["probabilities"]),
                        "mean_probability": comparison["raw"]["mean_probability"],
                        "median_probability": comparison["raw"]["median_probability"],
                        "std_probability": comparison["raw"]["std_probability"],
                        "min_probability": comparison["raw"]["min_probability"],
                        "max_probability": comparison["raw"]["max_probability"],
                        "mean_decision": comparison["raw"]["mean_decision"],
                        "median_decision": comparison["raw"]["median_decision"],
                        "calibrated_mean_probability": comparison["calibrated"]["mean_probability"],
                        "calibrated_mean_decision": comparison["calibrated"]["mean_decision"],
                        "total_latency_ms": comparison["total_latency_ms"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="*", default=list(DEFAULT_VIDEOS))
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "squat_ai_v4" / "averaged_inference",
    )
    args = parser.parse_args()
    videos = [path if path.is_absolute() else ROOT / path for path in args.videos]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {missing}")
    if not args.pose_model.is_file():
        raise FileNotFoundError(args.pose_model)

    orchestrator = SquatAIExperimentalOrchestrator(
        SquatAIMVPConfig.load(), allow_partial=False
    )
    config = AveragedCorrectnessConfig.load().with_enabled(True)
    experiment = AveragedCorrectnessExperiment(config, orchestrator.robustness_policy)
    try:
        results = [
            analyze_video(path, args.pose_model, orchestrator, experiment)
            for path in videos
        ]
    finally:
        orchestrator.close()

    payload = {
        "experiment": "Squat Averaged Correctness Inference A/B",
        "ground_truth_available": False,
        "accuracy_claim_allowed": False,
        "checkpoint_changed": False,
        "active_behavior_changed": False,
        "feature_flag_default": False,
        "raw_threshold": config.raw_threshold,
        "calibrated_threshold": config.calibrated_threshold,
        "sample_counts": list(config.comparison_sample_counts),
        "default_candidate_sample_count": config.default_sample_count,
        "crop_fraction": config.crop_fraction,
        "target_frames": config.target_frames,
        "videos": results,
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    write_json(output_dir / "averaged_inference_results.json", payload)
    write_flat_csv(output_dir / "per_rep_comparison.csv", results)
    print(
        json.dumps(
            {
                "status": "ok",
                "videos": len(results),
                "repetitions": sum(value["detected_repetitions"] for value in results),
                "output": str(output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
