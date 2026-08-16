from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.tools.audit_external_datasets import discover_dataset_roots
from datasets.adapters.physical_exercise_recognition_adapter import (
    PhysicalExerciseRecognitionAdapter,
)
from preprocessing.landmark_selector import H36M_EDGES


MEDIAPIPE_EDGES: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (0, 11), (0, 12),
)
PERCENTILES = (1, 5, 50, 95, 99)


def axis_statistics(values: np.ndarray) -> dict[str, float | dict[str, float]]:
    """Return finite scalar diagnostics for one coordinate channel."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan"), "percentiles": {}}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "percentiles": {
            str(value): float(np.percentile(finite, value)) for value in PERCENTILES
        },
    }


def _plot_skeleton(
    coordinates: np.ndarray,
    edges: Iterable[tuple[int, int]],
    title: str,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    for left, right in edges:
        axis.plot(
            [coordinates[left, 0], coordinates[right, 0]],
            [coordinates[left, 1], coordinates[right, 1]],
            color="#2563eb",
            linewidth=2,
        )
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c="#dc2626", s=22)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_trajectory_frames(
    coordinates: np.ndarray, frame_indices: np.ndarray, title: str, destination: Path
) -> None:
    selected = np.linspace(0, len(coordinates) - 1, min(6, len(coordinates)), dtype=int)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, index in zip(axes.flat, selected):
        frame = coordinates[index]
        for left, right in H36M_EDGES:
            axis.plot(
                [frame[left, 0], frame[right, 0]],
                [frame[left, 1], frame[right, 1]],
                color="#2563eb",
                linewidth=1.5,
            )
        axis.scatter(frame[:, 0], frame[:, 1], c="#dc2626", s=12)
        axis.set_title(f"frame {int(frame_indices[index])}")
        axis.set_aspect("equal", adjustable="datalim")
        axis.invert_yaxis()
        axis.grid(alpha=0.15)
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _plot_confidence(
    confidence: np.ndarray,
    frame_indices: np.ndarray,
    observed_mask: np.ndarray,
    outlier_mask: np.ndarray,
    title: str,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(frame_indices, confidence.mean(axis=1), label="mean joint confidence")
    axis.plot(frame_indices, confidence.min(axis=1), label="minimum joint confidence", alpha=0.8)
    if np.any(~observed_mask):
        axis.scatter(frame_indices[~observed_mask], confidence.mean(axis=1)[~observed_mask], label="interpolated", c="#f59e0b")
    if np.any(outlier_mask):
        axis.scatter(frame_indices[outlier_mask], confidence.mean(axis=1)[outlier_mask], label="outlier", c="#dc2626")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title(title)
    axis.set_xlabel("frame_order")
    axis.set_ylabel("confidence")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def generate_visualizations(
    adapter: PhysicalExerciseRecognitionAdapter,
    cache_manifest: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Save raw/mapped/normalized skeleton evidence for every exercise class."""

    result: dict[str, Any] = {}
    for label in adapter.class_vocabulary:
        row = cache_manifest[cache_manifest["exercise_label"] == label].sort_values("video_id").iloc[0]
        video_id = int(row["video_id"])
        segment = max(adapter.iter_processed_segments(video_id), key=lambda value: len(value["frame_indices"]))
        middle = len(segment["frame_indices"]) // 2
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        raw_path = class_dir / "01_raw_mediapipe33.png"
        mapped_path = class_dir / "02_raw_h36m17.png"
        normalized_path = class_dir / "03_normalized_h36m17.png"
        trajectory_path = class_dir / "04_normalized_trajectory_frames.png"
        confidence_path = class_dir / "05_confidence_timeline.png"
        _plot_skeleton(segment["geometry"][middle, :, :2], MEDIAPIPE_EDGES, f"{label}: raw MediaPipe-33", raw_path)
        _plot_skeleton(segment["raw_h36m"][middle, :, :2], H36M_EDGES, f"{label}: raw H36M-17", mapped_path)
        _plot_skeleton(segment["motionbert_input"][middle, :, :2], H36M_EDGES, f"{label}: normalized H36M-17", normalized_path)
        _plot_trajectory_frames(segment["motionbert_input"][..., :2], segment["frame_indices"], f"{label}: temporal sanity", trajectory_path)
        _plot_confidence(
            segment["motionbert_input"][..., 2], segment["frame_indices"],
            segment["observed_mask"], segment["outlier_mask"],
            f"{label}: confidence policy", confidence_path,
        )
        result[label] = {
            "video_id": str(video_id),
            "split": str(row["split"]),
            "frame_range": [int(segment["frame_indices"][0]), int(segment["frame_indices"][-1])],
            "raw_xyz": {
                "x": axis_statistics(segment["geometry"][..., 0]),
                "y": axis_statistics(segment["geometry"][..., 1]),
                "z": axis_statistics(segment["geometry"][..., 2]),
            },
            "normalized_xy": {
                "x": axis_statistics(segment["motionbert_input"][..., 0]),
                "y": axis_statistics(segment["motionbert_input"][..., 1]),
            },
            "sequence_scale": float(segment["sequence_scale"]),
            "interpolated_frames": segment["frame_indices"][~segment["observed_mask"]].astype(int).tolist(),
            "outlier_frames": segment["frame_indices"][segment["outlier_mask"]].astype(int).tolist(),
            "near_zero_scale_frames": segment["frame_indices"][segment["near_zero_scale_mask"]].astype(int).tolist(),
            "plots": [str(path.resolve()) for path in (raw_path, mapped_path, normalized_path, trajectory_path, confidence_path)],
        }
    return result


def build_numeric_diagnostics(
    adapter: PhysicalExerciseRecognitionAdapter,
    split_manifest: pd.DataFrame,
    cache_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate unique-frame coordinate diagnostics by split and exercise class."""

    split_map = {
        str(row.video_id): (str(row.split), str(row.exercise_label))
        for row in split_manifest.itertuples(index=False)
    }
    window_counts = {
        (str(split), str(label)): int(group["num_windows"].sum())
        for (split, label), group in cache_manifest.groupby(["split", "exercise_label"])
    }
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "videos": set(), "raw": [], "normalized": [], "confidence": [],
            "interpolated": 0, "outliers": 0, "clipped": 0, "near_zero": 0,
            "scales": [], "joint_std": [],
        }
    )
    for video_id in sorted(int(value) for value in adapter.labels["vid_id"]):
        split, label = split_map[str(video_id)]
        group = groups[(split, label)]
        group["videos"].add(str(video_id))
        for segment in adapter.iter_processed_segments(video_id):
            group["raw"].append(segment["geometry"][..., :3].reshape(-1, 3))
            group["normalized"].append(segment["motionbert_input"][..., :2].reshape(-1, 2))
            group["confidence"].append(segment["motionbert_input"][..., 2].reshape(-1))
            group["interpolated"] += int((~segment["observed_mask"]).sum())
            group["outliers"] += int(segment["outlier_mask"].sum())
            group["clipped"] += int(segment["clipped_mask"].sum())
            group["near_zero"] += int(segment["near_zero_scale_mask"].sum())
            group["scales"].append(float(segment["sequence_scale"]))
            group["joint_std"].extend(
                segment["motionbert_input"][..., :2].std(axis=1).mean(axis=1).tolist()
            )

    report_groups: dict[str, Any] = {}
    warnings: list[str] = []
    normalized_class_std: dict[str, list[float]] = defaultdict(list)
    class_scale_medians: dict[str, list[float]] = defaultdict(list)
    for (split, label), values in sorted(groups.items()):
        raw = np.concatenate(values["raw"], axis=0)
        normalized = np.concatenate(values["normalized"], axis=0)
        confidence = np.concatenate(values["confidence"], axis=0)
        key = f"{split}:{label}"
        finite_percentage = float(
            100.0 * (np.isfinite(raw).sum() + np.isfinite(normalized).sum() + np.isfinite(confidence).sum())
            / (raw.size + normalized.size + confidence.size)
        )
        joint_std = float(np.mean(values["joint_std"]))
        frame_count = int(len(confidence) // 17)
        outlier_percentage = 100.0 * values["outliers"] / max(1, frame_count)
        clipped_percentage = 100.0 * values["clipped"] / max(1, frame_count)
        normalized_std = float(normalized.std())
        normalized_class_std[label].append(normalized_std)
        class_scale_medians[label].extend(values["scales"])
        if finite_percentage != 100.0:
            warnings.append(f"{key}: non-finite values detected")
        if float(np.abs(normalized).max()) > 4.00001:
            warnings.append(f"{key}: normalized clipping contract exceeded")
        if joint_std <= 1e-4:
            warnings.append(f"{key}: normalized joints collapsed")
        if outlier_percentage > 10.0:
            warnings.append(f"{key}: {outlier_percentage:.2f}% frames flagged as outliers")
        if clipped_percentage > 5.0:
            warnings.append(f"{key}: {clipped_percentage:.2f}% frames required coordinate clipping")
        report_groups[key] = {
            "split": split,
            "class": label,
            "number_of_videos": len(values["videos"]),
            "number_of_windows": window_counts.get((split, label), 0),
            "raw": {axis: axis_statistics(raw[:, index]) for index, axis in enumerate(("x", "y", "z"))},
            "normalized": {axis: axis_statistics(normalized[:, index]) for index, axis in enumerate(("x", "y"))},
            "percentage_finite": finite_percentage,
            "percentage_confidence_0": float(100.0 * np.isclose(confidence, 0.0).mean()),
            "percentage_confidence_0_5": float(100.0 * np.isclose(confidence, 0.5).mean()),
            "percentage_confidence_1": float(100.0 * np.isclose(confidence, 1.0).mean()),
            "percentage_other_low_confidence": float(
                100.0 * (~np.isclose(confidence, 0.0) & ~np.isclose(confidence, 0.5) & ~np.isclose(confidence, 1.0)).mean()
            ),
            "interpolated_frame_count": values["interpolated"],
            "outlier_frame_count": values["outliers"],
            "outlier_frame_percentage": outlier_percentage,
            "clipped_frame_count": values["clipped"],
            "clipped_frame_percentage": clipped_percentage,
            "near_zero_scale_count": values["near_zero"],
            "sequence_scale": axis_statistics(np.asarray(values["scales"])),
            "mean_per_frame_joint_xy_std": joint_std,
        }

    class_summary: dict[str, Any] = {}
    scale_medians = {}
    for label in adapter.class_vocabulary:
        scale_values = np.asarray(class_scale_medians[label], dtype=np.float64)
        scale_median = float(np.median(scale_values))
        scale_medians[label] = scale_median
        class_summary[label] = {
            "raw_sequence_scale_median": scale_median,
            "normalized_xy_std_mean_across_splits": float(np.mean(normalized_class_std[label])),
        }
    positive_scales = [value for value in scale_medians.values() if value > 0]
    if positive_scales and max(positive_scales) / min(positive_scales) > 10.0:
        warnings.append("Raw median body scale differs by more than 10x between classes.")
    normalized_stds = [value["normalized_xy_std_mean_across_splits"] for value in class_summary.values()]
    if min(normalized_stds) <= 0 or max(normalized_stds) / min(normalized_stds) > 4.0:
        warnings.append("Normalized XY scale differs radically between classes.")
    return {
        "preprocessing_version": adapter.PREPROCESSING_VERSION,
        "h36m_mapping_version": adapter.H36M_MAPPING_VERSION,
        "coordinate_contract": {
            "root_joint": 0,
            "scale_method": adapter._coordinate_normalizer.SCALE_METHOD,
            "clipping_method": adapter._coordinate_normalizer.CLIPPING_METHOD,
            "confidence_policy": adapter._coordinate_normalizer.CONFIDENCE_POLICY,
            "global_train_statistics_used": False,
        },
        "groups": report_groups,
        "class_scale_summary": class_summary,
        "sanity_pass": not warnings,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual and numeric Physical skeleton sanity checks.")
    parser.add_argument("--external-root", type=Path, default=Path("datasets/external"))
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/window_cache/physical_exercise_recognition_v4"))
    parser.add_argument("--split-manifest", type=Path, default=Path("datasets/splits/physical_exercise_recognition_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_skeleton_sanity"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    resolve = lambda path: path if path.is_absolute() else project_root / path
    external_root = resolve(args.external_root)
    cache_dir = resolve(args.cache_dir)
    output_dir = resolve(args.output_dir)
    split_path = resolve(args.split_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = PhysicalExerciseRecognitionAdapter(
        discover_dataset_roots(external_root)["physical_exercise_recognition"]
    )
    cache_manifest = pd.read_csv(cache_dir / "cache_manifest.csv", dtype={"video_id": str})
    cache_statistics = json.loads(
        (cache_dir / "cache_statistics.json").read_text(encoding="utf-8")
    )
    if cache_statistics.get("preprocessing_version") != adapter.PREPROCESSING_VERSION:
        raise RuntimeError(
            "Cache preprocessing version does not match the active adapter: "
            f"{cache_statistics.get('preprocessing_version')!r} != "
            f"{adapter.PREPROCESSING_VERSION!r}"
        )
    split_manifest = pd.read_csv(split_path, dtype={"video_id": str})
    diagnostics = build_numeric_diagnostics(adapter, split_manifest, cache_manifest)
    samples = generate_visualizations(adapter, cache_manifest, output_dir)
    (output_dir / "numeric_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "sample_statistics.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "sanity_pass": diagnostics["sanity_pass"],
        "warnings": diagnostics["warnings"],
        "classes_visualized": list(samples),
        "output_dir": str(output_dir.resolve()),
    }, indent=2))
    if not diagnostics["sanity_pass"]:
        raise SystemExit("Coordinate sanity validation failed; pilot training was not started.")


if __name__ == "__main__":
    main()
