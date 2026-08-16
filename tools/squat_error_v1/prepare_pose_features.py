"""Extract cached MediaPipe/geometric features for static SquatDataset images."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.adapters.squat_dataset_adapter import SquatDatasetAdapter
from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS
from preprocessing.squat_posture_features import (
    MediaPipeImagePoseExtractor,
    POSTURE_JOINTS,
    SquatPostureFeatureExtractor,
)


CLASS_TO_INDEX = {"good": 0, "bad_back": 1, "bad_heel": 2}
EXPECTED_COUNTS = {
    ("train", "good"): 1001,
    ("train", "bad_back"): 984,
    ("train", "bad_heel"): 852,
    ("test", "good"): 310,
    ("test", "bad_back"): 338,
    ("test", "bad_heel"): 321,
}


def _decode(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _sample_id(relative_path: str) -> str:
    return "squat_error_" + hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def prepare(
    dataset_root: Path,
    output_dir: Path,
    pose_model: Path,
    max_images: int | None = None,
) -> dict[str, object]:
    """Extract every image independently and save finite arrays plus audit evidence."""

    adapter = SquatDatasetAdapter(dataset_root)
    paths = adapter.image_files
    if max_images is not None:
        paths = paths[: int(max_images)]
    extractor = SquatPostureFeatureExtractor()
    features = np.zeros((len(paths), extractor.feature_dim), dtype=np.float32)
    landmarks_cache = np.zeros((len(paths), 33, 4), dtype=np.float32)
    rows: list[dict[str, object]] = []
    resolution_counts: Counter[str] = Counter()
    readable = corrupted = detected = 0
    started = perf_counter()
    with MediaPipeImagePoseExtractor(pose_model) as pose:
        for index, path in enumerate(paths):
            relative = path.relative_to(dataset_root).as_posix()
            image = _decode(path)
            width = height = channels = 0
            landmarks = None
            if image is None:
                corrupted += 1
            else:
                readable += 1
                height, width, channels = image.shape
                resolution_counts[f"{width}x{height}"] += 1
                landmarks = pose.process(image)
            # Feature construction has no access to the class label or path.
            features[index] = extractor.extract(landmarks)
            pose_success = landmarks is not None
            if pose_success:
                assert landmarks is not None
                landmarks_cache[index] = landmarks
                detected += 1
            # Label is attached only after the label-independent feature vector exists.
            label = adapter.normalized_class(path)
            source_split = path.parent.parent.name.strip().lower()
            required_confidence = (
                landmarks[[MEDIAPIPE_LANDMARKS[name] for name in POSTURE_JOINTS], 3]
                if landmarks is not None
                else np.zeros(len(POSTURE_JOINTS), dtype=np.float32)
            )
            rows.append(
                {
                    "sample_id": _sample_id(relative),
                    "source_split": source_split,
                    "canonical_label": label,
                    "label_index": CLASS_TO_INDEX[label],
                    "image_path": str(path.resolve()),
                    "relative_path": relative,
                    "feature_index": index,
                    "readable": image is not None,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "pose_success": pose_success,
                    "mean_required_confidence": float(required_confidence.mean()),
                    "minimum_required_confidence": float(required_confidence.min()),
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(paths):
                print(
                    f"static pose extraction {index + 1}/{len(paths)} "
                    f"success={detected} failures={index + 1 - detected}",
                    flush=True,
                )
    manifest = pd.DataFrame(rows)
    if not np.isfinite(features).all() or features.shape[1] != extractor.feature_dim:
        raise FloatingPointError("Pose feature cache is non-finite or malformed.")
    if max_images is None:
        observed = {
            (str(split), str(label)): int(count)
            for (split, label), count in manifest.groupby(["source_split", "canonical_label"]).size().items()
        }
        if observed != EXPECTED_COUNTS:
            raise RuntimeError(f"SquatDataset counts changed: {observed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pose_features.npy", features)
    np.save(output_dir / "pose_landmarks.npy", landmarks_cache)
    manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
    (output_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "feature_version": extractor.VERSION,
                "feature_dim": extractor.feature_dim,
                "feature_names": list(extractor.feature_names),
                "class_to_index": CLASS_TO_INDEX,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    class_rows = []
    for (split, label), group in manifest.groupby(["source_split", "canonical_label"]):
        success = int(group["pose_success"].sum())
        class_rows.append(
            {
                "split": str(split),
                "class": str(label),
                "images": len(group),
                "pose_success": success,
                "pose_failure": len(group) - success,
                "pose_success_rate": success / len(group),
                "mean_landmark_confidence": float(group["mean_required_confidence"].mean()),
                "median_landmark_confidence": float(group["mean_required_confidence"].median()),
                "minimum_landmark_confidence_p05": float(group["minimum_required_confidence"].quantile(0.05)),
            }
        )
    class_summary = pd.DataFrame(class_rows)
    rates = class_summary["pose_success_rate"].to_numpy(float)
    summary: dict[str, object] = {
        "dataset": "SquatDataset",
        "task": "static mutually-exclusive posture error classification",
        "images": len(manifest),
        "readable_images": readable,
        "corrupted_images": corrupted,
        "rgb_channel_images": int((manifest["channels"] == 3).sum()),
        "resolution_distribution": dict(resolution_counts),
        "pose_success": detected,
        "pose_failure": len(manifest) - detected,
        "pose_success_rate": detected / max(len(manifest), 1),
        "class_pose_quality": class_rows,
        "maximum_class_success_rate_gap": float(rates.max() - rates.min()),
        "feature_version": extractor.VERSION,
        "feature_dim": extractor.feature_dim,
        "input_contract": "single_static_RGB_image_to_MediaPipe33_to_normalized_xy_confidence_geometry",
        "pose_model": str(pose_model),
        "max_decode_dimension_for_pose": 1024,
        "failed_pose_policy": "finite zero sentinel with pose_success=0; samples are not silently dropped",
        "subject_ids": None,
        "split_limitation": "Provided Train/Test only; no subject-wise claims are possible.",
        "elapsed_seconds": perf_counter() - started,
    }
    (output_dir / "pose_extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    class_summary.to_csv(output_dir / "pose_quality_by_class.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/external/SquatDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/squat_error_v1/data"))
    parser.add_argument("--pose-model", type=Path, default=Path(r"C:\MediaPipe\pose_landmarker_full.task"))
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    summary = prepare(resolve(args.dataset_root), resolve(args.output_dir), resolve(args.pose_model), args.max_images)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

