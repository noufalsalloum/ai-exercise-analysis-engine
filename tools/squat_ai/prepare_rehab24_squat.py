"""Build versioned REHAB24 Squat pose and repetition caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adapters.rehab24 import Rehab24RepetitionSample, Rehab24SquatAdapter
from datasets.adapters.rehab24_split import balanced_subject_split
from input_sources.pose_stream import PoseStreamProcessor
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector


CACHE_VERSION = "rehab24_squat_mp33_h36m_v4_rep60_v1"
TARGET_REP_FRAMES = 60
MAX_SHORT_GAP = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fill_missing_pose(raw: np.ndarray, detected: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Interpolate coordinates; only bounded gaps <=3 get confidence 0.5."""

    if raw.ndim != 3 or raw.shape[1:] != (33, 4) or detected.shape != (len(raw),):
        raise ValueError("Expected raw (T,33,4) and detected (T,).")
    if not detected.any():
        raise ValueError("Video contains no detected pose frame.")
    output = raw.copy(); indices = np.arange(len(raw)); observed = indices[detected]
    for joint in range(33):
        for channel in range(3):
            output[:, joint, channel] = np.interp(indices, observed, raw[detected, joint, channel])
    output[~detected, :, 3] = 0.0
    short_frames = 0; invalid_frames = 0
    starts = np.flatnonzero((~detected) & np.r_[True, detected[:-1]])
    for start in starts:
        end = start
        while end + 1 < len(detected) and not detected[end + 1]:
            end += 1
        length = end - start + 1
        if start > 0 and end + 1 < len(detected) and length <= MAX_SHORT_GAP:
            endpoint = np.minimum(output[start - 1, :, 3], output[end + 1, :, 3])
            output[start:end + 1, :, 3] = np.minimum(endpoint, 0.5)
            short_frames += int(length)
        else:
            invalid_frames += int(length)
    output[..., 3] = np.clip(output[..., 3], 0.0, 1.0)
    if not np.isfinite(output).all():
        raise FloatingPointError("Interpolation produced non-finite pose values.")
    return output.astype(np.float32), {"detected": int(detected.sum()), "short_gap": short_frames, "invalid": invalid_frames}


def resample_sequence(values: np.ndarray, target: int = TARGET_REP_FRAMES) -> np.ndarray:
    """Resample the complete sequence while preserving both endpoints."""

    if len(values) == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if len(values) == 1:
        return np.repeat(values, target, axis=0)
    old = np.linspace(0.0, 1.0, len(values)); new = np.linspace(0.0, 1.0, target)
    flat = values.reshape(len(values), -1); result = np.empty((target, flat.shape[1]), np.float32)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(new, old, flat[:, column])
    return result.reshape(target, *values.shape[1:])


def extract_video(path: Path, pose_model: Path, max_frames: int | None) -> tuple[np.ndarray, np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)); expected = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    limit = min(expected, max_frames) if max_frames else expected
    raw = np.full((limit, 33, 4), np.nan, np.float32); detected = np.zeros(limit, bool)
    processor = PoseStreamProcessor(pose_model)
    try:
        index = 0
        while index < limit:
            ok, frame = capture.read()
            if not ok: break
            landmarks = processor.process(frame, index / fps)
            if landmarks is not None: raw[index] = landmarks; detected[index] = True
            index += 1
    finally:
        processor.close(); capture.release()
    if max_frames is None and index != expected:
        raise ValueError(f"Decoded {index}/{expected} frames from {path.name}")
    return raw[:index], detected[:index], fps


def write_metadata(project: Path, adapter: Rehab24SquatAdapter, samples: list[Rehab24RepetitionSample], assignment: dict[str, str], evidence: dict[str, object]) -> Path:
    split_dir = project / "splits"; split_dir.mkdir(parents=True, exist_ok=True)
    names = {"train": "rehab24_squat_train_subjects.json", "validation": "rehab24_squat_val_subjects.json", "test": "rehab24_squat_test_subjects.json"}
    for split, filename in names.items():
        subjects = sorted((subject for subject, value in assignment.items() if value == split), key=int)
        adapter.write_subject_file(split_dir / filename, split, subjects,
                                   {"seed": evidence["seed"], "method": evidence["method"],
                                    "statistics": evidence["splits"][split], "assignment_sha256": evidence["assignment_sha256"]})
    data_dir = project / "results/squat_ai/data"; data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "repetition_manifest.csv"; adapter.write_manifest(manifest, assignment)
    rows = [{"split": split, **stats} for split, stats in evidence["splits"].items()]
    with (data_dir / "split_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=Path("datasets/external"))
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/window_cache/rehab24_squat_v1"))
    parser.add_argument("--pose-model", type=Path, default=Path(r"C:\MediaPipe\pose_landmarker_full.task"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(); project = PROJECT_ROOT
    external = args.external_root if args.external_root.is_absolute() else project / args.external_root
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else project / args.cache_dir
    full_dir = cache_dir / "full_videos"; rep_dir = cache_dir / "repetitions"
    full_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    adapter = Rehab24SquatAdapter(external); samples = adapter.samples()
    assignment, evidence = balanced_subject_split(samples, args.seed)
    manifest = write_metadata(project, adapter, samples, assignment, evidence)
    selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}}); normalizer = H36MCoordinateNormalizer()
    by_path: dict[str, list[Rehab24RepetitionSample]] = defaultdict(list)
    for sample in samples: by_path[sample.video_path].append(sample)
    selected = sorted(by_path)
    if args.max_videos is not None: selected = selected[:args.max_videos]
    diagnostics = Counter(); video_rows = []; cached_samples = 0; started = perf_counter()
    for number, path_text in enumerate(selected, 1):
        path = Path(path_text); stem = path.stem.replace("-30fps-transposed", "_cam18").replace("-30fps", "_cam17")
        full_path = full_dir / f"{stem}.npz"
        if full_path.is_file() and args.max_frames is None:
            with np.load(full_path, allow_pickle=False) as archive:
                filled = np.asarray(archive["landmarks"], np.float32); detected = np.asarray(archive["detected_mask"], bool)
                fps = float(archive["fps"]); gap_stats = json.loads(str(archive["gap_diagnostics_json"]))
        else:
            raw, detected, fps = extract_video(path, args.pose_model, args.max_frames)
            filled, gap_stats = fill_missing_pose(raw, detected)
            full_h36m = selector.to_h36m_17(filled); full_motion, _ = normalizer.normalize(full_h36m)
            if args.max_frames is None:
                np.savez_compressed(full_path, landmarks=filled, detected_mask=detected,
                                    motionbert_input=full_motion, fps=np.asarray(fps), cache_version=np.asarray(CACHE_VERSION),
                                    gap_diagnostics_json=np.asarray(json.dumps(gap_stats, sort_keys=True)))
        diagnostics.update(gap_stats)
        video_rows.append({"video_path": str(path), "frames": len(filled), "detected_frames": int(detected.sum()), "fps": fps, "cache_path": str(full_path.resolve())})
        for sample in by_path[path_text]:
            if sample.end_frame > len(filled):
                if args.max_frames is not None: continue
                raise ValueError(f"Sample exceeds decoded frames: {sample.sample_id}")
            segment = filled[sample.start_frame - 1:sample.end_frame]
            h36m = selector.to_h36m_17(segment); normalized, norm = normalizer.normalize(h36m)
            resampled = resample_sequence(normalized); resampled[..., 2] = np.clip(resampled[..., 2], 0.0, 1.0)
            temporal_mask = resampled[..., 2].mean(axis=1) > 0.01
            if not temporal_mask.any() or not np.isfinite(resampled).all(): raise ValueError(sample.sample_id)
            metadata = {**sample.to_dict(), "split": assignment[sample.subject_id], "cache_version": CACHE_VERSION,
                        "preprocessing_version": adapter.PREPROCESSING_VERSION, "h36m_mapping_version": adapter.H36M_MAPPING_VERSION,
                        "temporal_strategy": "full repetition endpoint-preserving linear resample to 60",
                        "coordinate_normalization": normalizer.SCALE_METHOD, "confidence_policy": normalizer.CONFIDENCE_POLICY,
                        "source_frames": len(segment), "sequence_scale": norm.sequence_scale}
            np.savez_compressed(rep_dir / f"{sample.sample_id}.npz", motionbert_input=resampled,
                                temporal_mask=temporal_mask, correctness=np.asarray(sample.correctness, np.int64),
                                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
            cached_samples += 1
        print(f"cached {number}/{len(selected)} {path.name} frames={len(filled)} detected={int(detected.sum())}", flush=True)
    if args.max_videos is None and cached_samples != len(samples):
        raise ValueError(f"Expected {len(samples)} caches, got {cached_samples}")
    summary = {"cache_version": CACHE_VERSION, "preprocessing_version": adapter.PREPROCESSING_VERSION,
               "h36m_mapping_version": adapter.H36M_MAPPING_VERSION, "target_repetition_frames": TARGET_REP_FRAMES,
               "source_camera_samples": len(samples), "cached_camera_samples": cached_samples, "cached_full_videos": len(selected),
               "split_manifest": str(manifest.resolve()), "split_manifest_sha256": sha256(manifest), "split_evidence": evidence,
               "gap_diagnostics": dict(diagnostics), "video_metadata": video_rows,
               "elapsed_seconds": perf_counter() - started, "max_videos": args.max_videos, "max_frames": args.max_frames}
    output = project / "results/squat_ai/data/preprocessing_summary.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (cache_dir / "cache_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "cached_samples": cached_samples, "cached_videos": len(selected),
                      "split": evidence["splits"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
