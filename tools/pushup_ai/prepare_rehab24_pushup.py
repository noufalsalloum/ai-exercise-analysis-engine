"""Build preprocessing-v4 caches for REHAB24 Ex3 table/incline Push-ups."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adapters.rehab24 import Rehab24RepetitionSample
from datasets.adapters.rehab24_pushup import Rehab24PushupAdapter
from datasets.adapters.rehab24_pushup_split import balanced_pushup_subject_split
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from tools.squat_ai.prepare_rehab24_squat import extract_video, fill_missing_pose, resample_sequence, sha256


CACHE_VERSION = "rehab24_ex3_table_pushup_mp33_h36m_v4_rep60_v1"


def write_metadata(project: Path, adapter: Rehab24PushupAdapter, samples: list[Rehab24RepetitionSample], assignment: dict[str, str], evidence: dict[str, object]) -> Path:
    split_dir = project / "splits"; split_dir.mkdir(parents=True, exist_ok=True)
    names = {"train": "rehab24_pushup_train_subjects.json", "validation": "rehab24_pushup_val_subjects.json", "test": "rehab24_pushup_test_subjects.json"}
    for split, filename in names.items():
        subjects = sorted((subject for subject, value in assignment.items() if value == split), key=int)
        adapter.write_subject_file(split_dir / filename, split, subjects, {"seed": evidence["seed"], "method": evidence["method"], "statistics": evidence["splits"][split], "assignment_sha256": evidence["assignment_sha256"], "exercise_scope": "REHAB24 Ex3 table/incline Push-up only"})
    output = project / "results" / "pushup_ai_v1" / "data_summary"; output.mkdir(parents=True, exist_ok=True)
    manifest = output / "repetition_manifest.csv"; adapter.write_manifest(manifest, assignment)
    rows = [{"split": split, **stats} for split, stats in evidence["splits"].items()]
    with (output / "split_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return manifest


def build_cache(project: Path, external: Path, cache_dir: Path, pose_model: Path, seed: int = 42) -> dict[str, object]:
    """Extract twenty physical camera videos without modifying source data."""
    full_dir = cache_dir / "full_videos"; rep_dir = cache_dir / "repetitions"
    full_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    adapter = Rehab24PushupAdapter(external); samples = adapter.samples()
    assignment, evidence = balanced_pushup_subject_split(samples, seed)
    manifest = write_metadata(project, adapter, samples, assignment, evidence)
    selector = LandmarkSelector({"landmarks": {"selected_landmarks": []}}); normalizer = H36MCoordinateNormalizer()
    by_path: dict[str, list[Rehab24RepetitionSample]] = defaultdict(list)
    for sample in samples: by_path[sample.video_path].append(sample)
    diagnostics: Counter[str] = Counter(); video_rows: list[dict[str, object]] = []; cached_samples = 0; started = perf_counter()
    for number, path_text in enumerate(sorted(by_path), 1):
        path = Path(path_text); stem = path.stem.replace("-30fps-transposed", "_cam18").replace("-30fps", "_cam17")
        full_path = full_dir / f"{stem}.npz"
        if full_path.is_file():
            with np.load(full_path, allow_pickle=False) as archive:
                filled = np.asarray(archive["landmarks"], np.float32); detected = np.asarray(archive["detected_mask"], bool)
                fps = float(archive["fps"]); gap_stats = json.loads(str(archive["gap_diagnostics_json"]))
        else:
            raw, detected, fps = extract_video(path, pose_model, None); filled, gap_stats = fill_missing_pose(raw, detected)
            full_h36m = selector.to_h36m_17(filled); full_motion, _ = normalizer.normalize(full_h36m)
            np.savez_compressed(full_path, landmarks=filled, detected_mask=detected, motionbert_input=full_motion, fps=np.asarray(fps), cache_version=np.asarray(CACHE_VERSION), gap_diagnostics_json=np.asarray(json.dumps(gap_stats, sort_keys=True)))
        diagnostics.update(gap_stats); video_rows.append({"video_path": str(path), "frames": len(filled), "detected_frames": int(detected.sum()), "fps": fps})
        for sample in by_path[path_text]:
            segment = filled[sample.start_frame - 1 : sample.end_frame]
            h36m = selector.to_h36m_17(segment); normalized, norm = normalizer.normalize(h36m); resampled = resample_sequence(normalized)
            resampled[..., 2] = np.clip(resampled[..., 2], 0.0, 1.0); mask = resampled[..., 2].mean(axis=1) > 0.01
            if not mask.any() or not np.isfinite(resampled).all(): raise ValueError(sample.sample_id)
            metadata = {**sample.to_dict(), "split": assignment[sample.subject_id], "cache_version": CACHE_VERSION, "preprocessing_version": adapter.PREPROCESSING_VERSION, "exercise_scope": "table/incline Push-up only", "source_frames": len(segment), "sequence_scale": norm.sequence_scale}
            np.savez_compressed(rep_dir / f"{sample.sample_id}.npz", motionbert_input=resampled, temporal_mask=mask, correctness=np.asarray(sample.correctness, np.int64), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
            cached_samples += 1
        print(f"cached {number}/20 {path.name} frames={len(filled)} detected={int(detected.sum())}", flush=True)
    if cached_samples != len(samples): raise ValueError(f"Expected {len(samples)} repetition caches, got {cached_samples}")
    summary: dict[str, object] = {"cache_version": CACHE_VERSION, "exercise_scope": "REHAB24 Ex3 table/incline Push-up only", "preprocessing_version": adapter.PREPROCESSING_VERSION, "source_camera_samples": len(samples), "cached_camera_samples": cached_samples, "cached_full_videos": len(video_rows), "split_manifest": str(manifest.resolve()), "split_manifest_sha256": sha256(manifest), "split_evidence": evidence, "gap_diagnostics": dict(diagnostics), "video_metadata": video_rows, "elapsed_seconds": perf_counter() - started}
    result_dir = project / "results" / "pushup_ai_v1" / "data_summary"
    (result_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (cache_dir / "cache_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=Path("datasets/external")); parser.add_argument("--cache-dir", type=Path, default=Path("datasets/window_cache/rehab24_pushup_v1")); parser.add_argument("--pose-model", type=Path, default=Path(r"C:\MediaPipe\pose_landmarker_full.task")); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); external = args.external_root if args.external_root.is_absolute() else PROJECT_ROOT / args.external_root; cache = args.cache_dir if args.cache_dir.is_absolute() else PROJECT_ROOT / args.cache_dir
    summary = build_cache(PROJECT_ROOT, external, cache, args.pose_model, args.seed)
    print(json.dumps({"status": "ok", "samples": summary["cached_camera_samples"], "seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__": main()
