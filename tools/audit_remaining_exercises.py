"""Lightweight metadata/label audit for Push-up, Pull-up, and Plank MVP gates."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.adapters.rehab24_pushup import Rehab24PushupAdapter


DATA = ROOT / "datasets"


def video_metadata(folder: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(folder.rglob("*.mp4")) if folder.is_dir() else []:
        capture = cv2.VideoCapture(str(path)); opened = capture.isOpened()
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0; capture.release()
        rows.append({"path": str(path.resolve()), "relative_path": str(path.relative_to(DATA)), "opened": opened, "frames": frames, "fps": fps, "duration_seconds": frames / fps if fps > 0 else None})
    return rows


def physical_class_counts() -> Counter[str]:
    path = DATA / "external/physical_exercise_recognition/labels.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return Counter(str(row.get("class") or row.get("label") or row.get("exercise")) for row in csv.DictReader(stream))


def pushup_audit() -> dict[str, object]:
    samples = Rehab24PushupAdapter(DATA / "external").samples(); unique = {value.pair_id: value for value in samples}.values()
    durations = np.asarray([value.end_frame - value.start_frame + 1 for value in unique], dtype=float)
    database = DATA / "external/PushUpDatabase"
    arrays = []
    for path in sorted(database.rglob("*.npy")):
        value = np.load(path, allow_pickle=False)
        arrays.append({"path": str(path.resolve()), "shape": list(value.shape), "dtype": str(value.dtype), "minimum": float(value.min()), "maximum": float(value.max()), "interpretation": "numeric pose sequence; folder supplies sequence-level correctness, not a label array"})
    return {
        "scientifically_supported_scope": "REHAB24 Ex3 table/incline Push-up only",
        "rehab24_ex3": {
            "subjects": sorted({value.subject_id for value in unique}, key=int), "subject_count": len({value.subject_id for value in unique}),
            "videos": len({value.video_path for value in samples}), "repetitions": len(list(unique)),
            "correct": sum(value.correctness == 1 for value in unique), "incorrect": sum(value.correctness == 0 for value in unique),
            "orientation_raw": dict(Counter(value.orientation_raw for value in unique)),
            "duration_frames": {"min": float(durations.min()), "mean": float(durations.mean()), "median": float(np.median(durations)), "p90": float(np.percentile(durations, 90)), "p95": float(np.percentile(durations, 95)), "max": float(durations.max())},
            "segmentation_gt": True, "correctness_gt": True, "detailed_error_gt": False,
        },
        "pushup_database": {"correct_videos": len(list((database / "Correct sequence").glob("*.mp4"))), "wrong_videos": len(list((database / "Wrong sequence").glob("*.mp4"))), "subject_ids": False, "rep_boundaries": False, "arrays": arrays, "limitation": "sequence folder labels exist but subject/source independence is undocumented"},
        "local_floor_smoke_videos": video_metadata(DATA / "pushup") + video_metadata(DATA / "easy_pushup"),
        "physical_exercise_recognition_pushup_samples": physical_class_counts().get("push_up", 0),
    }


def pullup_audit() -> dict[str, object]:
    videos = video_metadata(DATA / "raw/pull Up")
    return {"videos": videos, "video_count": len(videos), "subjects": None, "segmentation_gt": False, "correctness_gt": False, "detailed_error_gt": False, "physical_exercise_recognition_pullup_samples": physical_class_counts().get("pull_up", 0), "supported_mvp": "existing rule-based runtime plus truthful learned-assessment unavailable contract", "limitation": "No filename, manifest, or sidecar provides trustworthy repetition or correctness labels."}


def plank_audit() -> dict[str, object]:
    marching = video_metadata(DATA / "marching_plank"); cross = video_metadata(DATA / "cross_knee_plank"); raw = video_metadata(DATA / "raw/plank")
    return {"product_primary_variation": "marching_plank", "static_hold_claim_supported": False, "marching_plank_videos": marching, "cross_knee_plank_videos": cross, "raw_plank_videos": raw, "total_videos": len(marching) + len(cross) + len(raw), "subjects": None, "temporal_boundaries_gt": False, "correctness_gt": False, "detailed_error_gt": False, "supported_mvp": "existing Marching Plank rule-based leg-lift runtime; learned assessment unavailable", "limitation": "Static-hold and dynamic variations cannot be merged without labels."}


def write(name: str, payload: dict[str, object]) -> None:
    output = ROOT / "results" / f"{name}_ai_v1" / "data_summary" / "data_audit.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    write("pushup", pushup_audit()); write("pullup", pullup_audit()); write("plank", plank_audit())
    print(json.dumps({"pushup": "written", "pullup": "written", "plank": "written"}))


if __name__ == "__main__": main()
