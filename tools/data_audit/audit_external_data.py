"""Comprehensive read-only audit of external exercise datasets.

No MediaPipe extraction or full video decoding is performed. Video checks use
container metadata; CSV, skeleton, and image checks are streamed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


EXERCISES = {
    1: ("Arm abduction", "upper_body_rehabilitation", "arm_abduction"),
    2: ("Arm VW", "upper_body_rehabilitation", "arm_vw"),
    3: ("Push-ups with hands on a table", "pushup", "table_pushup"),
    4: ("Leg abduction", "lower_body_rehabilitation", "leg_abduction"),
    5: ("Leg lunge", "lunge", "rehab_leg_lunge"),
    6: ("Squats", "squat", "rehab_squat"),
}


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recursive_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def human_size(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {"count": len(finite), "min": min(finite), "max": max(finite),
            "mean": statistics.fmean(finite), "median": statistics.median(finite)}


@dataclass
class VideoMetadata:
    path: str
    opened: bool
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    width: int | None
    height: int | None


def video_metadata(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return VideoMetadata(str(path), False, None, None, None, None, None)
    fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
    frames_raw = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    fps = fps_raw if fps_raw > 0 and math.isfinite(fps_raw) else None
    frames = frames_raw if frames_raw >= 0 else None
    return VideoMetadata(str(path), True, fps, frames,
                         frames / fps if fps and frames is not None else None,
                         width or None, height or None)


def grouped(rows: Iterable[dict[str, Any]], key_fn: Any) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[key_fn(row)].append(row)
    return result


def audit_rehab24(external: Path, output: Path) -> dict[str, Any]:
    with (external / "Segmentation.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        columns = list(reader.fieldnames or [])
        raw_rows = list(reader)
    parsed: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    values = {column: Counter() for column in columns}
    numeric_columns = ("repetition_number", "exercise_id", "person_id", "first_frame", "last_frame",
                       "mocap_erroneous", "lights_on", "extra_person_in_cam17",
                       "extra_person_in_cam18", "correctness")
    for row_number, row in enumerate(raw_rows, 2):
        for column in columns:
            values[column][row.get(column, "")] += 1
        item: dict[str, Any] = dict(row)
        item["_row_number"] = row_number
        malformed = []
        for column in numeric_columns:
            try:
                item[column] = int(row[column])
            except (KeyError, TypeError, ValueError):
                malformed.append(column)
        if malformed:
            issues.append({"issue": "malformed_values", "row": row_number,
                           "video_id": row.get("video_id"), "details": ",".join(malformed)})
            continue
        item["duration_frames_inclusive"] = item["last_frame"] - item["first_frame"] + 1
        parsed.append(item)

    videos: dict[tuple[str, str], VideoMetadata] = {}
    for exercise_id in EXERCISES:
        for path in sorted((external / f"Ex{exercise_id}").glob("*.mp4")):
            match = re.fullmatch(r"(PM_[0-9A-Za-z]+)-Camera(17|18)-30fps(?:-transposed)?\.mp4", path.name)
            if not match:
                issues.append({"issue": "malformed_video_filename", "row": "", "video_id": "", "details": path.name})
                continue
            videos[(match.group(1), match.group(2))] = video_metadata(path)

    exact = Counter(tuple(row.get(column, "") for column in columns) for row in raw_rows)
    for key, count in exact.items():
        if count > 1:
            issues.append({"issue": "duplicate_rows", "row": "", "video_id": key[0], "details": f"count={count}"})
    segment_duplicates = Counter((r["video_id"], r["first_frame"], r["last_frame"]) for r in parsed)
    for key, count in segment_duplicates.items():
        if count > 1:
            issues.append({"issue": "duplicate_segments", "row": "", "video_id": key[0], "details": f"{key[1]}-{key[2]}, count={count}"})

    by_video = grouped(parsed, lambda row: row["video_id"])
    for row in parsed:
        if row["first_frame"] >= row["last_frame"]:
            issues.append({"issue": "invalid_frame_order", "row": row["_row_number"], "video_id": row["video_id"], "details": f"{row['first_frame']} >= {row['last_frame']}"})
        if row["correctness"] not in (0, 1):
            issues.append({"issue": "missing_or_invalid_correctness", "row": row["_row_number"], "video_id": row["video_id"], "details": str(row["correctness"])})
        for camera in ("17", "18"):
            meta = videos.get((row["video_id"], camera))
            if meta is None:
                issues.append({"issue": "segmentation_video_missing", "row": row["_row_number"], "video_id": row["video_id"], "details": f"Camera{camera}"})
            elif meta.frame_count is not None and row["last_frame"] > meta.frame_count:
                issues.append({"issue": "segment_outside_video", "row": row["_row_number"], "video_id": row["video_id"], "details": f"Camera{camera}: end={row['last_frame']}, frame_count={meta.frame_count}"})
    for video_id, rows in by_video.items():
        ordered = sorted(rows, key=lambda row: (row["first_frame"], row["last_frame"]))
        for previous, current in zip(ordered, ordered[1:]):
            if current["first_frame"] <= previous["last_frame"]:
                issues.append({"issue": "overlapping_segments", "row": current["_row_number"], "video_id": video_id, "details": f"{previous['first_frame']}-{previous['last_frame']} overlaps {current['first_frame']}-{current['last_frame']}"})
    for (video_id, camera), meta in videos.items():
        if video_id not in by_video:
            issues.append({"issue": "video_without_segmentation", "row": "", "video_id": video_id, "details": f"Camera{camera}"})
        if not meta.opened:
            issues.append({"issue": "video_metadata_unreadable", "row": "", "video_id": video_id, "details": f"Camera{camera}"})

    for row in parsed:
        meta = videos.get((row["video_id"], "17"))
        row["duration_seconds"] = row["duration_frames_inclusive"] / meta.fps if meta and meta.fps else None

    summary_rows = []
    for exercise_id, (name, family, variation) in EXERCISES.items():
        rows = [row for row in parsed if row["exercise_id"] == exercise_id]
        correct = sum(row["correctness"] == 1 for row in rows)
        incorrect = sum(row["correctness"] == 0 for row in rows)
        frames = [row["duration_frames_inclusive"] for row in rows]
        seconds = [row["duration_seconds"] for row in rows if row["duration_seconds"] is not None]
        summary_rows.append({
            "exercise_id": exercise_id, "exercise": name, "exercise_family": family,
            "variation": variation, "videos": len({row["video_id"] for row in rows}) * 2,
            "video_ids": len({row["video_id"] for row in rows}),
            "subjects": len({row["person_id"] for row in rows}), "repetitions": len(rows),
            "correct_repetitions": correct, "incorrect_repetitions": incorrect,
            "correct_percent": 100 * correct / len(rows) if rows else 0,
            "incorrect_percent": 100 * incorrect / len(rows) if rows else 0,
            "duration_frames_min": min(frames) if frames else None,
            "duration_frames_max": max(frames) if frames else None,
            "duration_frames_mean": statistics.fmean(frames) if frames else None,
            "duration_frames_median": statistics.median(frames) if frames else None,
            "duration_seconds_min": min(seconds) if seconds else None,
            "duration_seconds_max": max(seconds) if seconds else None,
            "duration_seconds_mean": statistics.fmean(seconds) if seconds else None,
            "duration_seconds_median": statistics.median(seconds) if seconds else None,
            "cam17_orientations": json.dumps(dict(Counter(row["cam17_orientation"] for row in rows)), sort_keys=True),
        })
    subject_rows = []
    for (exercise_id, subject_id), rows in sorted(grouped(parsed, lambda row: (row["exercise_id"], row["person_id"])).items()):
        subject_rows.append({"exercise_id": exercise_id, "exercise": EXERCISES[exercise_id][0],
                             "subject_id": subject_id, "video_ids": len({row["video_id"] for row in rows}),
                             "camera_files": len({(row["video_id"], camera) for row in rows for camera in (17, 18)}),
                             "repetitions": len(rows), "correct": sum(row["correctness"] == 1 for row in rows),
                             "incorrect": sum(row["correctness"] == 0 for row in rows),
                             "cam17_orientations": json.dumps(dict(Counter(row["cam17_orientation"] for row in rows)), sort_keys=True)})
    write_csv(output / "rehab24_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output / "rehab24_subject_summary.csv", subject_rows, list(subject_rows[0]))
    write_csv(output / "rehab24_validation_issues.csv", issues, ["issue", "row", "video_id", "details"])
    return {
        "dataset": "REHAB24-6",
        "size_bytes": sum(recursive_size(external / f"Ex{i}") for i in EXERCISES) + sum((external / name).stat().st_size for name in ("Segmentation.csv", "Segmentation.txt", "joints_names.txt")),
        "columns": columns, "rows": len(raw_rows), "valid_rows": len(parsed),
        "video_files": len(videos), "video_ids": len({key[0] for key in videos}),
        "subjects": len({row["person_id"] for row in parsed}),
        "unique_values": {column: dict(counter) for column, counter in values.items() if column not in {"first_frame", "last_frame"}},
        "summary": summary_rows, "video_metadata": [asdict(item) for item in videos.values()],
        "validation_issue_counts": dict(Counter(item["issue"] for item in issues)),
        "view_semantics": {"direct": "cam17_orientation is a direct label.",
                           "camera18": "Segmentation.txt states camera18 is orthogonal: front->side, half-profile->half-profile; local wording inconsistently uses profile versus side for the third orientation.",
                           "camera_id": "Filename Camera17/Camera18.", "subject": "person_id column."},
        "rep_boundary": "Integer, likely one-based inclusive first_frame/last_frame: minimum first_frame is 1, no zero start exists, and some valid-looking last_frame values equal frame_count. Durations audited as end-start+1.",
        "correctness": "Segmentation.txt explicitly defines 1=correct and 0=incorrect.",
    }


def audit_pushup_database(root: Path) -> dict[str, Any]:
    folders = {"correct": root / "Correct sequence", "incorrect": root / "Wrong sequence"}
    videos: dict[str, list[dict[str, Any]]] = {}
    for label, folder in folders.items():
        items = []
        for path in sorted(folder.glob("*.mp4"), key=lambda item: item.name.casefold()):
            metadata = asdict(video_metadata(path)); metadata["filename"] = path.name; items.append(metadata)
        videos[label] = items
    arrays = []
    for path in sorted((root / "labels").glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        unique = np.unique(array) if array.size else np.asarray([])
        arrays.append({"filename": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
                       "min": float(array.min()) if array.size and np.issubdtype(array.dtype, np.number) else None,
                       "max": float(array.max()) if array.size and np.issubdtype(array.dtype, np.number) else None,
                       "unique_count": int(unique.size), "unique_values": unique.tolist() if unique.size <= 30 else None,
                       "sample_flat": array.reshape(-1)[:30].tolist(),
                       "interpretation": "Continuous (50,150,66) tensor consistent with 50 fixed-length sequences and 33x2 values/frame; not a discrete label vector.",
                       "label_level": "UNKNOWN / NEEDS DOCUMENTATION: joint order and row-to-video mapping are absent."})
    all_videos = videos["correct"] + videos["incorrect"]
    return {"dataset": "PushUpDatabase", "size_bytes": recursive_size(root),
            "video_counts": {label: len(items) for label, items in videos.items()},
            "video_filenames": {label: [item["filename"] for item in items] for label, items in videos.items()},
            "video_metadata_summary": {"fps": numeric_summary(item["fps"] for item in all_videos if item["fps"] is not None),
                                       "frame_count": numeric_summary(item["frame_count"] for item in all_videos if item["frame_count"] is not None),
                                       "duration_seconds": numeric_summary(item["duration_seconds"] for item in all_videos if item["duration_seconds"] is not None),
                                       "unreadable": sum(not item["opened"] for item in all_videos)},
            "npy_files": arrays,
            "video_npy_relationship": "First dimension 50 matches each class folder, but there are no IDs/manifest; filesystem ordering is not a proven mapping.",
            "capabilities": {
                "repetition_segmentation": {"status": "UNKNOWN", "reason": "No explicit boundaries; whether each clip is exactly one rep is undocumented."},
                "repetition_counting": {"status": "PARTIALLY SUPPORTED", "reason": "Temporal clips exist, but ground-truth counts/boundaries do not."},
                "correct_incorrect": {"status": "SUPPORTED", "reason": "Correct/Wrong parent folder is a direct clip-level label."},
                "detailed_error_classification": {"status": "NOT SUPPORTED", "reason": "No wrong-form error types."},
                "phase_detection": {"status": "NOT SUPPORTED", "reason": "No frame-level phases."}}}


def parse_intelli_filename(path: Path) -> dict[str, str] | None:
    parts = path.stem.split("_")
    if len(parts) != 6:
        return None
    keys = ("subject_id", "session_id", "gesture_id", "repetition", "correctness_code", "position")
    return dict(zip(keys, parts))


def audit_intellirehab(root: Path, joints_file: Path) -> dict[str, Any]:
    raw_dir = root / "SkeletonData" / "RawData"
    simple_dir = root / "SkeletonData" / "Simplified"
    files = sorted(simple_dir.glob("*.txt")); metadata = []; invalid_names = []; lengths = []
    column_counts: Counter[int] = Counter(); invalid_files = []; blanks = 0; nonfinite = 0
    hashes: Counter[str] = Counter()
    for path in files:
        parsed = parse_intelli_filename(path)
        (metadata if parsed else invalid_names).append(parsed if parsed else path.name)
        line_count = 0; file_columns: set[int] = set(); non_numeric = False; digest = hashlib.sha256()
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw); text = raw.decode("utf-8", errors="replace").strip()
                if not text: blanks += 1; continue
                line_count += 1; parts = text.split(","); file_columns.add(len(parts))
                try: nonfinite += int((~np.isfinite(np.asarray([float(value) for value in parts]))).sum())
                except ValueError: non_numeric = True
        lengths.append(line_count)
        for count in file_columns: column_counts[count] += 1
        if file_columns != {75} or non_numeric: invalid_files.append({"file": path.name, "columns": sorted(file_columns), "non_numeric": non_numeric})
        hashes[digest.hexdigest()] += 1
    raw_files = sorted(raw_dir.glob("*.txt")); raw_stems = {p.stem for p in raw_files}; simple_stems = {p.stem for p in files}
    joints = [line.split(":", 1)[1].strip() for line in joints_file.read_text(encoding="utf-8", errors="replace").splitlines() if ":" in line]
    return {"dataset": "IntelliRehabDS", "size_bytes": recursive_size(root),
            "files": {"raw": len(raw_files), "simplified": len(files), "macos_metadata": len([p for p in (root / "__MACOSX").rglob("*") if p.is_file()])},
            "filename_schema": {"pattern": "subject_session_gesture_repetition_correctness_position.txt", "evidence": [p.name for p in files[:5]],
                                "subject_id": "field 1", "session_id": "field 2; semantic name requires documentation", "exercise_movement_id": "field 3, observed 0..8",
                                "repetition_trial": "field 4", "correctness": "field 5; existing project adapter maps 1=correct, 2=incorrect, 3=incorrect/unclassifiable, but no README inside extraction proves semantics",
                                "position": "field 6: stand/chair"},
            "field_values": {key: dict(Counter(item[key] for item in metadata)) for key in ("subject_id", "session_id", "gesture_id", "repetition", "correctness_code", "position")},
            "samples_per_exercise_id": dict(Counter(item["gesture_id"] for item in metadata)),
            "samples_per_subject": dict(Counter(item["subject_id"] for item in metadata)),
            "correctness_code_balance": dict(Counter(item["correctness_code"] for item in metadata)),
            "sequence_lengths": numeric_summary(lengths),
            "structure": {"simplified_shape": "(T,75)=25 Kinect-v2 joints x XYZ", "dimensions": 3, "joints": 25, "delimiter": "comma", "frame_order": "line order",
                          "column_count_distribution": dict(column_counts), "raw_format": "Version0.1 + per-frame IDs + 25 named joint tuples with tracking state, XYZ and projected XY",
                          "joints_names_file_count": len(joints), "joints_names_file": joints,
                          "joints_names_relation": "Root joints_names.txt is a different 26-joint mocap topology and does not match IntelliRehab's Kinect-v2 topology."},
            "quality": {"invalid_filenames": invalid_names, "invalid_simplified_files": invalid_files, "blank_lines": blanks, "nonfinite_values": nonfinite,
                        "exact_duplicate_files": sum(count - 1 for count in hashes.values() if count > 1),
                        "raw_missing_for_simplified": sorted(simple_stems - raw_stems), "simplified_missing_for_raw": sorted(raw_stems - simple_stems)},
            "motionbert": {"directly_compatible": False, "reason": "Kinect-v2 25x3 lacks exact H36M-17/confidence contract and nose; tracking state exists only in RawData.",
                           "requirement": "Audited Skeleton Adapter or native 25-joint encoder; never use XYZ as x,y,confidence."}}


def audit_squat_dataset(root: Path) -> dict[str, Any]:
    class_rows = []; bad_images = []; resolutions: Counter[str] = Counter(); modes: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    for split in ("train", "test"):
        for folder in sorted((root / split).iterdir(), key=lambda p: p.name.casefold()):
            if not folder.is_dir(): continue
            images = [path for path in folder.iterdir() if path.is_file()]; readable = 0
            for path in images:
                try:
                    hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(str(path.relative_to(root)))
                    with Image.open(path) as image: image.verify()
                    with Image.open(path) as image: resolutions[f"{image.width}x{image.height}"] += 1; modes[image.mode] += 1
                    readable += 1
                except Exception as exc: bad_images.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
            canonical = {"bad back": "bad_back", "bad heel": "bad_heel", "good": "good"}.get(folder.name.casefold(), "unknown")
            class_rows.append({"split": split, "folder_label": folder.name, "canonical_label": canonical,
                               "files": len(images), "readable": readable, "corrupted": len(images) - readable})
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {"dataset": "SquatDataset", "size_bytes": recursive_size(root), "class_counts": class_rows,
            "total_images": sum(row["files"] for row in class_rows), "corrupted_images": bad_images,
            "resolution_distribution": dict(resolutions), "color_mode_distribution": dict(modes),
            "exact_duplicate_groups": duplicate_groups, "exact_duplicate_extra_files": sum(len(g) - 1 for g in duplicate_groups),
            "capitalization_issue": "Train uses Bad back/Bad heel; test uses Bad Back/Bad Heel. Canonical proposal: bad_back/bad_heel.",
            "data_type": "static images, not sequences",
            "labels": {"ground_truth": ["Good", "Bad Back", "Bad Heel"], "correctness": "Derived Good vs bad", "error_type": "folder class", "repetitions": None, "phases": None, "scores": None},
            "motionbert": {"directly_compatible": False, "reason": "No temporal dimension; repeated frames would be fake motion."}}


def csv_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); columns = list(reader.fieldnames or []); rows = 0
        video_counts: Counter[str] = Counter(); first: dict[str, int] = {}; last: dict[str, int] = {}
        previous = None; adjacent_duplicates = 0; missing = 0
        for row in reader:
            rows += 1; video_id = row.get("vid_id") or row.get("video_id") or ""; video_counts[video_id] += 1
            frame = row.get("frame_order")
            if frame is not None:
                try:
                    value = int(frame); first[video_id] = min(first.get(video_id, value), value); last[video_id] = max(last.get(video_id, value), value)
                    key = (video_id, frame); adjacent_duplicates += int(key == previous); previous = key
                except ValueError: pass
            missing += sum(value == "" for value in row.values())
    return {"filename": path.name, "columns": columns, "row_count": rows, "video_ids": len(video_counts),
            "rows_per_video": numeric_summary(video_counts.values()), "frame_min_global": min(first.values()) if first else None,
            "frame_max_global": max(last.values()) if last else None, "missing_cells": missing,
            "adjacent_duplicate_keys": adjacent_duplicates, "video_id_values": sorted(video_counts)}


def audit_physical(root: Path) -> dict[str, Any]:
    schemas = {p.name: csv_schema(p) for p in sorted(root.glob("*.csv"))}
    with (root / "labels.csv").open("r", encoding="utf-8-sig", newline="") as stream: labels = list(csv.DictReader(stream))
    label_ids = {row["vid_id"] for row in labels}; coverage = {}
    for name in ("landmarks.csv", "angles.csv", "calculated_3d_distances.csv", "xyz_distances.csv"):
        ids = set(schemas[name].pop("video_id_values"))
        coverage[name] = {"all_labels_have_features": label_ids <= ids, "all_features_have_labels": ids <= label_ids,
                          "missing_feature_ids": sorted(label_ids - ids), "unlabelled_feature_ids": sorted(ids - label_ids)}
    manifest_ids = set(schemas["manifest.csv"].pop("video_id_values")); schemas["labels.csv"].pop("video_id_values", None)
    return {"dataset": "PhysicalExerciseRecognition", "size_bytes": recursive_size(root), "schemas": schemas,
            "videos_or_sequences": len(labels), "class_counts": dict(Counter(row["class"] for row in labels)),
            "join_keys": {"labels_to_features": "labels.vid_id = feature.vid_id", "frame_alignment": "feature tables join on (vid_id,frame_order)",
                          "manifest": "manifest.video_id is nonnumeric and is not a demonstrated join to numeric vid_id"},
            "join_coverage": coverage, "manifest_numeric_id_overlap": len(label_ids & manifest_ids),
            "labels": {"exercise_class": "sequence-level labels.csv", "correctness": None, "repetition_boundaries": None,
                       "error_type": None, "phase": None, "score": None, "subject": None, "view": None},
            "capabilities": {"exercise_representation": "YES", "exercise_classification": "YES", "temporal_motion_learning": "YES",
                             "correctness": "NO", "rep_segmentation": "NO", "phase": "NO", "specific_errors": "NO", "score": "NO"}}


def inventory_rows(rehab: dict[str, Any], pushup: dict[str, Any], intelli: dict[str, Any], squat: dict[str, Any], physical: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for s in rehab["summary"]:
        rows.append({"Dataset": "REHAB24-6", "Exercise": s["exercise"], "Exercise Family": s["exercise_family"], "Variation": s["variation"],
                     "Subjects": s["subjects"], "Videos/Samples": s["videos"], "Repetitions": s["repetitions"], "Correct Labels": s["correct_repetitions"],
                     "Incorrect Labels": s["incorrect_repetitions"], "Rep Boundaries": "YES", "Correctness Labels": "YES", "Detailed Error Labels": "NO",
                     "Phase Labels": "NO", "Score Labels": "NO", "Skeleton Data": "NO", "RGB Video": "YES", "Static Images": "NO",
                     "View Metadata": "PARTIAL", "Camera Metadata": "YES", "Recommended Use": "Rep correctness/boundary/counting and representation",
                     "Limitations": "No phase/error/score; camera18 third-view wording needs clarification"})
    rows.append({"Dataset": "PushUpDatabase", "Exercise": "Push-up", "Exercise Family": "pushup", "Variation": "unspecified", "Subjects": "UNKNOWN",
                 "Videos/Samples": 100, "Repetitions": "UNKNOWN", "Correct Labels": 50, "Incorrect Labels": 50, "Rep Boundaries": "UNKNOWN",
                 "Correctness Labels": "YES", "Detailed Error Labels": "NO", "Phase Labels": "NO", "Score Labels": "NO", "Skeleton Data": "PARTIAL",
                 "RGB Video": "YES", "Static Images": "NO", "View Metadata": "UNKNOWN", "Camera Metadata": "UNKNOWN",
                 "Recommended Use": "Clip-level correct/incorrect representation", "Limitations": "NPY mapping/coordinate semantics undocumented"})
    rows.append({"Dataset": "IntelliRehabDS", "Exercise": "Gesture IDs 0-8", "Exercise Family": "rehabilitation", "Variation": "nine Kinect gestures",
                 "Subjects": len(intelli["field_values"]["subject_id"]), "Videos/Samples": intelli["files"]["simplified"], "Repetitions": intelli["files"]["simplified"],
                 "Correct Labels": intelli["correctness_code_balance"].get("1", 0), "Incorrect Labels": intelli["correctness_code_balance"].get("2", 0),
                 "Rep Boundaries": "YES", "Correctness Labels": "PARTIAL", "Detailed Error Labels": "NO", "Phase Labels": "NO", "Score Labels": "NO",
                 "Skeleton Data": "YES", "RGB Video": "NO", "Static Images": "NO", "View Metadata": "NO", "Camera Metadata": "NO",
                 "Recommended Use": "Native 25-joint gesture/correctness encoder", "Limitations": "Correctness semantics need local documentation; not direct MotionBERT"})
    good = sum(r["files"] for r in squat["class_counts"] if r["canonical_label"] == "good")
    bad = sum(r["files"] for r in squat["class_counts"] if r["canonical_label"] != "good")
    rows.append({"Dataset": "SquatDataset", "Exercise": "Squat", "Exercise Family": "squat", "Variation": "static posture", "Subjects": "UNKNOWN",
                 "Videos/Samples": squat["total_images"], "Repetitions": "NO", "Correct Labels": good, "Incorrect Labels": bad, "Rep Boundaries": "NO",
                 "Correctness Labels": "PARTIAL", "Detailed Error Labels": "YES", "Phase Labels": "NO", "Score Labels": "NO", "Skeleton Data": "NO",
                 "RGB Video": "NO", "Static Images": "YES", "View Metadata": "UNKNOWN", "Camera Metadata": "UNKNOWN",
                 "Recommended Use": "Static three-class posture/error classification", "Limitations": "No temporal labels or subject IDs"})
    for exercise, count in physical["class_counts"].items():
        rows.append({"Dataset": "PhysicalExerciseRecognition", "Exercise": exercise,
                     "Exercise Family": {"push_up": "pushup", "pull_up": "pullup", "squat": "squat"}.get(exercise, "other"), "Variation": "unspecified",
                     "Subjects": "UNKNOWN", "Videos/Samples": count, "Repetitions": "UNKNOWN", "Correct Labels": "NO", "Incorrect Labels": "NO",
                     "Rep Boundaries": "NO", "Correctness Labels": "NO", "Detailed Error Labels": "NO", "Phase Labels": "NO", "Score Labels": "NO",
                     "Skeleton Data": "YES", "RGB Video": "NO", "Static Images": "NO", "View Metadata": "NO", "Camera Metadata": "NO",
                     "Recommended Use": "Exercise classification/representation", "Limitations": "No subject/view/correctness/boundary labels"})
    return rows


def capability_rows() -> list[dict[str, str]]:
    outputs = ["Exercise representation", "Repetition boundary detection", "Repetition counting", "Correct/Incorrect per rep", "Correct rep count",
               "Incorrect rep count", "Pass/Fail", "Detailed error detection", "Phase detection", "Quality score 0–100"]
    matrix: dict[str, dict[str, tuple[str, str]]] = {}
    matrix["Push-up"] = {
        outputs[0]: ("TRAINABLE NOW", "PhysicalExerciseRecognition push_up; PushUpDatabase; REHAB24 Ex3"), outputs[1]: ("TRAINABLE NOW", "REHAB24 Ex3"),
        outputs[2]: ("TRAINABLE NOW", "REHAB24 Ex3 boundaries"), outputs[3]: ("TRAINABLE NOW", "REHAB24 Ex3; PushUpDatabase clip-level only"),
        outputs[4]: ("TRAINABLE NOW", "REHAB24 Ex3"), outputs[5]: ("TRAINABLE NOW", "REHAB24 Ex3"),
        outputs[6]: ("NEEDS LABEL DEFINITION", "No session pass/fail policy"), outputs[7]: ("NOT TRAINABLE", "No error-type labels"),
        outputs[8]: ("PARTIALLY TRAINABLE", "Boundaries are weak temporal supervision, not phases"), outputs[9]: ("NOT TRAINABLE", "No score labels")}
    matrix["Pull-up"] = {output: (("TRAINABLE NOW", "PhysicalExerciseRecognition pull_up") if output == outputs[0] else ("NOT TRAINABLE", "No matching labels")) for output in outputs}
    matrix["Squat"] = {
        outputs[0]: ("TRAINABLE NOW", "PhysicalExerciseRecognition squat; REHAB24 Ex6"), outputs[1]: ("TRAINABLE NOW", "REHAB24 Ex6"),
        outputs[2]: ("TRAINABLE NOW", "REHAB24 Ex6 boundaries"), outputs[3]: ("TRAINABLE NOW", "REHAB24 Ex6"), outputs[4]: ("TRAINABLE NOW", "REHAB24 Ex6"),
        outputs[5]: ("TRAINABLE NOW", "REHAB24 Ex6"), outputs[6]: ("NEEDS LABEL DEFINITION", "No session policy"),
        outputs[7]: ("PARTIALLY TRAINABLE", "SquatDataset static Bad Back/Bad Heel only"), outputs[8]: ("PARTIALLY TRAINABLE", "REHAB24 boundaries, no phases"),
        outputs[9]: ("NOT TRAINABLE", "No score labels")}
    matrix["Lunge"] = {
        outputs[0]: ("TRAINABLE NOW", "REHAB24 Ex5"), outputs[1]: ("TRAINABLE NOW", "REHAB24 Ex5"), outputs[2]: ("TRAINABLE NOW", "REHAB24 Ex5"),
        outputs[3]: ("TRAINABLE NOW", "REHAB24 Ex5"), outputs[4]: ("TRAINABLE NOW", "REHAB24 Ex5"), outputs[5]: ("TRAINABLE NOW", "REHAB24 Ex5"),
        outputs[6]: ("NEEDS LABEL DEFINITION", "No session policy"), outputs[7]: ("NOT TRAINABLE", "No error labels"),
        outputs[8]: ("PARTIALLY TRAINABLE", "Boundaries only"), outputs[9]: ("NOT TRAINABLE", "No score labels")}
    matrix["Plank"] = {output: ("NO DATA", "No plank sample in audited external sources") for output in outputs}
    return [{"Exercise Family": family, "Output": output, "Status": status, "Source/Evidence": source}
            for family, tasks in matrix.items() for output, (status, source) in tasks.items()]


def proposed_schema() -> dict[str, Any]:
    return {"schema_version": "proposal_only_v1", "fields": [
        {"name": "sample_id", "type": "string"}, {"name": "dataset", "type": "string"}, {"name": "subject_id", "type": "string|null"},
        {"name": "exercise_family", "type": "string|null"}, {"name": "exercise_variation", "type": "string|null"},
        {"name": "camera_id", "type": "string|null"}, {"name": "orientation_raw", "type": "string|null"}, {"name": "view", "type": "string|null"},
        {"name": "input_path", "type": "string"}, {"name": "input_type", "type": "rgb_video|static_image|skeleton_sequence|landmark_sequence"},
        {"name": "start_frame", "type": "integer|null"}, {"name": "end_frame", "type": "integer|null"}, {"name": "rep_index", "type": "integer|string|null"},
        {"name": "correctness", "type": "0|1|null"}, {"name": "correctness_provenance", "type": "ground_truth|derived|unknown"},
        {"name": "error_label", "type": "list[string]|null"}, {"name": "phase_label", "type": "string|null"},
        {"name": "score_label", "type": "number|null"}, {"name": "split_group", "type": "string"}],
            "rules": ["Preserve raw labels.", "Do not infer view from camera ID alone.", "Absence of error is not correctness.", "Keep each subject in one split."]}


def write_reports(output: Path, rehab: dict[str, Any], pushup: dict[str, Any], intelli: dict[str, Any], squat: dict[str, Any], physical: dict[str, Any]) -> None:
    risks = """# Data risks

## CRITICAL

- REHAB24 and IntelliRehab expose subject IDs; random frame/repetition splits leak subjects.
- PushUpDatabase NPY arrays have no row-to-video manifest or documented joint/channel order.
- SquatDataset static images cannot train temporal MotionBERT, counting, or phases.

## HIGH

- REHAB24 local view documentation inconsistently says `profile` and `side` for the third camera17 orientation mapping.
- IntelliRehab correctness code meanings are not documented inside the extracted folder.
- SquatDataset has no subject IDs; duplicates/source frames may cross the supplied train/test split.
- PhysicalExerciseRecognition lacks subject/view metadata, so subject leakage cannot be audited.

## MEDIUM

- Squat labels differ in capitalization between train and test.
- REHAB24 nuisance flags (mocap errors, extra persons, lighting) must be retained.
- Sources use incompatible skeleton topologies.
- No audited source provides phase labels or quality scores.

## LOW

- IntelliRehab includes macOS metadata files.
- Confirm dataset licenses separately; this audit does not infer external terms.
"""
    (output / "data_risks.md").write_text(risks, encoding="utf-8")
    table = "\n".join(f"| {row['exercise_id']} | {row['exercise']} | {row['video_ids']} | {row['repetitions']} | {row['correct_repetitions']} | {row['incorrect_repetitions']} | {row['subjects']} |" for row in rehab["summary"])
    report = f"""# External data audit

No training, MediaPipe extraction, cache rebuild, production-code change, or dataset mutation was performed.

## Sizes

| Dataset | Size |
|---|---:|
| REHAB24-6 | {human_size(rehab['size_bytes'])} |
| PushUpDatabase | {human_size(pushup['size_bytes'])} |
| IntelliRehabDS | {human_size(intelli['size_bytes'])} |
| SquatDataset | {human_size(squat['size_bytes'])} |
| PhysicalExerciseRecognition | {human_size(physical['size_bytes'])} |

## REHAB24-6

`Segmentation.csv`: {rehab['rows']} rows, {len(rehab['columns'])} columns: `{', '.join(rehab['columns'])}`. Subjects: {rehab['subjects']}; video IDs: {rehab['video_ids']}; camera files: {rehab['video_files']}. Correctness is locally documented as 1=correct and 0=incorrect.

| ID | Exercise | Video IDs | Reps | Correct | Incorrect | Subjects |
|---:|---|---:|---:|---:|---:|---:|
{table}

Boundaries are integer, likely one-based inclusive: the minimum start is 1, no zero start exists, and some end values equal video frame count. Duration uses `last-first+1`. Camera IDs are in filenames. Camera17 orientation is direct. Camera18 is documented as orthogonal, but profile/side wording needs clarification.

## PushUpDatabase

{pushup['video_counts']['correct']} Correct and {pushup['video_counts']['incorrect']} Wrong videos. Both NPY files are continuous `{pushup['npy_files'][0]['shape']}` tensors, not label vectors. Their joint order and row-to-video mapping are undocumented locally.

## IntelliRehabDS

{intelli['files']['simplified']} Simplified and {intelli['files']['raw']} Raw sequences; filename pattern is `subject_session_gesture_repetition_correctness_position`. Simplified shape is `(T,75)=25xXYZ`; subjects: {len(intelli['field_values']['subject_id'])}; length summary: `{intelli['sequence_lengths']}`. Direct MotionBERT use is unsafe.

## SquatDataset

{squat['total_images']} static images with Good/Bad Back/Bad Heel folder labels. It supports static error classification only, not repetition/phase learning or direct MotionBERT.

## PhysicalExerciseRecognition

{physical['videos_or_sequences']} labelled sequences; classes: `{physical['class_counts']}`. Tables join by `(vid_id,frame_order)`, labels by `vid_id`. Only exercise class exists.

## Subject-wise split feasibility

- REHAB24: YES—group all cameras, orientations, exercises, and repetitions by `person_id`.
- IntelliRehabDS: YES—group all files by filename subject field.
- PushUpDatabase: UNKNOWN—no subject ID.
- SquatDataset: not verifiable—no subject ID; group hashes/near-duplicates before trusting split.
- PhysicalExerciseRecognition: not verifiable—group by `vid_id` at minimum, but person leakage remains possible.

## Recommendation

Next, train a subject-wise REHAB24 per-repetition correctness baseline for one product family. Squat Ex6 is the strongest first candidate because it provides explicit boundaries, binary correctness, subjects, and paired cameras, and can later complement the static SquatDataset error classifier. Do not start that model from this audit stage.
"""
    (output / "data_audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=Path("datasets/external"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_audit"))
    args = parser.parse_args(); external = args.external_root.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    rehab = audit_rehab24(external, output)
    pushup = audit_pushup_database(external / "PushUpDatabase")
    intelli = audit_intellirehab(external / "IntelliRehabDS", external / "joints_names.txt")
    squat = audit_squat_dataset(external / "SquatDataset")
    physical = audit_physical(external / "physical_exercise_recognition")
    inventory = inventory_rows(rehab, pushup, intelli, squat, physical); matrix = capability_rows()
    json_dump(output / "pushup_database_audit.json", pushup); json_dump(output / "intellirehab_audit.json", intelli)
    json_dump(output / "rehab24_audit.json", rehab)
    json_dump(output / "squat_dataset_audit.json", squat); json_dump(output / "physical_exercise_recognition_audit.json", physical)
    json_dump(output / "proposed_unified_schema.json", proposed_schema())
    inventory_fields = ["Dataset", "Exercise", "Exercise Family", "Variation", "Subjects", "Videos/Samples", "Repetitions", "Correct Labels", "Incorrect Labels", "Rep Boundaries", "Correctness Labels", "Detailed Error Labels", "Phase Labels", "Score Labels", "Skeleton Data", "RGB Video", "Static Images", "View Metadata", "Camera Metadata", "Recommended Use", "Limitations"]
    write_csv(output / "dataset_inventory.csv", inventory, inventory_fields)
    write_csv(output / "training_capability_matrix.csv", matrix, ["Exercise Family", "Output", "Status", "Source/Evidence"])
    write_reports(output, rehab, pushup, intelli, squat, physical)
    json_dump(output / "audit_run_summary.json", {"datasets": [rehab["dataset"], pushup["dataset"], intelli["dataset"], squat["dataset"], physical["dataset"]], "training_started": False})
    print(json.dumps({"status": "ok", "output_dir": str(output), "rehab_rows": rehab["rows"], "intelli_sequences": intelli["files"]["simplified"], "squat_images": squat["total_images"]}, indent=2))


if __name__ == "__main__":
    main()
