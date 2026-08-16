from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class ManifestRecord:
    video_id: str
    video_path: str
    exercise_id: str
    split_group: str
    subject_id: str = "unknown"
    pass_fail_label: str = ""
    phase_annotation_path: str = ""
    error_labels: str = ""


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read the project CSV manifest without inferring missing labels."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"video_id", "video_path", "exercise_id", "split_group"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Manifest is missing fields: {required - set(rows[0])}")
    return [ManifestRecord(**{key: row.get(key, "") for key in ManifestRecord.__dataclass_fields__}) for row in rows]


def group_aware_split(
    records: Sequence[ManifestRecord],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    prefer_subject_groups: bool = True,
) -> dict[str, list[ManifestRecord]]:
    """Split whole subjects when available, otherwise whole videos."""

    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be non-negative and sum below 1.")
    grouped: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        subject_known = record.subject_id not in {"", "unknown", "none", "None"}
        key = record.subject_id if prefer_subject_groups and subject_known else record.split_group
        grouped[key].append(record)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    count = len(keys)
    test_count = int(round(count * test_fraction))
    valid_count = int(round(count * validation_fraction))
    test_keys = set(keys[:test_count])
    valid_keys = set(keys[test_count:test_count + valid_count])
    result = {"train": [], "validation": [], "test": []}
    for key, items in grouped.items():
        split = "test" if key in test_keys else "validation" if key in valid_keys else "train"
        result[split].extend(items)
    return result


class AQAWindowDataset(Dataset[dict[str, Any]]):
    """Load numeric, pickle-free window caches and expose missing-label masks.

    Cache files are ``<cache_root>/<video_id>.npz``. Required array:
    ``motionbert_input`` with shape ``(N,T,17,3)``. Optional arrays are
    ``phase_labels`` (N,T), ``pass_fail_labels`` (N), and ``error_labels``
    (N,E). A missing array disables only its corresponding task.
    """

    def __init__(self, records: Sequence[ManifestRecord], cache_root: str | Path) -> None:
        self.records = list(records)
        self.cache_root = Path(cache_root)
        self.index: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            cache_path = self.cache_root / f"{record.video_id}.npz"
            if not cache_path.is_file():
                continue
            with np.load(cache_path, allow_pickle=False) as archive:
                if "motionbert_input" not in archive:
                    raise ValueError(f"Missing motionbert_input in {cache_path}")
                inputs = archive["motionbert_input"]
                if inputs.ndim != 4 or inputs.shape[2:] != (17, 3):
                    raise ValueError(f"Invalid MotionBERT cache shape {inputs.shape} in {cache_path}")
                self.index.extend((record_index, window) for window in range(len(inputs)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, window_index = self.index[index]
        record = self.records[record_index]
        path = self.cache_root / f"{record.video_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            motionbert_input = np.asarray(archive["motionbert_input"][window_index], dtype=np.float32)
            if not np.isfinite(motionbert_input).all():
                raise ValueError(f"Non-finite cached input in {path}")
            item: dict[str, Any] = {
                "motionbert_input": torch.from_numpy(motionbert_input),
                "temporal_mask": torch.ones(motionbert_input.shape[0], dtype=torch.bool),
                "exercise_id": record.exercise_id,
                "video_id": record.video_id,
                "phase_available": "phase_labels" in archive,
                "pass_fail_available": "pass_fail_labels" in archive,
                "errors_available": "error_labels" in archive,
            }
            if item["phase_available"]:
                item["phase_labels"] = torch.as_tensor(archive["phase_labels"][window_index], dtype=torch.long)
            if item["pass_fail_available"]:
                item["pass_fail_labels"] = torch.as_tensor(archive["pass_fail_labels"][window_index], dtype=torch.long)
            if item["errors_available"]:
                item["error_labels"] = torch.as_tensor(archive["error_labels"][window_index], dtype=torch.float32)
        return item


class ExerciseBatchSampler(Sampler[list[int]]):
    """Batch windows by exercise so one persistent expert handles each batch."""

    def __init__(
        self,
        dataset: AQAWindowDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, (record_index, _) in enumerate(self.dataset.index):
            groups[self.dataset.records[record_index].exercise_id].append(index)
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(indices[start:start + self.batch_size] for start in range(0, len(indices), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        groups: dict[str, int] = defaultdict(int)
        for record_index, _ in self.dataset.index:
            groups[self.dataset.records[record_index].exercise_id] += 1
        return sum((count + self.batch_size - 1) // self.batch_size for count in groups.values())


def collate_aqa_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad variable time lengths and retain per-task availability masks."""

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    exercise_ids = {str(item["exercise_id"]) for item in items}
    if len(exercise_ids) != 1:
        raise ValueError("A training batch must contain exactly one exercise.")
    max_frames = max(item["motionbert_input"].shape[0] for item in items)
    batch_size = len(items)
    inputs = torch.zeros(batch_size, max_frames, 17, 3, dtype=torch.float32)
    temporal_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    for row, item in enumerate(items):
        frames = item["motionbert_input"].shape[0]
        inputs[row, :frames] = item["motionbert_input"]
        temporal_mask[row, :frames] = True
    batch: dict[str, Any] = {
        "motionbert_input": inputs,
        "temporal_mask": temporal_mask,
        "exercise_id": next(iter(exercise_ids)),
        "video_id": [item["video_id"] for item in items],
    }
    for task, key in (("phase", "phase_labels"), ("pass_fail", "pass_fail_labels"), ("errors", "error_labels")):
        availability = torch.tensor([bool(item[f"{task}_available"]) for item in items], dtype=torch.bool)
        batch[f"{task}_available"] = availability
        if availability.any():
            if not availability.all():
                template = next(item[key] for item in items if key in item)
                labels = [item.get(key, torch.zeros_like(template)) for item in items]
            else:
                labels = [item[key] for item in items]
            if task == "phase":
                padded = torch.full((batch_size, max_frames), -100, dtype=torch.long)
                for row, label in enumerate(labels):
                    padded[row, :len(label)] = label
                batch[key] = padded
            else:
                batch[key] = torch.stack(labels)
    return batch
