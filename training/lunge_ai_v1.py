"""Subject-safe Lunge V1 training by reuse of validated generic training mechanics."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import training.pushup_ai_v1 as shared
from models.lunge_correctness import LungeCorrectnessModel
from models.lunge_rep_boundary import LungeRepBoundaryModel
from training.squat_correctness import load_manifest
from training.squat_rep_boundary import load_full_video_records


def run(project: Path, device: torch.device) -> dict:
    # The training mechanics are shared, while the model classes, data, split,
    # checkpoints and result paths remain strictly Lunge-specific.
    original_boundary=shared.PushupRepBoundaryModel; original_correctness=shared.PushupCorrectnessModel
    shared.PushupRepBoundaryModel=LungeRepBoundaryModel; shared.PushupCorrectnessModel=LungeCorrectnessModel
    try:
        manifest_path=project/"results/full_exercise_ai_parity/lunge/data/repetition_manifest.csv"
        cache=project/"datasets/window_cache/rehab24_lunge_v1"
        manifest=load_manifest(manifest_path,cache/"repetitions"); records=load_full_video_records(manifest,cache/"full_videos")
        boundary=shared.train_boundary(records,project/"checkpoints/lunge_ai_v1/boundary/best.pt",project/"results/full_exercise_ai_parity/lunge/boundary",device,epochs=8)
        correctness=shared.train_correctness_loso(manifest,cache/"repetitions",cache/"motionbert_features",project/"models/latest_epoch.bin",project/"checkpoints/lunge_ai_v1/correctness/final_dev.pt",project/"results/full_exercise_ai_parity/lunge/correctness",device,epochs=6)
    finally:
        shared.PushupRepBoundaryModel=original_boundary; shared.PushupCorrectnessModel=original_correctness
    result={"scope":"REHAB24 Ex5 Lunge","boundary":boundary,"correctness":correctness,"detailed_error":{"supported":False,"fallback":"Form Issue after learned FAIL only"}}
    (project/"results/full_exercise_ai_parity/lunge/model_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
