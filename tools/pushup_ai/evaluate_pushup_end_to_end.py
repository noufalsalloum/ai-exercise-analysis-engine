"""Evaluate predicted-boundary Push-up V1 once on locked Test subjects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from inference.pushup_ai_mvp import PushupTableAIOrchestrator
from training.squat_correctness import classification_metrics
from training.squat_rep_boundary import match_segments


def main() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "results/pushup_ai_v1/data_summary/repetition_manifest.csv", dtype={"subject_id": str})
    test = manifest[manifest["split"] == "test"]
    orchestrator = PushupTableAIOrchestrator(PROJECT_ROOT / "checkpoints/pushup_ai_v1/boundary/best.pt", PROJECT_ROOT / "checkpoints/pushup_ai_v1/correctness/final_dev.pt", PROJECT_ROOT / "models/latest_epoch.bin")
    rows = []; videos = []
    for (video_id, camera_id), group in test.groupby(["video_id", "camera_id"], sort=True):
        source_stem = Path(str(group.iloc[0]["video_path"])).stem; stem = source_stem.replace("-30fps-transposed", "_cam18").replace("-30fps", "_cam17")
        with np.load(PROJECT_ROOT / "datasets/window_cache/rehab24_pushup_v1/full_videos" / f"{stem}.npz", allow_pickle=False) as archive:
            raw = np.asarray(archive["landmarks"], np.float32); motion = np.asarray(archive["motionbert_input"], np.float32)
        result = orchestrator.analyze(raw, motion); predicted = [(value["start_frame"] - 1, value["end_frame"] - 1) for value in result["per_rep_results"]]
        ordered = group.sort_values(["start_frame", "end_frame"]).drop_duplicates("pair_id"); gt = [(int(value.start_frame) - 1, int(value.end_frame) - 1) for value in ordered.itertuples(index=False)]
        matches = match_segments(predicted, gt, 0.5); match_by_pred = {p: (g, iou) for p, g, iou in matches}
        for index, rep in enumerate(result["per_rep_results"]):
            match = match_by_pred.get(index); target = None if match is None else int(ordered.iloc[match[0]]["correctness"])
            rows.append({"video_id": video_id, "camera_id": camera_id, **rep, "matched_correctness_gt": target, "temporal_iou": None if match is None else match[1]})
        videos.append({"video_id": video_id, "camera_id": camera_id, "gt_reps": len(gt), "predicted_reps": len(predicted), "matched_reps": len(matches), "performance_score_no_gt_claim": result["performance_score"]})
    frame = pd.DataFrame(rows); matched = frame.dropna(subset=["matched_correctness_gt"]).copy(); predictions = (matched["assessment"] == "PASS").astype(int)
    correctness = classification_metrics(matched["matched_correctness_gt"].astype(int), predictions, matched["correctness_probability"]) if len(matched) else None
    output = {"scope": orchestrator.SCOPE, "test_opened_once": True, "boundary_source": "predicted", "videos": len(videos), "matched_repetitions": len(matched), "correctness_on_matched_predicted_segments": correctness, "warning": "Development Test benchmark for table/incline Push-ups; not floor-Push-up or production validation."}
    destination = PROJECT_ROOT / "results/pushup_ai_v1/end_to_end"; destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "per_rep_predictions.csv", index=False); pd.DataFrame(videos).to_csv(destination / "video_results.csv", index=False); (destination / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
