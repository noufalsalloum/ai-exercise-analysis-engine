from __future__ import annotations

import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from application.exercise_registry import ExerciseRegistry
from application.squat_shadow import SquatBoundaryShadow, SquatShadowConfig
from application.workers import AnalysisWorker
from evaluation.evaluate_squat_external import evaluate_video
from input_sources.frame_sources import FramePacket
from models.squat_correctness import SquatCorrectnessModel
from models.squat_rep_boundary_v2 import SquatRepBoundaryV2Model
from training.squat_correctness import load_checkpoint_strict
from training.squat_correctness_v3 import (
    DEVELOPMENT_SUBJECTS,
    build_loso_folds,
    development_only_manifest,
    subject_class_sample_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for subject in DEVELOPMENT_SUBJECTS:
        for correctness in (0, 1):
            pair = f"subject-{subject}-label-{correctness}"
            for camera in ("Camera17", "Camera18"):
                rows.append(
                    {
                        "sample_id": f"{pair}-{camera}",
                        "pair_id": pair,
                        "subject_id": subject,
                        "camera_id": camera,
                        "orientation_raw": "front" if correctness else "half-profile",
                        "correctness": correctness,
                        "repetition_cache_path": "unused.npz",
                    }
                )
    return pd.DataFrame(rows)


class FakeSource:
    fps = 10.0
    stream_lost = False
    backend_name = "TEST"
    camera_index = None

    def __init__(self) -> None:
        self.index = 0; self.closed = False

    def read(self) -> FramePacket | None:
        if self.closed or self.index >= 2:
            return None
        packet = FramePacket(
            np.zeros((32, 32, 3), np.uint8), self.index, self.index / self.fps, None
        )
        self.index += 1
        return packet

    def close(self) -> None:
        self.closed = True


class FakePose:
    def process(self, _frame: np.ndarray, _timestamp: float) -> np.ndarray:
        landmarks = np.zeros((33, 4), np.float32)
        landmarks[:, :2] = 0.5; landmarks[:, 3] = 1.0
        return landmarks

    def close(self) -> None:
        pass


class FakeRuntime:
    incomplete_cycles = 0

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def update_landmarks(self, *_args, **_kwargs):
        self.calls += 1
        cycle = None
        if self.calls == 1:
            cycle = SimpleNamespace(
                start_frame=0,
                end_frame=1,
                start_time=0.0,
                end_time=0.1,
                duration_seconds=0.1,
                phase_sequence=["STANDING", "BOTTOM", "STANDING"],
                confidence=1.0,
            )
        return SimpleNamespace(
            phase="STANDING",
            repetition_count=1,
            elbow_angle=90.0,
            selected_side="left",
            signal_confidence=1.0,
            last_repetition_duration=0.1,
            current_cycle_valid=True,
            completed_cycle=cycle,
            valid_frame=True,
        )

    def finish(self) -> None:
        pass


class FakeRouter:
    def __init__(self) -> None:
        self.runtime = FakeRuntime()

    def create(self, *_args, **_kwargs) -> FakeRuntime:
        return self.runtime


class FakeShadow:
    def __init__(self) -> None:
        self.frames = 0; self.rule_cycles = 0; self.finalized = False

    def record_frame(self, *_args) -> None:
        self.frames += 1

    def record_rule_cycle(self, _cycle) -> None:
        self.rule_cycles += 1

    def finalize_and_write(self, **kwargs) -> Path:
        self.finalized = True
        self.rule_total_seen = kwargs["rule_based_total_reps"]
        self.ai_total_reps = 99
        return Path("shadow.json")


class SquatAIV3Tests(unittest.TestCase):
    def test_loso_has_no_subject_or_camera_pair_leakage(self) -> None:
        folds = build_loso_folds(synthetic_manifest())
        self.assertEqual(len(folds), 7)
        held = []
        for fold in folds:
            held_subject = str(fold.attrs["held_subject"]); held.append(held_subject)
            validation = set(fold.loc[fold["split"] == "validation", "subject_id"])
            train = set(fold.loc[fold["split"] == "train", "subject_id"])
            self.assertEqual(validation, {held_subject})
            self.assertFalse(train & validation)
            self.assertEqual(fold.groupby("pair_id")["split"].nunique().max(), 1)
        self.assertEqual(tuple(held), DEVELOPMENT_SUBJECTS)
        self.assertNotIn("4", held); self.assertNotIn("7", held)

    def test_development_filter_excludes_historical_test_subjects(self) -> None:
        frame = synthetic_manifest()
        extra = frame.iloc[[0]].copy(); extra["subject_id"] = "4"
        filtered = development_only_manifest(pd.concat([frame, extra], ignore_index=True))
        self.assertEqual(set(filtered["subject_id"]), set(DEVELOPMENT_SUBJECTS))
        self.assertFalse({"4", "7"} & set(filtered["subject_id"]))

    def test_subject_class_balancing_equalizes_mass(self) -> None:
        frame = synthetic_manifest()
        duplicated = pd.concat([frame, frame[frame["subject_id"] == "1"]] * 3, ignore_index=True)
        weights = subject_class_sample_weights(duplicated)
        duplicated = duplicated.assign(weight=weights)
        subject_mass = duplicated.groupby("subject_id")["weight"].sum()
        self.assertLess(float(subject_mass.max() - subject_mass.min()), 1e-12)
        class_mass = duplicated.groupby(["subject_id", "correctness"])["weight"].sum()
        for subject in DEVELOPMENT_SUBJECTS:
            values = class_mass.loc[subject].to_numpy()
            self.assertLess(float(values.max() - values.min()), 1e-12)

    def test_shadow_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.json"
            path.write_text(
                json.dumps(
                    {
                        "enabled": False,
                        "checkpoint": "unused.pt",
                        "output_dir": "unused",
                        "device": "cpu",
                        "minimum_frames": 3,
                        "contract": "test",
                        "user_facing_count_source": "rule_based_only",
                    }
                ),
                encoding="utf-8",
            )
            config = SquatShadowConfig.load(path)
            self.assertFalse(config.enabled)
            with self.assertRaises(ValueError):
                SquatBoundaryShadow(config, model=SquatRepBoundaryV2Model(), postprocessing={})

    def test_shadow_log_is_observational(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = SquatRepBoundaryV2Model(channels=96, dropout=0.0)
            for parameter in model.parameters():
                parameter.data.zero_()
            config = SquatShadowConfig(
                True,
                Path("injected.pt"),
                Path(directory),
                "cpu",
                3,
                "preprocessing_v4_(T,17,3)_x_y_confidence",
                "rule_based_only",
            )
            post = {
                "enter_threshold": 0.35,
                "exit_threshold": 0.2,
                "smoothing_kernel": 1,
                "min_duration": 3,
                "max_duration": 20,
                "merge_gap": 0,
                "boundary_threshold": 0.9,
                "boundary_smoothing_kernel": 1,
                "boundary_cluster_gap": 2,
            }
            shadow = SquatBoundaryShadow(config, model=model, postprocessing=post)
            landmarks = np.zeros((33, 4), np.float32)
            landmarks[:, :2] = 0.5; landmarks[:, 3] = 1.0
            for index in range(5):
                shadow.record_frame(landmarks, index, index / 10)
            path = shadow.finalize_and_write(
                session_id="test",
                rule_based_total_reps=7,
                camera_view="side",
                input_mode="video",
                duration_seconds=0.5,
                video_path="sample.mp4",
                cancelled=False,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["rule_based_total_reps"], 7)
            self.assertEqual(payload["user_facing_count_source"], "rule_based_only")
            self.assertIn("difference_ai_minus_rule", payload)

    def test_worker_shadow_cannot_change_visible_rule_count(self) -> None:
        events: queue.Queue[dict] = queue.Queue(maxsize=20)
        shadow = FakeShadow()
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("squat"),
            input_mode="video",
            camera_view="side",
            source_factory=FakeSource,
            pose_factory=FakePose,
            events=events,
            runtime_router=FakeRouter(),
            preserve_video_timing=False,
            squat_shadow_factory=lambda: shadow,
        )
        worker.run_sync()
        complete = [item for item in list(events.queue) if item["type"] == "complete"][-1]
        self.assertEqual(complete["result"]["summary"]["total_repetitions"], 1)
        self.assertTrue(shadow.finalized)
        self.assertEqual(shadow.rule_total_seen, 1)
        self.assertEqual(shadow.ai_total_reps, 99)

    def test_v2_boundary_and_v3_correctness_load_strictly(self) -> None:
        boundary_checkpoint = torch.load(
            ROOT / "checkpoints/squat_ai_v2/rep_boundary/best.pt",
            map_location="cpu",
            weights_only=True,
        )
        boundary = SquatRepBoundaryV2Model()
        boundary.load_state_dict(boundary_checkpoint["model_state_dict"], strict=True)
        final_path = ROOT / "checkpoints/squat_ai_v3/correctness/final_dev.pt"
        self.assertTrue(final_path.is_file(), "Run V3 LOSO before the full suite.")
        correctness = SquatCorrectnessModel(ROOT / "models/latest_epoch.bin")
        checkpoint = load_checkpoint_strict(final_path, correctness, torch.device("cpu"))
        self.assertEqual(checkpoint["training_stage"], "development_final_model")
        self.assertTrue(checkpoint["motionbert_frozen"])
        self.assertEqual(set(checkpoint["development_subjects"]), set(DEVELOPMENT_SUBJECTS))
        self.assertFalse(checkpoint["historical_test_subjects_re_evaluated"] if "historical_test_subjects_re_evaluated" in checkpoint else False)

    def test_external_evaluator_blocks_historical_test_subjects_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "historical locked Test"):
            evaluate_video(
                Path("missing.mp4"), Path("missing.task"), Path("missing.pt"),
                Path("missing.pt"), Path("missing.pt"), torch.device("cpu"),
                subject_id="4",
            )


if __name__ == "__main__":
    unittest.main()
