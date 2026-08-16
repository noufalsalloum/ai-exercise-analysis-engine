from __future__ import annotations

import unittest
import json
from pathlib import Path

from inference.plank_runtime import MarchingPlankConfig, MarchingPlankRepetitionRuntime


def config(**overrides: object) -> MarchingPlankConfig:
    values = {"signal_smoothing_window": 1, "neutral_baseline_window": 1,
              "minimum_repetition_duration": .2, "cooldown_seconds": 0.0}
    values.update(overrides)
    return MarchingPlankConfig(**values)


class MarchingPlankRuntimeTests(unittest.TestCase):
    def test_active_project_config_loads_without_changing_thresholds(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "configs/plank.json").read_text(encoding="utf-8"))["runtime"]["repetition_counter"]
        loaded = MarchingPlankConfig.from_dict(payload)
        self.assertEqual(loaded.return_tolerance, 0.025)
        self.assertEqual(loaded.lift_start_displacement, 0.025)

    def test_leg_lift_state_does_not_replace_elapsed_second_count(self) -> None:
        runtime = MarchingPlankRepetitionRuntime(config())
        # Image coordinates decrease as the ankle rises vertically.
        values = [(0., 0., 0., 1., 0.), (-.04, 0., .01, 1., .1), (-.08, 0., .01, 1., .3), (-.01, 0., .01, 1., .5), (0., 0., .01, 1., .7)]
        results = [runtime.update_signals(*item, index, True) for index, item in enumerate(values)]
        self.assertEqual(results[-1].repetition_count, 0)
        self.assertEqual(results[-1].left_repetitions, 1)

    def test_valid_hold_counts_completed_integer_seconds(self) -> None:
        expected = ((0.9, 0), (1.0, 1), (1.9, 1), (2.0, 2), (3.0, 3), (8.7, 8))
        for duration, count in expected:
            with self.subTest(duration=duration):
                runtime = MarchingPlankRepetitionRuntime(config())
                runtime.update_signals(0.0, 0.0, 0.0, 1.0, 0.0, 0, True)
                result = runtime.update_signals(0.0, 0.0, 0.0, 1.0, duration, 1, True)
                self.assertEqual(result.repetition_count, count)
                self.assertAlmostEqual(result.hold_time_seconds, duration)

    def test_invalid_pose_pauses_hold_time_and_reset_clears_timer(self) -> None:
        runtime = MarchingPlankRepetitionRuntime(config())
        runtime.update_signals(0.0, 0.0, 0.0, 1.0, 0.0, 0, True)
        runtime.update_signals(0.0, 0.0, 0.0, 1.0, 2.0, 1, True)
        runtime.update_signals(None, None, None, 0.0, 5.0, 2, True)
        result = runtime.update_signals(0.0, 0.0, 0.0, 1.0, 8.0, 3, True)
        self.assertEqual(result.repetition_count, 2)
        self.assertAlmostEqual(result.hold_time_seconds, 2.0)
        runtime.reset()
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.valid_hold_seconds, 0.0)

    def test_uploaded_timing_uses_frame_over_fps_timestamps(self) -> None:
        runtime = MarchingPlankRepetitionRuntime(config())
        fps = 30.0
        for frame_index in range(61):
            runtime.update_signals(0.0, 0.0, 0.0, 1.0, frame_index / fps, frame_index, True)
        self.assertEqual(runtime.repetition_count, 2)

    def test_camera_timing_accepts_monotonic_relative_seconds(self) -> None:
        runtime = MarchingPlankRepetitionRuntime(config())
        for frame_index, timestamp in enumerate((0.0, 0.4, 1.0, 1.4, 2.0)):
            runtime.update_signals(0.0, 0.0, 0.0, 1.0, timestamp, frame_index, True)
        self.assertEqual(runtime.repetition_count, 2)

    def test_no_return_or_low_confidence_does_not_count(self) -> None:
        runtime = MarchingPlankRepetitionRuntime(config(max_low_confidence_gap_frames=1))
        for index, item in enumerate([(0.,0.,0.,1.,0.),(-.04,0.,0.,1.,.1),(-.08,0.,0.,1.,.2),(-.08,0.,0.,0.,.3),(-.08,0.,0.,0.,.4)]):
            runtime.update_signals(*item, index, True)
        self.assertEqual(runtime.repetition_count, 0)
        self.assertGreaterEqual(runtime.incomplete_cycles, 1)
