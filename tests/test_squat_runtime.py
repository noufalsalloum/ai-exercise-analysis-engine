from __future__ import annotations

import unittest

from application.runtime_router import FamilyRuntimeRouter
from inference.squat_runtime import SquatRepConfig, SquatRepetitionRuntime


def config(**overrides: object) -> SquatRepConfig:
    values = {"angle_smoothing_window": 1, "pelvis_smoothing_window": 1,
              "side_confidence_window": 1, "side_switch_confirm_frames": 1,
              "minimum_repetition_duration": 0.30, "cooldown_seconds": 0.0}
    values.update(overrides)
    return SquatRepConfig(**values)


def feed(runtime: SquatRepetitionRuntime, samples: list[tuple[float | None, float | None, float | None, float, float]], enabled: bool = True) -> list:
    return [runtime.update_signals(knee, hip, pelvis, confidence, timestamp, index, enabled)
            for index, (knee, hip, pelvis, confidence, timestamp) in enumerate(samples)]


FULL_CYCLE = [(170, 175, .40, 1., 0.), (140, 150, .43, 1., .2), (95, 120, .49, 1., .5), (128, 140, .46, 1., .7), (165, 172, .42, 1., 1.)]


class SquatRuntimeTests(unittest.TestCase):
    def test_router_uses_independent_squat_runtime(self) -> None:
        self.assertIsInstance(FamilyRuntimeRouter().create("squat", "video", "side"), SquatRepetitionRuntime)

    def test_complete_cycle_counts_once(self) -> None:
        results = feed(SquatRepetitionRuntime(config()), FULL_CYCLE)
        self.assertEqual(results[-1].repetition_count, 1)
        self.assertEqual(results[-1].completed_cycle.phase_sequence, ["STANDING", "DESCENDING", "BOTTOM", "ASCENDING", "STANDING"])

    def test_partial_or_low_confidence_cycle_does_not_count(self) -> None:
        runtime = SquatRepetitionRuntime(config(max_low_confidence_gap_frames=1))
        feed(runtime, [(170, 175, .40, 1., 0.), (140, 150, .43, 1., .2), (None, None, None, 0., .3), (None, None, None, 0., .4), (165, 172, .42, 1., .8)])
        self.assertEqual(runtime.repetition_count, 0)
        self.assertGreaterEqual(runtime.incomplete_cycles, 1)

    def test_countdown_and_bottom_hold_do_not_count(self) -> None:
        runtime = SquatRepetitionRuntime(config())
        feed(runtime, FULL_CYCLE, enabled=False)
        self.assertEqual(runtime.repetition_count, 0)
        feed(runtime, [(170, 175, .40, 1., 0.), (140, 150, .43, 1., .2)] + [(95, 120, .49, 1., .4 + i*.1) for i in range(20)])
        self.assertEqual(runtime.repetition_count, 0)

