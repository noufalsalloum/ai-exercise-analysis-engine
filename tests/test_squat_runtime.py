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

    # --- Regression coverage for the 2026-08-30 root-cause investigation ---
    # Real video showed a complete, honest squat (knee angle deep into the
    # low 70s, well past bottom_enter_angle=110) whose normalized pelvis-Y
    # displacement peaked at ~0.062-0.069 — short of the old 0.070 floor —
    # causing BOTTOM/ASCENDING to never trigger and the rep to silently
    # vanish (rep_count stayed 0 despite two genuine descend-then-return
    # cycles). See inference/squat_runtime.py's minimum_pelvis_displacement
    # comment for the full evidence (6-clip regression matrix).

    MODERATE_DEPTH_CYCLE = [
        (170, 175, .40, 1., 0.),
        (140, 150, .43, 1., .2),
        # Knee reaches deep flexion (100 <= bottom_enter_angle=110) while
        # displacement peaks at .062 - below the old 0.070 default but above
        # the new 0.060 default. This is the exact shape of the real,
        # previously-silently-dropped rep.
        (100, 115, .462, 1., .5),
        (130, 142, .44, 1., .7),
        (165, 172, .41, 1., 1.),
    ]

    def test_moderate_depth_cycle_with_below_old_threshold_displacement_now_counts(self) -> None:
        runtime = SquatRepetitionRuntime(config())
        results = feed(runtime, self.MODERATE_DEPTH_CYCLE)
        self.assertEqual(results[-1].repetition_count, 1)
        self.assertEqual(
            results[-1].completed_cycle.phase_sequence,
            ["STANDING", "DESCENDING", "BOTTOM", "ASCENDING", "STANDING"],
        )

    def test_moderate_depth_cycle_regresses_under_the_old_070_threshold(self) -> None:
        # Pins down the actual bug: the identical movement, only the OLD
        # (pre-fix) threshold applied, must fail to count it - proving this
        # test would have caught the regression before it shipped.
        runtime = SquatRepetitionRuntime(config(minimum_pelvis_displacement=0.070))
        results = feed(runtime, self.MODERATE_DEPTH_CYCLE)
        self.assertEqual(results[-1].repetition_count, 0)

    def test_knee_dip_without_real_pelvis_descent_still_does_not_count(self) -> None:
        # False-positive guard, preserved: knee alone dips well past
        # bottom_enter_angle=110, but displacement never exceeds .035 (the
        # deliberately-poor/mid-rep reference clip's real observed ceiling
        # was .0338) - this must NOT be counted as a rep even at the new,
        # lower threshold.
        runtime = SquatRepetitionRuntime(config())
        feed(runtime, [
            (170, 175, .40, 1., 0.),
            (140, 150, .43, 1., .2),
            (95, 120, .435, 1., .5),   # knee well past bottom_enter, displacement only .035
            (170, 175, .402, 1., .8),  # returns to standing without ever reaching BOTTOM
        ])
        self.assertEqual(runtime.repetition_count, 0)
        self.assertGreaterEqual(runtime.incomplete_cycles, 1)

    def test_displacement_just_under_the_new_threshold_still_does_not_count(self) -> None:
        # Locks in the exact new boundary (0.060) so a future silent change
        # to minimum_pelvis_displacement is caught by this test either way.
        runtime = SquatRepetitionRuntime(config())
        feed(runtime, [
            (170, 175, .40, 1., 0.),
            (140, 150, .43, 1., .2),
            (100, 115, .4599, 1., .5),  # displacement = .0599, just under .060
            (170, 175, .402, 1., .8),
        ])
        self.assertEqual(runtime.repetition_count, 0)

