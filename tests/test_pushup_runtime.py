from __future__ import annotations

import unittest

from inference.pushup_runtime import PushupRepConfig, PushupRepetitionRuntime


def test_config(**overrides: object) -> PushupRepConfig:
    values = {
        "angle_smoothing_window": 1,
        "side_confidence_window": 1,
        "side_switch_confirm_frames": 1,
        "min_repetition_duration": 0.30,
        "cooldown_seconds": 0.0,
    }
    values.update(overrides)
    return PushupRepConfig(**values)


def feed(runtime: PushupRepetitionRuntime, values: list[tuple[float | None, float, float]]) -> list:
    return [
        runtime.update_signal(angle, confidence, timestamp, index)
        for index, (angle, confidence, timestamp) in enumerate(values)
    ]


class PushupRuntimeTests(unittest.TestCase):
    def test_full_ordered_cycle_counts_exactly_once(self) -> None:
        runtime = PushupRepetitionRuntime(test_config())
        results = feed(
            runtime,
            [(170, 1.0, 0.0), (140, 1.0, 0.2), (90, 1.0, 0.5), (130, 1.0, 0.7), (165, 1.0, 1.0)],
        )
        self.assertEqual(runtime.repetition_count, 1)
        self.assertIsNotNone(results[-1].completed_cycle)
        self.assertEqual(
            results[-1].completed_cycle.phase_sequence,
            ["TOP", "DESCENDING", "BOTTOM", "ASCENDING", "TOP"],
        )

    def test_partial_cycle_does_not_count(self) -> None:
        runtime = PushupRepetitionRuntime(test_config())
        feed(runtime, [(170, 1.0, 0.0), (140, 1.0, 0.2), (125, 1.0, 0.4), (165, 1.0, 0.8)])
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.incomplete_cycles, 1)

    def test_stationary_top_does_not_repeat(self) -> None:
        runtime = PushupRepetitionRuntime(test_config())
        feed(runtime, [(170, 1.0, index * 0.1) for index in range(100)])
        self.assertEqual(runtime.repetition_count, 0)

    def test_confidence_changes_alone_never_count(self) -> None:
        runtime = PushupRepetitionRuntime(test_config())
        feed(
            runtime,
            [(170, confidence, index * 0.1) for index, confidence in enumerate([1.0, 0.2, 0.9, 0.0, 1.0] * 5)],
        )
        self.assertEqual(runtime.repetition_count, 0)

    def test_low_confidence_gap_invalidates_active_cycle_safely(self) -> None:
        runtime = PushupRepetitionRuntime(test_config(max_low_confidence_gap_frames=2))
        feed(
            runtime,
            [
                (170, 1.0, 0.0),
                (140, 1.0, 0.2),
                (None, 0.0, 0.3),
                (None, 0.0, 0.4),
                (None, 0.0, 0.5),
                (90, 1.0, 0.6),
                (120, 1.0, 0.8),
                (165, 1.0, 1.0),
            ],
        )
        self.assertEqual(runtime.repetition_count, 0)
        self.assertGreaterEqual(runtime.incomplete_cycles, 1)

    def test_disabled_counting_during_preparation_never_mutates_cycle(self) -> None:
        runtime = PushupRepetitionRuntime(test_config())
        for index, angle in enumerate((170, 140, 90, 120, 165)):
            result = runtime.update_signal(angle, 1.0, index * 0.2, index, counting_enabled=False)
            self.assertEqual(result.phase, "PREPARING")
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.phase, "READY")

    def test_reset_clears_session_state(self) -> None:
        runtime = PushupRepetitionRuntime(test_config())
        feed(runtime, [(170, 1.0, 0.0), (140, 1.0, 0.2), (90, 1.0, 0.5), (130, 1.0, 0.7), (165, 1.0, 1.0)])
        runtime.selected_side = "left"
        runtime.reset()
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.phase, "READY")
        self.assertIsNone(runtime.selected_side)
        self.assertIsNone(runtime.last_angle)
        self.assertEqual(runtime.incomplete_cycles, 0)


if __name__ == "__main__":
    unittest.main()
