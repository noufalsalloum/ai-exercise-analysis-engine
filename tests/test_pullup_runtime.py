from __future__ import annotations

import json
import unittest
from pathlib import Path

from application.exercise_registry import PROJECT_ROOT
from application.runtime_router import FamilyRuntimeRouter
from inference.pullup_runtime import PullupRepConfig, PullupRepetitionRuntime
from inference.pushup_runtime import PushupRepetitionRuntime


def test_config(**overrides: object) -> PullupRepConfig:
    values = {
        "angle_smoothing_window": 1,
        "vertical_smoothing_window": 1,
        "minimum_repetition_duration": 0.30,
        "cooldown_seconds": 0.0,
        "max_low_confidence_gap_frames": 2,
    }
    values.update(overrides)
    return PullupRepConfig(**values)


def feed(
    runtime: PullupRepetitionRuntime,
    values: list[tuple[float | None, float | None, bool, float, float]],
    *,
    counting_enabled: bool = True,
) -> list:
    return [
        runtime.update_signals(
            angle,
            body_y,
            wrists_above,
            confidence,
            timestamp,
            index,
            counting_enabled=counting_enabled,
        )
        for index, (angle, body_y, wrists_above, confidence, timestamp) in enumerate(values)
    ]


FULL_CYCLE = [
    (165.0, 0.62, True, 1.0, 0.0),
    (135.0, 0.59, True, 1.0, 0.2),
    (80.0, 0.51, True, 1.0, 0.5),
    (120.0, 0.55, True, 1.0, 0.7),
    (160.0, 0.61, True, 1.0, 1.0),
]


class PullupRuntimeTests(unittest.TestCase):
    def test_pullup_uses_independent_runtime_and_config(self) -> None:
        router = FamilyRuntimeRouter()
        pullup = router.create("pullup", "video", camera_view="front")
        pushup = router.create("pushup", "video", camera_view="side")
        self.assertIsInstance(pullup, PullupRepetitionRuntime)
        self.assertIsInstance(pushup, PushupRepetitionRuntime)
        self.assertIsNot(type(pullup), type(pushup))
        pullup_payload = json.loads((PROJECT_ROOT / "configs" / "pullup.json").read_text(encoding="utf-8"))
        pushup_payload = json.loads((PROJECT_ROOT / "configs" / "pushup.json").read_text(encoding="utf-8"))
        self.assertNotEqual(
            pullup_payload["runtime"]["repetition_counter"],
            pushup_payload["runtime"]["repetition_counter"],
        )

    def test_complete_hang_to_hang_cycle_counts_once(self) -> None:
        runtime = PullupRepetitionRuntime(test_config())
        results = feed(runtime, FULL_CYCLE)
        self.assertEqual(runtime.repetition_count, 1)
        cycle = results[-1].completed_cycle
        self.assertIsNotNone(cycle)
        self.assertEqual(cycle.phase_sequence, ["HANG", "ASCENDING", "TOP", "DESCENDING", "HANG"])

    def test_partial_ascent_does_not_count(self) -> None:
        runtime = PullupRepetitionRuntime(test_config())
        feed(
            runtime,
            [
                (165.0, 0.62, True, 1.0, 0.0),
                (135.0, 0.59, True, 1.0, 0.2),
                (115.0, 0.56, True, 1.0, 0.5),
                (160.0, 0.61, True, 1.0, 0.9),
            ],
        )
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.incomplete_cycles, 1)

    def test_top_hold_never_repeats_and_hang_return_is_required(self) -> None:
        runtime = PullupRepetitionRuntime(test_config())
        values = FULL_CYCLE[:3] + [(75.0, 0.50, True, 1.0, 0.6 + i * 0.1) for i in range(20)]
        feed(runtime, values)
        self.assertEqual(runtime.repetition_count, 0)
        runtime.finish()
        self.assertEqual(runtime.incomplete_cycles, 1)

    def test_swing_without_elbow_rom_does_not_count(self) -> None:
        runtime = PullupRepetitionRuntime(test_config())
        feed(
            runtime,
            [
                (165.0, body_y, True, 1.0, index * 0.2)
                for index, body_y in enumerate((0.62, 0.52, 0.45, 0.52, 0.62) * 3)
            ],
        )
        self.assertEqual(runtime.repetition_count, 0)

    def test_low_confidence_gap_invalidates_cycle(self) -> None:
        runtime = PullupRepetitionRuntime(test_config())
        feed(
            runtime,
            [
                (165.0, 0.62, True, 1.0, 0.0),
                (135.0, 0.59, True, 1.0, 0.2),
                (80.0, 0.51, True, 1.0, 0.5),
                (80.0, 0.50, True, 0.1, 0.6),
                (80.0, 0.50, True, 0.1, 0.7),
                (80.0, 0.50, True, 0.1, 0.8),
                (120.0, 0.55, True, 1.0, 0.9),
                (160.0, 0.61, True, 1.0, 1.2),
            ],
        )
        self.assertEqual(runtime.repetition_count, 0)
        self.assertGreaterEqual(runtime.incomplete_cycles, 1)

    def test_countdown_signals_do_not_mutate_pullup_cycle(self) -> None:
        runtime = PullupRepetitionRuntime(test_config())
        results = feed(runtime, FULL_CYCLE, counting_enabled=False)
        self.assertTrue(all(item.phase == "PREPARING" for item in results))
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.phase, "READY")
        self.assertEqual(runtime.incomplete_cycles, 0)

    def test_reset_clears_family_specific_history(self) -> None:
        runtime = PullupRepetitionRuntime(test_config(), camera_view="side")
        feed(runtime, FULL_CYCLE)
        runtime.selected_side = "left"
        runtime.reset()
        self.assertEqual(runtime.repetition_count, 0)
        self.assertEqual(runtime.phase, "READY")
        self.assertIsNone(runtime.selected_side)
        self.assertIsNone(runtime.last_vertical_body_motion)
        self.assertEqual(runtime.incomplete_cycles, 0)


if __name__ == "__main__":
    unittest.main()
