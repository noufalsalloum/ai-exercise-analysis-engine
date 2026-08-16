from __future__ import annotations

import json
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from application.exercise_registry import ExerciseRegistry
from application.feedback_events import FeedbackEvent
from application.workers import AnalysisWorker
from audio_feedback.catalog import phrase_for, prewarm_phrases
from audio_feedback.engine import HybridTTSEngine, NEURAL_VOICES
from audio_feedback.service import AudioFeedbackService
from input_sources.frame_sources import FramePacket


ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    def __init__(self) -> None:
        self.language = "ar"
        self.spoken: list[tuple[str, str]] = []
        self.prewarmed: list[tuple[str, tuple[str, ...]]] = []
        self.closed = False

    def set_language(self, language: str) -> None:
        self.language = language

    def speak(self, text: str, *, language: str | None = None) -> bool:
        self.spoken.append((language or self.language, text))
        return True

    def prewarm(self, phrases: list[str], language: str) -> None:
        self.prewarmed.append((language, tuple(phrases)))
        return None

    def shutdown(self) -> None:
        self.closed = True

    def status(self) -> dict[str, object]:
        return {"language": self.language, "tier": "fake"}


class FakeAudioService:
    def __init__(self) -> None:
        self.events: list[FeedbackEvent] = []
        self.closed = False

    def submit(self, event: FeedbackEvent) -> bool:
        self.events.append(event)
        return True

    def shutdown(self) -> None:
        self.closed = True


class FakeSource:
    fps = 30.0
    stream_lost = False
    backend_name = "TEST"
    camera_index = None

    def __init__(self) -> None:
        self.index = 0
        self.closed = False

    def read(self) -> FramePacket | None:
        if self.closed or self.index >= 2:
            return None
        packet = FramePacket(
            np.zeros((32, 32, 3), dtype=np.uint8),
            self.index,
            self.index / self.fps,
            None,
        )
        self.index += 1
        return packet

    def close(self) -> None:
        self.closed = True


class FakePose:
    def process(self, _frame: np.ndarray, _timestamp: float) -> np.ndarray:
        landmarks = np.zeros((33, 4), dtype=np.float32)
        landmarks[:, 0:2] = 0.5
        landmarks[:, 3] = 1.0
        return landmarks

    def close(self) -> None:
        pass


class TwoCycleRuntime:
    def __init__(self) -> None:
        self.count = 0
        self.incomplete_cycles = 0
        self.phase = "TOP"

    def reset(self) -> None:
        self.count = 0

    def update_landmarks(self, _landmarks, timestamp, frame_index, counting_enabled=True):
        self.count += int(bool(counting_enabled))
        cycle = SimpleNamespace(
            start_frame=frame_index,
            end_frame=frame_index,
            start_time=timestamp,
            end_time=timestamp,
            duration_seconds=0.1,
            phase_sequence=("TOP", "BOTTOM", "TOP"),
            confidence=1.0,
        )
        return SimpleNamespace(
            phase="TOP",
            repetition_count=self.count,
            elbow_angle=170.0,
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
    def __init__(self, runtime: TwoCycleRuntime) -> None:
        self.runtime = runtime

    def create(self, *_args, **_kwargs) -> TwoCycleRuntime:
        return self.runtime


class AudioFeedbackTests(unittest.TestCase):
    def test_bilingual_first_rep_phrases(self) -> None:
        self.assertEqual(phrase_for("first_repetition_completed", "ar"), "أحسنت")
        self.assertEqual(phrase_for("first_repetition_completed", "en"), "Great job!")

    def test_service_prewarm_and_async_submission_contract(self) -> None:
        engine = FakeEngine()
        service = AudioFeedbackService(engine, event_cooldown_seconds=0.0)
        expected = prewarm_phrases(("ar", "en"))
        self.assertEqual(dict(engine.prewarmed), {key: tuple(value) for key, value in expected.items()})
        event = FeedbackEvent.first_repetition_completed(
            session_id="session",
            exercise_id="squat",
            family_id="squat",
            timestamp_seconds=1.0,
            language="en",
        )
        started = time.perf_counter()
        self.assertTrue(service.submit(event))
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertEqual(engine.spoken, [("en", "Great job!")])
        self.assertFalse(service.submit(event))
        service.shutdown()
        self.assertTrue(engine.closed)

    def test_cache_key_separates_language_and_reports_cached_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = HybridTTSEngine(cache_dir=directory, enabled=False)
            arabic = engine.cache_path("أحسنت", "ar")
            english = engine.cache_path("Great job!", "en")
            self.assertNotEqual(arabic, english)
            arabic.write_bytes(b"x" * 513)
            self.assertEqual(engine.cache_status(["أحسنت"], "ar"), (1, 1))
            self.assertEqual(engine.cache_status(["Great job!"], "en"), (0, 1))
            engine.shutdown()

    def test_fallback_order_neural_sapi_beep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = HybridTTSEngine(cache_dir=directory, enabled=False)
            with patch.object(engine, "_speak_neural", return_value=True), patch.object(
                engine, "_speak_sapi", return_value=True
            ), patch.object(engine, "_beep", return_value=True):
                self.assertEqual(engine._deliver("Great job!"), "neural")
            with patch.object(engine, "_speak_neural", return_value=False), patch.object(
                engine, "_speak_sapi", return_value=True
            ), patch.object(engine, "_beep", return_value=True):
                self.assertEqual(engine._deliver("Great job!"), "sapi")
            with patch.object(engine, "_speak_neural", return_value=False), patch.object(
                engine, "_speak_sapi", return_value=False
            ), patch.object(engine, "_beep", return_value=True):
                self.assertEqual(engine._deliver("Great job!"), "beep")
            engine.shutdown()

    def test_active_worker_remains_busy_after_provisional_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = HybridTTSEngine(cache_dir=directory, enabled=False)
            engine._active.set()
            engine._busy_until = time.monotonic() - 1.0
            self.assertTrue(engine.is_speaking())
            self.assertFalse(engine.speak("Great job!"))
            engine._active.clear()
            engine.shutdown()

    def test_worker_emits_only_first_completed_rep_and_closes_audio(self) -> None:
        service = FakeAudioService()
        events: queue.Queue[dict] = queue.Queue(maxsize=20)
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("pushup"),
            input_mode="video",
            camera_view="side",
            source_factory=FakeSource,
            pose_factory=FakePose,
            events=events,
            runtime_router=FakeRouter(TwoCycleRuntime()),
            preserve_video_timing=False,
            audio_feedback_factory=lambda: service,
        )
        worker.run_sync()
        self.assertEqual(len(service.events), 1)
        self.assertEqual(service.events[0].event_type, "first_repetition_completed")
        self.assertEqual(service.events[0].unit_index, 1)
        self.assertTrue(service.closed)

    def test_config_preserves_hamed_and_fallback_order(self) -> None:
        config = json.loads((ROOT / "configs" / "audio_feedback.json").read_text(encoding="utf-8"))
        self.assertEqual(NEURAL_VOICES["ar"], "ar-SA-HamedNeural")
        self.assertEqual(config["voices"]["ar"], "ar-SA-HamedNeural")
        self.assertEqual(config["fallback_order"], ["neural_or_cache", "windows_sapi", "beep"])


if __name__ == "__main__":
    unittest.main()
