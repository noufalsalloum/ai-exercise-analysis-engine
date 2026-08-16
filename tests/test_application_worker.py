from __future__ import annotations

import queue
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from application.exercise_registry import ExerciseRegistry
from application.workers import AnalysisWorker
from input_sources.frame_sources import FramePacket
from preprocessing.landmark_selector import MEDIAPIPE_LANDMARKS


def landmarks_for_angle(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    landmarks = np.zeros((33, 4), dtype=np.float32)
    landmarks[:, 0] = 0.5
    landmarks[:, 1] = 0.5
    landmarks[:, 3] = 1.0
    for side in ("LEFT", "RIGHT"):
        shoulder = MEDIAPIPE_LANDMARKS[f"{side}_SHOULDER"]
        elbow = MEDIAPIPE_LANDMARKS[f"{side}_ELBOW"]
        wrist = MEDIAPIPE_LANDMARKS[f"{side}_WRIST"]
        landmarks[shoulder, :2] = (0.25, 0.5)
        landmarks[elbow, :2] = (0.45, 0.5)
        landmarks[wrist, :2] = (0.45 - 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle))
    return landmarks


class FakeSource:
    def __init__(self, timestamps: list[float]) -> None:
        self.timestamps = timestamps
        self.index = 0
        self.fps = 1.0
        self.closed = False

    def read(self) -> FramePacket | None:
        if self.closed or self.index >= len(self.timestamps):
            return None
        timestamp = self.timestamps[self.index]
        packet = FramePacket(
            np.zeros((240, 320, 3), dtype=np.uint8),
            self.index,
            timestamp,
            None,
        )
        self.index += 1
        return packet

    def close(self) -> None:
        self.closed = True


class FakePose:
    def __init__(self, angles: list[float]) -> None:
        self.angles = angles
        self.index = 0
        self.closed = False

    def process(self, _frame: np.ndarray, _timestamp: float) -> np.ndarray:
        angle = self.angles[min(self.index, len(self.angles) - 1)]
        self.index += 1
        return landmarks_for_angle(angle)

    def close(self) -> None:
        self.closed = True


class BlockingSource(FakeSource):
    def __init__(self) -> None:
        super().__init__([0.0])
        self.started_waiting = threading.Event()

    def read(self) -> FramePacket | None:
        if self.index == 0:
            return super().read()
        self.started_waiting.set()
        while not self.closed:
            time.sleep(0.005)
        return None


class ContinuousSource(FakeSource):
    def __init__(self) -> None:
        super().__init__([])
        self.continuous = True
        self.stream_lost = False
        self.backend_name = "FAKE_STREAM"
        self.camera_index = 0
        self.captured_frame_count = 0
        self.delivered_frame_count = 0
        self.dropped_frame_count = 0

    def read(self) -> FramePacket | None:
        if self.closed:
            return None
        time.sleep(0.003)
        packet = FramePacket(
            np.full((240, 320, 3), self.index % 255, dtype=np.uint8),
            self.index,
            self.index * 0.25,
            None,
        )
        self.index += 1
        self.captured_frame_count += 1
        self.delivered_frame_count += 1
        return packet


class ApplicationWorkerTests(unittest.TestCase):
    def test_countdown_frames_do_not_count_repetitions(self) -> None:
        source = FakeSource([0, 1, 2, 3, 4, 5])
        pose = FakePose([170, 140, 90, 120, 165, 170])
        events: queue.Queue[dict] = queue.Queue(maxsize=20)
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("pushup"),
            input_mode="realtime",
            camera_view="side",
            source_factory=lambda: source,
            pose_factory=lambda: pose,
            events=events,
        )
        worker.run_sync()
        complete = [event for event in list(events.queue) if event.get("type") == "complete"][-1]
        self.assertEqual(complete["result"]["summary"]["total_repetitions"], 0)
        self.assertTrue(source.closed)
        self.assertTrue(pose.closed)

    def test_stop_releases_camera_and_pose_resources(self) -> None:
        source = BlockingSource()
        pose = FakePose([170])
        events: queue.Queue[dict] = queue.Queue(maxsize=20)
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("pushup"),
            input_mode="realtime",
            camera_view="side",
            source_factory=lambda: source,
            pose_factory=lambda: pose,
            events=events,
        )
        worker.start()
        self.assertTrue(source.started_waiting.wait(timeout=2.0))
        worker.stop()
        worker.join(timeout=2.0)
        self.assertFalse(worker.running)
        self.assertTrue(source.closed)
        self.assertTrue(pose.closed)

    def test_untrained_neural_heads_are_not_called_by_rule_runtime(self) -> None:
        source = FakeSource([0.0, 0.2, 0.5, 0.7, 1.0])
        pose = FakePose([170, 140, 90, 120, 165])
        events: queue.Queue[dict] = queue.Queue(maxsize=20)
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("pushup"),
            input_mode="video",
            camera_view="side",
            source_factory=lambda: source,
            pose_factory=lambda: pose,
            events=events,
        )
        with patch("heads.phase_head.PhaseHead.forward", side_effect=AssertionError("untrained head called")), patch(
            "heads.passfail_head.PassFailHead.forward",
            side_effect=AssertionError("untrained head called"),
        ), patch("heads.error_head.ErrorHead.forward", side_effect=AssertionError("untrained head called")):
            worker.run_sync()
        complete = [event for event in list(events.queue) if event.get("type") == "complete"][-1]
        self.assertFalse(complete["result"]["capabilities"]["pass_fail"])
        self.assertFalse(complete["result"]["capabilities"]["errors"])
        self.assertFalse(complete["result"]["capabilities"]["score"])

    def test_realtime_preview_and_analysis_continue_across_countdown(self) -> None:
        source = ContinuousSource()
        pose = FakePose([170, 140, 90, 120, 165] * 20)
        events: queue.Queue[dict] = queue.Queue(maxsize=200)
        worker = AnalysisWorker(
            exercise=ExerciseRegistry().get("pushup"),
            input_mode="realtime",
            camera_view="side",
            source_factory=lambda: source,
            pose_factory=lambda: pose,
            events=events,
        )
        worker.start()
        deadline = time.monotonic() + 3.0
        while source.index < 30 and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.stop()
        worker.join(timeout=3.0)
        self.assertFalse(worker.running)
        frame_events = [event for event in list(events.queue) if event.get("type") == "frame"]
        self.assertGreaterEqual(len(frame_events), 25)
        self.assertGreater(len({event["frame_index"] for event in frame_events}), 20)
        self.assertTrue(any(event["metrics"].get("countdown") for event in frame_events))
        self.assertTrue(any(event["metrics"].get("session_time", 0.0) > 0 for event in frame_events))
        self.assertTrue(source.closed)
        self.assertTrue(pose.closed)


if __name__ == "__main__":
    unittest.main()
