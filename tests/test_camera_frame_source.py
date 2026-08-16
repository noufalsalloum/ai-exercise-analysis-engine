from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from input_sources.frame_sources import CameraFrameSource


class FakeCapture:
    def __init__(self, opened: bool, backend_name: str, fail_after: int | None = None) -> None:
        self.opened = opened
        self.backend_name = backend_name
        self.fail_after = fail_after
        self.read_count = 0
        self.released = False
        self.lock = threading.Lock()

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.003)
        with self.lock:
            if self.released or not self.opened:
                return False, None
            if self.fail_after is not None and self.read_count >= self.fail_after:
                return False, None
            value = self.read_count % 255
            self.read_count += 1
        return True, np.full((24, 32, 3), value, dtype=np.uint8)

    def get(self, property_id: int) -> float:
        return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

    def getBackendName(self) -> str:
        return self.backend_name

    def release(self) -> None:
        self.released = True


class CameraFrameSourceTests(unittest.TestCase):
    def test_windows_backend_and_index_fallback_requires_valid_frame(self) -> None:
        calls: list[tuple[int, int | None]] = []
        captures: list[FakeCapture] = []

        def factory(index: int, backend: int | None = None) -> FakeCapture:
            calls.append((index, backend))
            succeeds = index == 2 and backend == int(cv2.CAP_MSMF)
            capture = FakeCapture(succeeds, "MSMF")
            captures.append(capture)
            return capture

        with patch("input_sources.frame_sources.os.name", "nt"):
            source = CameraFrameSource(
                camera_index=0,
                capture_factory=factory,
                candidate_indices=(0, 1, 2, 3),
                initial_read_attempts=1,
            )
        try:
            self.assertTrue(source.first_frame_verified)
            self.assertEqual(source.camera_index, 2)
            self.assertEqual(source.backend_name, "MSMF")
            self.assertIn((0, int(cv2.CAP_DSHOW)), calls)
            self.assertIn((0, int(cv2.CAP_MSMF)), calls)
            self.assertIn((2, int(cv2.CAP_MSMF)), calls)
            self.assertIsNotNone(source.read())
        finally:
            source.close()
        self.assertTrue(all(capture.released for capture in captures))

    def test_continuous_capture_returns_multiple_new_frames_and_drops_backlog(self) -> None:
        capture = FakeCapture(True, "FAKE")
        source = CameraFrameSource(
            capture_factory=lambda *_args: capture,
            candidate_indices=(0,),
            initial_read_attempts=1,
        )
        try:
            first = source.read()
            time.sleep(0.06)
            second = source.read()
            time.sleep(0.03)
            third = source.read()
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertIsNotNone(third)
            assert first is not None and second is not None and third is not None
            self.assertLess(first.frame_index, second.frame_index)
            self.assertLess(second.frame_index, third.frame_index)
            self.assertGreater(source.captured_frame_count, 3)
            self.assertGreater(source.dropped_frame_count, 0)
        finally:
            source.close()
        self.assertTrue(capture.released)
        self.assertFalse(source._thread.is_alive())

    def test_open_capture_without_decodable_frame_is_rejected(self) -> None:
        captures: list[FakeCapture] = []

        def factory(*_args: object) -> FakeCapture:
            capture = FakeCapture(True, "EMPTY", fail_after=0)
            captures.append(capture)
            return capture

        with patch("input_sources.frame_sources.os.name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "No working camera stream"):
                CameraFrameSource(
                    capture_factory=factory,
                    candidate_indices=(0,),
                    initial_read_attempts=1,
                )
        self.assertTrue(all(capture.released for capture in captures))


if __name__ == "__main__":
    unittest.main()
