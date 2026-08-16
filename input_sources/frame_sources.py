from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Protocol

import cv2
import numpy as np


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray]: ...
    def get(self, property_id: int) -> float: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame with source-preserving timing."""

    frame: np.ndarray
    frame_index: int
    timestamp_seconds: float
    progress: float | None


def _valid_frame(success: bool, frame: np.ndarray | None) -> bool:
    return bool(success and frame is not None and frame.ndim == 3 and frame.size > 0)


class VideoFrameSource:
    """Uploaded-video source with deterministic frame/FPS timestamps."""

    def __init__(
        self,
        video_path: str | Path,
        capture_factory: Callable[[str], CaptureLike] = cv2.VideoCapture,
        max_frames: int | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        self._capture = capture_factory(str(self.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"OpenCV cannot open this video or codec: {self.video_path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= self.fps <= 240.0:
            self.fps = 30.0
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = frame_count if frame_count > 0 else None
        self._frame_index = 0
        if max_frames is not None and max_frames <= 0:
            self._capture.release()
            raise ValueError("max_frames must be positive when provided.")
        self.max_frames = max_frames
        self.closed = False
        self.stream_lost = False
        self.continuous = False
        self.backend_name = "video_file"
        self.camera_index: int | None = None

    def read(self) -> FramePacket | None:
        if self.closed or (
            self.max_frames is not None and self._frame_index >= self.max_frames
        ):
            return None
        success, frame = self._capture.read()
        if not _valid_frame(success, frame):
            return None
        index = self._frame_index
        self._frame_index += 1
        progress = (
            min(1.0, self._frame_index / self.frame_count)
            if self.frame_count is not None
            else None
        )
        return FramePacket(frame, index, index / self.fps, progress)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._capture.release()


class CameraFrameSource:
    """Low-latency Windows camera stream that always exposes the latest frame.

    A dedicated capture thread prevents OpenCV/backend buffers from building a
    multi-second delay while MediaPipe is processing. Old frames are replaced,
    never queued, so :meth:`read` returns the newest verified frame available.
    """

    def __init__(
        self,
        camera_index: int = 0,
        capture_factory: Callable[..., CaptureLike] = cv2.VideoCapture,
        candidate_indices: tuple[int, ...] | None = None,
        max_read_failures: int = 20,
        initial_read_attempts: int = 4,
    ) -> None:
        if max_read_failures <= 0 or initial_read_attempts <= 0:
            raise ValueError("Camera retry limits must be positive.")
        ordered_indices = [int(camera_index)]
        for index in candidate_indices or (0, 1, 2, 3):
            if int(index) not in ordered_indices:
                ordered_indices.append(int(index))
        self.attempt_log: list[dict[str, object]] = []
        self._capture_factory = capture_factory
        self._capture: CaptureLike | None = None
        self.camera_index = int(camera_index)
        self.backend_name = "unknown"
        self.max_read_failures = int(max_read_failures)
        first_frame: np.ndarray | None = None

        backends: list[tuple[str, int | None]] = []
        if os.name == "nt":
            backends.extend(
                [
                    ("DSHOW", int(cv2.CAP_DSHOW)),
                    ("MSMF", int(cv2.CAP_MSMF)),
                ]
            )
        backends.append(("DEFAULT", None))

        for index in ordered_indices:
            for backend_name, backend in backends:
                capture = self._create_capture(index, backend)
                opened = bool(capture.isOpened())
                frame_ok = False
                frame: np.ndarray | None = None
                if opened:
                    for _ in range(initial_read_attempts):
                        success, candidate = capture.read()
                        if _valid_frame(success, candidate):
                            frame = candidate
                            frame_ok = True
                            break
                        sleep(0.03)
                self.attempt_log.append(
                    {
                        "index": index,
                        "backend": backend_name,
                        "opened": opened,
                        "valid_frame": frame_ok,
                    }
                )
                if opened and frame_ok and frame is not None:
                    self._capture = capture
                    self.camera_index = index
                    self.backend_name = self._backend_label(capture, backend_name)
                    first_frame = frame
                    break
                capture.release()
            if self._capture is not None:
                break

        if self._capture is None or first_frame is None:
            attempts = ", ".join(
                f"index {item['index']} / {item['backend']}"
                for item in self.attempt_log
            )
            raise RuntimeError(
                "No working camera stream was found. OpenCV tried "
                f"{attempts}. Close other camera applications and verify Windows "
                "camera privacy permissions."
            )

        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self._started = monotonic()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._latest = FramePacket(first_frame, 0, 0.0, None)
        self._last_delivered_index = -1
        self._next_frame_index = 1
        self.captured_frame_count = 1
        self.delivered_frame_count = 0
        self.read_failure_count = 0
        self.stream_lost = False
        self.closed = False
        self.first_frame_verified = True
        self.continuous = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()

    def _create_capture(self, index: int, backend: int | None) -> CaptureLike:
        if backend is None:
            return self._capture_factory(index)
        try:
            return self._capture_factory(index, backend)
        except TypeError:
            return self._capture_factory(index)

    @staticmethod
    def _backend_label(capture: CaptureLike, fallback: str) -> str:
        getter = getattr(capture, "getBackendName", None)
        if callable(getter):
            try:
                value = str(getter())
                if value:
                    return value
            except (cv2.error, RuntimeError):
                pass
        return fallback

    def _capture_loop(self) -> None:
        capture = self._capture
        if capture is None:
            return
        consecutive_failures = 0
        while not self._stop_event.is_set():
            success, frame = capture.read()
            if not _valid_frame(success, frame):
                consecutive_failures += 1
                self.read_failure_count += 1
                if consecutive_failures >= self.max_read_failures:
                    with self._condition:
                        self.stream_lost = True
                        self._condition.notify_all()
                    return
                sleep(0.03)
                continue
            consecutive_failures = 0
            packet = FramePacket(
                frame=frame,
                frame_index=self._next_frame_index,
                timestamp_seconds=monotonic() - self._started,
                progress=None,
            )
            self._next_frame_index += 1
            self.captured_frame_count += 1
            with self._condition:
                self._latest = packet
                self._condition.notify_all()

    @property
    def dropped_frame_count(self) -> int:
        return max(0, self.captured_frame_count - self.delivered_frame_count - 1)

    def read(self, timeout_seconds: float = 1.0) -> FramePacket | None:
        deadline = monotonic() + max(0.01, float(timeout_seconds))
        with self._condition:
            while not self.closed and not self.stream_lost:
                packet = self._latest
                if packet is not None and packet.frame_index > self._last_delivered_index:
                    self._last_delivered_index = packet.frame_index
                    self.delivered_frame_count += 1
                    return packet
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
        return None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        with self._condition:
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
