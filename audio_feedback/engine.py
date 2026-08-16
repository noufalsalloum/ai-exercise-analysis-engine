"""Asynchronous bilingual TTS with neural cache, SAPI, and beep fallbacks."""

from __future__ import annotations

import asyncio
import hashlib
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .catalog import normalize_language


NEURAL_VOICES = {
    "ar": "ar-SA-HamedNeural",
    "en": "en-US-AriaNeural",
}
SAPI_VOICE_HINTS = {"ar": "Naayf", "en": "Zira"}
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "ai_fitness_tts_cache"

try:
    import winsound
except Exception:  # pragma: no cover - non-Windows platforms
    winsound = None


class HybridTTSEngine:
    """Non-blocking TTS retaining the package's three-tier degradation policy."""

    def __init__(
        self,
        language: str = "ar",
        *,
        enabled: bool = True,
        cache_dir: str | Path | None = None,
        completion_cooldown_seconds: float = 2.75,
        maximum_synthesis_seconds: float = 8.0,
    ) -> None:
        self.language = normalize_language(language)
        self.enabled = bool(enabled)
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.completion_cooldown_seconds = max(0.0, float(completion_cooldown_seconds))
        self.maximum_synthesis_seconds = max(0.1, float(maximum_synthesis_seconds))
        self.tier = "unknown"
        self.prewarm_stats: dict[str, Any] = {}

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._busy_until = 0.0
        self._state_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._stop = threading.Event()
        self._active = threading.Event()
        self._sapi_process: subprocess.Popen[str] | None = None
        self._mixer_ready = False
        self._edge_available: bool | None = None
        self._prewarm_threads: list[threading.Thread] = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(
            target=self._run,
            name="audio-feedback-worker",
            daemon=True,
        )
        self._worker.start()

    def set_language(self, language: str) -> None:
        self.language = normalize_language(language, self.language)

    def speak(self, text: str, *, language: str | None = None) -> bool:
        """Queue one utterance and return immediately."""

        cleaned = " ".join(str(text or "").split())
        if not self.enabled or self._stop.is_set() or not cleaned:
            return False
        if language is not None:
            self.set_language(language)
        now = time.monotonic()
        with self._state_lock:
            if self._active.is_set() or now < self._busy_until:
                return False
            self._busy_until = now + 1.0
        try:
            self._queue.put_nowait(cleaned)
            return True
        except queue.Full:
            with self._state_lock:
                self._busy_until = now
            return False

    def is_speaking(self) -> bool:
        with self._state_lock:
            return self._active.is_set() or time.monotonic() < self._busy_until

    def cache_path(self, text: str, language: str | None = None) -> Path:
        lang = normalize_language(language, self.language)
        voice = NEURAL_VOICES[lang]
        digest = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.mp3"

    @staticmethod
    def _valid_cache_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 512

    def cache_status(self, phrases: Iterable[str], language: str | None = None) -> tuple[int, int]:
        lang = normalize_language(language, self.language)
        total = have = 0
        for raw in phrases:
            text = " ".join(str(raw or "").split())
            if not text:
                continue
            total += 1
            have += int(self._valid_cache_file(self.cache_path(text, lang)))
        return have, total

    def prewarm(self, phrases: Iterable[str], language: str | None = None) -> threading.Thread | None:
        """Populate one language's neural cache on a daemon thread without playback."""

        lang = normalize_language(language, self.language)
        values = tuple(dict.fromkeys(" ".join(str(item or "").split()) for item in phrases))
        values = tuple(item for item in values if item)
        if not self.enabled or not values or self._stop.is_set():
            return None

        def work() -> None:
            synthesised = cached = failed = 0
            for text in values:
                if self._stop.is_set():
                    break
                path = self.cache_path(text, lang)
                if self._valid_cache_file(path):
                    cached += 1
                    continue
                while self.is_speaking() and not self._stop.wait(0.05):
                    pass
                if self._stop.is_set():
                    break
                if self._synthesize(text, NEURAL_VOICES[lang], path):
                    synthesised += 1
                else:
                    failed += 1
                    if self._edge_available is False:
                        break
                self._stop.wait(0.02)
            self.prewarm_stats[lang] = {
                "synthesised": synthesised,
                "cached": cached,
                "failed": failed,
            }

        thread = threading.Thread(
            target=work,
            name=f"audio-prewarm-{lang}",
            daemon=True,
        )
        self._prewarm_threads.append(thread)
        thread.start()
        return thread

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop queued work and release mixer/SAPI resources; safe to repeat."""

        if self._stop.is_set():
            return
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._stop_playback()
        self._shutdown_sapi()
        if self._worker.is_alive() and threading.current_thread() is not self._worker:
            self._worker.join(timeout=max(0.0, float(timeout)))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if text is None or self._stop.is_set():
                continue
            self._active.set()
            try:
                self._deliver(text)
            finally:
                self._active.clear()
                with self._state_lock:
                    self._busy_until = time.monotonic() + self.completion_cooldown_seconds

    def _deliver(self, text: str) -> str:
        if self._speak_neural(text):
            self.tier = "neural"
        elif self._speak_sapi(text):
            self.tier = "sapi"
        elif self._beep():
            self.tier = "beep"
        else:
            self.tier = "silent"
        return self.tier

    def _synthesize(self, text: str, voice: str, path: Path) -> bool:
        try:
            import edge_tts
        except Exception:
            self._edge_available = False
            return False
        temporary = path.with_suffix(path.suffix + ".part")
        with self._synthesis_lock:
            if self._valid_cache_file(path):
                self._edge_available = True
                return True

            async def synthesize() -> None:
                await edge_tts.Communicate(text, voice).save(str(temporary))

            try:
                asyncio.run(
                    asyncio.wait_for(
                        synthesize(),
                        timeout=self.maximum_synthesis_seconds,
                    )
                )
                if not self._valid_cache_file(temporary):
                    raise RuntimeError("Neural TTS returned an invalid cache file.")
                os.replace(temporary, path)
                self._edge_available = True
                return True
            except Exception:
                self._edge_available = False
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                return False

    def _ensure_mixer(self) -> bool:
        if self._mixer_ready:
            return True
        try:
            import pygame

            pygame.mixer.init()
            self._mixer_ready = True
            return True
        except Exception:
            self._mixer_ready = False
            return False

    def _speak_neural(self, text: str) -> bool:
        language = self.language
        path = self.cache_path(text, language)
        if not self._valid_cache_file(path):
            if self._edge_available is False or not self._synthesize(text, NEURAL_VOICES[language], path):
                return False
        if not self._ensure_mixer():
            return False
        try:
            import pygame

            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            deadline = time.monotonic() + 30.0
            while (
                not self._stop.is_set()
                and pygame.mixer.music.get_busy()
                and time.monotonic() < deadline
            ):
                self._stop.wait(0.02)
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            return not self._stop.is_set()
        except Exception:
            return False

    def _ensure_sapi(self) -> subprocess.Popen[str] | None:
        process = self._sapi_process
        if process is not None and process.poll() is None:
            return process
        script = Path(__file__).with_name("speak_server.ps1")
        if not script.is_file():
            return None
        try:
            self._sapi_process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return self._sapi_process
        except Exception:
            self._sapi_process = None
            return None

    @staticmethod
    def _readline_with_timeout(stream: Any, timeout: float) -> str | None:
        responses: queue.Queue[str | None] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                responses.put(stream.readline())
            except Exception:
                responses.put(None)

        threading.Thread(target=read, name="sapi-response", daemon=True).start()
        try:
            return responses.get(timeout=timeout)
        except queue.Empty:
            return None

    def _speak_sapi(self, text: str) -> bool:
        process = self._ensure_sapi()
        if process is None or process.stdin is None or process.stdout is None:
            return False
        try:
            process.stdin.write(f"{self.language}|{text}\n")
            process.stdin.flush()
            response = self._readline_with_timeout(process.stdout, 30.0)
            if response is not None and response.strip() == "DONE":
                return True
        except Exception:
            pass
        self._shutdown_sapi()
        return False

    def _beep(self) -> bool:
        if winsound is None:
            return False
        try:
            winsound.Beep(880, 120)
            return True
        except Exception:
            return False

    def _stop_playback(self) -> None:
        if not self._mixer_ready:
            return
        try:
            import pygame

            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

    def _shutdown_sapi(self) -> None:
        process = self._sapi_process
        self._sapi_process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("__QUIT__\n")
                process.stdin.flush()
                process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "language": self.language,
            "tier": self.tier,
            "neural_voice": NEURAL_VOICES[self.language],
            "cache_dir": str(self.cache_dir),
            "prewarm": dict(self.prewarm_stats),
        }


def dependency_probe() -> dict[str, bool]:
    try:
        import edge_tts  # noqa: F401

        edge_available = True
    except Exception:
        edge_available = False
    try:
        import pygame  # noqa: F401

        pygame_available = True
    except Exception:
        pygame_available = False
    return {
        "edge_tts": edge_available,
        "pygame": pygame_available,
        "winsound": winsound is not None,
        "sapi_script": Path(__file__).with_name("speak_server.ps1").is_file(),
    }
