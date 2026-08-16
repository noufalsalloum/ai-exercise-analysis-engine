"""Semantic feedback-event adapter for the asynchronous hybrid TTS engine."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from application.feedback_events import FeedbackEvent

from .catalog import normalize_language, phrase_for, prewarm_phrases
from .engine import HybridTTSEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "audio_feedback.json"


class AudioFeedbackService:
    """Translate runtime events into queued speech without exposing TTS to runtimes."""

    def __init__(
        self,
        engine: HybridTTSEngine,
        *,
        language: str = "ar",
        event_cooldown_seconds: float = 3.0,
        prewarm_languages: tuple[str, ...] = ("ar", "en"),
        prewarm_enabled: bool = True,
    ) -> None:
        self.engine = engine
        self.language = normalize_language(language)
        self.event_cooldown_seconds = max(0.0, float(event_cooldown_seconds))
        self._lock = threading.Lock()
        self._seen_event_ids: set[str] = set()
        self._cooldown_until: dict[str, float] = {}
        self._closed = False
        self._prewarm_threads: list[threading.Thread] = []
        self.engine.set_language(self.language)
        if prewarm_enabled:
            for lang, phrases in prewarm_phrases(prewarm_languages).items():
                thread = self.engine.prewarm(phrases, lang)
                if thread is not None:
                    self._prewarm_threads.append(thread)

    @classmethod
    def from_default_config(cls, path: str | Path = DEFAULT_CONFIG) -> "AudioFeedbackService":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        environment_name = str(config.get("language_environment_variable", "AI_ENGINE_AUDIO_LANGUAGE"))
        language = normalize_language(os.getenv(environment_name), str(config.get("default_language", "ar")))
        cache_dir = config.get("cache_dir")
        engine = HybridTTSEngine(
            language,
            enabled=bool(config.get("enabled", True)),
            cache_dir=cache_dir,
            completion_cooldown_seconds=float(config.get("completion_cooldown_seconds", 2.75)),
            maximum_synthesis_seconds=float(config.get("maximum_synthesis_seconds", 8.0)),
        )
        languages = tuple(config.get("prewarm_languages", ["ar", "en"]))
        return cls(
            engine,
            language=language,
            event_cooldown_seconds=float(config.get("event_cooldown_seconds", 3.0)),
            prewarm_languages=languages,
            prewarm_enabled=bool(config.get("prewarm_enabled", True)),
        )

    def submit(self, event: FeedbackEvent) -> bool:
        """Resolve and enqueue one event; unsupported/duplicate events are ignored."""

        language = normalize_language(event.language, self.language)
        text = phrase_for(event.event_type, language)
        if text is None:
            return False
        now = time.monotonic()
        with self._lock:
            if self._closed or event.event_id in self._seen_event_ids:
                return False
            if now < self._cooldown_until.get(event.event_type, 0.0):
                return False
            accepted = self.engine.speak(text, language=language)
            if not accepted:
                return False
            self._seen_event_ids.add(event.event_id)
            self._cooldown_until[event.event_type] = now + self.event_cooldown_seconds
            return True

    def reset(self) -> None:
        """Clear event deduplication for a new session without changing configuration."""

        with self._lock:
            self._seen_event_ids.clear()
            self._cooldown_until.clear()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.engine.shutdown()

    def status(self) -> dict[str, Any]:
        result = self.engine.status()
        result.update(
            {
                "service_closed": self._closed,
                "event_cooldown_seconds": self.event_cooldown_seconds,
                "seen_event_count": len(self._seen_event_ids),
            }
        )
        return result
