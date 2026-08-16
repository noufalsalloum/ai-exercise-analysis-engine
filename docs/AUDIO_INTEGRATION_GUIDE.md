# Bilingual Audio Feedback Integration

## Purpose

The audio integration is an optional output channel. It never participates in pose extraction, repetition counting, model inference, thresholds, scoring, or UI rendering.

```text
AI / Rule Runtime
  -> FeedbackEvent
  -> AudioFeedbackService
  -> HybridTTSEngine queue
  -> Tier 1 neural/cache
  -> Tier 2 Windows SAPI
  -> Tier 3 beep
```

The current proof event is emitted once, after the first officially completed repetition:

- Arabic: `أحسنت`
- English: `Great job!`

No detailed exercise error is spoken because the application does not have validated detailed feedback for every family.

## Components

- `application.feedback_events.FeedbackEvent`: transport-neutral semantic event.
- `audio_feedback.service.AudioFeedbackService`: language resolution, deduplication, cooldown, and lifecycle.
- `audio_feedback.engine.HybridTTSEngine`: asynchronous queue, cache, synthesis, playback, and fallback tiers.
- `audio_feedback.catalog`: supported bilingual event text.
- `audio_feedback/speak_server.ps1`: persistent Windows SAPI server.
- `configs/audio_feedback.json`: operational configuration.
- `docs/audio_feedback_contract.json`: machine-readable event contract.

## Initialization

```python
from audio_feedback.service import AudioFeedbackService

audio = AudioFeedbackService.from_default_config()
```

The desktop application injects this factory into each `AnalysisWorker`. The worker owns the service, starts cache pre-warming in background threads, and calls `shutdown()` in its existing `finally` cleanup path.

Default language is Arabic. Override it without changing UI or runtime code:

```powershell
$env:AI_ENGINE_AUDIO_LANGUAGE = "en"
python run_application.py
```

Accepted values are `ar` and `en`; invalid values fall back to the configured default.

## Publishing Events

```python
from application.feedback_events import FeedbackEvent

event = FeedbackEvent.first_repetition_completed(
    session_id="session-123",
    exercise_id="squat",
    family_id="squat",
    timestamp_seconds=4.2,
    language="ar",  # optional
)
audio.submit(event)
```

`submit()` returns immediately. It returns `False` for unsupported events, duplicates, cooldown rejection, a closed service, or a busy one-item TTS queue.

## Backend / Mobile / Web Mapping

Other clients do not need any knowledge of MediaPipe, MotionBERT, experts, or checkpoints. Serialize the event described by `audio_feedback_contract.json`, then map it to the platform audio adapter:

```text
Backend event bus -> first_repetition_completed payload
Mobile/Web client -> localized phrase catalog -> native/cloud TTS
Desktop Python -> AudioFeedbackService -> HybridTTSEngine
```

Keep `event_id` stable for retries; it is the deduplication key.

## Cache and Pre-warming

Tier 1 uses Edge-TTS voices:

- Arabic: `ar-SA-HamedNeural`
- English: `en-US-AriaNeural`

Files are cached under `%TEMP%/ai_fitness_tts_cache` by a hash of voice and normalized text. Arabic and English never share a cache entry. On service initialization, the supported phrases for both languages are pre-warmed on daemon threads. Pre-warm yields while live audio is busy and never touches camera/video processing.

Use the API directly when diagnostics need cache coverage:

```python
from audio_feedback.catalog import prewarm_phrases

for language, phrases in prewarm_phrases().items():
    have, total = audio.engine.cache_status(phrases, language)
    print(language, have, total)
```

## Fallback Tiers

1. **Neural/cache:** play a valid cached MP3, or synthesize it online with Edge-TTS and cache it.
2. **Windows SAPI:** persistent PowerShell/System.Speech process resolves an installed voice by language and replies `DONE`, `NOVOICE`, or `FAIL`.
3. **Beep:** `winsound.Beep` if neural playback and SAPI are unavailable.

Failures are contained within audio. They never stop a worker or change official/AI results.

## Queue, Cooldown, and Lifecycle

- Queue capacity is one phrase.
- A phrase in progress is not interrupted.
- New phrases during playback/cooldown are dropped, preventing delayed narration and repetition.
- `event_id` is accepted once per service/session.
- Stop, Back, New Session, normal completion, errors, and application Close all reach the worker `finally` block and call `shutdown()`.
- `shutdown()` clears queued work, stops playback, terminates SAPI safely, and is idempotent.

## Configuration

`configs/audio_feedback.json` controls:

- enable/disable;
- default language and environment override;
- event and completion cooldowns;
- synthesis timeout;
- cache directory;
- pre-warm languages;
- documented fallback order and voices.

Setting `enabled` to `false` disables speech without changing analysis behavior.

## Adding New Feedback

1. Add a new semantic event type to `FeedbackEventType`.
2. Add both Arabic and English text to `audio_feedback/catalog.py`.
3. Add the event to `configs/audio_feedback.json` and the JSON contract.
4. Emit it only from a verified runtime fact.
5. Add deduplication, cooldown, cache, language, and fallback tests.

Do not add named form errors unless the corresponding exercise has validated labels and a reliable source for that event.
