"""Small, explicit bilingual catalog for currently supported audio events."""

from __future__ import annotations

from collections.abc import Iterable


SUPPORTED_LANGUAGES = ("ar", "en")
PHRASES: dict[str, dict[str, str]] = {
    "first_repetition_completed": {
        "ar": "أحسنت",
        "en": "Great job!",
    },
}


def normalize_language(language: str | None, default: str = "ar") -> str:
    value = str(language or default).strip().lower()
    return value if value in SUPPORTED_LANGUAGES else default


def phrase_for(event_type: str, language: str) -> str | None:
    translations = PHRASES.get(event_type)
    if translations is None:
        return None
    return translations.get(normalize_language(language))


def prewarm_phrases(languages: Iterable[str] = SUPPORTED_LANGUAGES) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for raw_language in languages:
        language = normalize_language(raw_language)
        phrases = [translations[language] for translations in PHRASES.values()]
        output[language] = list(dict.fromkeys(phrases))
    return output
