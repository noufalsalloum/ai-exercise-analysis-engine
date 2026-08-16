"""Truthful cross-family experimental AI availability contracts."""

from __future__ import annotations

from typing import Any


STATUS_BY_FAMILY: dict[str, dict[str, Any]] = {
    "pushup": {
        "scope": "REHAB24 Ex3 table/incline Push-up only",
        "reason": "The learned development model is not applied to standard floor Push-up sessions.",
        "detection": "Boundary V1 Development — Table/Incline only",
        "correctness": "Correctness V1 Development — Table/Incline only",
        "errors": "No validated detailed-error labels",
    },
    "pullup": {
        "scope": "Rule-based standard Pull-up",
        "reason": "No local repetition-boundary or correctness ground truth is available for learned assessment.",
        "detection": "Not Available — rule-based count remains official",
        "correctness": "Not Available — no ground truth",
        "errors": "Not Available — no detailed-error ground truth",
    },
    "plank": {
        "scope": "Marching Plank primary variation",
        "reason": "Local clips have no subject, boundary, or correctness annotations; static-hold AI is not inferred from Marching Plank data.",
        "detection": "Not Available — rule-based leg-lift count remains official",
        "correctness": "Not Available — no ground truth",
        "errors": "Not Available — no detailed-error ground truth",
    },
}


def unavailable_family_ai(family_id: str, official_count: int = 0) -> dict[str, Any] | None:
    """Return a backward-compatible experimental contract without fake outputs."""
    status = STATUS_BY_FAMILY.get(family_id)
    if status is None:
        return None
    return {
        "exercise": family_id,
        "experimental": True,
        "status": "Development / External Validation Required",
        "available": False,
        "scope": status["scope"],
        "reason": status["reason"],
        "official_count": int(official_count),
        "ai_detected_count": None,
        "ai_detected_reps": None,
        "per_rep_results": [],
        "pass_count": None,
        "fail_count": None,
        "needs_review_count": None,
        "performance_score": None,
        "assessment_coverage": None,
        "model_status": {
            "detection": status["detection"],
            "correctness": status["correctness"],
            "errors": status["errors"],
        },
    }


def deterministic_rep_score(pass_count: int, fail_count: int) -> float | None:
    """MVP score for assessed reps only; NEEDS_REVIEW is excluded."""
    assessed = int(pass_count) + int(fail_count)
    return None if assessed <= 0 else 100.0 * int(pass_count) / assessed


def rule_based_parity_status(
    family_id: str,
    official_count: int,
    per_unit_results: list[dict[str, Any]] | None = None,
    *,
    current_phase: str | None = None,
) -> dict[str, Any]:
    """Product-parity contract for families with rule events but no learned labels."""
    if family_id not in {"pullup", "plank"}:
        raise ValueError("Rule-based parity is only defined for Pull-up and Marching Plank.")
    units = list(per_unit_results or [])
    latest = units[-1] if units else None
    plank = family_id == "plank"
    return {
        "exercise": family_id,
        "experimental": False,
        "available": True,
        "assessment_kind": "rule_based",
        "scope": "Marching Plank" if plank else "Standard Pull-up",
        "reason": "Learned correctness is not available because trustworthy ground truth is unavailable.",
        "official_count": int(official_count),
        "unit_count": len(units),
        "current_phase": current_phase,
        "latest_unit": latest,
        "per_unit_results": units,
        "pass_count": None,
        "fail_count": None,
        "needs_review_count": None,
        "performance_score": None,
        "assessment_coverage": None,
        "model_status": {
            "official_counter": "Rule-Based",
            "detection": "Not Available — official rule-based events are shown",
            "correctness": "Not Available",
            "errors": "Not Available",
            "score": "Not Available",
            "variation": "Marching Plank — Rule-Based MVP" if plank else "Standard Pull-up",
            **({"cross_knee_plank": "In Development"} if plank else {}),
        },
    }
