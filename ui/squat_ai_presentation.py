"""Pure presentation helpers for the Experimental Squat AI contract.

This module formats existing MVP output for Tkinter. It never changes model
decisions, repetition counting, thresholds, or persisted session contracts.
"""

from __future__ import annotations

from typing import Any

from application.presentation_contract import NOT_AVAILABLE, PROCESSING, WAITING_REPETITION, completed_unit


ERROR_LABELS = {
    "bad_back": "Bad Back",
    "bad_heel": "Bad Heel",
    "form_issue": "Form Issue",
}

DEFAULT_MODEL_STATUSES = {
    "rep_detection": "Boundary V2 — Development / Shadow Candidate",
    "correctness": "V3 Development Final — External Validation Pending",
    "error_detection": "V1 Development Static Error Model",
}


def _percentage(value: Any) -> str | None:
    if value is None:
        return None
    try:
        probability = min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None
    return f"{probability * 100.0:.0f}%"


def present_ai_rep(rep: dict[str, Any]) -> dict[str, Any]:
    """Return one truthful, user-facing repetition view."""

    rep_index = int(rep.get("rep_index", 0))
    pass_fail = rep.get("assessment", rep.get("pass_fail"))
    pass_fail = (
        str(pass_fail).upper()
        if pass_fail in {"PASS", "FAIL", "NEEDS_REVIEW"}
        else None
    )

    error_key = rep.get("error_class") if pass_fail == "FAIL" else None
    error_label = ERROR_LABELS.get(str(error_key)) if error_key is not None else None
    if pass_fail == "FAIL" and error_label is None:
        error_label = ERROR_LABELS["form_issue"]

    correct_probability = rep.get("correct_probability")
    decision_confidence: float | None = None
    if rep.get("assessment_confidence") is not None:
        decision_confidence = float(rep["assessment_confidence"])
    elif correct_probability is not None and pass_fail in {"PASS", "FAIL"}:
        probability = min(1.0, max(0.0, float(correct_probability)))
        decision_confidence = probability if pass_fail == "PASS" else 1.0 - probability

    correctness_confidence = _percentage(decision_confidence)
    error_confidence = (
        _percentage(rep.get("error_confidence"))
        if error_key in {"bad_back", "bad_heel"}
        else None
    )
    if pass_fail == "PASS":
        summary = f"Rep {rep_index} — PASS"
        detail = "Correct form"
    elif pass_fail == "FAIL":
        summary = f"Rep {rep_index} — FAIL — {error_label}"
        detail = error_label
    elif pass_fail == "NEEDS_REVIEW":
        summary = f"Rep {rep_index} — NEEDS REVIEW"
        detail = str(rep.get("needs_review_reason") or "Unable to assess confidently")
    else:
        summary = f"Rep {rep_index} - {NOT_AVAILABLE}"
        detail = NOT_AVAILABLE

    return {
        "rep_index": rep_index,
        "pass_fail": pass_fail,
        "error": error_label,
        "correctness_confidence": correctness_confidence,
        "error_confidence": error_confidence,
        "needs_review_reason": rep.get("needs_review_reason"),
        "summary": summary,
        "detail": detail,
        "score": None,
    }


def present_live_squat_ai(status: dict[str, Any] | None) -> dict[str, str]:
    """Format the lightweight live status without rebuilding any rep table."""

    if not isinstance(status, dict):
        return {
            "heading": "AI ANALYSIS — EXPERIMENTAL",
            "detected": "0",
            "processing": WAITING_REPETITION,
            "last_rep": WAITING_REPETITION,
            "confidence": NOT_AVAILABLE,
            "warning": NOT_AVAILABLE,
        }
    if not status.get("boundary_available", True):
        return {
            "heading": "AI ANALYSIS — EXPERIMENTAL",
            "detected": NOT_AVAILABLE,
            "processing": NOT_AVAILABLE,
            "last_rep": NOT_AVAILABLE,
            "confidence": NOT_AVAILABLE,
            "warning": NOT_AVAILABLE,
        }

    detected = int(status.get("ai_detected_reps", 0))
    last = status.get("last_ai_rep")
    rep_view = present_ai_rep(last) if isinstance(last, dict) else None
    analyzing = bool(status.get("analyzing"))
    processing = (
        PROCESSING
        if analyzing
        else WAITING_REPETITION
    )
    if rep_view is None:
        last_text = WAITING_REPETITION
        confidence = NOT_AVAILABLE
    else:
        last_text = rep_view["summary"]
        confidence_parts = []
        if rep_view["correctness_confidence"]:
            confidence_parts.append(
                f"Correctness {rep_view['correctness_confidence']}"
            )
        if rep_view["error_confidence"]:
            confidence_parts.append(f"Error {rep_view['error_confidence']}")
        confidence = " | ".join(confidence_parts) or NOT_AVAILABLE

    if detected and not status.get("correctness_available", True):
        last_text = NOT_AVAILABLE
    warning = rep_view["error"] if rep_view and rep_view["pass_fail"] == "FAIL" else NOT_AVAILABLE
    return {
        "heading": "AI ANALYSIS — EXPERIMENTAL",
        "detected": str(detected),
        "processing": processing,
        "last_rep": last_text,
        "confidence": confidence,
        "warning": warning,
    }


def present_squat_dashboard(
    family_id: str,
    official_count: int | None,
    experimental_ai: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a Squat-only dashboard view from the existing aggregate contract."""

    if family_id != "squat":
        return None
    ai = experimental_ai if isinstance(experimental_ai, dict) else {}
    boundary_available = bool(ai.get("available", ai.get("boundary_available", False)))
    detected_value = ai.get("ai_detected_reps") if boundary_available else None
    detected = int(detected_value) if detected_value is not None else None

    unique_reps: dict[int, dict[str, Any]] = {}
    for rep in ai.get("per_rep_results") or []:
        if not isinstance(rep, dict):
            continue
        index = int(rep.get("rep_index", 0))
        if index > 0 and index not in unique_reps:
            unique_reps[index] = present_ai_rep(rep)
    rows = [unique_reps[index] for index in sorted(unique_reps)]

    errors = ai.get("error_counts") or {}
    error_counts = {
        "bad_back": int(errors.get("bad_back", 0)),
        "bad_heel": int(errors.get("bad_heel", 0)),
        "form_issue": int(errors.get("form_issue", 0)),
    }
    pass_rate = ai.get("ai_pass_rate")
    model_statuses = ai.get("model_statuses") or {}
    boundary_status = model_statuses.get("boundary")
    correctness_status = model_statuses.get("correctness")
    error_status = model_statuses.get("error")
    models = {
        "rep_detection": (
            f"Boundary V2 - {boundary_status}"
            if boundary_status
            else DEFAULT_MODEL_STATUSES["rep_detection"]
        ),
        "correctness": (
            f"V3 {correctness_status}"
            if correctness_status
            else DEFAULT_MODEL_STATUSES["correctness"]
        ),
        "error_detection": (
            f"V1 {error_status}"
            if error_status
            else DEFAULT_MODEL_STATUSES["error_detection"]
        ),
        "score": "Deterministic Experimental Performance Score",
    }
    difference = (
        None
        if official_count is None or detected is None
        else int(detected - int(official_count))
    )
    return {
        "heading": "AI ANALYSIS — EXPERIMENTAL",
        "warning": "Experimental AI — Results may require validation",
        "official_count": official_count,
        "ai_detected_reps": detected,
        "difference": difference,
        "correct_reps": int(ai.get("ai_correct_reps", 0)),
        "incorrect_reps": int(ai.get("ai_incorrect_reps", 0)),
        "needs_review_reps": int(ai.get("needs_review_count", 0)),
        "pass_count": int(ai.get("pass_count", 0)),
        "fail_count": int(ai.get("fail_count", 0)),
        "pass_rate": _percentage(pass_rate),
        "error_counts": error_counts,
        "no_form_issues": not any(error_counts.values()),
        "per_rep_results": rows,
        "models": models,
        "score": ai.get("performance_score", ai.get("score")),
        "assessment_coverage": ai.get("assessment_coverage"),
        "assessed_reps": int(ai.get("assessed_reps", 0)),
        "score_unavailable_reason": ai.get("score_unavailable_reason"),
        "available": boundary_available,
        "failures": list(ai.get("failures") or []),
    }
