"""Pure presentation mapping for Push-up, Pull-up and Marching Plank parity."""

from __future__ import annotations

from typing import Any

from application.presentation_contract import (
    COMPLETED,
    HOLDING,
    NOT_AVAILABLE,
    PROCESSING,
    READY,
    WAITING_REPETITION,
    completed_unit,
)


def _percent(value: Any) -> str:
    return NOT_AVAILABLE if value is None else f"{100.0 * float(value):.0f}%"


def _score(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE
    return f"{float(value):.1f}".rstrip("0").rstrip(".") + " / 100"


def normalize_pushup_live_snapshot(
    status: dict[str, Any],
    *,
    completed: bool = False,
) -> dict[str, str]:
    """Map the canonical Push-up AI state to user-facing live values."""

    detected_value = status.get("ai_detected_reps")
    detected = (
        NOT_AVAILABLE
        if detected_value is None
        else str(int(detected_value))
    )

    last = status.get("last_ai_rep")
    if not isinstance(last, dict):
        per_rep_results = status.get("per_rep_results")
        last = (
            per_rep_results[-1]
            if isinstance(per_rep_results, list)
            and per_rep_results
            and isinstance(per_rep_results[-1], dict)
            else None
        )

    if last is not None:
        assessment_value = last.get("assessment")
        assessment = (
            NOT_AVAILABLE
            if assessment_value is None
            else str(assessment_value)
        )
        latest = f"Rep {int(last.get('rep_index', 0))} — {assessment}"
        confidence_value = _percent(last.get("confidence"))
        confidence = (
            NOT_AVAILABLE
            if confidence_value == NOT_AVAILABLE
            else f"Correctness {confidence_value}"
        )
    else:
        assessment = None
        latest = WAITING_REPETITION
        confidence = NOT_AVAILABLE

    coverage_value = status.get("assessment_coverage")
    if coverage_value is None:
        coverage = NOT_AVAILABLE
    else:
        pass_value = status.get("pass_count")
        fail_value = status.get("fail_count")
        assessed = (
            int(pass_value) + int(fail_value)
            if pass_value is not None and fail_value is not None
            else 0
        )
        denominator = int(detected_value) if detected_value is not None else 0
        coverage = (
            f"{assessed} / {denominator} "
            f"({100.0 * float(coverage_value):.0f}%)"
        )

    if completed:
        processing = COMPLETED
    elif bool(status.get("analyzing")):
        processing = PROCESSING
    else:
        processing = WAITING_REPETITION

    return {
        "heading": "AI ANALYSIS — EXPERIMENTAL",
        "detected": detected,
        "processing": processing,
        "last": latest,
        "confidence": confidence,
        "errors": "Form Issue" if assessment == "FAIL" else NOT_AVAILABLE,
        "score": _score(status.get("performance_score")),
        "coverage": coverage,
    }


def present_live_family_status(
    family_id: str,
    status: dict[str, Any] | None,
    *,
    variation_id: str | None = None,
    completed: bool = False,
) -> dict[str, str]:
    status = status if isinstance(status, dict) else {}
    pushup_scope_is_supported = (
        family_id == "pushup"
        and (variation_id is None or variation_id == "table_incline")
    )
    if pushup_scope_is_supported and status.get("available"):
        return normalize_pushup_live_snapshot(status, completed=completed)
    if family_id in {"pullup", "plank"} and status.get("assessment_kind") == "rule_based":
        latest = status.get("latest_unit") if isinstance(status.get("latest_unit"), dict) else None
        label = "Marching Plank — Rule-Based MVP" if family_id == "plank" else "Pull-up — Rule-Based"
        latest_text = WAITING_REPETITION
        confidence = NOT_AVAILABLE
        if latest:
            latest_text = completed_unit(latest.get('repetition_index'), plank=family_id == "plank")
            confidence = _percent(latest.get("confidence"))
        return {
            "heading": label,
            "detected": str(status.get("unit_count", status.get("official_count", 0))),
            "processing": HOLDING if family_id == "plank" and status.get("current_phase") else PROCESSING,
            "last": latest_text,
            "confidence": confidence,
            "errors": NOT_AVAILABLE,
            "score": NOT_AVAILABLE,
            "coverage": NOT_AVAILABLE,
        }
    if family_id == "lunge" and status.get("available"):
        last=status.get("last_ai_rep") if isinstance(status.get("last_ai_rep"),dict) else None
        return {"heading":"AI DETECTION - EXPERIMENTAL","detected":str(status.get("ai_detected_reps",0)),"processing":PROCESSING if status.get("analyzing") else READY,"last":completed_unit(last.get('rep_index')) if last else WAITING_REPETITION,"confidence":NOT_AVAILABLE,"errors":NOT_AVAILABLE,"score":NOT_AVAILABLE,"coverage":NOT_AVAILABLE}
    return {
        "heading": NOT_AVAILABLE,
        "detected": NOT_AVAILABLE, "processing": NOT_AVAILABLE,
        "last": NOT_AVAILABLE, "confidence": NOT_AVAILABLE,
        "errors": NOT_AVAILABLE, "score": NOT_AVAILABLE,
        "coverage": NOT_AVAILABLE,
    }


def present_family_dashboard(family_id: str, ai: dict[str, Any]) -> dict[str, Any]:
    learned = family_id == "pushup" and bool(ai.get("available"))
    detection_only = family_id == "lunge" and bool(ai.get("available"))
    units = ai.get("per_rep_results") if learned or detection_only else ai.get("per_unit_results")
    units = list(units) if isinstance(units, list) else []
    rows: list[str] = []
    for unit in units:
        if learned:
            assessment = str(unit.get("assessment") or "Not Available")
            error = " — Form Issue" if assessment == "FAIL" else ""
            rows.append(
                f"Rep {unit.get('rep_index')}   {assessment}{error}   Confidence {_percent(unit.get('confidence'))}"
            )
        elif detection_only:
            rows.append(f"AI Segment {unit.get('rep_index')}   Frames {unit.get('start_frame')}–{unit.get('end_frame')}   Assessment Not Available")
        else:
            duration = unit.get("duration_seconds")
            duration_text = f"{float(duration):.2f} s" if duration is not None else "Not Available"
            rows.append(f"Rep {unit.get('repetition_index')}   Completed   {duration_text}")
    return {
        "learned": learned,
        "detection_only": detection_only,
        "scope": ai.get("scope") or "Not Available",
        "ai_detected": ai.get("ai_detected_reps", ai.get("ai_detected_count")) if learned or detection_only else None,
        "pass_count": ai.get("pass_count") if learned else None,
        "fail_count": ai.get("fail_count") if learned else None,
        "score": ai.get("performance_score") if learned else None,
        "coverage": ai.get("assessment_coverage") if learned else None,
        "rows": rows,
        "models": ai.get("model_status") if isinstance(ai.get("model_status"), dict) else {},
        "reason": ai.get("reason"),
    }


def present_shared_dashboard(result: dict[str, Any]) -> dict[str, Any]:
    """Map every family into one stable dashboard field order."""
    from ui.squat_ai_presentation import present_squat_dashboard

    family_id = str(result.get("family_id") or result.get("exercise_id") or "")
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    ai = result.get("experimental_ai") if isinstance(result.get("experimental_ai"), dict) else {}
    official = summary.get("total_repetitions")
    rows: list[str] = []
    models: dict[str, Any] = {}
    ai_detected = pass_count = fail_count = needs_review = score = coverage = None
    error = confidence = NOT_AVAILABLE

    if family_id == "squat":
        squat = present_squat_dashboard("squat", official, ai) or {}
        ai_detected = squat.get("ai_detected_reps")
        pass_count = squat.get("pass_count")
        fail_count = squat.get("fail_count")
        needs_review = squat.get("needs_review_reps")
        score = squat.get("score")
        coverage = squat.get("assessment_coverage")
        rows = [str(item.get("summary")) for item in squat.get("per_rep_results", [])]
        if squat.get("per_rep_results"):
            latest = squat["per_rep_results"][-1]
            error = latest.get("error") or NOT_AVAILABLE
            confidence = latest.get("correctness_confidence") or NOT_AVAILABLE
        models = dict(squat.get("models") or {})
    else:
        view = present_family_dashboard(family_id, ai)
        ai_detected = view.get("ai_detected")
        pass_count = view.get("pass_count")
        fail_count = view.get("fail_count")
        needs_review = ai.get("needs_review_count") if view.get("learned") else None
        score = view.get("score")
        coverage = view.get("coverage")
        rows = list(view.get("rows") or [])
        models = dict(view.get("models") or {})
        if view.get("learned") and ai.get("per_rep_results"):
            latest = ai["per_rep_results"][-1]
            error = "Form Issue" if latest.get("assessment") == "FAIL" else NOT_AVAILABLE
            confidence = _percent(latest.get("confidence"))

    variation_by_exercise = {
        "pushup": "Standard / Floor Push-up",
        "table_incline_pushup": "Table / Incline Push-up",
        "pullup": "Standard Pull-up",
        "squat": "Air Squat",
        "plank": "Marching Plank",
        "lunge": "Lunge",
    }
    exercise_id = str(result.get("exercise_id") or family_id)
    scope = variation_by_exercise.get(exercise_id, exercise_id.replace("_", " ").title())
    hold_duration = summary.get("hold_duration")
    official_label = "Official Result" if family_id == "plank" else "Official Reps"
    official_value = f"{int(official or 0)} s" if family_id == "plank" else official
    session_fields = [
        ("Exercise", exercise_id.replace("_", " ").title()),
        ("Variation", str(scope)),
        (official_label, official_value),
        ("Duration", f"{float(result.get('duration_seconds', 0.0)):.1f} s"),
    ]
    if family_id == "plank":
        session_fields.append(("Valid Hold Time", None if hold_duration is None else f"{int(float(hold_duration))} s"))
    coverage_display = None
    if coverage is not None and ai_detected is not None:
        assessed = int(pass_count or 0) + int(fail_count or 0)
        coverage_display = f"{assessed} / {int(ai_detected)} ({100.0 * float(coverage):.0f}%)"
    analysis_fields = [
        ("AI Detected Reps", ai_detected),
        ("PASS", pass_count),
        ("FAIL", fail_count),
        ("Needs Review", needs_review),
        ("Performance Score", None if score is None else f"{float(score):g} / 100"),
        ("Assessment Coverage", coverage_display),
    ]
    form_fields = [("Error", error), ("Confidence", confidence)]
    return {
        "field_order": [name for name, _ in session_fields + analysis_fields],
        "session_fields": session_fields,
        "analysis_fields": analysis_fields,
        "rows": rows,
        "form_fields": form_fields,
        "models": models,
    }
