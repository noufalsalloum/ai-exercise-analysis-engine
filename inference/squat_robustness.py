"""Post-hoc calibration, abstention, and deterministic Squat performance score.

The active V3/Error V1 weights and their raw predictions remain unchanged.
This module only adds a transparent Experimental assessment policy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "squat_ai_robustness.json"


@dataclass(frozen=True)
class PerformanceScoreResult:
    """Explainable session score and its assessment coverage."""

    score: float | None
    assessed_reps: int
    total_reps: int
    needs_review_reps: int
    coverage: float | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "performance_score": self.score,
            "assessed_reps": self.assessed_reps,
            "total_reps": self.total_reps,
            "needs_review_count": self.needs_review_reps,
            "assessment_coverage": self.coverage,
            "score_unavailable_reason": self.reason,
            "score_name": "Performance Score",
            "score_status": "Experimental AI-derived performance score",
        }


class SquatRobustnessPolicy:
    """Apply Development-derived calibration and conservative view abstention."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        calibration = config["calibration"]
        if calibration["method"] != "platt":
            raise ValueError("Only the validated Platt calibration is supported.")
        self.coefficient = float(calibration["coefficient"])
        self.intercept = float(calibration["intercept"])
        self.raw_threshold = float(calibration["raw_threshold"])
        self.calibrated_threshold = float(calibration["calibrated_threshold"])
        self.margin_enabled = bool(config["abstention"]["margin_enabled"])
        self.margin = float(config["abstention"]["derived_margin"])
        self.supported_views = frozenset(
            str(value).lower() for value in config["abstention"]["supported_camera_views"]
        )
        self.error_supported_views = frozenset(
            str(value).lower()
            for value in config["detailed_error"]["supported_camera_views"]
        )
        self.minimum_error_confidence = float(
            config["detailed_error"]["minimum_confidence"]
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "SquatRobustnessPolicy":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def calibrate(self, raw_probability: float) -> float:
        probability = min(1.0 - 1e-6, max(1e-6, float(raw_probability)))
        logit = math.log(probability / (1.0 - probability))
        return float(1.0 / (1.0 + math.exp(-(self.coefficient * logit + self.intercept))))

    def assess_rep(self, rep: dict[str, Any], camera_view: str | None) -> dict[str, Any]:
        """Return a backward-compatible rep with an explicit assessment field."""

        result = dict(rep)
        raw_probability = result.get("correct_probability")
        if raw_probability is None:
            result.update(
                {
                    "assessment": None,
                    "assessment_confidence": None,
                    "calibrated_correct_probability": None,
                    "needs_review_reason": "Correctness analysis unavailable",
                }
            )
            return result
        calibrated = self.calibrate(float(raw_probability))
        view = None if camera_view is None else str(camera_view).lower()
        needs_review_reason: str | None = None
        if view is not None and view not in self.supported_views:
            needs_review_reason = self.config["abstention"]["unsupported_view_reason"]
        elif self.margin_enabled and abs(calibrated - self.calibrated_threshold) < self.margin:
            needs_review_reason = "Prediction is inside the Development-derived uncertainty margin"

        raw_decision = "PASS" if float(raw_probability) >= self.raw_threshold else "FAIL"
        assessment = "NEEDS_REVIEW" if needs_review_reason else raw_decision
        # This is calibrated class probability, not model accuracy.
        class_confidence = calibrated if raw_decision == "PASS" else 1.0 - calibrated
        result.update(
            {
                "raw_model_decision": raw_decision,
                "calibrated_correct_probability": calibrated,
                "assessment_confidence": class_confidence,
                "assessment": assessment,
                "pass_fail": assessment,
                "needs_review_reason": needs_review_reason,
                "calibration_method": "platt_development_oof",
            }
        )
        if assessment != "FAIL":
            result["error_class"] = None
            result["error_confidence"] = None
        elif (
            (view is not None and view not in self.error_supported_views)
            or result.get("error_confidence") is None
            or float(result["error_confidence"]) < self.minimum_error_confidence
        ):
            result["error_class"] = "form_issue"
            result["error_confidence"] = None
        return result

    def performance_score(self, reps: Iterable[dict[str, Any]]) -> PerformanceScoreResult:
        rows = list(reps)
        total = len(rows)
        pass_count = sum(row.get("assessment", row.get("pass_fail")) == "PASS" for row in rows)
        fail_count = sum(row.get("assessment", row.get("pass_fail")) == "FAIL" for row in rows)
        assessed = pass_count + fail_count
        needs_review = sum(
            row.get("assessment", row.get("pass_fail")) == "NEEDS_REVIEW" for row in rows
        )
        coverage = None if total == 0 else float(assessed / total)
        score_config = self.config["performance_score"]
        reason: str | None = None
        if total == 0:
            reason = "No AI repetitions were detected"
        elif assessed < int(score_config["minimum_assessed_reps"]):
            reason = "Insufficient confident repetitions"
        elif coverage is None or coverage < float(score_config["minimum_coverage"]):
            reason = "Assessment coverage is below the required two-thirds"
        score = None if reason else float(round(100.0 * pass_count / assessed, 1))
        return PerformanceScoreResult(score, assessed, total, needs_review, coverage, reason)
