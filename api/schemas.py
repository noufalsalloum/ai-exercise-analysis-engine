"""Response/error contracts for the /analyze endpoint.

Mirrors what expo/utils/exerciseAnalysis.ts (RawAnalysisResponse) and the
exercise-analysis Edge Function expect back — see that file's own comment
pointing here. `result` is intentionally a raw dict, not a strict Pydantic
model: it is a genuine pass-through of application/contracts.py's
SessionResult.to_dict(), which differs by exercise family, and forcing a
strict shape here would risk silently dropping real fields (same reasoning
the frontend's own type applies to itself).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PoseCoverage(BaseModel):
    rate: float | None
    pose_rarely_detected: bool | None


class AnalyzeResult(BaseModel):
    session_id: str
    exercise_id: str
    family_id: str
    pose_coverage: PoseCoverage
    result: dict[str, Any]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
