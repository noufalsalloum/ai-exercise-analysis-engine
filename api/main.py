"""FastAPI service exposing the AI exercise-analysis engine over HTTP for
Cloud Run. The only caller in production is the Mulhim Supabase Edge
Function `exercise-analysis` (see that function's own top-of-file comment,
and expo/services/exerciseAnalysisApi.ts) — the mobile app never calls this
directly, so the only auth this needs is the shared secret that Edge
Function relay already assumes.

Request shape (multipart/form-data, matching the Edge Function's relay):
    exercise_id: str   (the engine's own family id, e.g. "squat")
    video_url:   str   (a short-lived signed Supabase Storage URL)
    session_id:  str, optional (not used by the engine; accepted and ignored)
    camera_view: str, optional

Response shape: see api/schemas.py — mirrors exactly what
utils/exerciseAnalysis.ts's parseAnalysisResponse() reads.
"""

from __future__ import annotations

import hmac
import logging
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import torch
from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config
from .pipeline import (
    UnsupportedExerciseError,
    VideoProcessingError,
    ensure_runnable,
    probe_video_duration_seconds,
    run_analysis,
    supported_exercise_ids,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exercise-analysis-api")

app = FastAPI(title="Mulhim Exercise Analysis Engine", version="1.0.0")


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("[exercise-analysis-api] Unhandled error on %s", request.url.path)
    return _error(500, "inference_failed", "The analysis service hit an unexpected internal error.")


def _health_payload() -> tuple[int, dict[str, Any]]:
    pose_model_present = config.POSE_MODEL_PATH.is_file()
    motionbert_present = config.MOTIONBERT_CHECKPOINT_PATH.is_file()
    squat_checkpoints = {
        name: path.is_file() for name, path in config.SQUAT_CHECKPOINT_PATHS.items()
    }
    ok = pose_model_present and motionbert_present and all(squat_checkpoints.values())
    body = {
        "status": "ok" if ok else "degraded",
        "pose_model_present": pose_model_present,
        "motionbert_checkpoint_present": motionbert_present,
        "squat_checkpoints": squat_checkpoints,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "supported_exercise_families": supported_exercise_ids(),
    }
    return (200 if ok else 503, body)


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness/readiness probe. Deliberately independent of Supabase (this
    service never talks to Supabase — it only receives a signed URL to fetch
    a video from) and independent of the API_SHARED_SECRET auth gate, so
    Cloud Run and operators can always see real status here."""

    status_code, body = _health_payload()
    return JSONResponse(status_code=status_code, content=body)


@app.get("/ping")
async def ping() -> JSONResponse:
    """Identical to /health, under the path RunPod's load-balancing workers
    poll by default (docs.runpod.io/serverless/load-balancing/overview) —
    Cloud Run keeps using /health via Dockerfile's HEALTHCHECK; this exists
    purely so a RunPod endpoint works without changing its console-configured
    health check path away from the default."""

    status_code, body = _health_payload()
    return JSONResponse(status_code=status_code, content=body)


async def _download_video(video_url: str) -> Path:
    parsed = urlparse(video_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("video_url must be an http(s) URL.")

    handle = tempfile.NamedTemporaryFile(prefix="exercise-analysis-", suffix=".mp4", delete=False)
    destination = Path(handle.name)
    total_bytes = 0
    try:
        async with httpx.AsyncClient(timeout=config.VIDEO_DOWNLOAD_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", video_url) as response:
                if response.status_code != 200:
                    raise ValueError(f"video_url returned HTTP {response.status_code}.")
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > config.MAX_VIDEO_SIZE_BYTES:
                        raise ValueError(f"Video exceeds the {config.MAX_VIDEO_SIZE_MB}MB limit.")
                    handle.write(chunk)
    finally:
        handle.close()

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("Downloaded video is empty.")
    return destination


@app.post("/analyze")
async def analyze(
    exercise_id: str = Form(...),
    video_url: str = Form(...),
    session_id: str | None = Form(None),
    camera_view: str | None = Form(None),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> JSONResponse:
    if not x_api_key or not hmac.compare_digest(x_api_key, config.API_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")

    if not config.POSE_MODEL_PATH.is_file():
        return _error(503, "service_unavailable", "The analysis service is not fully configured yet.")

    try:
        ensure_runnable(exercise_id)
    except UnsupportedExerciseError as exc:
        return _error(404, "unsupported_exercise", str(exc))

    video_path: Path | None = None
    try:
        try:
            video_path = await _download_video(video_url)
        except ValueError as exc:
            return _error(400, "video_invalid", str(exc))
        except httpx.HTTPError as exc:
            return _error(400, "video_invalid", f"Could not download the video: {exc}")

        # Cheap metadata-only check (no frame decode) before any real
        # processing starts — rejects a video whose analysis would very
        # likely exceed the platform's synchronous request-timeout window
        # (see api.config.MAX_VIDEO_DURATION_SECONDS for how the cap was
        # derived from real RunPod timing). None means OpenCV couldn't read
        # duration metadata at all — let the real pipeline's own decoder
        # raise a clear, honest error for that instead of guessing here.
        duration_seconds = probe_video_duration_seconds(video_path)
        if duration_seconds is not None and duration_seconds > config.MAX_VIDEO_DURATION_SECONDS:
            return _error(
                400,
                "video_invalid",
                f"Video is {duration_seconds:.1f}s long; the maximum for synchronous "
                f"analysis is {config.MAX_VIDEO_DURATION_SECONDS:.0f}s.",
            )

        try:
            outcome: dict[str, Any] = run_analysis(
                video_path=video_path,
                exercise_id=exercise_id,
                pose_model_path=config.POSE_MODEL_PATH,
                camera_view=camera_view,
            )
        except UnsupportedExerciseError as exc:
            return _error(404, "unsupported_exercise", str(exc))
        except VideoProcessingError as exc:
            return _error(400, "video_invalid", str(exc))

        session_result = outcome["result"]
        coverage_rate = outcome["pose_coverage_rate"]
        pose_rarely_detected = (
            coverage_rate is not None and coverage_rate < config.POSE_RARELY_DETECTED_THRESHOLD
        )

        return JSONResponse(
            status_code=200,
            content={
                "session_id": session_result["session_id"],
                "exercise_id": session_result["exercise_id"],
                "family_id": session_result["family_id"],
                "pose_coverage": {"rate": coverage_rate, "pose_rarely_detected": pose_rarely_detected},
                "result": session_result,
            },
        )
    finally:
        if video_path is not None:
            video_path.unlink(missing_ok=True)
