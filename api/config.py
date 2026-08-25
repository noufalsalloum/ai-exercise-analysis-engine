from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The one production secret this service requires (see CLOUD_RUN.md). Fails
# fast at import time rather than serving an unauthenticated /analyze if the
# Cloud Run secret binding is missing or misnamed.
API_SHARED_SECRET = os.environ.get("API_SHARED_SECRET", "").strip()
if not API_SHARED_SECRET:
    raise RuntimeError(
        "API_SHARED_SECRET is not set. This service refuses to start without it "
        "(it is the only thing that gates /analyze). Set it via `--set-secrets` "
        "on Cloud Run, or in a local .env for development."
    )

# MediaPipe Tasks pose model. Downloaded into the image at build time (see
# Dockerfile) — never bundled in the git repo. Overridable so a local dev
# environment can point at a manually-downloaded copy.
POSE_MODEL_PATH = Path(
    os.environ.get("POSE_MODEL_PATH", str(PROJECT_ROOT / "mediapipe_models" / "pose_landmarker_full.task"))
)

# MotionBERT backbone checkpoint. Resolved by application/exercise_registry.py
# itself (PROJECT_ROOT-relative) for everything else; exposed here only so
# /health can report on it without importing the full registry.
MOTIONBERT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "latest_epoch.bin"

# Same PROJECT_ROOT-relative paths application/exercise_registry.py uses for
# the squat family's learned checkpoints — duplicated here (not imported
# from there) purely so /health can report on them without constructing a
# full ExerciseRegistry on every health check.
SQUAT_CHECKPOINT_PATHS = {
    "rep_boundary_v2": PROJECT_ROOT / "checkpoints" / "squat_ai_v2" / "rep_boundary" / "best.pt",
    "correctness_v3": PROJECT_ROOT / "checkpoints" / "squat_ai_v3" / "correctness" / "final_dev.pt",
    "error_v1": PROJECT_ROOT / "checkpoints" / "squat_error_v1" / "best.pt",
}

# Mirrors expo/utils/exerciseAnalysis.ts's MAX_VIDEO_SIZE_MB — the client
# already rejects an oversized file before upload, but the backend enforces
# its own limit regardless of what the client checked, since video_url is
# fetched server-side and a client isn't the only thing that could call this.
MAX_VIDEO_SIZE_MB = 200
MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024

# Budget for downloading video_url from Supabase Storage. The Edge Function's
# own BACKEND_TIMEOUT_MS is 180s total for download + inference combined, so
# this leaves the remainder for MediaPipe/MotionBERT/expert inference.
VIDEO_DOWNLOAD_TIMEOUT_SECONDS = 60.0

# Below this fraction of processed frames having a detected pose, the result
# is flagged pose_rarely_detected=true so the client can warn the user rather
# than silently show a low-confidence result as if it were normal. A starting
# heuristic, not a validated threshold — tune once real sessions are observed.
POSE_RARELY_DETECTED_THRESHOLD = 0.5
