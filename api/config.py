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

# Deployment-integrity fingerprint (2026-08-31): written by Dockerfile.runpod
# at build time from the build context's own .git/HEAD — proves what source
# commit is actually baked into a given running image, independent of
# whatever a deploy platform's UI *claims* was deployed. Never set locally
# (no /app/GIT_SHA file outside a container), so this is "unknown" in dev by
# design — that's expected, not an error.
_GIT_SHA_FILE = PROJECT_ROOT / "GIT_SHA"
GIT_SHA = (
    _GIT_SHA_FILE.read_text().strip() if _GIT_SHA_FILE.is_file() else os.environ.get("GIT_SHA", "unknown")
) or "unknown"


def squat_runtime_diagnostics() -> dict[str, object]:
    """Reports the *actually effective* squat rep-counter values, not just a
    source file's default. minimum_pelvis_displacement (and every other
    SquatRepConfig field) is NOT read from inference/squat_runtime.py's
    dataclass default at runtime — application/runtime_router.py's
    FamilyRuntimeRouter.create() (the exact path api/pipeline.py's
    AnalysisWorker uses for every /analyze video) loads configs/squat.json
    fresh on every call and overrides every field via
    SquatRepConfig.from_dict(...). A change to the dataclass default alone
    is therefore invisible in production unless configs/squat.json agrees —
    this function builds the real runtime the same way AnalysisWorker does,
    specifically so /health and /ping can never report a value more
    optimistic than what a real request actually uses. Import is local
    (not at module load) since application.runtime_router transitively
    pulls in the heavier inference stack; failures are caught so a
    misconfigured diagnostic can never take down a liveness probe.
    """
    try:
        import inspect

        from application.runtime_router import FamilyRuntimeRouter
        from inference import squat_runtime as squat_runtime_module

        runtime = FamilyRuntimeRouter().create("squat", "video", "side")
        return {
            "minimum_pelvis_displacement": runtime.config.minimum_pelvis_displacement,
            "return_pelvis_tolerance": runtime.config.return_pelvis_tolerance,
            "squat_runtime_module_path": inspect.getfile(squat_runtime_module),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic only, must never crash /health
        return {"error": f"{type(exc).__name__}: {exc}"}

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

# Rejects a video whose analysis would very likely run past the RunPod Load
# Balancer's synchronous request-timeout window (documented default ~60s;
# see RUNPOD.md) — checked cheaply (video metadata only, no frame decode)
# before any real processing starts. Derived from measured, real RunPod
# timing, not picked arbitrarily:
#   - a 23s / 690-frame (30fps) clip's rule-based-only analysis (no
#     experimental AI — see api/pipeline.py) completed in ~49-58s of real
#     wall-clock time on a warm worker: a ~2.13x video-duration processing
#     ratio (49.1s / 23s).
#   - the same clip also once measured 75.2s under cold-start conditions —
#     a real, observed ~26s of extra variance on top of the warm baseline,
#     not hypothetical.
#   - targeting total processing at or below 40s (leaving ~20s of margin
#     under the ~60s window — enough to absorb the cold-start variance
#     actually observed) gives a duration ceiling of 40 / 2.13 ~= 18.8s.
# 18s is that, rounded down for a clean, deliberately conservative margin.
# This is calibrated against 30fps source video specifically (both real
# clips used for calibration were 30fps) — processing cost scales with
# frame count, so a much-higher-fps video of the same nominal duration
# would take longer than this ratio assumes; if that turns out to matter in
# practice, the cheap probe already available in api/pipeline.py exposes
# frame count too and this could be swapped to a frame-count cap instead.
# The other, complementary lever for the cold-start variance specifically
# (not covered by this cap at all) is RunPod's own min-workers setting —
# see RUNPOD.md.
MAX_VIDEO_DURATION_SECONDS = 18.0
