# Cloud Run image for the exercise-analysis API (api/main.py).
#
# NOTE: this Dockerfile has not been built or run anywhere yet — there is no
# Docker/Python-with-torch environment available in the workspace that wrote
# it. Build and smoke-test it locally (`docker build . && docker run -e
# API_SHARED_SECRET=test -p 8080:8080 <image>` then `curl localhost:8080/health`)
# before the first real Cloud Run deploy. See CLOUD_RUN.md.

FROM python:3.11-slim

# libgl1/libglib2.0-0: runtime shared libs opencv-python-headless and
# mediapipe dlopen even in headless mode. libgomp1: PyTorch's OpenMP runtime.
# curl: fetches the MediaPipe model below and backs the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying the source tree so this layer is
# cached across code changes that don't touch dependencies.
COPY requirements-api.lock.txt .
# CPU-only torch from PyTorch's own index — Cloud Run has no GPU, and the
# default PyPI torch wheel drags in multi-GB CUDA libraries for nothing.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.lock.txt

# Official MediaPipe Tasks pose model — never committed to git, fetched at
# build time. If Google's model path changes, get the current URL from
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker and
# update this line; `curl -f` fails the build loudly instead of shipping an
# image with a missing model.
RUN mkdir -p /app/mediapipe_models && \
    curl -f -sSL \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
    -o /app/mediapipe_models/pose_landmarker_full.task

# Engine source + checkpoints (see .dockerignore for what's excluded —
# training/evaluation/tools/tests/ui/datasets are not needed at runtime).
COPY . .

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Documentation only — Cloud Run ignores EXPOSE and injects its own PORT.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8080}/health" || exit 1

# Shell form (not exec-array) so ${PORT} actually expands — Cloud Run sets
# PORT at runtime and the container must listen on *that*, not a hardcoded
# 8080. The ${PORT:-8080} fallback only matters for `docker run` without -e PORT.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
