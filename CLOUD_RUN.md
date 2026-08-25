# Cloud Run deployment

## Status: written, never built or run

Everything in this file and in `api/`, `Dockerfile`, `.dockerignore`,
`requirements-api.lock.txt`, `deploy-cloud-run.sh` was written without access
to a Docker daemon or a Python environment with `torch`/`mediapipe`/`opencv`
installed — there was no way to build the image or execute the pipeline to
confirm any of it actually works end to end. Treat this as a first draft that
needs a real build-and-run pass, not a verified deployment. Before pointing
production traffic at it:

1. `docker build -t exercise-analysis-api .` locally (or `gcloud builds submit`)
   and confirm it completes — this is the first real test of the MediaPipe
   download URL, the pinned dependency versions, and the `.dockerignore` list
   actually including everything the pipeline imports at runtime.
2. `docker run -e API_SHARED_SECRET=test -p 8080:8080 exercise-analysis-api`,
   then `curl localhost:8080/health` — confirms the container starts and
   both checkpoints/model files are present.
3. Send one real video through `/analyze` for each exercise family you care
   about first (at minimum squat, since it's the most fully wired family) and
   check the response against what `expo/utils/exerciseAnalysis.ts`'s
   `parseAnalysisResponse()` expects.
4. Only then run `./deploy-cloud-run.sh`.

If step 1 fails on a dependency conflict, regenerate `requirements-api.lock.txt`
for real: `pip install -r requirements-api.lock.txt` in a clean venv, fix
whatever breaks, then `pip freeze > requirements-api.lock.txt`.

## Architecture

```
Mulhim app --(auth'd)--> Supabase Edge Function "exercise-analysis"
                              --(X-API-Key: API_SHARED_SECRET)-->
                          Cloud Run: this service
                              GET  /health   — liveness, no auth, no Supabase
                              POST /analyze  — downloads video_url, runs the
                                               pipeline, returns SessionResult
```

This service never talks to Supabase. It receives a signed Storage URL in
`video_url`, downloads the video itself, and returns JSON. The Edge Function
is the only thing that holds Supabase credentials; this service holds none.

## What actually runs `/analyze`

`api/pipeline.py` reuses `application.workers.AnalysisWorker` — the same
worker class the desktop app (`ui/app.py`) uses for its live camera view —
run synchronously (`worker.run_sync()`) against `VideoFrameSource` instead of
a camera, exactly like `ui/app.py`'s own `run_headless_video()` does for its
headless smoke-test path. No engine logic, model, or threshold was changed;
`api/` only supplies frames from an uploaded file instead of a camera and
observes MediaPipe's per-frame detection rate from the outside (for the
`pose_coverage` field) via the `pose_factory` injection point the worker
already exposed as a public constructor argument.

`inference/predict.py`'s `ExercisePredictor` (used by
`application/legacy_inference_cli.py`) is a different, older pipeline with a
different JSON shape (`prototype_similarity`, head-level `phase`/`pass_fail`)
— it is not what the Mulhim app's contract expects and `api/` does not use it.

## Required Google Cloud services/APIs

- Cloud Run (`run.googleapis.com`)
- Cloud Build (`cloudbuild.googleapis.com`) — builds the Dockerfile when you
  deploy with `--source .`
- Artifact Registry (`artifactregistry.googleapis.com`) — stores the built image
- Secret Manager (`secretmanager.googleapis.com`) — holds `API_SHARED_SECRET`

`deploy-cloud-run.sh` enables all four; it's safe to re-run.

## What must be uploaded/built

The whole `ai-exercise-analysis-engine` repo root (this directory) — that's
where `Dockerfile` lives, and `application/exercise_registry.py` resolves
`checkpoints/`, `models/`, and `configs/` relative to this same root at
runtime, so nothing here can be built from a subdirectory. `.dockerignore`
strips `training/`, `evaluation/`, `tools/`, `tests/`, `ui/`, `datasets/`,
`archive/` from the Cloud Build upload — none of them are imported by `api/`.
`gcloud run deploy --source .` (what `deploy-cloud-run.sh` runs) uploads this
directory and has Cloud Build build the Dockerfile automatically.

## Git LFS and Cloud Build

Cloud Build receives whatever is in the local working tree when
`gcloud run deploy --source .` runs `gcloud builds submit` under the hood —
it tars up the working directory as-is over the gcloud API, it does not do
its own `git clone`. So: **as long as `git lfs pull` has been run locally
first** (already done for this checkout — all 7 checkpoints under
`checkpoints/` are real binaries, verified via `git lfs status` / `file`),
Cloud Build receives the real model bytes, not LFS pointer stubs. There is no
LFS-specific Cloud Build configuration needed. If you ever deploy from a
fresh clone, run `git lfs pull` before `./deploy-cloud-run.sh`, or the build
will silently ship 130-byte pointer files instead of the real checkpoints.

## MediaPipe model

`preprocessing`/`input_sources.pose_stream.PoseStreamProcessor` both use the
modern MediaPipe Tasks API, which requires an explicit
`pose_landmarker_full.task` file — it is not bundled by the `mediapipe` pip
package and was never committed to this repo (no `.task` file exists in git
history). The `Dockerfile` downloads it at build time from Google's official
model host into `/app/mediapipe_models/pose_landmarker_full.task`
(`api/config.py`'s `POSE_MODEL_PATH` default). **This URL was not verified
against a live network in this session** — if the build's `curl -f` step
fails, get the current URL from
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker and
update the `Dockerfile`.

## Environment / secrets

**Cloud Run side — exactly one required secret:**

| Name | Where it's read | Notes |
|---|---|---|
| `API_SHARED_SECRET` | `api/config.py` | Fails the process at startup if unset. Gates `/analyze` via the `X-API-Key` header; `/health` needs no auth. |

`PORT` is injected by Cloud Run automatically — `Dockerfile`'s `CMD` reads it
(`${PORT:-8080}`); nothing in `api/` hardcodes a port.

**Supabase Edge Function side (already implemented, see that repo) — exactly two:**

| Name | Points at |
|---|---|
| `EXERCISE_ANALYSIS_API_URL` | This service's Cloud Run URL |
| `EXERCISE_ANALYSIS_API_SHARED_SECRET` | The same value as `API_SHARED_SECRET` above |

## Resource sizing

`--memory 4Gi --cpu 2 --timeout 300s` in `deploy-cloud-run.sh` are starting
points, not measured values — torch + mediapipe + a MotionBERT forward pass
on CPU need real headroom, and 300s leaves margin over the Edge Function's
own 180s `BACKEND_TIMEOUT_MS`. Watch Cloud Run's memory/CPU utilization
metrics after the first real traffic and adjust; `--min-instances 0` means
the first request after idle pays a cold-start cost (model loading) — raise
it to 1 if that latency becomes a problem, at the cost of always-on billing.

## Why `--allow-unauthenticated`

The Edge Function calls this service as a plain HTTPS relay with no GCP
identity of its own — it authenticates with the `X-API-Key` header, not IAM.
That means this Cloud Run URL is reachable by anyone who finds it; the
shared secret is the only gate. That matches the design already built into
the Edge Function and is an accepted tradeoff for now, not an oversight — if
tighter access is wanted later, look at Cloud Run's IAM invoker auth plus an
OIDC token minted from the Edge Function, but that's new scope beyond what
exists today.
