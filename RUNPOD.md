# RunPod Serverless deployment

## Status: written, never built or run

Same caveat as `CLOUD_RUN.md`: no Docker daemon and no NVIDIA GPU were
available in the environment that wrote this. `Dockerfile.runpod` has not
been built, and nothing here has been deployed. Build and smoke-test it
(`docker build -f Dockerfile.runpod -t exercise-analysis-runpod .` — a CPU
machine can build it; you just can't exercise the CUDA path without a GPU)
before trusting it in production. Everything else — the FastAPI app itself,
`/health`/`/analyze`, the pipeline — is the same code already verified
against a real venv and a real test run; see the rest of this conversation
for that evidence.

## The architecture decision: Load Balancer endpoint, not Queue endpoint

RunPod Serverless has two fundamentally different worker types
(docs.runpod.io/serverless/overview,
docs.runpod.io/serverless/load-balancing/overview — verified live in this
session, not from memory):

- **Queue endpoints** (the "traditional"/default type): your container must
  call `runpod.serverless.start({"handler": fn})`. RunPod queues jobs and
  calls `fn(event)` one at a time; callers post to `/run` or `/runsync` with
  a `{"input": {...}}` envelope and poll `/status/{id}` for async jobs. This
  requires the `runpod` SDK and a request/response shape nothing in this
  codebase already speaks.
- **Load Balancer endpoints**: no handler function, no `runpod` SDK. Your
  container just runs a normal HTTP server — "you can define your own
  custom API endpoints without a handler function, using any HTTP framework
  of your choice (like FastAPI or Flask)" — and RunPod's edge routes HTTP
  requests straight to it, method/path/body preserved, at
  `https://<ENDPOINT_ID>.api.runpod.ai/<your-path>`.

**This deployment uses a Load Balancer endpoint**, running the exact same
`api/main.py` FastAPI app Cloud Run runs — `/health`, `/ping`, and `/analyze`
unchanged. This is what "reuse the existing validated pipeline, don't
duplicate inference logic" concretely means here: there is no second
implementation of the analysis flow, no handler.py wrapping it — RunPod just
becomes a second place the same container can run.

## Files added/changed for this task

| File | What / why |
|---|---|
| `Dockerfile.runpod` | New. Same image as `./Dockerfile` (same source tree, same `requirements-api.lock.txt`, same MediaPipe download), except torch is installed from PyPI's CUDA wheel index (`cu121`) instead of the CPU-only one — RunPod Serverless is GPU infrastructure. `PORT/PORT_HEALTH` handling matches RunPod's injected env vars. |
| `api/main.py` | Added a `/ping` route (RunPod's default Load Balancer health-check path) that returns the exact same payload as `/health` — refactored the shared logic into `_health_payload()` so neither route duplicates the other. `/health` is untouched and still what Cloud Run's `HEALTHCHECK` polls. |
| `RUNPOD.md` | New — this file. |

**Not changed:** `Dockerfile`, `.dockerignore`, `requirements-api.lock.txt`,
`deploy-cloud-run.sh`, `CLOUD_RUN.md`, `.env.example`, `api/config.py`,
`api/pipeline.py`, `api/schemas.py`, and nothing under `application/`,
`inference/`, `models/`, `checkpoints/`, `preprocessing/`, etc. was touched
by this task (the one earlier fix to `tools/squat_ai/prepare_rehab24_squat.py`
— moving a stray import — was from the *previous* task, not this one, and
was a packaging fix, not a model-logic change).

`requirements-api.lock.txt` is reused as-is — a Load Balancer endpoint needs
no extra dependency (no `runpod` SDK), so a separate RunPod requirements
file would have been a pure duplicate.

## What must be uploaded/built

The same repo root as Cloud Run: `C:\Users\Khalid\Desktop\Mulhim\ai-exercise-analysis-engine`.
RunPod's GitHub integration builds whatever Dockerfile you point it at from
your repo — set the console's **Dockerfile Path** field to `Dockerfile.runpod`
(default is `Dockerfile`, which is the Cloud Run one; picking the wrong one
would build a CPU-only image on a GPU worker — wasted GPU spend, not a
crash, since the app still runs fine on CPU).

**Before RunPod can see any of this: none of it is on GitHub yet.** Everything
in this session — including the earlier Cloud Run files — is uncommitted, by
your own instruction not to commit or push. RunPod's GitHub integration reads
from GitHub, not this local working tree. You need to commit and push these
files yourself before setting up the RunPod endpoint.

**Also verify:** RunPod's GitHub integration authorizes against repos you
(or your GitHub org) own — "each RunPod account can link to only one GitHub
account." If `ai-exercise-analysis-engine` lives under `JAshehri`'s account
and that's not the account you'll authorize RunPod with, you'll need it
forked/transferred to an account RunPod can see, or have `JAshehri` grant the
RunPod GitHub App access to this specific repo.

## Required RunPod environment variables

| Name | Where | Notes |
|---|---|---|
| `API_SHARED_SECRET` | RunPod console → endpoint → Settings → Environment Variables | Identical mechanism to Cloud Run — `api/config.py` fails startup without it. Set as a RunPod **secret**, not a plain env var (RunPod's own guidance: "Secrets should only be set as runtime variables in the console... never hardcode"). |

`PORT` / `PORT_HEALTH` are injected by RunPod itself — do not set them
manually; `Dockerfile.runpod`'s `CMD` already reads `PORT`.

## Authentication: two layers, not one

RunPod's own docs example custom routes being called with
`Authorization: Bearer <RUNPOD_API_KEY>` — RunPod's account-level API key,
required by RunPod's own gateway to route the request to your endpoint at
all (the docs didn't explicitly confirm this is enforced *before* reaching
the container rather than just a documented convention, but it's consistent
with how RunPod's Queue endpoints are documented to work, and the load
balancer example code checks a *separate*, app-defined key — implying
RunPod's key is already handled upstream).

This deployment keeps **both** layers rather than relying on RunPod's key
alone:
1. RunPod's own `Authorization: Bearer <RUNPOD_API_KEY>` — required to reach
   the endpoint at all.
2. This app's existing `X-API-Key: <API_SHARED_SECRET>` check — unchanged
   from Cloud Run.

Reason: RunPod account API keys are account-wide by default (docs didn't
surface an endpoint-scoped key option) — handing Supabase's Edge Function a
key that could also manage unrelated RunPod endpoints/pods is a bigger blast
radius than necessary. Keeping `X-API-Key` means the credential Supabase
holds for this feature stays scoped to exactly this feature, same as today.
**Check the RunPod console for a scoped/restricted API key option before
wiring this up** — if one exists, use it instead of a full account key.

## Recommended GPU and worker configuration

This model is small — MotionBERT-lite backbone (~64MB) plus a handful of
per-family heads a few MB to tens of MB each, single video in, one forward
pass per window. It does not need a top-tier GPU. `squat_ai_mvp.py`,
`pushup_ai_mvp.py`, and `lunge_ai_mvp.py` all already do
`torch.device("cuda" if torch.cuda.is_available() else "cpu")` — confirmed
by reading the code, not assumed — so any CUDA-capable tier RunPod offers
will be picked up automatically with zero code changes.

- **GPU**: the smallest/cheapest tier available (RunPod's entry GPU class —
  e.g. RTX 4090 or equivalent 16-24GB card — was quoted around $0.7-1.1/hr
  serverless in a search done in this session; verify current pricing in
  the console, tiers and names change). This workload is nowhere near
  VRAM-bound; paying for A100/H100-class hardware would be pure waste.
- **Workers**: start with `min workers = 0` (scale-to-zero, no idle cost)
  and `max workers = 2-3` to start. Watch actual concurrent-request volume
  and raise max if requests queue.
- **Cold starts**: with `min = 0`, the first request after idle pays
  container-start + model-loading latency (torch/mediapipe import,
  MotionBERT + family checkpoints loading) on top of inference — this
  wasn't measured here (no GPU available to time it). If that latency is a
  problem for the user-facing flow, set `min workers = 1` to keep one warm
  worker (at constant per-hour cost instead of pay-per-request) — measure
  actual cold-start time in the RunPod console after the first real deploy
  before deciding.

## Does the Supabase Edge Function need contract changes?

**Yes, but minimally — none of it is the request/response body.** The
`exercise-analysis` Edge Function needs to change:
1. **Base URL**: `EXERCISE_ANALYSIS_API_URL` → `https://<ENDPOINT_ID>.api.runpod.ai` instead of the Cloud Run URL.
2. **Add one header**: `Authorization: Bearer <RUNPOD_API_KEY>` alongside
   the existing `X-API-Key` header it already sends — both are required now.

Everything else — `POST /analyze` as multipart/form-data with
`exercise_id`/`video_url`/`session_id`/`camera_view`, the `X-API-Key`
header itself, and the full JSON response shape
(`session_id`/`exercise_id`/`family_id`/`pose_coverage`/`result`) — is
byte-for-byte the same FastAPI app, so nothing about how the Edge Function
parses the response changes. This is a direct consequence of choosing a
Load Balancer endpoint over a Queue endpoint: a Queue endpoint would have
forced a real contract rewrite (the `{"input": {...}}` job envelope, and
either polling `/status/{id}` or blocking on `/runsync`).

Also re-examine the Edge Function's `BACKEND_TIMEOUT_MS` (180s on the Cloud
Run side) against RunPod's actual cold-start + inference latency once
measured — it may need adjusting either direction.

## Exact remaining manual steps

1. Commit and push `Dockerfile.runpod`, the `/ping` addition in
   `api/main.py`, and this file to the branch RunPod will watch. (Not done
   by this session, per your instruction not to commit/push.)
2. In the RunPod console: **New Endpoint → GitHub Repo** (authorize RunPod's
   GitHub App against the account that owns/can access this repo first, if
   not already done).
3. Select repository `ai-exercise-analysis-engine`, the branch you pushed
   to, **Dockerfile Path = `Dockerfile.runpod`**.
4. **Endpoint Type = Load Balancer** (not Queue) — this is the setting that
   makes RunPod treat the container as a plain HTTP server instead of
   expecting a handler.
5. Set health check path to `/ping` (should be the console default; verify
   it, since a wrong path leaves the endpoint permanently "unhealthy" even
   though the app is fine).
6. GPU: pick the smallest tier available; Workers: min 0 / max 2-3 (see
   above) — adjust after watching real traffic.
7. Add `API_SHARED_SECRET` as an environment variable/secret (Settings →
   Environment Variables) — same value used on the Cloud Run side, or a
   different one if you want the two deployments cryptographically
   independent (your call; nothing here requires them to match).
8. Deploy, then `curl https://<ENDPOINT_ID>.api.runpod.ai/ping -H "Authorization: Bearer <RUNPOD_API_KEY>"` and confirm the same JSON `/health` shape comes back.
9. Send one real video through `POST /analyze` (both headers, same
   multipart form as Cloud Run) before pointing the Edge Function at it.
10. Update Supabase's `EXERCISE_ANALYSIS_API_URL` and add the Edge
    Function's `Authorization: Bearer` header (see above) — this is a
    Mulhim-repo change, out of scope for this session per your instruction
    not to touch that repo here.
11. Check for a RunPod endpoint-scoped API key option (see Authentication
    section) before handing Supabase a full account key.
