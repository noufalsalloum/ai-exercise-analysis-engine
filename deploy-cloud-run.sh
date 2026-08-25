#!/usr/bin/env bash
# Deploys api/ to Cloud Run from source (Cloud Build builds the Dockerfile in
# this directory). Run this from the repo root:
#   ./deploy-cloud-run.sh
#
# Prerequisites (see CLOUD_RUN.md):
#   - gcloud CLI installed and `gcloud auth login` done
#   - The Secret Manager secret API_SHARED_SECRET already created:
#       printf '%s' 'your-real-secret' | gcloud secrets create API_SHARED_SECRET --data-file=-
#   - Billing enabled on the target project
#
# This script does not run automatically and does not commit/push anything —
# it only talks to Google Cloud when you invoke it.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-exercise-analysis-api}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "No project set. Run 'gcloud config set project YOUR_PROJECT_ID' or pass PROJECT_ID=... ./deploy-cloud-run.sh" >&2
  exit 1
fi

echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo

echo "Enabling required APIs (no-op if already enabled)..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "${PROJECT_ID}"

if ! gcloud secrets describe API_SHARED_SECRET --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo
  echo "Secret API_SHARED_SECRET does not exist yet in project ${PROJECT_ID}." >&2
  echo "Create it first, then re-run this script:" >&2
  echo "  printf '%s' 'your-real-secret' | gcloud secrets create API_SHARED_SECRET --data-file=- --project ${PROJECT_ID}" >&2
  exit 1
fi

echo
echo "Deploying (this uploads the repo root as the Cloud Build context — see"
echo ".dockerignore for what's excluded — and builds Dockerfile here)..."

# --allow-unauthenticated: this endpoint's only gate is the X-API-Key shared
# secret the exercise-analysis Edge Function sends (see api/main.py) — the
# Edge Function has no GCP identity to use IAM-based invoker auth instead.
# --memory/--cpu: torch + mediapipe + a MotionBERT forward pass need real
# headroom; start here and adjust from observed Cloud Run metrics.
# --timeout: the Edge Function's own BACKEND_TIMEOUT_MS is 180s: this must
# stay >= that or Cloud Run will cut the request before the caller gives up.
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --min-instances 0 \
  --max-instances 3 \
  --set-secrets "API_SHARED_SECRET=API_SHARED_SECRET:latest"

echo
SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)')"
echo "Deployed: ${SERVICE_URL}"
echo
echo "Next: set on the Supabase Edge Function side —"
echo "  supabase secrets set EXERCISE_ANALYSIS_API_URL=${SERVICE_URL}"
echo "  supabase secrets set EXERCISE_ANALYSIS_API_SHARED_SECRET=<the same value you put in API_SHARED_SECRET>"
echo
echo "Then smoke test: curl ${SERVICE_URL}/health"
