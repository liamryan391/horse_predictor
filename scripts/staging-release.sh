#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${DATABASE_URL:?DATABASE_URL must be set before running the staging release.}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m alembic upgrade head

if [ "${SEED_SAMPLE_DATA:-false}" = "true" ]; then
  "$PYTHON_BIN" data_pipeline.py --provider sample --no-csv --disable-lock
fi

if [ "${RUN_ACCEPTANCE_CHECKS:-false}" = "true" ]; then
  : "${API_BASE_URL:?API_BASE_URL must be set when RUN_ACCEPTANCE_CHECKS=true.}"
  READINESS_ARGS=(
    "scripts/production-readiness-check.py"
    "--base-url"
    "$API_BASE_URL"
    "--require-approved-model"
    "--require-approved-artifact"
    "--require-prediction-run"
    "--require-monitoring"
    "--require-admin-governance"
  )
  if [ "${ALLOW_STALE_DATA:-false}" = "true" ]; then READINESS_ARGS+=("--allow-stale"); fi
  if [ "${REQUIRE_POLICY_LINKS:-false}" = "true" ]; then READINESS_ARGS+=("--require-policy-links"); fi
  if [ "${REQUIRE_ENRICHED_DATA:-true}" != "false" ]; then READINESS_ARGS+=("--require-enriched-data"); fi
  if [ "${REQUIRE_NO_CRITICAL_ALERTS:-true}" != "false" ]; then READINESS_ARGS+=("--require-no-critical-alerts"); fi
  "$PYTHON_BIN" "${READINESS_ARGS[@]}"
fi

if [ -n "${RELEASE_RECORD_PATH:-}" ]; then
  : "${API_BASE_URL:?API_BASE_URL must be set when RELEASE_RECORD_PATH is set.}"
  "$PYTHON_BIN" scripts/release-record.py --base-url "$API_BASE_URL" --output "$RELEASE_RECORD_PATH"
fi

echo "Staging release checks completed."
