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

echo "Staging release checks completed."
