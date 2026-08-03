#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" data_pipeline.py --repeat-hourly --no-csv --lock-name ingestion-worker
