#!/usr/bin/env bash
set -euo pipefail

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
SKIP_DOCTOR="${SKIP_DOCTOR:-0}"

if ! command -v agent-browser >/dev/null 2>&1; then
  echo "agent-browser is not installed or not on PATH." >&2
  exit 127
fi

echo "Closing agent-browser sessions..."
timeout "${TIMEOUT_SECONDS}s" agent-browser close --all >/dev/null 2>&1 || true

if command -v pgrep >/dev/null 2>&1; then
  while read -r pid _; do
    if [ -n "${pid:-}" ] && [ "$pid" != "$$" ]; then
      echo "Stopping agent-browser helper process $pid"
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done < <(pgrep -af "agent-browser" || true)
fi

if [ "$SKIP_DOCTOR" != "1" ]; then
  echo "Running agent-browser doctor --fix..."
  timeout "${TIMEOUT_SECONDS}s" agent-browser doctor --fix || true
fi

echo "agent-browser reset complete."
