#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

".venv/bin/python" -m pip install -r requirements.txt
".venv/bin/python" -m alembic upgrade head
".venv/bin/python" data_pipeline.py --provider sample

if [ ! -d "frontend/node_modules" ]; then
  (cd frontend && npm install)
fi

cleanup() {
  kill "${API_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

".venv/bin/python" -m uvicorn api:app --reload --host 127.0.0.1 --port 8000 &
API_PID=$!

(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo "Horse Predictor is starting."
echo "Backend:  http://127.0.0.1:8000/api/health"
echo "Frontend: http://127.0.0.1:5173"
echo "Press Ctrl+C to stop both services."

wait -n "$API_PID" "$FRONTEND_PID"
