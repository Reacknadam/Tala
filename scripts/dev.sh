#!/usr/bin/env bash
# TALA dev launcher (macOS/Linux): backend on :8000, frontend on :5173
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/backend/.venv/bin/python"
[ -f "$PY" ] || PY=python

(cd "$ROOT/backend" && exec "$PY" -m uvicorn app.main:app --reload --port 8000) &
BACK=$!
trap "kill $BACK 2>/dev/null" EXIT

echo "Backend  -> http://127.0.0.1:8000 (docs: /docs)"
echo "Frontend -> http://localhost:5173"
cd "$ROOT/frontend" && npm run dev
