#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

echo "=== CroqTuner Agent Bot ==="

echo "[1/2] Starting backend on port ${CROQTUNER_PORT:-8642}..."
cd "$PROJECT_DIR/backend"
if [ ! -d ".venv" ]; then
    echo "  Creating Python virtualenv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
else
    source .venv/bin/activate
fi
uvicorn app.main:app --host "${CROQTUNER_HOST:-0.0.0.0}" --port "${CROQTUNER_PORT:-8642}" &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

echo "[2/2] Starting frontend dev server on port 5173..."
cd "$PROJECT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "  Installing npm dependencies..."
    npm install -q
fi
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "Dashboard: http://localhost:5173"
echo "API:       http://localhost:${CROQTUNER_PORT:-8642}/api"
echo "Press Ctrl+C to stop."
echo ""

cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
