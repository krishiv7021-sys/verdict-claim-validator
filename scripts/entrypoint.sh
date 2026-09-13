#!/usr/bin/env bash
set -eo pipefail

echo "=================================================="
echo "⚖️  Starting VERDICT Production Service"
echo "=================================================="

# Ensure persistent directories exist
mkdir -p "${DATA_DIR:-/app/data}"
if [ -d "/app/data" ] && [ ! -L "/app/certificates" ]; then
    mkdir -p /app/data/certificates
    rm -rf /app/certificates 2>/dev/null || true
    ln -s /app/data/certificates /app/certificates
else
    mkdir -p "${CERTIFICATES_DIR:-/app/certificates}"
fi

# Resolve target port and endpoints
PORT="${PORT:-8501}"
export PORT
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"
export DATABASE_PATH="${DATABASE_PATH:-/app/data/verdict.db}"
export PYTHONPATH="/app:${PYTHONPATH:-.}"

echo "[1/4] Ensuring SQLite database schema is initialized..."
python -c "from backend.database.schema import init_db; init_db()"

echo "[2/4] Starting FastAPI backend service on 127.0.0.1:8000..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info &
FASTAPI_PID=$!

# Trap signals for graceful shutdown of background services
cleanup() {
    echo "Shutting down VERDICT services..."
    if [ -n "$FASTAPI_PID" ] && kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "Stopping internal FastAPI backend (PID: $FASTAPI_PID)..."
        kill -TERM "$FASTAPI_PID" 2>/dev/null || true
    fi
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "Stopping Streamlit frontend (PID: $STREAMLIT_PID)..."
        kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT EXIT

echo "[3/4] Checking internal FastAPI readiness..."
FASTAPI_READY=false
for i in {1..20}; do
    if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "❌ FATAL: Internal FastAPI process terminated unexpectedly."
        exit 1
    fi
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)" 2>/dev/null; then
        FASTAPI_READY=true
        echo "✅ Internal FastAPI backend is ready and healthy on 127.0.0.1:8000."
        break
    fi
    sleep 0.5
done

if [ "$FASTAPI_READY" != "true" ]; then
    echo "⚠️ Warning: Internal FastAPI health check timed out. Proceeding with in-process fallback."
fi

echo "[4/4] Starting Streamlit frontend on 0.0.0.0:${PORT}..."
streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to exit, preserving exit code
wait "$STREAMLIT_PID"
EXIT_CODE=$?
exit "$EXIT_CODE"
