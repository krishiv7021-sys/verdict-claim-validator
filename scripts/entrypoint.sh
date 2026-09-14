#!/usr/bin/env bash
set -eo pipefail

echo "=================================================="
echo "⚖️  Starting VERDICT Service (Render Free Tier)"
echo "=================================================="

# Ensure runtime directories exist (ephemeral filesystem)
mkdir -p "${DATA_DIR:-/app/data}" "${CERTIFICATES_DIR:-/app/certificates}"

# Resolve target port and environment defaults
PORT="${PORT:-8501}"
export PORT
export DATABASE_PATH="${DATABASE_PATH:-/app/data/verdict.db}"
export PYTHONPATH="/app:${PYTHONPATH:-.}"
export USE_TFIDF_EMBEDDINGS="${USE_TFIDF_EMBEDDINGS:-}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[1/2] Ensuring SQLite database schema is initialized..."
python -c "from backend.database.schema import init_db; init_db()"

echo "[2/2] Starting Streamlit frontend on 0.0.0.0:${PORT} (Single-Process In-Process Engine)..."
exec streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true
