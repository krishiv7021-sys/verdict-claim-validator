# =====================================================================
# VERDICT: Claim Verification & Evidence Analysis — Production Container
# Combined service: Internal FastAPI (127.0.0.1:8000) + Public Streamlit ($PORT)
# =====================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies for health check and process utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache BGE embedding model weights into the image layer for instant cold starts.
# Does not fail the build if HuggingFace Hub is unreachable (runtime fallback preserved).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" || true

# Copy project files (.dockerignore excludes local virtualenvs, databases, secrets, and caches)
COPY . .

# Ensure storage directories exist and entrypoint is executable
RUN mkdir -p /app/data /app/certificates && \
    chmod +x /app/scripts/entrypoint.sh

# Default environment configuration (overridable at container runtime)
ENV PORT=8501 \
    PYTHONPATH=/app \
    DATABASE_PATH=/app/data/verdict.db \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    LLM_PROVIDER=""

# Streamlit port exposed externally
EXPOSE 8501

ENTRYPOINT ["/bin/bash", "/app/scripts/entrypoint.sh"]
