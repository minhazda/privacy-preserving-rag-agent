# syntax=docker/dockerfile:1.7
# ===========================================================================
# Multi-stage build for the Privacy-Preserving RAG Agent.
#   Stage 1 (builder): install pinned wheels into a venv.
#   Stage 2 (runtime): copy venv + source -> small, non-root image.
# Embeddings run on-device (ONNX MiniLM via chromadb); no GPU/torch needed.
# ===========================================================================

FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="privacy-rag-agent" \
      org.opencontainers.image.description="Privacy-preserving RAG agent with tool-calling over synthetic data." \
      org.opencontainers.image.authors="Md Minhazur Rahman <minhazurrahman.ds@gmail.com>" \
      org.opencontainers.image.source="https://github.com/minhazda/privacy-preserving-rag-agent"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    RAG_CONFIG=/app/configs/config.yaml \
    API_PORT=8080

COPY --from=builder /opt/venv /opt/venv

# Unprivileged user + writable runtime dirs (chroma store + model cache).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma /app/data/documents \
    && chown -R appuser:appuser /app

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser data/documents/ ./data/documents/
COPY --chown=appuser:appuser docker/entrypoint.sh ./docker/entrypoint.sh
COPY --chown=appuser:appuser pyproject.toml README.md ./
RUN chmod +x ./docker/entrypoint.sh

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${API_PORT}/health" || exit 1

ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["serve"]
