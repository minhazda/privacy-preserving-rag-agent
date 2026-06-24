#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Container entrypoint. One image, multiple roles via the first argument:
#   ingest  -> index the corpus into Chroma
#   serve   -> start the FastAPI app (default)
#   agent   -> one-shot question, e.g. `agent "what are the key results?"`
# Any other command is exec'd verbatim (e.g. bash, pytest).
# ---------------------------------------------------------------------------
set -euo pipefail

CMD="${1:-serve}"
shift || true

case "$CMD" in
  ingest)
    exec python -m rag_agent.ingest "$@"
    ;;
  serve)
    exec uvicorn rag_agent.api.main:app \
      --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8080}" "$@"
    ;;
  agent)
    exec python -m rag_agent.agent "$@"
    ;;
  *)
    exec "$CMD" "$@"
    ;;
esac
