# Deployment

## Prerequisites

- Python 3.11 (the project targets 3.11; a `.venv` is used locally).
- An `ANTHROPIC_API_KEY` for the live agent (the service starts without one,
  but `/chat` returns 503 until it is set).
- Optional: Project 1's forecasting API running for the `forecast_demand` tool.

## Configuration

`configs/config.yaml` is the single source of truth (paths, model, retrieval,
privacy, DP). The config path can be overridden with the `RAG_CONFIG` env var.
Secrets are read **only** from the environment.

```bash
export ANTHROPIC_API_KEY=sk-...
export RAG_CONFIG=configs/config.yaml   # optional
```

## Option A — Docker Compose (recommended)

```bash
docker compose run --rm ingest      # build the vector index
docker compose up api               # serve UI + API on :8080
```

The image is built and smoke-tested in CI (it imports the app and loads config
with no key). A container healthcheck can hit `GET /health`.

## Option B — Local Python

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m rag_agent.ingest                          # build the index
uvicorn rag_agent.api.main:app --host 0.0.0.0 --port 8080
```

## Indexing the corpus

Place `dissertation.pdf` and `preprint.pdf` in `data/documents/` (a synthetic
sample doc ships so the stack works without them), then run ingestion via
`POST /ingest`, `python -m rag_agent.ingest`, or the compose `ingest` service.
The store persists under `data/chroma/`.

## Evaluation gate

```bash
python -m rag_agent.eval            # prints metrics table; exits 1 if below thresholds
```

Wire this into CI to block regressions in retrieval/answer quality.

## Health, scaling & ops

- **Health**: `GET /health` for liveness/readiness probes.
- **Stateless-ish**: conversation memory is in-process. For multi-replica
  deployments, pin sessions (sticky routing) or back memory with a shared store.
- **Audit log**: hash-chained JSONL at `paths.audit_log` (default
  `data/audit/`); ship it to durable storage and verify integrity with
  `AuditLog.verify()`.
- **Privacy/DP**: enable `privacy.dp_enabled` and tune `dp_epsilon` to perturb
  forecast outputs.

## CI/CD

GitHub Actions runs three jobs: (1) lint + type-check + mocked unit tests on a
light profile (no key), (2) build & push the full image to GHCR, (3) image smoke
test. Quality gates: ruff, black, mypy, pytest (80%+ coverage target).
