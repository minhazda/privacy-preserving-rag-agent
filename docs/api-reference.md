# API Reference

Base URL (local): `http://localhost:8080`. Interactive docs are served at
`/docs` (Swagger) and `/redoc`. All request/response bodies are validated by
Pydantic models.

## `GET /health`

Liveness/readiness. Always 200.

```json
{ "status": "ok", "agent_ready": true }
```

## `GET /tools`

Lists the tools the agent can call (available even when the agent is degraded).

```json
{ "tools": [ { "name": "retrieve_research", "description": "..." }, ... ] }
```

## `POST /chat`

Ask a question. Records the turn to history + the audit log.

Request:

```json
{ "message": "What MAE did the model achieve?", "session_id": "optional-id" }
```

| Field | Type | Notes |
| --- | --- | --- |
| `message` | string | 1–2000 chars, required. |
| `session_id` | string\|null | ≤64 chars; generated if omitted. |

Response `200`:

```json
{ "answer": "…privacy-filtered answer…", "session_id": "abc123" }
```

Errors: `422` invalid body; `503` agent unavailable; `500` agent error.

## `GET /history?session_id=...`

Returns stored turns for a session.

```json
{ "session_id": "abc123",
  "turns": [ { "role": "user", "content": "…", "ts": "2026-…Z" },
             { "role": "assistant", "content": "…", "ts": "2026-…Z" } ] }
```

## `WS /ws/chat`

Streaming chat. **Privacy-safe filter-then-stream**: the agent produces a fully
privacy-filtered answer, which is then streamed in chunks; unfiltered tokens are
never sent.

Client → server: `{ "message": "...", "session_id": "optional" }`

Server → client frames:

```json
{ "type": "start", "session_id": "abc123" }
{ "type": "chunk", "data": "partial text " }
{ "type": "done" }
{ "type": "error", "detail": "..." }
```

## `POST /ingest`

(Re)indexes the configured corpus.

```json
{ "chunks_indexed": 128 }
```

## `GET /`

Serves the minimal chat frontend.

## Tools

| Tool | Input | Output |
| --- | --- | --- |
| `retrieve_research` | question (str) | Cited context, or an "I don't know" signal if confidence is low. |
| `forecast_demand` | rows (list of synthetic feature dicts) | Predicted demand (optionally ε-DP perturbed). |
| `explain_method` | method name (str) | Concise explanation of a synthetic-data/privacy/forecasting method. |

## Authentication & secrets

The agent reads `ANTHROPIC_API_KEY` from the environment only; it is never
accepted in a request body or logged.
