"""FastAPI inference service exposing the privacy-preserving RAG agent.

Endpoints:
* ``GET  /health``  — liveness/readiness probe (always 200; reports agent state)
* ``POST /chat``    — ask the agent a question
* ``POST /ingest``  — (re)index the configured corpus
* ``GET  /``        — minimal chat frontend

The agent is built lazily on startup. If its dependencies or the
``ANTHROPIC_API_KEY`` are absent, the service still starts and ``/chat`` returns
503 until the agent is available — so the container is always probeable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import load_config
from ..exceptions import RagAgentError
from ..logging_config import configure_logging, get_logger

log = get_logger(__name__)

_STATIC = Path(__file__).parent / "static"
# Populated on startup; replaceable in tests.
_STATE: dict[str, Any] = {"agent": None}


def _try_build_agent() -> None:
    """Attempt to build the agent, degrading gracefully on failure."""
    from ..agent import build_agent

    try:
        _STATE["agent"] = build_agent()
        log.info("agent_ready")
    except (RagAgentError, Exception) as exc:  # noqa: BLE001 - never crash startup
        log.warning("agent_unavailable", error=str(exc))
        _STATE["agent"] = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure logging and build the agent on startup."""
    cfg = load_config()
    configure_logging(cfg.log_level)
    _try_build_agent()
    yield


app = FastAPI(
    title="Privacy-Preserving RAG Agent",
    version="0.1.0",
    description="Research Q&A with tool-calling over synthetic-only data.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    """A single user question."""

    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """The agent's privacy-filtered answer."""

    answer: str


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe."""
    return {"status": "ok", "agent_ready": _STATE["agent"] is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question via the agent.

    Raises:
        HTTPException: 503 if the agent is not available; 500 on agent error.
    """
    agent = _STATE["agent"]
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not available (missing ANTHROPIC_API_KEY or index).",
        )
    try:
        return ChatResponse(answer=agent.answer(req.message))
    except RagAgentError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


@app.post("/ingest")
def ingest_corpus() -> dict[str, int]:
    """(Re)index the configured corpus and report the chunk count."""
    from ..ingest import ingest

    cfg = load_config()
    count = ingest(cfg)
    if _STATE["agent"] is None and os.environ.get("ANTHROPIC_API_KEY"):
        _try_build_agent()
    return {"chunks_indexed": count}


@app.get("/")
def index() -> FileResponse:
    """Serve the minimal chat frontend."""
    return FileResponse(_STATIC / "index.html")
