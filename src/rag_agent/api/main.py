"""FastAPI inference service exposing the privacy-preserving RAG agent.

Endpoints:
* ``GET  /health``   — liveness/readiness probe (always 200; reports agent state)
* ``GET  /tools``    — list the tools available to the agent
* ``POST /chat``     — ask the agent a question (records history + audit)
* ``GET  /history``  — return conversation history for a session
* ``WS   /ws/chat``  — streaming chat over WebSocket (privacy filter-then-stream)
* ``POST /ingest``   — (re)index the configured corpus
* ``GET  /``         — minimal chat frontend

The agent is built lazily on startup. If its dependencies or the
``ANTHROPIC_API_KEY`` are absent, the service still starts; ``/chat`` returns 503
until the agent is available — so the container is always probeable. The
conversation store and audit log are always available.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..audit import AuditLog
from ..config import load_config
from ..exceptions import RagAgentError
from ..logging_config import configure_logging, get_logger
from ..memory import ConversationStore
from ..tools import tool_specs

log = get_logger(__name__)

_STATIC = Path(__file__).parent / "static"
# Populated on startup; individually replaceable in tests.
_STATE: dict[str, Any] = {"agent": None, "audit": None, "history": None}


def _try_build_agent() -> None:
    """Attempt to build the agent, degrading gracefully on failure."""
    from ..agent import build_agent

    try:
        _STATE["agent"] = build_agent()
        log.info("agent_ready")
    except Exception as exc:  # noqa: BLE001 - startup must never crash
        log.warning("agent_unavailable", error=str(exc))
        _STATE["agent"] = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure logging, init history + audit, and build the agent on startup."""
    cfg = load_config()
    configure_logging(cfg.log_level)
    _STATE["history"] = ConversationStore()
    _STATE["audit"] = AuditLog(cfg.paths.audit_log)
    _try_build_agent()
    yield


app = FastAPI(
    title="Privacy-Preserving RAG Agent",
    version="0.1.0",
    description="Research Q&A with tool-calling over synthetic-only data.",
    lifespan=lifespan,
)


# --- Pydantic models --------------------------------------------------------
class ChatRequest(BaseModel):
    """A single user question, optionally tied to a conversation."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    """The agent's privacy-filtered answer and its session id."""

    answer: str
    session_id: str


class ToolInfo(BaseModel):
    """Public metadata for a single agent tool."""

    name: str
    description: str


class ToolsResponse(BaseModel):
    """The catalogue of tools available to the agent."""

    tools: list[ToolInfo]


class Turn(BaseModel):
    """A single stored conversation turn."""

    role: str
    content: str
    ts: str


class HistoryResponse(BaseModel):
    """Conversation history for a session."""

    session_id: str
    turns: list[Turn]


# --- Helpers ----------------------------------------------------------------
def _answer_and_record(message: str, session_id: str) -> str:
    """Run the agent, persist the turn pair to history + audit, return the answer."""
    answer: str = _STATE["agent"].answer(message, session_id=session_id)
    history: ConversationStore | None = _STATE.get("history")
    if history is not None:
        history.add(session_id, "user", message)
        history.add(session_id, "assistant", answer)
    audit: AuditLog | None = _STATE.get("audit")
    if audit is not None:
        audit.record("chat", session_id=session_id, message=message, answer=answer)
    return answer


def _chunks(text: str) -> list[str]:
    """Split ``text`` into word chunks (with spaces) for incremental delivery."""
    parts = text.split(" ")
    return [p + " " if i < len(parts) - 1 else p for i, p in enumerate(parts)]


# --- Endpoints --------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe."""
    return {"status": "ok", "agent_ready": _STATE["agent"] is not None}


@app.get("/tools", response_model=ToolsResponse)
def tools() -> ToolsResponse:
    """List the tools the agent can call (available even if the agent is down)."""
    return ToolsResponse(tools=[ToolInfo(**t) for t in tool_specs()])


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question via the agent, recording history and an audit entry.

    Raises:
        HTTPException: 503 if the agent is not available; 500 on agent error.
    """
    if _STATE["agent"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not available (missing ANTHROPIC_API_KEY or index).",
        )
    session_id = req.session_id or uuid.uuid4().hex
    try:
        answer = _answer_and_record(req.message, session_id)
    except RagAgentError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    return ChatResponse(answer=answer, session_id=session_id)


@app.get("/history", response_model=HistoryResponse)
def history(session_id: str = Query(..., min_length=1, max_length=64)) -> HistoryResponse:
    """Return the stored conversation turns for ``session_id``."""
    store: ConversationStore | None = _STATE.get("history")
    turns = store.history(session_id) if store is not None else []
    return HistoryResponse(session_id=session_id, turns=[Turn(**t) for t in turns])


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Streaming chat over WebSocket.

    Privacy-safe streaming: the agent first produces a *fully privacy-filtered*
    answer, which is then streamed to the client in word chunks. Unfiltered model
    tokens are never sent over the wire.

    Protocol (JSON frames):
        client -> {"message": str, "session_id"?: str}
        server -> {"type": "start", "session_id": str}
               -> {"type": "chunk", "data": str}  (repeated)
               -> {"type": "done"}
               -> {"type": "error", "detail": str}
    """
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            message = (payload or {}).get("message", "")
            if not isinstance(message, str) or not message.strip():
                await websocket.send_json({"type": "error", "detail": "empty message"})
                continue
            if _STATE["agent"] is None:
                await websocket.send_json({"type": "error", "detail": "agent unavailable"})
                continue
            session_id = (payload.get("session_id") or uuid.uuid4().hex)[:64]
            await websocket.send_json({"type": "start", "session_id": session_id})
            try:
                answer = await run_in_threadpool(_answer_and_record, message, session_id)
            except RagAgentError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
                continue
            for chunk in _chunks(answer):
                await websocket.send_json({"type": "chunk", "data": chunk})
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        log.info("ws_disconnect")


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
