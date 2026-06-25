"""Tests for the FastAPI surface, including the degraded (no-agent) path."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import main as api_main

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture(autouse=True)
def _point_at_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the API's load_config() find the project config regardless of CWD."""
    monkeypatch.setenv("RAG_CONFIG", str(_CONFIG_PATH))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


class _FakeAgent:
    def answer(self, question: str, session_id: str | None = None) -> str:
        return f"answer to: {question}"


def test_health_reports_agent_state() -> None:
    with TestClient(api_main.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_chat_503_when_agent_unavailable() -> None:
    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = None
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 503


def test_chat_returns_answer_with_injected_agent() -> None:
    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = _FakeAgent()
        resp = client.post("/chat", json={"message": "what is synthetic data?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "answer to: what is synthetic data?"
    api_main._STATE["agent"] = None


def test_chat_validates_empty_message() -> None:
    with TestClient(api_main.app) as client:
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422


def test_tools_lists_all_three_tools() -> None:
    with TestClient(api_main.app) as client:
        resp = client.get("/tools")
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()["tools"]}
        assert {"retrieve_research", "forecast_demand", "explain_method"} <= names


def test_chat_returns_session_id_and_records_history() -> None:
    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = _FakeAgent()
        resp = client.post("/chat", json={"message": "hello", "session_id": "s1"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "s1"
        hist = client.get("/history", params={"session_id": "s1"})
        assert hist.status_code == 200
        turns = hist.json()["turns"]
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert turns[0]["content"] == "hello"
    api_main._STATE["agent"] = None


def test_chat_generates_session_id_when_absent() -> None:
    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = _FakeAgent()
        resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 200
        assert len(resp.json()["session_id"]) > 0
    api_main._STATE["agent"] = None


def test_history_unknown_session_is_empty() -> None:
    with TestClient(api_main.app) as client:
        resp = client.get("/history", params={"session_id": "does-not-exist"})
        assert resp.status_code == 200
        assert resp.json()["turns"] == []


def test_chat_writes_redacted_audit_entry(tmp_path: Path) -> None:
    from rag_agent.audit import AuditLog

    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = _FakeAgent()
        api_main._STATE["audit"] = AuditLog(tmp_path / "audit.jsonl")
        client.post("/chat", json={"message": "reach me at x@y.com", "session_id": "s2"})
        audit = api_main._STATE["audit"]
        assert "chat" in [e["event"] for e in audit.entries()]
        assert audit.verify() is True
        raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
        assert "x@y.com" not in raw  # PII redacted in the audit trail
    api_main._STATE["agent"] = None


def test_ws_streams_privacy_filtered_answer() -> None:
    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = _FakeAgent()
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "stream this", "session_id": "s3"})
            start = ws.receive_json()
            assert start["type"] == "start"
            assert start["session_id"] == "s3"
            chunks: list[str] = []
            while True:
                frame = ws.receive_json()
                if frame["type"] == "done":
                    break
                assert frame["type"] == "chunk"
                chunks.append(frame["data"])
            assert "".join(chunks) == "answer to: stream this"
    api_main._STATE["agent"] = None


def test_ws_rejects_empty_message() -> None:
    with TestClient(api_main.app) as client:
        api_main._STATE["agent"] = _FakeAgent()
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "   "})
            assert ws.receive_json()["type"] == "error"
    api_main._STATE["agent"] = None
