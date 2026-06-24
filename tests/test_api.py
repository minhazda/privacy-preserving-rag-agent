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
    def answer(self, question: str) -> str:
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
