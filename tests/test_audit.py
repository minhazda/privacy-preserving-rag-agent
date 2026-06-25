"""Tests for the append-only, hash-chained audit log."""

from __future__ import annotations

from pathlib import Path

from rag_agent.audit import AuditLog


def test_record_and_read_in_order(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("query", message="hello")
    log.record("answer", text="world")
    assert [e["event"] for e in log.entries()] == ["query", "answer"]


def test_pii_is_redacted_before_write(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record("query", message="reach me at alice@example.com")
    raw = path.read_text(encoding="utf-8")
    assert "alice@example.com" not in raw
    assert "[REDACTED:email]" in raw


def test_intact_chain_verifies(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.record("event", i=i)
    assert log.verify() is True


def test_tampering_breaks_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.record("event", value="original")
    log.record("event", value="second")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("original", "forged")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert AuditLog(path).verify() is False


def test_chain_continues_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record("first", a=1)
    reopened = AuditLog(path)
    reopened.record("second", b=2)
    entries = reopened.entries()
    assert entries[1]["prev"] == entries[0]["hash"]
    assert reopened.verify() is True


def test_entries_limit_returns_tail(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(10):
        log.record("event", i=i)
    tail = log.entries(limit=3)
    assert [e["data"]["i"] for e in tail] == [7, 8, 9]
