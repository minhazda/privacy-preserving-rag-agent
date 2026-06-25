"""In-memory conversation store backing the ``/history`` endpoint.

Keeps an ordered list of turns per session. Deliberately dependency-free and
process-local: it gives the API a conversation view and lets the agent be
addressed per session, without persisting user content to disk (the durable,
privacy-redacted record is the responsibility of :mod:`rag_agent.audit`).
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Turn:
    """A single message in a conversation."""

    role: str  # "user" | "assistant"
    content: str
    ts: str


class ConversationStore:
    """Thread-safe, per-session history of conversation turns."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, list[Turn]] = {}

    def add(self, session_id: str, role: str, content: str) -> Turn:
        """Append a turn to ``session_id`` and return it."""
        turn = Turn(
            role=role, content=content, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        with self._lock:
            self._sessions.setdefault(session_id, []).append(turn)
        return turn

    def history(self, session_id: str) -> list[dict[str, str]]:
        """Return the turns for ``session_id`` (empty list if unknown)."""
        with self._lock:
            return [asdict(t) for t in self._sessions.get(session_id, [])]

    def sessions(self) -> list[str]:
        """Return all known session ids in insertion order."""
        with self._lock:
            return list(self._sessions)

    def clear(self) -> None:
        """Drop all sessions (used by tests)."""
        with self._lock:
            self._sessions.clear()
