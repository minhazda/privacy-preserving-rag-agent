"""Append-only, hash-chained audit log for queries, tool calls, and answers.

Every entry is one JSON line: an ISO timestamp, an event name, a redacted
payload, the previous entry's hash, and this entry's SHA-256 hash. The chain
makes tampering detectable — altering or deleting any line breaks every hash
that follows. All free text is passed through the privacy redactor before being
written, so the audit trail itself never stores PII.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .logging_config import get_logger
from .privacy import redact_text

log = get_logger(__name__)

_GENESIS = "0" * 64
_BODY_KEYS = ("ts", "event", "data", "prev")


def _redact(value: Any) -> Any:
    """Recursively redact PII from any strings inside ``value``."""
    if isinstance(value, str):
        return redact_text(value)[0]
    if isinstance(value, Mapping):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(v) for v in value]
    return value


def _hash(body: str) -> str:
    """SHA-256 hex digest of ``body``."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _canonical(entry: Mapping[str, Any]) -> str:
    """Deterministic JSON of the hashable body fields (excludes ``hash``)."""
    return json.dumps({k: entry[k] for k in _BODY_KEYS}, sort_keys=True, ensure_ascii=False)


class AuditLog:
    """Thread-safe, append-only JSONL audit trail with a tamper-evident chain.

    Args:
        path: Destination JSONL file; parent directories are created.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """Read the final entry's hash so new appends chain correctly."""
        if not self.path.is_file():
            return _GENESIS
        last = _GENESIS
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    last = json.loads(stripped)["hash"]
                except (json.JSONDecodeError, KeyError):  # pragma: no cover - defensive
                    continue
        return last

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append a redacted, hash-chained entry and return it.

        Args:
            event: Short event name (e.g. ``"query"``, ``"tool_call"``).
            **fields: Arbitrary payload; strings are PII-redacted recursively.

        Returns:
            The full entry written, including its ``hash``.
        """
        with self._lock:
            entry: dict[str, Any] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                "data": _redact(fields),
                "prev": self._last_hash,
            }
            entry["hash"] = _hash(_canonical(entry))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._last_hash = entry["hash"]
            log.info("audit_event", kind=event)
            return entry

    def entries(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return audit entries in order (optionally only the last ``limit``)."""
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    out.append(json.loads(stripped))
        return out[-limit:] if limit is not None else out

    def verify(self) -> bool:
        """Recompute the hash chain; return True iff every link is intact."""
        prev = _GENESIS
        for entry in self.entries():
            if entry.get("prev") != prev:
                return False
            if _hash(_canonical(entry)) != entry.get("hash"):
                return False
            prev = entry["hash"]
        return True
