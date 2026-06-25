"""Tests for retrieval confidence gating (low-relevance -> honest fallback)."""

from __future__ import annotations

from typing import Any

from rag_agent.config import Config
from rag_agent.privacy import PrivacyGuard
from rag_agent.tools import retrieve_research


class _LowConfidenceCollection:
    """A collection whose best hit is far away (high cosine distance)."""

    def query(self, query_texts: list[str], n_results: int) -> dict[str, Any]:
        return {
            "documents": [["loosely related text"]],
            "metadatas": [[{"source": "preprint.pdf"}]],
            "distances": [[0.95]],  # relevance 0.05, below the 0.25 threshold
        }


def test_low_confidence_triggers_idk_fallback(cfg: Config) -> None:
    out = retrieve_research(
        "an unrelated question", cfg, _LowConfidenceCollection(), PrivacyGuard()
    )
    assert "do not have this information" in out


def test_high_confidence_returns_cited_context(cfg: Config, fake_collection: Any) -> None:
    out = retrieve_research("privacy?", cfg, fake_collection, PrivacyGuard())
    assert "[1]" in out
    assert "dissertation.pdf" in out
