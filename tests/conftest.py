"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.config import Config, load_config

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture
def cfg() -> Config:
    """The real project configuration, loaded from configs/config.yaml."""
    return load_config(_CONFIG_PATH)


class FakeCollection:
    """Minimal stand-in for a Chroma collection used in retrieval tests."""

    def __init__(self, docs: list[str], sources: list[str]) -> None:
        self._docs = docs
        self._sources = sources

    def query(self, query_texts: list[str], n_results: int) -> dict[str, object]:
        n = min(n_results, len(self._docs))
        return {
            "documents": [self._docs[:n]],
            "metadatas": [[{"source": s} for s in self._sources[:n]]],
            "distances": [[0.1 * (i + 1) for i in range(n)]],
        }


@pytest.fixture
def fake_collection() -> FakeCollection:
    """A fake collection returning two synthetic passages."""
    return FakeCollection(
        docs=["Synthetic data preserves privacy.", "LightGBM cut MAE vs baseline."],
        sources=["dissertation.pdf", "preprint.pdf"],
    )
