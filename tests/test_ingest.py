"""Tests for the pure-Python chunker and document loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.exceptions import IngestionError
from rag_agent.ingest import chunk_text, load_documents


def test_chunk_text_overlap_and_coverage() -> None:
    text = " ".join(str(i) for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_rejects_bad_overlap() -> None:
    with pytest.raises(IngestionError):
        chunk_text("hello world", size=10, overlap=10)


def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("   ", size=50, overlap=5) == []


def test_load_documents_reads_text_and_skips_readme(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("synthetic research notes", encoding="utf-8")
    (tmp_path / "README.md").write_text("not a source doc", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("a,b,c", encoding="utf-8")
    loaded = load_documents(tmp_path)
    names = {name for name, _ in loaded}
    assert names == {"doc.txt"}


def test_load_documents_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_documents(tmp_path / "nope")
