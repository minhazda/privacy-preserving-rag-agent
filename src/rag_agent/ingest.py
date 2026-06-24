"""Corpus ingestion: load documents, chunk them, and index into Chroma.

The character chunker is pure-Python (no heavy deps) so it is unit-testable in
isolation. PDF parsing uses ``pypdf`` lazily; ``.md`` / ``.txt`` files are read
directly. Chunk IDs are content-stable so re-ingesting is idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .config import Config, load_config
from .exceptions import IngestionError
from .logging_config import configure_logging, get_logger
from .vectorstore import add_chunks, get_collection

log = get_logger(__name__)

_SUPPORTED = {".pdf", ".md", ".txt"}


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping character windows.

    Args:
        text: Raw document text.
        size: Target chunk length in characters (must be > 0).
        overlap: Characters shared between consecutive chunks (< ``size``).

    Returns:
        Non-empty, stripped chunks in document order.

    Raises:
        IngestionError: If ``size``/``overlap`` are invalid.
    """
    if size <= 0:
        raise IngestionError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise IngestionError("overlap must satisfy 0 <= overlap < size")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    step = size - overlap
    chunks = [cleaned[i : i + size] for i in range(0, len(cleaned), step)]
    return [c.strip() for c in chunks if c.strip()]


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using ``pypdf`` (imported lazily)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - env guard
        raise IngestionError("pypdf is required to read PDF files.") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(documents_dir: Path) -> list[tuple[str, str]]:
    """Load supported documents from ``documents_dir``.

    Returns:
        A list of (source_filename, full_text) pairs.

    Raises:
        IngestionError: If the directory does not exist.
    """
    if not documents_dir.is_dir():
        raise IngestionError(f"documents_dir not found: {documents_dir}")
    out: list[tuple[str, str]] = []
    for path in sorted(documents_dir.iterdir()):
        if path.suffix.lower() not in _SUPPORTED or path.name == "README.md":
            continue
        text = (
            _read_pdf(path)
            if path.suffix.lower() == ".pdf"
            else path.read_text(encoding="utf-8", errors="ignore")
        )
        if text.strip():
            out.append((path.name, text))
    return out


def ingest(cfg: Config) -> int:
    """Load, chunk, and index the corpus. Returns the number of chunks indexed."""
    docs = load_documents(cfg.paths.documents_dir)
    if not docs:
        log.warning("no_documents_found", dir=str(cfg.paths.documents_dir))
        return 0

    collection = get_collection(cfg)
    total = 0
    for source, text in docs:
        chunks = chunk_text(text, cfg.ingest.chunk_size, cfg.ingest.chunk_overlap)
        ids, metadatas = [], []
        for i, chunk in enumerate(chunks):
            digest = hashlib.sha1(f"{source}:{i}:{chunk}".encode()).hexdigest()[:16]
            ids.append(f"{source}-{i}-{digest}")
            metadatas.append({"source": source, "chunk_index": i})
        add_chunks(collection, ids, chunks, metadatas)
        total += len(chunks)
        log.info("document_ingested", source=source, chunks=len(chunks))
    log.info("ingestion_complete", documents=len(docs), chunks=total)
    return total


def main() -> None:
    """CLI entrypoint: ingest the configured corpus."""
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.log_level)
    ingest(cfg)


if __name__ == "__main__":
    main()
