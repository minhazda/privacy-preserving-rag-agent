"""ChromaDB-backed vector store with local, on-device embeddings.

Embeddings use ChromaDB's bundled ONNX ``all-MiniLM-L6-v2`` model: no API key,
no network, nothing leaves the machine — consistent with the privacy design.

Heavy imports (``chromadb``) are done lazily inside functions so that the
lightweight modules (config, privacy, tool logic) can be imported and unit
tested without installing the full vector stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import Config
from .exceptions import VectorStoreError
from .logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chromadb.api.models.Collection import Collection

log = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk with its provenance and distance score."""

    text: str
    source: str
    distance: float


def get_collection(cfg: Config) -> Collection:
    """Open (or create) the persistent Chroma collection for the corpus.

    Raises:
        VectorStoreError: If Chroma cannot be initialised.
    """
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:  # pragma: no cover - env guard
        raise VectorStoreError(
            "chromadb is not installed. Install requirements.txt to use the " "vector store."
        ) from exc

    cfg.paths.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.paths.chroma_dir))
    embedder = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=cfg.ingest.collection_name,
        embedding_function=embedder,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    collection: Collection,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert chunks into the collection (idempotent on ``ids``)."""
    if not ids:
        return
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)  # type: ignore[arg-type]
    log.info("chunks_indexed", count=len(ids))


def query(collection: Collection, text: str, k: int) -> list[RetrievedChunk]:
    """Return the ``k`` nearest chunks to ``text``."""
    res = collection.query(query_texts=[text], n_results=k)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists, strict=False):
        source = str((meta or {}).get("source", "unknown"))
        out.append(RetrievedChunk(text=doc, source=source, distance=float(dist)))
    return out
