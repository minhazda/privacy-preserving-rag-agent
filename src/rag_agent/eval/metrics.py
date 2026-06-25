"""Deterministic, offline RAG evaluation metrics (RAGAS-style proxies).

Production RAGAS uses an LLM judge plus embeddings. These are *lexical proxies*
that run with no API key and no heavy dependencies, so they can gate CI. They
are pure and deterministic, each scoring in ``[0, 1]``:

* :func:`faithfulness`      — share of answer sentences supported by the contexts.
* :func:`answer_relevance`  — share of the question's content terms the answer covers.
* :func:`context_precision` — share of retrieved contexts relevant to the reference.

A light stemmer (trailing-``s`` stripping) and a small stop-word list make the
lexical overlap robust to trivial morphology without pulling in NLP libraries.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD = re.compile(r"[A-Za-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "for",
        "with",
        "that",
        "this",
        "it",
        "as",
        "by",
        "at",
        "be",
        "from",
        "has",
        "have",
        "had",
        "but",
        "not",
        "its",
        "their",
        "there",
        "which",
        "what",
        "why",
        "how",
        "does",
        "do",
        "did",
        "when",
        "where",
        "who",
        "whom",
        "can",
        "will",
        "would",
        "should",
        "if",
        "no",
    }
)


def _stem(token: str) -> str:
    """Strip a single trailing plural/3rd-person ``s`` (very light stemming)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokens(text: str) -> list[str]:
    """Lowercase, split to word tokens, drop stop-words, and light-stem."""
    return [
        _stem(m.group().lower())
        for m in _WORD.finditer(text)
        if m.group().lower() not in _STOPWORDS
    ]


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty sentences on terminal punctuation."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _coverage(target: set[str], source: set[str]) -> float:
    """Share of ``target`` tokens present in ``source`` (0 if target empty)."""
    if not target:
        return 0.0
    return len(target & source) / len(target)


def faithfulness(answer: str, contexts: Sequence[str], threshold: float = 0.5) -> float:
    """Share of answer sentences whose tokens are mostly covered by the contexts.

    A sentence counts as supported when at least ``threshold`` of its content
    tokens appear in the union of the context tokens. Returns 1.0 when the
    answer has no scorable sentences (nothing to contradict).
    """
    sentences = _split_sentences(answer)
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens |= set(tokens(c))
    supported = 0
    scorable = 0
    for sentence in sentences:
        st = set(tokens(sentence))
        if not st:
            continue
        scorable += 1
        if _coverage(st, ctx_tokens) >= threshold:
            supported += 1
    return 1.0 if scorable == 0 else supported / scorable


def answer_relevance(answer: str, question: str) -> float:
    """Share of the question's content tokens that the answer addresses."""
    return _coverage(set(tokens(question)), set(tokens(answer)))


def context_precision(reference: str, contexts: Sequence[str], threshold: float = 0.2) -> float:
    """Share of retrieved contexts that are relevant to the reference answer.

    A context is relevant when it covers at least ``threshold`` of the
    reference's content tokens. Returns 0.0 when there are no contexts.
    """
    if not contexts:
        return 0.0
    ref = set(tokens(reference))
    if not ref:
        return 0.0
    relevant = sum(1 for c in contexts if _coverage(ref, set(tokens(c))) >= threshold)
    return relevant / len(contexts)
