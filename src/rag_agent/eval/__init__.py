"""Offline, deterministic RAG evaluation (RAGAS-style proxies).

Public API:
    * metrics: faithfulness, answer_relevance, context_precision
    * dataset: EvalCase, GOLD_CASES, load_cases
    * harness: evaluate, EvalReport, CaseScore, DEFAULT_THRESHOLDS
"""

from __future__ import annotations

from .dataset import GOLD_CASES, EvalCase, load_cases
from .harness import (
    DEFAULT_THRESHOLDS,
    CaseScore,
    EvalReport,
    evaluate,
    score_case,
)
from .metrics import answer_relevance, context_precision, faithfulness

__all__ = [
    "DEFAULT_THRESHOLDS",
    "GOLD_CASES",
    "CaseScore",
    "EvalCase",
    "EvalReport",
    "answer_relevance",
    "context_precision",
    "evaluate",
    "faithfulness",
    "load_cases",
    "score_case",
]
