"""Tests for the offline evaluation harness and its metrics."""

from __future__ import annotations

import json
from pathlib import Path

from rag_agent.eval import (
    GOLD_CASES,
    answer_relevance,
    context_precision,
    evaluate,
    faithfulness,
    load_cases,
)

_CTX = ["Synthetic data is artificially generated and contains no real records."]


def test_faithfulness_high_when_answer_grounded() -> None:
    answer = "Synthetic data is artificially generated and contains no real records."
    assert faithfulness(answer, _CTX) == 1.0


def test_faithfulness_low_when_answer_unsupported() -> None:
    answer = "The model teleports groceries using quantum entanglement spells."
    assert faithfulness(answer, _CTX) < 0.5


def test_faithfulness_empty_answer_is_vacuously_true() -> None:
    assert faithfulness("", _CTX) == 1.0


def test_answer_relevance_rewards_question_coverage() -> None:
    q = "What is synthetic data?"
    assert answer_relevance("Synthetic data is generated data.", q) == 1.0
    assert answer_relevance("Bananas are yellow fruit.", q) < 0.5


def test_context_precision_counts_relevant_contexts() -> None:
    reference = "synthetic data preserves privacy without real records"
    contexts = [
        "Synthetic data preserves privacy without real records.",  # relevant
        "The weather in Paris is mild in spring.",  # irrelevant
    ]
    assert context_precision(reference, contexts) == 0.5


def test_context_precision_empty_contexts_is_zero() -> None:
    assert context_precision("anything", []) == 0.0


def test_gold_set_passes_default_thresholds() -> None:
    report = evaluate(GOLD_CASES)
    assert report.passes() is True
    means = report.means()
    assert 0.0 <= means["faithfulness"] <= 1.0


def test_report_fails_impossible_thresholds() -> None:
    report = evaluate(GOLD_CASES)
    assert report.passes({"faithfulness": 1.01}) is False


def test_load_cases_round_trip(tmp_path: Path) -> None:
    data = [
        {
            "question": "q?",
            "reference": "r",
            "answer": "a",
            "contexts": ["c1", "c2"],
        }
    ]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].contexts == ("c1", "c2")
