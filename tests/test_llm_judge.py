"""Tests for the LLM-as-judge scorer using an injected fake client (no API)."""

from __future__ import annotations

import pytest

from rag_agent.eval.dataset import GOLD_CASES
from rag_agent.eval.harness import evaluate
from rag_agent.eval.llm_judge import JudgeScore, judge_case, parse_judgement


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Msg:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _FakeClient:
    """Mimics anthropic.Anthropic().messages.create(...)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.messages = self

    def create(self, **_: object) -> _Msg:
        return _Msg(self._text)


def test_parse_judgement_maps_1_5_to_unit_interval() -> None:
    s = parse_judgement(
        '{"faithfulness":5,"answer_relevance":3,"context_precision":1,"rationale":"ok"}'
    )
    assert s.faithfulness == 1.0
    assert s.answer_relevance == 0.5
    assert s.context_precision == 0.0


def test_parse_judgement_tolerates_surrounding_prose() -> None:
    s = parse_judgement('Result: {"faithfulness":4,"answer_relevance":4,"context_precision":4} done')
    assert round(s.faithfulness, 2) == 0.75


def test_parse_judgement_clamps_out_of_range() -> None:
    s = parse_judgement('{"faithfulness":9,"answer_relevance":0,"context_precision":3}')
    assert s.faithfulness == 1.0
    assert s.answer_relevance == 0.0


def test_parse_judgement_raises_without_json() -> None:
    with pytest.raises(ValueError):
        parse_judgement("no json here")


def test_judge_case_with_fake_client() -> None:
    fake = _FakeClient('{"faithfulness":5,"answer_relevance":5,"context_precision":5}')
    score = judge_case(GOLD_CASES[0], client=fake)
    assert isinstance(score, JudgeScore)
    assert score.faithfulness == 1.0


def test_evaluate_lexical_still_works_without_keys() -> None:
    report = evaluate(GOLD_CASES)
    assert report.scorer == "lexical"
    assert report.passes()
