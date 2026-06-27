"""LLM-as-judge scoring for RAG answers (Anthropic Claude).

Unlike the deterministic lexical proxies in :mod:`rag_agent.eval.metrics`, this
grades each case with a Claude model on three rubric dimensions (integer 1-5,
normalised to ``[0, 1]``): faithfulness, answer relevance, context precision.

Requires ``ANTHROPIC_API_KEY``. The judge model is configurable via
``RAG_JUDGE_MODEL`` (default ``claude-sonnet-4-6``). When Langfuse env vars are
present, each judgement is traced; tracing failures never break evaluation.

The Anthropic client is injectable so the unit tests run with a fake (no key,
no network).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .dataset import EvalCase

JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", "claude-sonnet-4-6")

_RUBRIC = (
    "You are a strict RAG evaluation judge. Score the answer-under-test on three "
    "dimensions, each an INTEGER 1-5 (5 = best):\n"
    "- faithfulness: is every claim in the answer supported by the contexts "
    "(no hallucination or unsupported claims)?\n"
    "- answer_relevance: does the answer directly and completely address the "
    "question?\n"
    "- context_precision: are the retrieved contexts relevant to the reference "
    "answer (little irrelevant material)?\n"
    'Return ONLY compact JSON: {"faithfulness":n,"answer_relevance":n,'
    '"context_precision":n,"rationale":"<=15 words"}'
)


@dataclass(frozen=True)
class JudgeScore:
    """Normalised (0-1) LLM-judge scores for one case."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    rationale: str = ""


def _norm(value: Any) -> float:
    """Map a 1-5 rating onto [0, 1]; clamp out-of-range values."""
    n = float(value)
    n = max(1.0, min(5.0, n))
    return (n - 1.0) / 4.0


def _build_prompt(case: EvalCase) -> str:
    ctx = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(case.contexts))
    return (
        f"Question:\n{case.question}\n\n"
        f"Reference answer:\n{case.reference}\n\n"
        f"Retrieved contexts:\n{ctx}\n\n"
        f"Answer under test:\n{case.answer}"
    )


def parse_judgement(text: str) -> JudgeScore:
    """Parse the judge's JSON (tolerating surrounding prose) into a JudgeScore."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge output: {text!r}")
    data = json.loads(match.group())
    return JudgeScore(
        faithfulness=_norm(data["faithfulness"]),
        answer_relevance=_norm(data["answer_relevance"]),
        context_precision=_norm(data["context_precision"]),
        rationale=str(data.get("rationale", "")),
    )


def judge_case(
    case: EvalCase, *, client: Any | None = None, model: str | None = None
) -> JudgeScore:
    """Grade a single case with the LLM judge.

    Args:
        case: the evaluation case.
        client: an Anthropic-compatible client (injected in tests). When None,
            a real ``anthropic.Anthropic()`` is constructed (needs the API key).
        model: override the judge model id.
    """
    model = model or JUDGE_MODEL
    if client is None:
        from anthropic import Anthropic

        client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        system=_RUBRIC,
        messages=[{"role": "user", "content": _build_prompt(case)}],
    )
    text = "".join(
        getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
    )
    score = parse_judgement(text)
    _maybe_trace(case, score, model)
    return score


def _maybe_trace(case: EvalCase, score: JudgeScore, model: str) -> None:
    """Best-effort Langfuse trace; no-op without keys, never raises."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return
    try:  # pragma: no cover - optional integration
        from langfuse import Langfuse

        Langfuse().trace(
            name="rag-judge",
            input=case.question,
            output={
                "faithfulness": score.faithfulness,
                "answer_relevance": score.answer_relevance,
                "context_precision": score.context_precision,
            },
            metadata={"model": model, "rationale": score.rationale},
        )
    except Exception:  # noqa: BLE001 - tracing must never break evaluation
        pass
