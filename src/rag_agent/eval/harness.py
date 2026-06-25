"""Evaluation harness: score cases, aggregate, and gate on thresholds.

``evaluate`` scores a list of :class:`~rag_agent.eval.dataset.EvalCase` with the
deterministic metrics in :mod:`rag_agent.eval.metrics` and returns an
:class:`EvalReport`. The CLI (``python -m rag_agent.eval``) runs the built-in
gold set (or a JSON dataset), prints a table, and exits non-zero if any mean
metric is below its threshold — so it can fail a CI job.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from .dataset import GOLD_CASES, EvalCase, load_cases
from .metrics import answer_relevance, context_precision, faithfulness

DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.70,
    "answer_relevance": 0.60,
    "context_precision": 0.50,
}


@dataclass(frozen=True)
class CaseScore:
    """Per-case metric scores."""

    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation results over a set of cases."""

    scores: list[CaseScore] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        if not self.scores:
            return 0.0
        return sum(getattr(s, attr) for s in self.scores) / len(self.scores)

    def means(self) -> dict[str, float]:
        """Return the mean of each metric across all cases."""
        return {
            "faithfulness": round(self._mean("faithfulness"), 4),
            "answer_relevance": round(self._mean("answer_relevance"), 4),
            "context_precision": round(self._mean("context_precision"), 4),
        }

    def passes(self, thresholds: dict[str, float] | None = None) -> bool:
        """True iff every mean metric meets its threshold."""
        thr = thresholds or DEFAULT_THRESHOLDS
        means = self.means()
        return all(means[k] >= v for k, v in thr.items())


def score_case(case: EvalCase) -> CaseScore:
    """Compute all metrics for a single case."""
    return CaseScore(
        question=case.question,
        faithfulness=round(faithfulness(case.answer, case.contexts), 4),
        answer_relevance=round(answer_relevance(case.answer, case.question), 4),
        context_precision=round(context_precision(case.reference, case.contexts), 4),
    )


def evaluate(cases: Sequence[EvalCase]) -> EvalReport:
    """Score every case and return an :class:`EvalReport`."""
    return EvalReport(scores=[score_case(c) for c in cases])


def _format_table(report: EvalReport) -> str:
    """Render a compact, fixed-width results table."""
    lines = [f"{'faith':>7} {'ans_rel':>8} {'ctx_prec':>9}  question"]
    for s in report.scores:
        q = s.question if len(s.question) <= 50 else s.question[:47] + "..."
        lines.append(
            f"{s.faithfulness:7.2f} {s.answer_relevance:8.2f} " f"{s.context_precision:9.2f}  {q}"
        )
    means = report.means()
    lines.append(
        f"{means['faithfulness']:7.2f} {means['answer_relevance']:8.2f} "
        f"{means['context_precision']:9.2f}  == MEAN =="
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint. Returns a process exit code (0 pass, 1 fail)."""
    parser = argparse.ArgumentParser(description="Evaluate RAG quality (offline).")
    parser.add_argument("--dataset", default=None, help="JSON dataset (default: gold set).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    cases = load_cases(args.dataset) if args.dataset else list(GOLD_CASES)
    report = evaluate(cases)
    passed = report.passes()

    if args.json:
        print(json.dumps({"means": report.means(), "passed": passed}, indent=2))
    else:
        print(_format_table(report))
        print(f"\nThresholds: {DEFAULT_THRESHOLDS}")
        print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
