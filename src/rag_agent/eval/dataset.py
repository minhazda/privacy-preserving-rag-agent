"""Evaluation dataset: a typed case schema, a built-in gold set, and a loader.

Each :class:`EvalCase` bundles a question, a reference answer, the answer that
was produced, and the contexts that were retrieved. The built-in
:data:`GOLD_CASES` are grounded in the project's research domain and let the
harness compute real metric values offline (no agent, no API key) so CI can
gate on them. Live evaluation can instead supply cases whose ``answer`` and
``contexts`` were produced by the running agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """One evaluation example (question + reference + produced answer/contexts)."""

    question: str
    reference: str
    answer: str
    contexts: tuple[str, ...]


GOLD_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        question="What is synthetic data and why does it preserve privacy?",
        reference=(
            "Synthetic data is artificially generated data that preserves the "
            "statistical structure of a real dataset without containing real "
            "records; it preserves privacy because it contains no real "
            "individuals, which removes identifiers and reduces re-identification."
        ),
        answer=(
            "Synthetic data is artificially generated data that preserves the "
            "statistical structure of a real dataset without containing real "
            "records. It preserves privacy because it contains no real "
            "individuals, which removes identifiers and reduces re-identification "
            "risk."
        ),
        contexts=(
            "Synthetic data is artificially generated data that preserves the "
            "statistical structure of a real dataset without containing any real "
            "records.",
            "Because synthetic data contains no real individuals, it removes "
            "identifiers and reduces re-identification risk, which is how it "
            "preserves privacy.",
        ),
    ),
    EvalCase(
        question="Which model was used for demand forecasting and how was accuracy measured?",
        reference=(
            "Gradient-boosted decision trees were used for demand forecasting; "
            "accuracy was measured with mean absolute error."
        ),
        answer=(
            "Demand forecasting used gradient-boosted decision trees, and "
            "accuracy was measured with mean absolute error (MAE)."
        ),
        contexts=(
            "Demand forecasting used gradient-boosted decision trees, which fit "
            "an additive ensemble of trees to reduce error.",
            "Forecast accuracy was measured with mean absolute error (MAE) " "against a baseline.",
        ),
    ),
    EvalCase(
        question="How does differential privacy protect forecast outputs?",
        reference=(
            "Differential privacy protects outputs by adding Laplace noise "
            "scaled by sensitivity over epsilon, bounding each record's effect, "
            "controlled by a privacy budget epsilon."
        ),
        answer=(
            "Differential privacy protects forecast outputs by adding calibrated "
            "Laplace noise scaled by sensitivity over epsilon, bounding any "
            "single record's effect on the output."
        ),
        contexts=(
            "Differential privacy adds calibrated noise so that any single "
            "record has a bounded effect on an output, controlled by a privacy "
            "budget epsilon.",
            "The Laplace mechanism adds noise scaled by sensitivity over epsilon "
            "to numeric forecast outputs.",
        ),
    ),
    EvalCase(
        question="What is CTGAN and what does it do for tabular data?",
        reference=(
            "CTGAN is a conditional tabular GAN that uses mode-specific "
            "normalisation for continuous columns and training-by-sampling for "
            "imbalanced categorical columns."
        ),
        answer=(
            "CTGAN is a conditional tabular GAN that uses mode-specific "
            "normalisation for multi-modal continuous columns and "
            "training-by-sampling for imbalanced categorical columns in tabular "
            "data."
        ),
        contexts=(
            "CTGAN is a conditional tabular GAN that uses mode-specific "
            "normalisation for multi-modal continuous columns.",
            "CTGAN uses a conditional generator with training-by-sampling for "
            "imbalanced categorical columns in tabular data.",
        ),
    ),
)


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file (list of case objects).

    Each object must have ``question``, ``reference``, ``answer``, and
    ``contexts`` (a list of strings).

    Raises:
        ValueError: If the JSON is not a list or a case is missing fields.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval dataset must be a JSON list of case objects.")
    cases: list[EvalCase] = []
    for i, obj in enumerate(raw):
        try:
            cases.append(
                EvalCase(
                    question=obj["question"],
                    reference=obj["reference"],
                    answer=obj["answer"],
                    contexts=tuple(obj["contexts"]),
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed eval case at index {i}: {exc}") from exc
    return cases
