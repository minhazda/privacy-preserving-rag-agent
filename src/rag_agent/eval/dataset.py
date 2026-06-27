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
    EvalCase(
        question="What is MASE and why is it used to compare forecasts?",
        reference=(
            "MASE is the mean absolute scaled error: forecast MAE divided by the "
            "MAE of a naive baseline, so a value below 1 beats the baseline and "
            "it is comparable across series of different scales."
        ),
        answer=(
            "MASE (mean absolute scaled error) scales forecast MAE by a naive "
            "baseline's MAE; below 1 means the model beats the baseline, and it "
            "is scale-free so series of different magnitudes are comparable."
        ),
        contexts=(
            "MASE divides the forecast's mean absolute error by the MAE of a "
            "naive baseline; a value under 1 means the model beats the baseline.",
            "Because it is scaled, MASE is comparable across time series with "
            "different magnitudes.",
        ),
    ),
    EvalCase(
        question="How is data leakage avoided when building forecasting features?",
        reference=(
            "Leakage is avoided with a time-ordered train/test split and lag and "
            "rolling features computed per series without looking ahead, so no "
            "future information enters training."
        ),
        answer=(
            "A time-ordered split is used instead of random shuffling, and lag "
            "and rolling-mean features are computed per series using only past "
            "values, so future information never leaks into training."
        ),
        contexts=(
            "The split is time-ordered rather than random so the test period is "
            "strictly in the future.",
            "Lag and rolling features are computed per series from past values "
            "only, with no look-ahead, preventing leakage.",
        ),
    ),
    EvalCase(
        question="What does the agent's privacy guard do to tool inputs and outputs?",
        reference=(
            "The privacy guard validates every forecast row against a "
            "synthetic-only allow-list and redacts PII patterns from outputs, "
            "failing closed when an identifying field appears."
        ),
        answer=(
            "The guard checks each tool input row against a synthetic-only "
            "allow-list and rejects identifying fields (fail-closed), and it "
            "redacts PII patterns such as emails and card numbers from outputs."
        ),
        contexts=(
            "Every forecast row is checked against a synthetic-only allow-list; "
            "identifying keys are rejected, failing closed.",
            "Outputs are scanned and PII patterns (email, phone, card numbers) "
            "are redacted before reaching the user.",
        ),
    ),
    EvalCase(
        question="Why are document embeddings computed on-device?",
        reference=(
            "Embeddings are computed on-device with an ONNX MiniLM model so that "
            "document text never leaves the machine, keeping the corpus private."
        ),
        answer=(
            "The agent runs an on-device ONNX MiniLM embedding model, so document "
            "text is never sent to an external API and the corpus stays private."
        ),
        contexts=(
            "Embeddings use an on-device ONNX MiniLM model so no document text is "
            "sent to any external API.",
            "Keeping embedding local means the research corpus never leaves the "
            "machine.",
        ),
    ),
    EvalCase(
        question="What is rolling-origin cross-validation for time series?",
        reference=(
            "Rolling-origin cross-validation repeatedly trains on data up to a "
            "cutoff and tests on the next window, advancing the origin forward, "
            "so evaluation always respects temporal order."
        ),
        answer=(
            "Rolling-origin cross-validation moves the training cutoff forward in "
            "steps, each time training on the past and testing on the following "
            "window, preserving temporal order across folds."
        ),
        contexts=(
            "Rolling-origin cross-validation trains up to a cutoff and tests on "
            "the next window, then advances the origin.",
            "Each fold respects temporal order: training is always on the past, "
            "testing on the future.",
        ),
    ),
    EvalCase(
        question="Why can MAPE be misleading for sparse, low-count demand?",
        reference=(
            "MAPE is inflated when actual values are near zero because the "
            "percentage error divides by tiny denominators, so MAE or RMSE are "
            "preferred for sparse low-count demand."
        ),
        answer=(
            "With sparse low counts, MAPE divides by near-zero actuals and blows "
            "up, so it is misleading; scale-robust metrics like MAE and RMSE are "
            "preferred instead."
        ),
        contexts=(
            "When actual demand is near zero, MAPE's division by a tiny "
            "denominator inflates the percentage error.",
            "For sparse, low-count series, MAE and RMSE are more reliable "
            "headline metrics than MAPE.",
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
