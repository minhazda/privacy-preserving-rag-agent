"""Agent tool implementations (framework-agnostic, hence directly testable).

Two capabilities are exposed to the agent:

* :func:`retrieve_research` — semantic search over the indexed corpus, with all
  returned text passed through the privacy guard.
* :func:`forecast_demand` — calls the Project 1 forecasting API, but only after
  every input row is validated as synthetic-only by the privacy guard.

``agent.py`` adapts these into LangChain tools; keeping the logic here means the
privacy-critical paths are unit-tested without spinning up an LLM.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

import httpx

from .config import Config
from .exceptions import ForecastToolError
from .logging_config import get_logger
from .privacy import PrivacyGuard
from .vectorstore import RetrievedChunk

log = get_logger(__name__)


# --- Query rewriting --------------------------------------------------------
_ACRONYMS: dict[str, str] = {
    "mae": "mean absolute error",
    "rmse": "root mean squared error",
    "mape": "mean absolute percentage error",
    "dp": "differential privacy",
    "gan": "generative adversarial network",
    "ctgan": "conditional tabular generative adversarial network",
    "tvae": "tabular variational autoencoder",
    "kpi": "key performance indicator",
    "eda": "exploratory data analysis",
}


def rewrite_query(question: str) -> str:
    """Deterministically normalise and expand a query for better retrieval.

    Collapses whitespace and appends spelled-out forms of any known acronyms so
    dense retrieval matches both the abbreviation and its expansion. Pure and
    deterministic — no LLM call — so it is fast and unit-testable.

    Args:
        question: The raw user question.

    Returns:
        The rewritten query (unchanged if empty or no acronyms present).
    """
    cleaned = " ".join(question.split()).strip()
    if not cleaned:
        return cleaned
    expansions: list[str] = []
    for token in re.findall(r"[A-Za-z]+", cleaned):
        expanded = _ACRONYMS.get(token.lower())
        if expanded:
            expansions.append(expanded)
    if expansions:
        unique = list(dict.fromkeys(expansions))
        cleaned = f"{cleaned} ({'; '.join(unique)})"
    return cleaned


# --- Method explanations ----------------------------------------------------
_METHOD_LIBRARY: dict[str, str] = {
    "synthetic_data": (
        "Synthetic data is artificially generated data that preserves the "
        "statistical structure of a real dataset without containing any real "
        "records. It enables model development and sharing while removing direct "
        "and indirect identifiers — the foundation of this project's privacy model."
    ),
    "differential_privacy": (
        "Differential privacy (DP) gives a mathematical guarantee that the "
        "presence or absence of any single record has a bounded effect on an "
        "output, controlled by a privacy budget epsilon (smaller epsilon = more "
        "privacy). It is typically enforced by adding calibrated noise (e.g. the "
        "Laplace or Gaussian mechanism) to query results or gradients."
    ),
    "laplace_mechanism": (
        "The Laplace mechanism achieves epsilon-DP for numeric outputs by adding "
        "noise drawn from a Laplace distribution with scale = sensitivity / "
        "epsilon, where sensitivity bounds how much one record can change the "
        "output. This project applies it as opt-in output perturbation on forecasts."
    ),
    "ctgan": (
        "CTGAN (Conditional Tabular GAN) is a generative adversarial network "
        "specialised for tabular data: mode-specific normalisation handles "
        "multi-modal continuous columns and a conditional generator with "
        "training-by-sampling addresses imbalanced categorical columns."
    ),
    "tvae": (
        "TVAE (Tabular Variational Autoencoder) learns a latent representation of "
        "tabular data and samples from it to synthesise new rows. It often trains "
        "more stably than a GAN but can blur sharp distributional modes."
    ),
    "smote": (
        "SMOTE (Synthetic Minority Over-sampling Technique) creates new minority-"
        "class examples by interpolating between a sample and its nearest "
        "neighbours. It rebalances classes but is an augmentation method, not a "
        "privacy mechanism."
    ),
    "k_anonymity": (
        "k-anonymity generalises or suppresses quasi-identifiers so each record is "
        "indistinguishable from at least k-1 others. It mitigates re-identification "
        "but, unlike differential privacy, gives no guarantee against attribute "
        "inference from background knowledge."
    ),
    "gradient_boosting": (
        "Gradient-boosted decision trees (e.g. LightGBM, XGBoost) fit an additive "
        "ensemble where each tree corrects the residual errors of the previous "
        "ones. They are strong, efficient baselines for tabular demand forecasting."
    ),
}

_METHOD_ALIASES: dict[str, str] = {
    "dp": "differential_privacy",
    "differential privacy": "differential_privacy",
    "conditional tabular gan": "ctgan",
    "tabular gan": "ctgan",
    "variational autoencoder": "tvae",
    "laplace": "laplace_mechanism",
    "k anonymity": "k_anonymity",
    "k-anonymity": "k_anonymity",
    "lightgbm": "gradient_boosting",
    "xgboost": "gradient_boosting",
    "gbdt": "gradient_boosting",
    "boosting": "gradient_boosting",
}


def explain_method(name: str, guard: PrivacyGuard) -> str:
    """Explain a synthetic-data, privacy, or forecasting method by name.

    Resolves aliases and tolerant matching (normalised key, then fuzzy match),
    returning a concise, privacy-filtered explanation. Unknown methods return a
    helpful list of supported names rather than a hard error, so the agent can
    recover or fall back to :func:`retrieve_research`.

    Args:
        name: The method name or alias (case/spacing insensitive).
        guard: Privacy guard applied to the returned text.

    Returns:
        A short explanation, or a "not found" message listing known methods.
    """
    raw = name.strip().lower()
    key = _METHOD_ALIASES.get(raw, raw.replace(" ", "_").replace("-", "_"))
    if key not in _METHOD_LIBRARY:
        close = difflib.get_close_matches(key, _METHOD_LIBRARY, n=1, cutoff=0.6)
        if close:
            key = close[0]
    explanation = _METHOD_LIBRARY.get(key)
    if explanation is None:
        known = ", ".join(sorted(_METHOD_LIBRARY))
        log.info("explain_method_miss", requested=raw)
        return (
            f"'{name}' is not in the curated method library. Known methods: "
            f"{known}. For detail specific to the research, use retrieve_research."
        )
    log.info("explain_method", method=key)
    return guard.filter_output(explanation)


def format_context(chunks: list[RetrievedChunk], guard: PrivacyGuard) -> str:
    """Render retrieved chunks into a privacy-filtered, cited context block."""
    if not chunks:
        return "No relevant passages were found in the indexed research corpus."
    parts = []
    for i, c in enumerate(chunks, start=1):
        safe = guard.filter_output(c.text)
        parts.append(f"[{i}] (source: {c.source})\n{safe}")
    return "\n\n".join(parts)


def retrieve_research(
    question: str,
    cfg: Config,
    collection: Any,
    guard: PrivacyGuard,
) -> str:
    """Retrieve and privacy-filter the most relevant research passages.

    Args:
        question: The user's natural-language query.
        cfg: Active configuration (uses ``agent.retrieval_k``).
        collection: A Chroma collection (or compatible object with ``query``).
        guard: Privacy guard applied to every returned passage.

    Returns:
        A cited, redacted context string suitable for the LLM.
    """
    from .vectorstore import query  # local import keeps chroma optional

    effective = rewrite_query(question) if cfg.agent.enable_query_rewrite else question
    chunks = query(collection, effective, cfg.agent.retrieval_k)
    log.info("retrieved", n=len(chunks), rewritten=effective != question)
    if not chunks:
        return "No relevant passages were found in the indexed research corpus."
    # Chroma returns cosine distance; relevance = 1 - distance, clamped to [0, 1].
    top_relevance = max(0.0, 1.0 - chunks[0].distance)
    if top_relevance < cfg.agent.min_relevance:
        log.info("low_confidence", top_relevance=round(top_relevance, 3))
        return (
            "No sufficiently relevant passage was found in the indexed research "
            f"corpus (top relevance {top_relevance:.2f} is below the "
            f"{cfg.agent.min_relevance:.2f} threshold). Tell the user you do not "
            "have this information rather than guessing."
        )
    return format_context(chunks, guard)


def forecast_demand(
    rows: list[dict[str, float]],
    cfg: Config,
    guard: PrivacyGuard,
    client: httpx.Client | None = None,
) -> list[float]:
    """Run live demand forecasts via the Project 1 API (synthetic rows only).

    Each row is validated against the synthetic-feature allow-list *before*
    leaving the process, so the tool cannot be used to send or surface real
    customer data.

    Args:
        rows: Pre-engineered synthetic feature rows.
        cfg: Active configuration (forecasting API url + timeout).
        guard: Privacy guard used to validate each row.
        client: Optional injected httpx client (used by tests).

    Returns:
        The list of predicted demand values.

    Raises:
        ForecastToolError: On empty input, API failure, or malformed response.
        PrivacyViolationError: If any row contains non-synthetic fields.
    """
    if not rows:
        raise ForecastToolError("No feature rows supplied to forecast_demand.")
    for row in rows:
        guard.assert_synthetic_record(row)

    owns_client = client is None
    client = client or httpx.Client(timeout=cfg.forecasting.timeout_seconds)
    try:
        resp = client.post(f"{cfg.forecasting.api_url}/predict", json={"rows": rows})
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError as exc:
        raise ForecastToolError(f"Forecasting API call failed: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    preds = payload.get("predictions")
    if not isinstance(preds, list):
        raise ForecastToolError("Forecasting API returned no 'predictions' list.")
    values = guard.privatize_forecast([float(p) for p in preds])
    log.info("forecast_done", n=len(values), dp=guard.dp_enabled)
    return values
