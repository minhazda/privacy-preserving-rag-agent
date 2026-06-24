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

from typing import Any

import httpx

from .config import Config
from .exceptions import ForecastToolError
from .logging_config import get_logger
from .privacy import PrivacyGuard
from .vectorstore import RetrievedChunk

log = get_logger(__name__)


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

    chunks = query(collection, question, cfg.agent.retrieval_k)
    log.info("retrieved", n=len(chunks))
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
    log.info("forecast_done", n=len(preds))
    return [float(p) for p in preds]
