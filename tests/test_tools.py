"""Tests for tool logic: forecasting (mocked HTTP) and guarded retrieval."""

from __future__ import annotations

from typing import Any

import pytest

from rag_agent.config import Config
from rag_agent.exceptions import ForecastToolError, PrivacyViolationError
from rag_agent.privacy import PrivacyGuard
from rag_agent.tools import forecast_demand, retrieve_research


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - mimic httpx
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Records the last request and returns a canned forecast response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.last_json: dict[str, Any] | None = None

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        self.last_json = json
        return _FakeResponse(self._payload)

    def close(self) -> None:
        return None


def test_forecast_demand_returns_predictions(cfg: Config) -> None:
    client = _FakeClient({"predictions": [1.5, 2.0]})
    rows = [{"product_id": "P001", "sales_volume": 1, "promo_flag": 0}]
    preds = forecast_demand(rows, cfg, PrivacyGuard(fail_closed=False), client=client)
    assert preds == [1.5, 2.0]
    assert client.last_json == {"rows": rows}


def test_forecast_demand_blocks_non_synthetic_row(cfg: Config) -> None:
    client = _FakeClient({"predictions": [1.0]})
    with pytest.raises(PrivacyViolationError):
        forecast_demand(
            [{"customer_email": "a@b.com"}], cfg, PrivacyGuard(fail_closed=True), client=client
        )
    assert client.last_json is None  # never left the process


def test_forecast_demand_empty_rows_raises(cfg: Config) -> None:
    with pytest.raises(ForecastToolError):
        forecast_demand([], cfg, PrivacyGuard())


def test_forecast_demand_bad_payload_raises(cfg: Config) -> None:
    client = _FakeClient({"oops": True})
    with pytest.raises(ForecastToolError):
        forecast_demand(
            [{"product_id": "P001"}], cfg, PrivacyGuard(fail_closed=False), client=client
        )


def test_retrieve_research_formats_with_sources(cfg: Config, fake_collection: Any) -> None:
    out = retrieve_research("privacy?", cfg, fake_collection, PrivacyGuard())
    assert "dissertation.pdf" in out
    assert "[1]" in out
