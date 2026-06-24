"""Tests for configuration loading and secret handling."""

from __future__ import annotations

import pytest

from rag_agent.config import Config, api_key, load_config
from rag_agent.exceptions import ConfigError


def test_load_config_real_file(cfg: Config) -> None:
    assert cfg.ingest.collection_name == "research"
    assert cfg.agent.retrieval_k >= 1
    assert cfg.paths.documents_dir.name == "documents"
    assert cfg.forecasting.api_url.startswith("http")


def test_missing_config_raises() -> None:
    with pytest.raises(ConfigError):
        load_config("does/not/exist.yaml")


def test_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        api_key()


def test_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    assert api_key() == "sk-test-123"
