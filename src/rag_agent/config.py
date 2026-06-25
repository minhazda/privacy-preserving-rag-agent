"""Typed, YAML-driven configuration.

No values are hard-coded in business logic; everything flows from
``configs/config.yaml`` (overridable via ``RAG_CONFIG``). Filesystem paths are
resolved relative to the config file so the same file works on a laptop, in
Docker, and in CI. Secrets are read from the environment, never from YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError

DEFAULT_CONFIG_ENV = "RAG_CONFIG"
DEFAULT_CONFIG_PATH = "configs/config.yaml"
API_KEY_ENV = "ANTHROPIC_API_KEY"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations for the corpus, vector store, and audit log."""

    documents_dir: Path
    chroma_dir: Path
    audit_log: Path


@dataclass(frozen=True)
class IngestConfig:
    """Chunking and collection settings for ingestion."""

    collection_name: str = "research"
    chunk_size: int = 1000
    chunk_overlap: int = 150


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings. The API key is taken from the environment."""

    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class AgentConfig:
    """Agent loop and retrieval settings."""

    max_iterations: int = 6
    retrieval_k: int = 4
    system_prompt: str = ""
    # Minimum top-1 relevance (0..1, = 1 - cosine distance) required to trust a
    # retrieval; below this the agent is told no confident match was found so it
    # can answer "I don't know" instead of hallucinating.
    min_relevance: float = 0.0
    # Apply lightweight, deterministic query rewriting before retrieval.
    enable_query_rewrite: bool = True


@dataclass(frozen=True)
class ForecastingConfig:
    """Connection settings for the Project 1 forecasting API."""

    api_url: str = "http://localhost:8000"
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy-guard behaviour."""

    fail_closed: bool = True
    max_output_chars: int = 8000
    # Differential-privacy (output perturbation) for forecast values. When
    # enabled, each prediction is clipped to [0, dp_clip_max] (bounding
    # sensitivity) and Laplace(sensitivity / epsilon) noise is added. Disabled
    # by default; the underlying data is already synthetic, so this is an
    # opt-in, demonstrable privacy primitive (defence in depth).
    dp_enabled: bool = False
    dp_epsilon: float = 1.0
    dp_clip_max: float = 1000.0


@dataclass(frozen=True)
class ApiConfig:
    """HTTP server bind settings."""

    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(frozen=True)
class Config:
    """Top-level application configuration."""

    paths: PathsConfig
    ingest: IngestConfig
    llm: LLMConfig
    agent: AgentConfig
    forecasting: ForecastingConfig
    privacy: PrivacyConfig
    api: ApiConfig
    log_level: str = "INFO"


def _resolve(base: Path, value: str) -> Path:
    """Resolve ``value`` against ``base`` unless it is already absolute."""
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def api_key() -> str:
    """Return the Anthropic API key from the environment.

    Raises:
        ConfigError: If ``ANTHROPIC_API_KEY`` is not set.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise ConfigError(
            f"{API_KEY_ENV} is not set. Export it before running the agent; "
            "it is never read from config files."
        )
    return key


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration from YAML.

    Args:
        path: Explicit config path; falls back to ``$RAG_CONFIG`` then
            ``configs/config.yaml`` relative to the working directory.

    Returns:
        A fully populated, immutable :class:`Config`.

    Raises:
        ConfigError: If the file is missing or a section is malformed.
    """
    cfg_path = Path(path or os.environ.get(DEFAULT_CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")

    try:
        raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc

    base = cfg_path.parent
    try:
        paths_raw = raw["paths"]
        paths = PathsConfig(
            documents_dir=_resolve(base, paths_raw["documents_dir"]),
            chroma_dir=_resolve(base, paths_raw["chroma_dir"]),
            audit_log=_resolve(base, paths_raw.get("audit_log", "../data/audit/audit.jsonl")),
        )
        ingest = IngestConfig(**raw.get("ingest", {}))
        llm = LLMConfig(**raw.get("llm", {}))
        agent = AgentConfig(**raw.get("agent", {}))
        forecasting = ForecastingConfig(**raw.get("forecasting", {}))
        privacy = PrivacyConfig(**raw.get("privacy", {}))
        api = ApiConfig(**raw.get("api", {}))
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Malformed config section: {exc}") from exc

    return Config(
        paths=paths,
        ingest=ingest,
        llm=llm,
        agent=agent,
        forecasting=forecasting,
        privacy=privacy,
        api=api,
        log_level=raw.get("log_level", "INFO"),
    )
