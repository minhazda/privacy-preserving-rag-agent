"""Custom exception hierarchy for the RAG agent.

A single base (:class:`RagAgentError`) lets callers catch every domain error
while still allowing precise handling of individual failure modes.
"""

from __future__ import annotations


class RagAgentError(Exception):
    """Base class for all RAG-agent errors."""


class ConfigError(RagAgentError):
    """Raised when configuration is missing or malformed."""


class IngestionError(RagAgentError):
    """Raised when documents cannot be loaded or indexed."""


class VectorStoreError(RagAgentError):
    """Raised when the vector store cannot be opened or queried."""


class ForecastToolError(RagAgentError):
    """Raised when the forecasting API call fails or returns bad data."""


class PrivacyViolationError(RagAgentError):
    """Raised when content would expose non-synthetic / sensitive data."""
