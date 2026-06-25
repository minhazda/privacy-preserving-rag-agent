"""Tests for query rewriting and the method-explanation tool."""

from __future__ import annotations

from rag_agent.privacy import PrivacyGuard
from rag_agent.tools import explain_method, rewrite_query


def test_rewrite_expands_known_acronyms() -> None:
    out = rewrite_query("What MAE did the model achieve?")
    assert "mean absolute error" in out
    assert "MAE" in out  # original text preserved


def test_rewrite_collapses_whitespace() -> None:
    assert rewrite_query("  explain   synthetic   data ") == "explain synthetic data"


def test_rewrite_empty_string_is_empty() -> None:
    assert rewrite_query("   ") == ""


def test_explain_known_method() -> None:
    out = explain_method("differential privacy", PrivacyGuard())
    assert "epsilon" in out.lower()


def test_explain_alias_resolution() -> None:
    out = explain_method("LightGBM", PrivacyGuard())
    assert "boost" in out.lower()


def test_explain_fuzzy_match_tolerates_typo() -> None:
    out = explain_method("ctga", PrivacyGuard())
    assert "gan" in out.lower()


def test_explain_unknown_lists_known_methods() -> None:
    out = explain_method("quantum teleportation", PrivacyGuard())
    assert "not in the curated method library" in out
    assert "synthetic_data" in out
