"""Tests for the privacy guard — PII redaction and synthetic-only enforcement."""

from __future__ import annotations

import pytest

from rag_agent.exceptions import PrivacyViolationError
from rag_agent.privacy import (
    PrivacyGuard,
    redact_text,
    scan_text,
)


def test_scan_detects_common_pii() -> None:
    kinds = {f.kind for f in scan_text("mail a@b.com call 415-555-1234 ssn 123-45-6789")}
    assert {"email", "phone", "ssn"} <= kinds


def test_redact_masks_email_and_keeps_other_text() -> None:
    out, findings = redact_text("Contact alice@example.com for details.")
    assert "alice@example.com" not in out
    assert "[REDACTED:email]" in out
    assert any(f.kind == "email" for f in findings)


def test_card_requires_luhn_validity() -> None:
    # Valid Visa test number (Luhn-valid) is flagged; random 16 digits are not.
    assert any(f.kind == "card" for f in scan_text("4111 1111 1111 1111"))
    assert not any(f.kind == "card" for f in scan_text("1234 5678 9012 3456"))


def test_filter_output_truncates() -> None:
    guard = PrivacyGuard(max_output_chars=10)
    assert guard.filter_output("x" * 50).startswith("x" * 10)
    assert "truncated" in guard.filter_output("x" * 50)


def test_synthetic_record_allows_allowlisted_fields() -> None:
    guard = PrivacyGuard(fail_closed=True)
    row = {"product_id": "P001", "sales_volume": 3, "promo_flag": 0}
    assert guard.assert_synthetic_record(row) == row


def test_synthetic_record_rejects_forbidden_field() -> None:
    guard = PrivacyGuard(fail_closed=True)
    with pytest.raises(PrivacyViolationError):
        guard.assert_synthetic_record({"customer_name": "Jane"})


def test_fail_closed_rejects_unknown_field() -> None:
    assert PrivacyGuard(fail_closed=True)  # sanity
    with pytest.raises(PrivacyViolationError):
        PrivacyGuard(fail_closed=True).assert_synthetic_record({"weird_col": 1})


def test_fail_open_allows_unknown_but_safe_field() -> None:
    guard = PrivacyGuard(fail_closed=False)
    assert guard.assert_synthetic_record({"weird_col": 1}) == {"weird_col": 1}
