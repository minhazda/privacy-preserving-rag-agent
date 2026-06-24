"""Privacy guard: ensures only synthetic, non-identifying data is ever exposed.

This module is deliberately dependency-free and pure-Python so it is fast,
auditable, and exhaustively unit-testable. It provides two layers:

1. **PII redaction** — emails, phone numbers, SSNs, IPv4 addresses, and
   Luhn-valid card numbers are detected and masked in any outbound text.
2. **Synthetic-only enforcement** — feature rows returned by tools are checked
   to contain only the allow-listed synthetic schema; anything resembling a
   real identifier is rejected (fail-closed by default).

The forecasting pipeline (Project 1) is synthetic by construction, so in normal
operation nothing is ever redacted — this is defence in depth, not a crutch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .exceptions import PrivacyViolationError

# --- PII patterns ----------------------------------------------------------
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

# Allow-listed synthetic feature columns the agent may expose verbatim.
SYNTHETIC_FEATURE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "product_id",
        "category",
        "sales_volume",
        "stock_level",
        "promo_flag",
        "holiday_flag",
        "foot_traffic",
        "weather",
        "hour",
        "day_of_week",
        "timestamp",
    }
)

# Keys that must never appear in a "synthetic" record.
_FORBIDDEN_KEY_HINTS: tuple[str, ...] = (
    "name",
    "email",
    "phone",
    "address",
    "ssn",
    "dob",
    "customer",
    "card",
    "account",
)


@dataclass(frozen=True)
class Finding:
    """A single detected piece of sensitive data."""

    kind: str
    value: str


def _luhn_valid(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn checksum (card sanity check)."""
    nums = [int(d) for d in digits if d.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def scan_text(text: str) -> list[Finding]:
    """Return all PII findings in ``text`` (emails, phones, SSNs, IPs, cards)."""
    findings: list[Finding] = []
    findings += [Finding("email", m.group()) for m in _EMAIL.finditer(text)]
    findings += [Finding("ssn", m.group()) for m in _SSN.finditer(text)]
    findings += [Finding("phone", m.group()) for m in _PHONE.finditer(text)]
    findings += [Finding("ipv4", m.group()) for m in _IPV4.finditer(text)]
    for m in _CARD_CANDIDATE.finditer(text):
        if _luhn_valid(m.group()):
            findings.append(Finding("card", m.group()))
    return findings


def redact_text(text: str) -> tuple[str, list[Finding]]:
    """Mask any PII in ``text``.

    Returns:
        A tuple of (redacted_text, findings). Each match becomes
        ``[REDACTED:<kind>]``.
    """
    findings = scan_text(text)
    redacted = text
    # Replace longest values first to avoid partial-overlap corruption.
    for f in sorted(findings, key=lambda x: len(x.value), reverse=True):
        redacted = redacted.replace(f.value, f"[REDACTED:{f.kind}]")
    return redacted, findings


class PrivacyGuard:
    """Stateful guard applying PII redaction and synthetic-only checks.

    Args:
        fail_closed: If True, ambiguous/forbidden records raise rather than
            being silently dropped.
        max_output_chars: Hard cap on returned text length.
    """

    def __init__(self, fail_closed: bool = True, max_output_chars: int = 8000) -> None:
        self.fail_closed = fail_closed
        self.max_output_chars = max_output_chars

    def filter_output(self, text: str) -> str:
        """Redact PII and truncate ``text`` for safe outbound display."""
        redacted, _ = redact_text(text)
        if len(redacted) > self.max_output_chars:
            redacted = redacted[: self.max_output_chars] + "…[truncated]"
        return redacted

    def assert_synthetic_record(self, record: Mapping[str, object]) -> Mapping[str, object]:
        """Validate that ``record`` contains only allow-listed synthetic fields.

        Args:
            record: A single feature row destined for the user.

        Returns:
            The record, unchanged, if it is safe.

        Raises:
            PrivacyViolationError: If a forbidden/identifying key is present, or
                (when ``fail_closed``) an unknown key appears.
        """
        for key in record:
            lowered = key.lower()
            if any(hint in lowered for hint in _FORBIDDEN_KEY_HINTS):
                raise PrivacyViolationError(f"Refusing to expose record: forbidden field '{key}'.")
            if self.fail_closed and lowered not in SYNTHETIC_FEATURE_ALLOWLIST:
                raise PrivacyViolationError(
                    f"Refusing to expose record: '{key}' is not in the synthetic "
                    "feature allow-list (fail-closed)."
                )
        return record
