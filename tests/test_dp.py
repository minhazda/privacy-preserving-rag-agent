"""Tests for the differential-privacy (Laplace) primitives."""

from __future__ import annotations

import random
import statistics

import pytest

from rag_agent.privacy import PrivacyGuard, laplace_noise, privatize_values


def test_laplace_noise_is_seed_reproducible() -> None:
    assert laplace_noise(1.0, random.Random(0)) == laplace_noise(1.0, random.Random(0))


def test_laplace_noise_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError):
        laplace_noise(0.0)


def test_privatize_values_is_nonnegative_and_same_length() -> None:
    out = privatize_values([10.0, -5.0, 999.0], epsilon=10.0, clip_max=100.0, rng=random.Random(1))
    assert len(out) == 3
    assert all(v >= 0.0 for v in out)


def test_privatize_values_rejects_bad_epsilon() -> None:
    with pytest.raises(ValueError):
        privatize_values([1.0], epsilon=0.0, clip_max=10.0)


def test_privatize_values_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        privatize_values([1.0], epsilon=1.0, clip_max=0.0, clip_min=5.0)


def test_privatize_mean_is_approximately_unbiased() -> None:
    # Low-noise regime (large epsilon): the zero-mean Laplace noise should leave
    # the sample mean close to the true value.
    samples = privatize_values([50.0] * 4000, epsilon=25.0, clip_max=100.0, rng=random.Random(42))
    assert abs(statistics.mean(samples) - 50.0) < 2.0


def test_guard_privatize_forecast_disabled_is_identity() -> None:
    assert PrivacyGuard(dp_enabled=False).privatize_forecast([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_guard_privatize_forecast_enabled_perturbs() -> None:
    guard = PrivacyGuard(dp_enabled=True, dp_epsilon=0.5, dp_clip_max=100.0)
    out = guard.privatize_forecast([50.0, 50.0, 50.0], rng=random.Random(7))
    assert len(out) == 3
    assert out != [50.0, 50.0, 50.0]
