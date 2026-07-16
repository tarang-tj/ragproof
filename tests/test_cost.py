"""Unit tests for ragproof.cost -- hand-computed dollar amounts."""

from __future__ import annotations

import pytest

from ragproof.cost import DEFAULT_RATES, estimate_cost


def test_estimate_cost_matches_hand_computation():
    rates = {"input_usd_per_million": 3.0, "output_usd_per_million": 15.0, "embed_usd_per_million": 0.1}
    result = estimate_cost(input_tokens=1000, output_tokens=200, embed_tokens=100, rates=rates)

    assert result["input_usd"] == pytest.approx(1000 * 3.0 / 1_000_000)
    assert result["output_usd"] == pytest.approx(200 * 15.0 / 1_000_000)
    assert result["embed_usd"] == pytest.approx(100 * 0.1 / 1_000_000)
    assert result["total_usd"] == pytest.approx(result["input_usd"] + result["output_usd"] + result["embed_usd"])
    assert result["total_tokens"] == 1300


def test_estimate_cost_uses_default_rates_when_none_given():
    result = estimate_cost(1000, 1000, 1000, rates=None)
    expected_total = (
        1000 * DEFAULT_RATES["input_usd_per_million"] / 1_000_000
        + 1000 * DEFAULT_RATES["output_usd_per_million"] / 1_000_000
        + 1000 * DEFAULT_RATES["embed_usd_per_million"] / 1_000_000
    )
    assert result["total_usd"] == pytest.approx(expected_total)


def test_estimate_cost_zero_tokens_is_zero_cost():
    result = estimate_cost(0, 0, 0)
    assert result["total_usd"] == 0.0
    assert result["total_tokens"] == 0


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost(-1, 0, 0)


def test_estimate_cost_partial_rate_override_falls_back_to_default():
    result = estimate_cost(1000, 0, 0, rates={"input_usd_per_million": 1.0})
    assert result["input_usd"] == pytest.approx(1000 * 1.0 / 1_000_000)
