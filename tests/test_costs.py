# tests/test_costs.py

from guardrail_eval.costs import estimate_cost_usd


def test_estimate_cost_usd_zero_tokens():
    assert estimate_cost_usd(0, 0) == 0.0


def test_estimate_cost_usd_prompt_only():
    assert estimate_cost_usd(1_000_000, 0) == 0.15


def test_estimate_cost_usd_completion_only():
    assert estimate_cost_usd(0, 1_000_000) == 0.6


def test_estimate_cost_usd_mixed_tokens():
    assert estimate_cost_usd(1000, 500) == 0.00045


def test_estimate_cost_usd_rounding_matches_original_runner():
    assert estimate_cost_usd(123, 456) == round(
        123 * 0.00000015 + 456 * 0.00000060,
        6,
    )