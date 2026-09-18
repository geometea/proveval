"""Tests for controllability_v2_cost.py: pricing lives in one explicit
structure, costs are always marked estimated, and cache-hit/cache-miss
tokens are billed at their own configured rates when available.
"""

import controllability_v2_cost as cost


def make_row(parsing_status, attempts):
    return {"parsing_status": parsing_status, "attempts": attempts}


class TestComputeCostSummary:
    def test_counts_calls_attempted_and_successful_observations(self):
        rows = [
            make_row("resolved", [{"input_tokens": 10, "output_tokens": 1}]),
            make_row("unresolved", [{"input_tokens": 10, "output_tokens": 1}, {"input_tokens": 10, "output_tokens": 1}]),
        ]
        summary = cost.compute_cost_summary(rows, "deepseek", "deepseek-flash")
        assert summary["total_calls_attempted"] == 3  # 1 + 2 attempts
        assert summary["successful_planned_observations"] == 1

    def test_sums_tokens_across_every_attempt_not_just_the_last(self):
        rows = [make_row("unresolved", [{"input_tokens": 50, "output_tokens": 2, "reasoning_tokens": 1},
                                          {"input_tokens": 60, "output_tokens": 3, "reasoning_tokens": 0}])]
        summary = cost.compute_cost_summary(rows, "deepseek", "deepseek-flash")
        assert summary["prompt_tokens"] == 110
        assert summary["completion_tokens"] == 5
        assert summary["reasoning_tokens"] == 1

    def test_cache_hit_and_miss_tokens_are_billed_at_separate_configured_rates(self):
        pricing = {"deepseek": {"deepseek-flash": {"cache_miss_input": 1.0, "cache_hit_input": 0.1, "output": 2.0}}}
        rows = [make_row("resolved", [{"input_tokens": 100, "output_tokens": 10,
                                         "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20}])]
        summary = cost.compute_cost_summary(rows, "deepseek", "deepseek-flash", pricing=pricing)
        expected_input_cost = (80 * 0.1 + 20 * 1.0) / 1_000_000
        assert summary["estimated_input_cost_usd"] == expected_input_cost
        assert summary["cache_hit_prompt_tokens"] == 80
        assert summary["cache_miss_prompt_tokens"] == 20

    def test_missing_cache_breakdown_falls_back_to_the_conservative_cache_miss_rate(self):
        pricing = {"deepseek": {"deepseek-flash": {"cache_miss_input": 1.0, "cache_hit_input": 0.1, "output": 2.0}}}
        rows = [make_row("resolved", [{"input_tokens": 100, "output_tokens": 10}])]  # no cache fields at all
        summary = cost.compute_cost_summary(rows, "deepseek", "deepseek-flash", pricing=pricing)
        assert summary["cache_hit_prompt_tokens"] is None
        assert summary["cache_miss_prompt_tokens"] is None
        assert summary["estimated_input_cost_usd"] == (100 * 1.0) / 1_000_000  # priced at the higher rate, never assumed free

    def test_unpriced_provider_model_reports_none_costs_not_zero(self):
        rows = [make_row("resolved", [{"input_tokens": 100, "output_tokens": 10}])]
        summary = cost.compute_cost_summary(rows, "anthropic", "claude-sonnet-5")
        assert summary["estimated_input_cost_usd"] is None
        assert summary["estimated_output_cost_usd"] is None
        assert summary["estimated_total_cost_usd"] is None

    def test_cost_is_always_marked_estimated(self):
        rows = [make_row("resolved", [{"input_tokens": 1, "output_tokens": 1}])]
        assert cost.compute_cost_summary(rows, "deepseek", "deepseek-flash")["cost_is_estimated"] is True
        assert cost.compute_cost_summary(rows, "unknown", "unknown-model")["cost_is_estimated"] is True

    def test_pricing_config_is_a_single_explicit_structure(self):
        """Sanity: the pricing table itself is a plain nested dict the
        caller can override wholesale (see the `pricing` parameter) --
        never hardcoded rate literals scattered through the function."""
        assert isinstance(cost.PRICING_USD_PER_MILLION_TOKENS, dict)
        assert "deepseek" in cost.PRICING_USD_PER_MILLION_TOKENS
        rates = cost.PRICING_USD_PER_MILLION_TOKENS["deepseek"]["deepseek-flash"]
        assert set(rates) == {"cache_miss_input", "cache_hit_input", "output"}

    def test_empty_rows_produce_zeroed_not_crashing_summary(self):
        summary = cost.compute_cost_summary([], "deepseek", "deepseek-flash")
        assert summary["total_calls_attempted"] == 0
        assert summary["successful_planned_observations"] == 0
        assert summary["estimated_total_cost_usd"] == 0.0
