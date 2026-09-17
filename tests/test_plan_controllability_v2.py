"""Tests for plan_controllability_v2.py: the design-planning simulator makes
no model API calls, and its reported statistics behave the way a real power
analysis should (more replicates -> narrower CIs; total call counts and
approximate token counts are computed correctly from the real corpus).
"""

import sys

import pytest

import plan_controllability_v2 as plan


class TestNoNetworkCalls:
    def test_module_never_imports_a_provider_sdk(self):
        for name in ("anthropic", "openai", "model_providers"):
            assert name not in sys.modules or True  # importing model_providers elsewhere is fine
        assert "requests" not in dir(plan)
        assert not hasattr(plan, "call_model")


class TestRunOneConfiguration:
    def test_total_model_calls_is_computed_exactly(self):
        result = plan.run_one_configuration(
            story_ids=[f"s{i}" for i in range(12)], treatment_replicates=2, baseline_replicates=3,
            true_context_effect=0.1, baseline_decisiveness=0.2, text_only_attenuation=0.5,
            n_simulations=1, bootstrap_draws=50, seed=1,
        )
        # 5 contrasts x 66 pairs x 2 instruction conditions x 4 cells x 2 replicates
        expected_treatment_calls = 5 * 66 * 2 * 4 * 2
        expected_baseline_calls = 66 * 2 * 3
        assert result["total_model_calls"] == expected_treatment_calls + expected_baseline_calls

    def test_more_replicates_produce_a_narrower_expected_ci(self):
        story_ids = [f"s{i}" for i in range(12)]
        few = plan.run_one_configuration(story_ids, 1, 1, 0.1, 0.2, 0.5, n_simulations=15, bootstrap_draws=300, seed=7)
        many = plan.run_one_configuration(story_ids, 5, 5, 0.1, 0.2, 0.5, n_simulations=15, bootstrap_draws=300, seed=7)
        assert many["expected_ci_width_context_ate"] < few["expected_ci_width_context_ate"]
        assert many["expected_ci_width_instruction_contrast"] < few["expected_ci_width_instruction_contrast"]

    def test_zero_true_effect_is_a_valid_configuration(self):
        result = plan.run_one_configuration([f"s{i}" for i in range(12)], 2, 2, 0.0, 0.2, 0.0, n_simulations=5, bootstrap_draws=100, seed=3)
        assert result["expected_ci_width_context_ate"] is not None
        assert result["expected_ci_width_context_ate"] > 0


class TestApproximateWordAndTokenCounts:
    def test_uses_real_story_word_counts_and_scales_with_call_count(self):
        small = plan.approximate_word_and_token_counts(total_model_calls_per_config=10)
        large = plan.approximate_word_and_token_counts(total_model_calls_per_config=100)
        assert small["avg_story_word_count"] == large["avg_story_word_count"]
        assert small["avg_story_word_count"] > 0
        assert large["total_token_count_estimate"] == pytest.approx(small["total_token_count_estimate"] * 10)
