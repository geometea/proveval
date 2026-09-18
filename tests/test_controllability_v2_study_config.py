"""Tests for the v2 production study config: DeepSeek Flash as primary
evaluator, 10 replicates everywhere, and the resulting planned production
observation counts.
"""

import model_providers
from controllability_v2_study_config import build_study_config


class TestDeepSeekFlashIsPrimary:
    def test_default_deepseek_model_is_deepseek_flash(self):
        assert model_providers.DEFAULT_MODELS["deepseek"] == "deepseek-flash"

    def test_primary_evaluator_is_deepseek_flash(self):
        config = build_study_config()
        primary = config["primary_evaluator"]
        assert primary["provider"] == "deepseek"
        assert primary["requested_model"] == "deepseek-flash"
        assert primary["reasoning_profile"] == "low"
        assert primary["evaluator_id"] == "deepseek__deepseek-flash__low"
        assert primary["role"] == "primary"

    def test_claude_is_a_replication_evaluator_not_primary(self):
        config = build_study_config()
        assert config["primary_evaluator"]["provider"] != "anthropic"
        replication_providers = [e["provider"] for e in config["replication_evaluators"]]
        assert "anthropic" in replication_providers
        for evaluator in config["replication_evaluators"]:
            assert evaluator["role"] == "replication"

    def test_exactly_one_primary_evaluator(self):
        config = build_study_config()
        assert len([e for e in [config["primary_evaluator"], *config["replication_evaluators"]] if e["role"] == "primary"]) == 1


class TestTenReplicates:
    def test_treatment_replicate_count_is_10(self):
        assert build_study_config()["treatment_replicate_count"] == 10

    def test_baseline_replicate_count_is_10(self):
        assert build_study_config()["baseline_replicate_count"] == 10

    def test_planned_production_observation_counts(self):
        """2,640 unique treatment prompts x 10 + 132 unique baseline prompts
        x 10 = 27,720 total planned observations -- manifest sizes
        themselves are unchanged; replication happens at execution time."""
        import controllability_v2_trials as t
        from context_trials import load_items, story_pairs

        items = load_items()
        contrasts = t.load_contrasts()
        treatment_trials = t.build_all_treatment_trials(contrasts, items, t.PRIMARY_INSTRUCTION_CONDITIONS)
        baseline_trials = t.build_baseline_trials(items)

        assert len(treatment_trials) == 2640
        assert len(baseline_trials) == 132

        config = build_study_config()
        planned_treatment = len(treatment_trials) * config["treatment_replicate_count"]
        planned_baseline = len(baseline_trials) * config["baseline_replicate_count"]
        assert planned_treatment == 26400
        assert planned_baseline == 1320
        assert planned_treatment + planned_baseline == 27720

    def test_max_output_tokens_is_recorded_in_the_config(self):
        config = build_study_config()
        assert config["max_output_tokens"] == model_providers.default_max_output_tokens("deepseek")
