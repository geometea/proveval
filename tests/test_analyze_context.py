"""Tests for analyze_context.py's core statistics and pairwise diagnostics.

Persists the synthetic-fixture verification (items B-G) that was previously
run ad hoc and deleted, so these invariants are checked on every run rather
than only once, by hand, right before a commit.
"""

import statistics

import pytest

import analyze_context as ac

RATING_FIELDS = ac.RATING_FIELDS


# ---------------------------------------------------------------------------
# Tie-aware rank statistics
# ---------------------------------------------------------------------------

class TestTieAwareStats:
    def test_average_ranks_no_ties(self):
        values = {"a": 3, "b": 1, "c": 2}
        ranks = ac.average_ranks(values, ["a", "b", "c"])
        assert ranks == {"a": 1, "b": 3, "c": 2}

    def test_average_ranks_ties_share_mean_rank(self):
        values = {"a": 5, "b": 5, "c": 1}
        ranks = ac.average_ranks(values, ["a", "b", "c"])
        assert ranks["a"] == ranks["b"] == 1.5
        assert ranks["c"] == 3

    def test_average_ranks_order_of_items_argument_does_not_matter(self):
        values = {"a": 5, "b": 5, "c": 1}
        r1 = ac.average_ranks(values, ["a", "b", "c"])
        r2 = ac.average_ranks(values, ["c", "b", "a"])
        assert r1 == r2

    def test_tied_groups_reports_shared_rank_position(self):
        values = {"a": 5, "b": 5, "c": 1}
        groups = ac.tied_groups(values, ["a", "b", "c"])
        assert groups[0]["rank_position"] == 1.5
        assert set(groups[0]["story_ids"]) == {"a", "b"}
        assert groups[1]["story_ids"] == ["c"]

    def test_kendall_tau_b_perfect_agreement(self):
        a = {"x": 1, "y": 2, "z": 3}
        b = {"x": 1, "y": 2, "z": 3}
        assert ac.kendall_tau_b(a, b, ["x", "y", "z"]) == pytest.approx(1.0)

    def test_kendall_tau_b_perfect_disagreement(self):
        a = {"x": 1, "y": 2, "z": 3}
        b = {"x": 3, "y": 2, "z": 1}
        assert ac.kendall_tau_b(a, b, ["x", "y", "z"]) == pytest.approx(-1.0)

    def test_kendall_tau_b_handles_ties_without_arbitrary_ordering(self):
        a = {"x": 1, "y": 1, "z": 2}
        b = {"x": 1, "y": 2, "z": 1}
        # x/y tied on side a; must not silently break the tie in either direction
        tau = ac.kendall_tau_b(a, b, ["x", "y", "z"])
        assert tau is not None

    def test_kendall_tau_b_none_for_fewer_than_two_items(self):
        assert ac.kendall_tau_b({"x": 1}, {"x": 1}, ["x"]) is None

    def test_spearman_tie_aware_matches_pearson_of_average_ranks(self):
        a = {"x": 1, "y": 2, "z": 3}
        b = {"x": 3, "y": 1, "z": 2}
        expected = ac.pearson_correlation([1, 2, 3], [3, 1, 2])
        assert ac.spearman_tie_aware(a, b, ["x", "y", "z"]) == pytest.approx(expected)

    def test_pairwise_diagnostics_from_scores_never_fabricates_a_winner_from_a_tie(self):
        values = {"a": 5, "b": 5}
        concordant, discordant, model_tied, checked = ac.pairwise_diagnostics_from_scores(values, [("a", "b")])
        assert concordant == 0 and discordant == 0 and model_tied == 1 and checked == 1


class TestDescribeAttenuation:
    def test_no_naturalistic_effect(self):
        assert ac.describe_attenuation(0, 0) == "no_naturalistic_effect"

    def test_invariance_effect_only(self):
        assert ac.describe_attenuation(0, 0.5) == "invariance_effect_only"

    def test_reversed(self):
        assert ac.describe_attenuation(1.0, -0.5) == "reversed"

    def test_attenuated(self):
        assert ac.describe_attenuation(1.0, 0.2) == "attenuated"

    def test_amplified(self):
        assert ac.describe_attenuation(1.0, 2.0) == "amplified"

    def test_unchanged(self):
        assert ac.describe_attenuation(1.0, 0.9) == "unchanged"


# ---------------------------------------------------------------------------
# collapse_attempts / filter_by_sampling_regime
# ---------------------------------------------------------------------------

class TestCollapseAttempts:
    def _trials_by_id(self):
        return {"t1": {"trial_id": "t1", "type": "context_single", "story_id": "s1"}}

    def test_keeps_the_latest_successful_attempt(self):
        rows = [
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
             "parsed_response": None, "validation_error": "bad", "sampling_regime": "low_variance_primary"},
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 2,
             "parsed_response": {"x": 1}, "validation_error": None, "sampling_regime": "low_variance_primary"},
        ]
        observations, unresolved, absorbed, missing = ac.collapse_attempts(rows, self._trials_by_id())
        assert len(observations) == 1
        assert absorbed == 1
        assert not unresolved

    def test_sampling_regime_is_part_of_the_observation_key(self):
        """Matches the identity run_batch.py must also use -- see
        tests/test_run_batch.py's sampling_regime regression tests."""
        rows = [
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
             "parsed_response": {"x": 1}, "validation_error": None, "sampling_regime": "low_variance_primary"},
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
             "parsed_response": {"x": 2}, "validation_error": None, "sampling_regime": "provider_default_secondary"},
        ]
        observations, *_ = ac.collapse_attempts(rows, self._trials_by_id())
        assert len(observations) == 2  # never collapsed into one observation

    def test_unresolved_when_every_attempt_failed(self):
        rows = [
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
             "parsed_response": None, "validation_error": "bad", "sampling_regime": "low_variance_primary"},
        ]
        observations, unresolved, absorbed, missing = ac.collapse_attempts(rows, self._trials_by_id())
        assert not observations
        assert len(unresolved) == 1

    def test_filter_by_sampling_regime(self):
        observations = {
            "a": {"sampling_regime": "low_variance_primary"},
            "b": {"sampling_regime": "provider_default_secondary"},
        }
        kept, excluded = ac.filter_by_sampling_regime(observations, "low_variance_primary")
        assert list(kept) == ["a"]
        assert excluded == 1

    def test_execution_identity_fields_survive_into_the_observation(self):
        """provider/requested_model/response_model/reasoning_profile/
        provider_reasoning_settings/execution_mode/request_id/reasoning_tokens
        must never be dropped by collapse_attempts -- downstream analysis
        relies on them to detect (and refuse to silently pool) results from
        different evaluator configurations that happen to share a
        trial_id/replicate_id."""
        rows = [{
            "trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
            "parsed_response": {"x": 1}, "validation_error": None, "sampling_regime": "low_variance_primary",
            "provider": "anthropic", "requested_model": "claude-sonnet-5", "response_model": "claude-sonnet-5-20250929",
            "reasoning_profile": "low", "provider_reasoning_settings": {"thinking": {"type": "adaptive"}},
            "execution_mode": "direct", "request_id": "msg_abc", "reasoning_tokens": None,
        }]
        observations, *_ = ac.collapse_attempts(rows, self._trials_by_id())
        obs = next(iter(observations.values()))
        assert obs["provider"] == "anthropic"
        assert obs["requested_model"] == "claude-sonnet-5"
        assert obs["response_model"] == "claude-sonnet-5-20250929"
        assert obs["reasoning_profile"] == "low"
        assert obs["provider_reasoning_settings"] == {"thinking": {"type": "adaptive"}}
        assert obs["execution_mode"] == "direct"
        assert obs["request_id"] == "msg_abc"
        assert obs["reasoning_tokens"] is None

    def test_execution_identity_fields_default_to_none_when_absent(self):
        """Pre-existing rows saved before multi-provider support have none
        of these fields -- collapse_attempts must not raise, and the
        observation just carries None for each, same as before this field
        set existed."""
        rows = [{
            "trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
            "parsed_response": {"x": 1}, "validation_error": None, "sampling_regime": "low_variance_primary",
        }]
        observations, *_ = ac.collapse_attempts(rows, self._trials_by_id())
        obs = next(iter(observations.values()))
        for field in ("provider", "requested_model", "response_model", "reasoning_profile",
                      "execution_mode", "request_id", "reasoning_tokens"):
            assert obs[field] is None


# ---------------------------------------------------------------------------
# Pairwise cell-rate / directional-effect fixtures
# ---------------------------------------------------------------------------

def make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, assignment, position, wins, n, prefix, choice_mode="forced"):
    """wins = number of replicates (out of n) where story_1 is the chosen story."""
    obs = {}
    for rep in range(1, n + 1):
        story_1_chosen = rep <= wins
        if position == "story1_as_a":
            story_a_id, story_b_id = s1, s2
            choice = "A" if story_1_chosen else "B"
        else:
            story_a_id, story_b_id = s2, s1
            choice = "B" if story_1_chosen else "A"
        obs[f"{prefix}_{rep}"] = {
            "type": "context_pairwise", "model": model, "evaluation_regime": evaluation_regime,
            "choice_mode": choice_mode, "contrast_id": contrast_id, "story_1_id": s1, "story_2_id": s2,
            "story_a_id": story_a_id, "story_b_id": story_b_id, "assignment": assignment, "position": position,
            "replicate_id": rep, "parsed_response": {f: choice for f in RATING_FIELDS},
        }
    return obs


def make_full_block(model, evaluation_regime, contrast_id, s1, s2, fwd_a, fwd_b, flp_a, flp_b, n=10, prefix="x", choice_mode="forced"):
    obs = {}
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "forward", "story1_as_a", fwd_a, n, f"{prefix}_fa", choice_mode))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "forward", "story2_as_a", fwd_b, n, f"{prefix}_fb", choice_mode))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "flipped", "story1_as_a", flp_a, n, f"{prefix}_pa", choice_mode))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "flipped", "story2_as_a", flp_b, n, f"{prefix}_pb", choice_mode))
    return obs


class TestDirectionalPairwiseEffect:
    def test_position_confound_exposes_both_context_and_position_effects(self):
        # forward: story1_as_a=8/10, story1_as_b=4/10; flipped: 6/10, 2/10
        obs = make_full_block("m", "naturalistic", "c1", "s1", "s2", fwd_a=8, fwd_b=4, flp_a=6, flp_b=2)

        directional = ac.analyze_directional_pairwise_effects(obs)
        assert directional[0]["directional_effect_a_minus_b"] == pytest.approx(0.2)

        cell_rows = ac.analyze_pairwise_cell_rates(obs)
        position_rows = ac.analyze_position_and_interaction_effects(cell_rows)
        prow = position_rows[0]
        assert prow["position_effect_pooled"] == pytest.approx(0.4)
        # the position effect (0.4) is larger than the pooled context effect (0.2)
        # -- pooling alone would have hidden this.
        assert abs(prow["position_effect_pooled"]) > abs(directional[0]["directional_effect_a_minus_b"])

    def test_context_by_position_interaction_is_visible(self):
        # forward: A=9/10, B=7/10; flipped: A=6/10, B=1/10
        obs = make_full_block("m", "naturalistic", "c2", "s1", "s2", fwd_a=9, fwd_b=7, flp_a=6, flp_b=1)

        directional = ac.analyze_directional_pairwise_effects(obs)
        assert directional[0]["directional_effect_a_minus_b"] == pytest.approx(0.45)

        cell_rows = ac.analyze_pairwise_cell_rates(obs)
        prow = ac.analyze_position_and_interaction_effects(cell_rows)[0]
        assert prow["position_effect_under_forward"] != prow["position_effect_under_flipped"]
        assert prow["context_x_position_interaction"] == pytest.approx(-0.3)

    def test_choice_mode_tie_allowed_never_leaks_into_forced_directional_effect(self):
        forced = make_full_block("m", "naturalistic", "c3", "s1", "s2", fwd_a=10, fwd_b=10, flp_a=0, flp_b=0, choice_mode="forced")
        tie_allowed = make_full_block("m", "naturalistic", "c3", "s1", "s2", fwd_a=5, fwd_b=5, flp_a=5, flp_b=5, choice_mode="tie_allowed", prefix="y")
        combined = {**forced, **tie_allowed}

        directional = ac.analyze_directional_pairwise_effects(combined)
        row = directional[0]
        assert row["directional_effect_a_minus_b"] == pytest.approx(1.0)  # unaffected by the tie_allowed rows
        assert row["n_value_a"] == 20  # forced rows only (10 fwd_a + 10 fwd_b), not 40

    def test_evaluation_regimes_are_never_pooled(self):
        naturalistic = make_full_block("m", "naturalistic", "c4", "s1", "s2", fwd_a=10, fwd_b=10, flp_a=0, flp_b=0, prefix="nat")
        invariance = make_full_block("m", "text_only_invariance", "c4", "s1", "s2", fwd_a=5, fwd_b=5, flp_a=5, flp_b=5, prefix="inv")
        combined = {**naturalistic, **invariance}

        directional = ac.analyze_directional_pairwise_effects(combined)
        by_regime = {r["evaluation_regime"]: r["directional_effect_a_minus_b"] for r in directional}
        assert by_regime["naturalistic"] == pytest.approx(1.0)
        assert by_regime["text_only_invariance"] == pytest.approx(0.0)


class TestPerStoryAndLeaveOneOut:
    def test_one_unusual_story_is_identified_by_per_story_summary_and_collapses_on_leave_one_out(self):
        stories = ["santa", "s2", "s3", "s4"]
        obs = {}
        idx = 0
        for i, s1 in enumerate(stories):
            for s2 in stories[i + 1:]:
                idx += 1
                involves_santa = "santa" in (s1, s2)
                fwd, flp = (9, 1) if involves_santa else (5, 5)
                obs.update(make_full_block("m", "naturalistic", "c5", s1, s2, fwd, fwd, flp, flp, prefix=f"p{idx}"))

        directional = ac.analyze_directional_pairwise_effects(obs)

        per_story = ac.analyze_per_story_context_effects(directional)
        santa_row = next(r for r in per_story if r["story_id"] == "santa")
        other_rows = [r for r in per_story if r["story_id"] != "santa"]
        assert santa_row["mean_effect"] > max(abs(r["mean_effect"]) for r in other_rows)

        loo = ac.analyze_leave_one_story_out(directional)
        by_excluded = {r["excluded_story_id"]: r["mean_directional_effect"] for r in loo}
        assert abs(by_excluded["(none -- full aggregate)"]) > 0.1
        assert by_excluded["santa"] == pytest.approx(0.0, abs=1e-9)

    def test_per_pair_values_are_preserved_not_just_the_mean(self):
        obs = {}
        obs.update(make_full_block("m", "naturalistic", "c6", "santa", "gilbert", 9, 9, 1, 1, prefix="p1"))
        obs.update(make_full_block("m", "naturalistic", "c6", "santa", "buddy", 5, 5, 5, 5, prefix="p2"))
        directional = ac.analyze_directional_pairwise_effects(obs)
        per_story = ac.analyze_per_story_context_effects(directional)
        santa_row = next(r for r in per_story if r["story_id"] == "santa")
        assert santa_row["n_opponents"] == 2
        assert "gilbert=" in santa_row["per_opponent_effects"]
        assert "buddy=" in santa_row["per_opponent_effects"]
        assert santa_row["range_effect"] > 0  # the two opponents' effects genuinely differ


class TestTieAllowedDiagnostic:
    def test_reports_tie_rate_separately_and_never_pools_with_forced(self):
        obs = {}
        obs.update(make_full_block("m", "naturalistic", "c7", "s1", "s2", 10, 10, 0, 0, choice_mode="forced", prefix="f"))

        # tie-allowed cells: half tie, half A
        tie_obs = {}
        for rep in range(1, 5):
            choice = "tie" if rep <= 2 else "A"
            tie_obs[f"tie_{rep}"] = {
                "type": "context_pairwise", "model": "m", "evaluation_regime": "naturalistic",
                "choice_mode": "tie_allowed", "contrast_id": "c7", "story_1_id": "s1", "story_2_id": "s2",
                "story_a_id": "s1", "story_b_id": "s2", "assignment": "forward", "position": "story1_as_a",
                "replicate_id": rep, "parsed_response": {f: choice for f in RATING_FIELDS},
            }
        combined = {**obs, **tie_obs}

        tie_rows = ac.analyze_tie_allowed_diagnostic(combined)
        assert len(tie_rows) == len(RATING_FIELDS)  # one row per category, forward assignment only
        assert all(r["p_tie"] == pytest.approx(0.5) for r in tie_rows)

        # forced results must be totally unaffected by the tie_allowed rows
        directional = ac.analyze_directional_pairwise_effects(combined)
        assert directional[0]["directional_effect_a_minus_b"] == pytest.approx(1.0)
        assert directional[0]["n_value_a"] == 20  # only the forced-choice cells


class TestSingleVsPairwiseDisagreement:
    def test_zero_single_text_effect_alongside_large_pairwise_effect_is_not_an_error(self):
        obs = {
            "neutral_1": {
                "type": "context_single", "model": "m", "evaluation_regime": "naturalistic",
                "story_id": "s1", "dimension": "neutral", "value": "neutral", "condition_id": "neutral",
                "replicate_id": 1, "parsed_response": {f: 5.0 for f in RATING_FIELDS},
            },
            "treat_1": {
                "type": "context_single", "model": "m", "evaluation_regime": "naturalistic",
                "story_id": "s1", "dimension": "provenance", "value": "ai_claude",
                "condition_id": "provenance__ai_claude",
                "replicate_id": 1, "parsed_response": {f: 5.0 for f in RATING_FIELDS},
            },
        }
        obs.update(make_full_block("m", "naturalistic", "c8", "s1", "s2", 10, 10, 0, 0))

        delta_rows = ac.analyze_treatment_vs_neutral(obs)
        assert all(r["delta"] == 0 for r in delta_rows)

        directional = ac.analyze_directional_pairwise_effects(obs)
        assert directional[0]["directional_effect_a_minus_b"] == pytest.approx(1.0)
        # No exception, no cross-format consistency check enforced -- this
        # combination is a legitimate design outcome, not a bug.
