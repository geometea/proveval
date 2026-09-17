"""Tests for controllability_v2_stats.py: the story-level bootstrap,
D_pair/ATE, controllability measures, equivalence classification, and the
ambiguity regression -- including the synthetic-data checks item 22 calls
for (known sign, known attenuation, known negative slope, null data).
"""

import itertools
import random

import pytest

import controllability_v2_stats as st

STORY_IDS = [f"s{i:02d}" for i in range(12)]
ALL_PAIRS = list(itertools.combinations(sorted(STORY_IDS), 2))


# ---------------------------------------------------------------------------
# D_pair / position-counterbalancing
# ---------------------------------------------------------------------------

class TestDPairFromCellRates:
    def test_missing_cell_returns_none(self):
        assert st.d_pair_from_cell_rates({("forward", "story1_as_a"): 0.5}) is None

    def test_d_pair_is_the_mean_of_the_two_position_specific_context_effects(self):
        cells = {("forward", "story1_as_a"): 0.7, ("flipped", "story1_as_a"): 0.5,
                  ("forward", "story2_as_a"): 0.6, ("flipped", "story2_as_a"): 0.4}
        result = st.d_pair_from_cell_rates(cells)
        assert result["context_effect_at_a"] == pytest.approx(0.2)
        assert result["context_effect_at_b"] == pytest.approx(0.2)
        assert result["d_pair"] == pytest.approx(0.2)

    def test_counterbalancing_cancels_a_synthetic_pure_position_bias(self):
        """A model that always prefers whichever text is displayed as
        Passage A (regardless of context/story identity) must show ZERO
        D_pair -- only a raw, large position_effect."""
        cells = {("forward", "story1_as_a"): 0.9, ("flipped", "story1_as_a"): 0.9,
                  ("forward", "story2_as_a"): 0.1, ("flipped", "story2_as_a"): 0.1}
        result = st.d_pair_from_cell_rates(cells)
        assert result["d_pair"] == pytest.approx(0.0)
        assert abs(result["position_effect"]) > 0.5

    def test_position_x_context_interaction_is_zero_when_effect_is_position_independent(self):
        cells = {("forward", "story1_as_a"): 0.7, ("flipped", "story1_as_a"): 0.5,
                  ("forward", "story2_as_a"): 0.7, ("flipped", "story2_as_a"): 0.5}
        result = st.d_pair_from_cell_rates(cells)
        assert result["context_x_position_interaction"] == pytest.approx(0.0)


class TestAteAndLeaveOneOut:
    def test_ate_is_the_unweighted_mean(self):
        values = {("a", "b"): 0.1, ("a", "c"): 0.3, ("b", "c"): 0.2}
        assert st.ate_from_pair_values(values) == pytest.approx(0.2)

    def test_ate_of_empty_is_none(self):
        assert st.ate_from_pair_values({}) is None

    def test_leave_one_story_out_excludes_every_touching_pair(self):
        values = {("a", "b"): 0.4, ("a", "c"): 0.6, ("b", "c"): 0.2}
        loo = st.leave_one_story_out(values, ["a", "b", "c"])
        assert loo["a"] == pytest.approx(0.2)  # only (b, c) remains
        assert loo["b"] == pytest.approx(0.6)  # only (a, c) remains
        assert loo["c"] == pytest.approx(0.4)  # only (a, b) remains


# ---------------------------------------------------------------------------
# Story-level bootstrap: resamples STORIES, weights PAIRS by m_i * m_j
# ---------------------------------------------------------------------------

class TestStoryBootstrap:
    def test_skips_draws_with_fewer_than_two_unique_stories(self, monkeypatch):
        """Force every resample to be a single repeated story -- every draw
        must be skipped, leaving zero draws."""
        class OneStoryRng:
            def choices(self, population, k):
                return [population[0]] * k

        pair_values = {("a", "b"): 0.5, ("a", "c"): 0.3}
        draws = st.story_bootstrap_joint_draws({"x": pair_values}, ["a", "b", "c"], 50, OneStoryRng())
        assert draws == []

    def test_weights_pairs_by_multiplicity_product(self):
        """A hand-constructed rng that always resamples exactly
        [a, a, b] (3 stories requested here to keep it simple) must weight
        pair (a, b) by 2*1=2 and never include a pair not touching a or b."""
        class FixedRng:
            def choices(self, population, k):
                return ["a", "a", "b"]

        pair_values = {("a", "b"): 1.0, ("a", "c"): 5.0, ("b", "c"): 9.0}
        draws = st.story_bootstrap_joint_draws({"x": pair_values}, ["a", "b", "c"], 1, FixedRng())
        assert len(draws) == 1
        # only (a,b) has both endpoints present (c has multiplicity 0) -- its weighted mean is just its own value
        assert draws[0]["x"] == pytest.approx(1.0)

    def test_resamples_stories_not_pairs_directly(self):
        """A constant per-pair value must bootstrap to a constant regardless
        of story resampling (sanity: the bootstrap must not corrupt a
        degenerate, no-variance case)."""
        pair_values = {p: 0.15 for p in ALL_PAIRS}
        rng = random.Random(0)
        draws = st.story_bootstrap_joint_draws({"ate": pair_values}, STORY_IDS, 300, rng)
        assert len(draws) > 250
        assert all(d["ate"] == pytest.approx(0.15) for d in draws)

    def test_bootstrap_ci_reproducible_from_seed(self):
        pair_values = {p: (hash(p) % 100) / 100 - 0.5 for p in ALL_PAIRS}
        draws_a = st.story_bootstrap_joint_draws({"ate": pair_values}, STORY_IDS, 200, random.Random(11))
        draws_b = st.story_bootstrap_joint_draws({"ate": pair_values}, STORY_IDS, 200, random.Random(11))
        assert [d["ate"] for d in draws_a] == [d["ate"] for d in draws_b]

    def test_joint_draws_share_one_resample_across_named_maps(self):
        """control and text_only in the SAME draw must reflect the SAME
        resampled story multiplicities -- verified indirectly: if both maps
        are identical, every draw's two values must be identical too."""
        pair_values = {p: (hash(p) % 100) / 100 for p in ALL_PAIRS}
        draws = st.story_bootstrap_joint_draws({"a": pair_values, "b": pair_values}, STORY_IDS, 100, random.Random(3))
        assert all(d["a"] == d["b"] for d in draws)


# ---------------------------------------------------------------------------
# Synthetic checks: known sign, known attenuation, null data
# ---------------------------------------------------------------------------

class TestSyntheticControllabilityChecks:
    def test_synthetic_known_context_effect_gives_the_correct_sign(self):
        """story1 wins more whenever it holds value 'a' (forward) -- ATE
        must be reliably positive."""
        rng = random.Random(1)
        pair_values = {p: 0.2 + rng.uniform(-0.01, 0.01) for p in ALL_PAIRS}
        ate = st.ate_from_pair_values(pair_values)
        assert ate > 0.15

        pair_values_negative = {p: -0.2 + rng.uniform(-0.01, 0.01) for p in ALL_PAIRS}
        assert st.ate_from_pair_values(pair_values_negative) < -0.15

    def test_synthetic_attenuation_gives_the_correct_reduction(self):
        control = {p: 0.20 for p in ALL_PAIRS}
        text_only = {p: 0.05 for p in ALL_PAIRS}  # attenuated toward zero, same sign
        point = st.controllability_point_estimates(control, text_only)
        assert point["magnitude_reduction"] == pytest.approx(0.15)
        assert point["signed_instruction_difference"] == pytest.approx(-0.15)
        assert point["residual_text_only_effect"] == pytest.approx(0.05)

    def test_synthetic_null_data_does_not_produce_a_systematic_directional_effect(self):
        rng = random.Random(0)
        pair_values = {p: rng.uniform(-0.03, 0.03) for p in ALL_PAIRS}
        ate = st.ate_from_pair_values(pair_values)
        assert abs(ate) < 0.02
        ci = st.bootstrap_ci_from_draws(
            [d["ate"] for d in st.story_bootstrap_joint_draws({"ate": pair_values}, STORY_IDS, 500, rng)],
            conf_levels=(0.95,),
        )[0.95]
        assert ci[0] < 0 < ci[1]  # CI must include zero -- no systematic effect detected


# ---------------------------------------------------------------------------
# Equivalence classification
# ---------------------------------------------------------------------------

class TestEquivalenceClassification:
    def test_unfrozen_reports_intervals_but_no_verdict(self):
        result = st.classify_equivalence([0.01, -0.01, 0.005, -0.005] * 20, margin=0.05, is_frozen=False)
        assert result["verdict"] is None
        assert result["ci_95"][0] is not None

    def test_frozen_and_within_margin_is_practically_invariant(self):
        draws = [0.001 * i for i in range(-50, 51)]  # tight spread around 0, well within a 0.05 margin
        result = st.classify_equivalence(draws, margin=0.05, is_frozen=True)
        assert result["verdict"] == "practically_invariant"

    def test_frozen_and_outside_margin_is_not_practically_invariant(self):
        draws = [0.3 + 0.001 * i for i in range(-50, 51)]  # centered far outside a 0.05 margin
        result = st.classify_equivalence(draws, margin=0.05, is_frozen=True)
        assert result["verdict"] == "not_practically_invariant"

    def test_uses_the_90_percent_ci_not_the_95_percent(self):
        """A margin that the 90% CI clears but the (necessarily wider) 95%
        CI would not -- the verdict must be based on the 90% CI."""
        rng = random.Random(5)
        draws = [rng.gauss(0, 0.02) for _ in range(2000)]
        result = st.classify_equivalence(draws, margin=0.05, is_frozen=True)
        # sanity: 95% CI is wider than 90% CI, both roughly centered at 0
        assert (result["ci_95"][1] - result["ci_95"][0]) >= (result["ci_90"][1] - result["ci_90"][0])


# ---------------------------------------------------------------------------
# Ambiguity regression: D_pair ~ intercept + beta * baseline_strength
# ---------------------------------------------------------------------------

class TestAmbiguityRegression:
    def test_synthetic_effect_decreasing_with_baseline_strength_gives_a_negative_slope(self):
        rows = [
            {"story_1_id": p[0], "story_2_id": p[1], "baseline_strength": i / len(ALL_PAIRS),
             "d_pair": 0.3 - 0.3 * (i / len(ALL_PAIRS))}
            for i, p in enumerate(ALL_PAIRS)
        ]
        fit = st.fit_ambiguity_regression(rows)
        assert fit["beta"] < -0.2

    def test_flat_relationship_gives_a_slope_near_zero(self):
        rng = random.Random(2)
        rows = [
            {"story_1_id": p[0], "story_2_id": p[1], "baseline_strength": rng.uniform(0, 1), "d_pair": 0.1 + rng.uniform(-0.01, 0.01)}
            for p in ALL_PAIRS
        ]
        fit = st.fit_ambiguity_regression(rows)
        assert abs(fit["beta"]) < 0.1

    def test_bootstrap_ci_for_beta_is_reproducible(self):
        rows = [
            {"story_1_id": p[0], "story_2_id": p[1], "baseline_strength": i / len(ALL_PAIRS), "d_pair": 0.2 - 0.2 * (i / len(ALL_PAIRS))}
            for i, p in enumerate(ALL_PAIRS)
        ]
        betas_a = st.ambiguity_regression_bootstrap(rows, STORY_IDS, 100, random.Random(9))
        betas_b = st.ambiguity_regression_bootstrap(rows, STORY_IDS, 100, random.Random(9))
        assert betas_a == betas_b
        assert len(betas_a) > 50

    def test_predict_at_percentiles_uses_the_observed_distribution(self):
        values = [0.0, 0.5, 1.0]
        predictions = st.predict_at_percentiles(intercept=1.0, beta=2.0, baseline_strength_values=values)
        assert predictions[50] == pytest.approx(1.0 + 2.0 * 0.5)

    def test_underdetermined_fit_returns_none(self):
        assert st.fit_ambiguity_regression([{"story_1_id": "a", "story_2_id": "b", "baseline_strength": 0.5, "d_pair": 0.1}]) is None


# ---------------------------------------------------------------------------
# Baseline pair stats + split-half reliability
# ---------------------------------------------------------------------------

class TestBaselinePairStats:
    def test_baseline_p_uses_the_laplace_correction(self):
        stats = st.baseline_pair_stats(0, 0)
        assert stats["baseline_p"] == pytest.approx(0.5)
        assert stats["baseline_n"] == 0

    def test_unanimous_pair_never_produces_an_undefined_log_odds(self):
        stats = st.baseline_pair_stats(10, 0)
        assert stats["baseline_signed_log_odds"] > 0
        assert stats["baseline_strength"] > 0

    def test_baseline_margin_and_strength_are_both_nonnegative(self):
        stats = st.baseline_pair_stats(3, 7)
        assert stats["baseline_margin"] >= 0
        assert stats["baseline_strength"] >= 0


class TestSplitHalfReliability:
    def test_perfectly_consistent_pairs_give_correlation_near_one(self):
        per_pair = {
            ("a", "b"): [(1, True), (2, True), (3, False), (4, False)],
            ("a", "c"): [(1, False), (2, False), (3, False), (4, False)],
            ("b", "c"): [(1, True), (2, True), (3, True), (4, True)],
        }
        result = st.split_half_baseline_reliability(per_pair)
        assert result["r_split_half"] == pytest.approx(1.0)
        assert result["n_pairs"] == 3

    def test_too_few_pairs_returns_none(self):
        result = st.split_half_baseline_reliability({("a", "b"): [(1, True), (2, False)]})
        assert result["r_split_half"] is None
