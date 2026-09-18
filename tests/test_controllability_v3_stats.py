"""Pure-statistics tests for v3: shared story resamples, suppression and
its guarded ratio, noise-corrected drift, Pareto flags, ambiguity
regressions/bins, held-out transfer, rank correlation, split-half."""

import math
import random
from collections import Counter

import pytest

import controllability_v3_stats as st

STORIES = [f"s{i}" for i in range(6)]
PAIRS = [(a, b) for i, a in enumerate(STORIES) for b in STORIES[i + 1:]]


def resamples(n=200, seed=0):
    return st.make_story_resamples(STORIES, n, random.Random(seed))


class TestResamples:
    def test_resamples_are_deterministic_and_sum_to_story_count(self):
        a, b = resamples(50, 1), resamples(50, 1)
        assert a == b and all(sum(c.values()) == len(STORIES) for c in a) and len(a) <= 50
        assert all(len(c) >= 2 for c in a)

    def test_joint_draws_share_one_resample_per_draw(self):
        values = {p: 0.3 for p in PAIRS}
        draws = st.joint_draws({"a": values, "b": {p: 2 * v for p, v in values.items()}}, resamples())
        assert all(abs(d["b"] - 2 * d["a"]) < 1e-12 for d in draws)

    def test_zero_weight_draw_is_skipped(self):
        draws = st.joint_draws({"a": {("s0", "s1"): 1.0}}, [Counter({"s2": 3, "s3": 3}), Counter({"s0": 3, "s1": 3})])
        assert len(draws) == 1

    def test_pooled_over_cues_is_per_pair_mean(self):
        pooled = st.pooled_over_cues({"c1": {("a", "b"): 0.2}, "c2": {("a", "b"): 0.4}})
        assert pooled == {("a", "b"): pytest.approx(0.3)}


class TestSuppression:
    def test_point_estimates_and_ratio(self):
        control = {p: 0.4 for p in PAIRS}
        treated = {p: 0.1 for p in PAIRS}
        out = st.suppression_estimates(control, treated, resamples())
        assert out["CE_control"] == pytest.approx(0.4) and out["residual_context_effect"] == pytest.approx(0.1)
        assert out["signed_suppression"] == pytest.approx(0.3) and out["magnitude_suppression"] == pytest.approx(0.3)
        assert out["relative_suppression"] == pytest.approx(0.75) and out["relative_suppression_reported"]
        assert out["signed_suppression_ci_95_lo"] == pytest.approx(0.3)

    def test_magnitude_vs_signed_when_the_sign_flips(self):
        out = st.suppression_estimates({p: 0.2 for p in PAIRS}, {p: -0.2 for p in PAIRS}, resamples())
        assert out["signed_suppression"] == pytest.approx(0.4) and out["magnitude_suppression"] == pytest.approx(0.0)

    def test_ratio_withheld_for_near_zero_denominator(self):
        out = st.suppression_estimates({p: 0.01 for p in PAIRS}, {p: 0.0 for p in PAIRS}, resamples())
        assert out["relative_suppression"] is None and not out["relative_suppression_reported"]

    def test_ratio_withheld_when_control_ci_includes_zero(self):
        control = {p: (0.5 if i % 2 else -0.5) for i, p in enumerate(PAIRS)}
        out = st.suppression_estimates(control, {p: 0.0 for p in PAIRS}, resamples())
        assert not out["relative_suppression_reported"]

    def test_no_common_pairs_returns_none(self):
        assert st.suppression_estimates({("a", "b"): 1}, {("c", "d"): 1}, resamples()) is None


def rates(p_by_pair, n=10, p_other=None):
    return {pair: {"story1_as_a": {"p": p, "n": n}, "story2_as_a": {"p": p if p_other is None else p_other[pair], "n": n}} for pair, p in p_by_pair.items()}


class TestDrift:
    def test_same_sample_is_exactly_zero(self):
        r = rates({p: 0.3 for p in PAIRS})
        values = st.drift_pair_values(r, r, same_sample=True)
        assert all(v == 0.0 for name in ("abs_prob_shift", "squared_shift_noise_corrected", "ab_disagreement_excess") for v in values[name].values())
        out = st.drift_estimates(r, r, resamples(), same_sample=True)
        assert out["rms_prob_shift_noise_corrected"] == 0.0 and out["story_win_rate_rms_shift_noise_corrected"] == 0.0 and out["story_win_rate_spearman"] == pytest.approx(1.0)

    def test_noise_correction_formula(self):
        r0, r1 = rates({("a", "b"): 0.3}), rates({("a", "b"): 0.7})
        values = st.drift_pair_values(r0, r1)
        v = (0.21 / 9 + 0.21 / 9) / 4  # unbiased variance of the pair rate (mean over two positions, n=10 each)
        assert values["abs_prob_shift"][("a", "b")] == pytest.approx(0.4)
        assert values["squared_shift_noise_corrected"][("a", "b")] == pytest.approx(0.16 - 2 * v)
        assert values["abs_prob_shift_null_floor"][("a", "b")] == pytest.approx(math.sqrt(2 / math.pi) * math.sqrt(2 * v))
        assert values["ab_disagreement"][("a", "b")] == pytest.approx(0.7 * 0.7 + 0.3 * 0.3)
        assert values["ab_disagreement_noise_floor"][("a", "b")] == pytest.approx(2 * 0.3 * 0.7 * 10 / 9)
        assert values["pair_preference_flip"][("a", "b")] == 1.0

    def test_independent_samples_of_the_same_truth_have_near_zero_corrected_drift(self):
        rng = random.Random(3)
        truth = {p: rng.uniform(0.2, 0.8) for p in PAIRS}
        n = 10

        def sample():
            return {p: {pos: {"p": sum(rng.random() < truth[p] for _ in range(n)) / n, "n": n} for pos in ("story1_as_a", "story2_as_a")} for p in PAIRS}
        corrected, raw = [], []
        for _ in range(300):
            v = st.drift_pair_values(sample(), sample())
            corrected.append(st.mean(v["squared_shift_noise_corrected"].values()))
            raw.append(st.mean(v["abs_prob_shift"].values()))
        assert abs(st.mean(corrected)) < 0.004   # unbiased around zero
        assert st.mean(raw) > 0.1                 # the raw metric's noise floor

    def test_planted_shift_is_recovered_by_the_corrected_estimator(self):
        r0 = rates({p: 0.3 for p in PAIRS})
        r1 = rates({p: 0.7 for p in PAIRS})
        out = st.drift_estimates(r0, r1, resamples())
        assert 0.35 < out["rms_prob_shift_noise_corrected"] < 0.4
        assert out["pair_preference_flip"] == 1.0 and out["signed_prob_shift"] == pytest.approx(0.4)

    def test_story_win_rates_and_spearman(self):
        wr = st.story_win_rates(rates({("a", "b"): 1.0, ("a", "c"): 1.0, ("b", "c"): 1.0}))
        assert wr["a"][0] == 1.0 and wr["b"][0] == 0.5 and wr["c"][0] == 0.0
        assert st.spearman_rank_correlation({"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}) == pytest.approx(1.0)
        assert st.spearman_rank_correlation({"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 2, "c": 1}) == pytest.approx(-1.0)
        assert st.spearman_rank_correlation({"a": 1}, {"a": 1}) is None

    def test_nocontext_position_effect(self):
        r = rates({("a", "b"): 0.8}, p_other={("a", "b"): 0.6})
        assert st.nocontext_position_effect(r) == {("a", "b"): pytest.approx(0.2)}
        out = st.drift_estimates(r, r, resamples())
        assert out["nocontext_position_effect"] == pytest.approx(0.2)


class TestFrontier:
    def test_pareto_flags(self):
        flags = st.pareto_flags({"I0": (0.0, 0.0), "I1": (0.1, 0.2), "I2": (0.05, 0.3), "I3": (0.3, 0.3), "I4": (0.0, 0.0)})
        assert flags["I2"]["pareto_efficient"] and flags["I0"]["pareto_efficient"]
        assert not flags["I1"]["pareto_efficient"] and flags["I1"]["dominated_by"] == ["I2"]
        assert not flags["I3"]["pareto_efficient"] and flags["I3"]["dominated_by"] == ["I2"]
        assert flags["I4"]["pareto_efficient"]  # identical to I0: neither dominates the other


class TestAmbiguity:
    def test_regression_recovers_a_linear_relation(self):
        rows = [{"story_1_id": p[0], "story_2_id": p[1], "x": i / 10, "y": 0.1 + 0.5 * i / 10} for i, p in enumerate(PAIRS)]
        fit = st.regression_with_story_bootstrap(rows, resamples())
        assert fit["beta"] == pytest.approx(0.5) and fit["intercept"] == pytest.approx(0.1)
        assert fit["beta_ci_95_lo"] == pytest.approx(0.5, abs=1e-9) and fit["predicted_at_p50"] == pytest.approx(0.1 + 0.5 * 0.7)
        assert st.regression_with_story_bootstrap(rows[:1], resamples()) is None

    def test_tertile_bins_and_binned_means(self):
        strength = {p: i for i, p in enumerate(PAIRS)}
        bins = st.bin_by_strength(strength, 3)
        assert Counter(bins.values()) == {0: 5, 1: 5, 2: 5}
        assert bins[PAIRS[0]] == 0 and bins[PAIRS[-1]] == 2
        means = st.binned_means({p: float(bins[p]) for p in PAIRS}, bins, resamples())
        assert [means[b]["mean_signed_suppression"] for b in (0, 1, 2)] == [0.0, 1.0, 2.0]


class TestHoldout:
    def test_transfer_arithmetic(self):
        control = {p: 0.5 for p in PAIRS}
        out = st.holdout_transfer(control, {p: 0.4 for p in PAIRS}, {p: 0.1 for p in PAIRS}, {p: 0.3 for p in PAIRS}, resamples())
        assert out["suppression_full_enumeration"] == pytest.approx(0.4) and out["suppression_holdout"] == pytest.approx(0.2)
        assert out["transfer_gap_full_minus_holdout"] == pytest.approx(0.2) and out["transfer_fraction"] == pytest.approx(0.5)
        assert out["suppression_generic_I1"] == pytest.approx(0.1) and out["holdout_minus_generic"] == pytest.approx(0.1)

    def test_transfer_fraction_withheld_for_tiny_full_suppression(self):
        c = {p: 0.5 for p in PAIRS}
        out = st.holdout_transfer(c, c, {p: 0.495 for p in PAIRS}, {p: 0.49 for p in PAIRS}, resamples())
        assert out["transfer_fraction"] is None and not out["transfer_fraction_reported"]


class TestSplitHalf:
    def test_split_half(self):
        a = {p: i for i, p in enumerate(PAIRS)}
        assert st.split_half_pair_correlation(a, a)["r_split_half"] == pytest.approx(1.0)
        assert st.split_half_pair_correlation({PAIRS[0]: 1}, {PAIRS[0]: 1})["r_split_half"] is None
