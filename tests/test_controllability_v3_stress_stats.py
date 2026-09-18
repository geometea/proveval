"""Stress estimators on synthetic data: attack recovery + guarded ratios,
flip rate, dose curve, susceptibility slope and +10pp threshold, slope
difference, logistic IRLS recovery, reversal dose, overflow safety."""

import math
import random

import pytest

import controllability_v3_stats as st
import controllability_v3_stress_stats as ss

STORIES = [f"s{i}" for i in range(6)]
PAIRS = [(a, b) for i, a in enumerate(STORIES) for b in STORIES[i + 1:]]


def resamples(n=200, seed=0):
    return st.make_story_resamples(STORIES, n, random.Random(seed))


class TestAdversarial:
    def test_attack_recovery_arithmetic(self):
        out = ss.attack_recovery({p: 0.1 for p in PAIRS}, {p: 0.4 for p in PAIRS}, {p: 0.5 for p in PAIRS}, resamples())
        assert out["attack_recovery"] == pytest.approx(0.3) and out["abs_attack_recovery"] == pytest.approx(0.3)
        assert out["robust_suppression"] == pytest.approx(0.2) and out["ordinary_suppression"] == pytest.approx(0.8)
        assert out["attack_recovery_ci_95_lo"] == pytest.approx(0.3) and out["robust_suppression_reported"]

    def test_ratio_guard(self):
        out = ss.attack_recovery({p: 0.0 for p in PAIRS}, {p: 0.1 for p in PAIRS}, {p: 0.005 for p in PAIRS}, resamples())
        assert not out["robust_suppression_reported"] and out["robust_suppression"] is None
        assert ss.attack_recovery({("a", "b"): 1}, {("c", "d"): 1}, {}, resamples()) is None

    def test_choice_flip_rate(self):
        ordinary = {(("a", "b"), "forward", "story1_as_a"): 0.8, (("a", "b"), "flipped", "story1_as_a"): 0.3, (("c", "d"), "forward", "story1_as_a"): 0.5}
        attack = {(("a", "b"), "forward", "story1_as_a"): 0.2, (("a", "b"), "flipped", "story1_as_a"): 0.3, (("c", "d"), "forward", "story1_as_a"): 0.9}
        flips = ss.choice_flip_rate(ordinary, attack)
        assert flips == {("a", "b"): 0.5, ("c", "d"): 0.0}


def planted(slope, intercept=0.0, noise=0.0, seed=1):
    rng = random.Random(seed)
    return {d: {p: intercept + slope * ss.dose_x_of(d) + rng.gauss(0, noise) for p in PAIRS} for d in (51, 60, 70, 80, 90, 99)}


class TestDose:
    def test_curve_and_slope_recover_planted_values(self):
        pbd = planted(0.08, 0.02)
        curve = ss.dose_curve(pbd, resamples())
        assert [c["dose"] for c in curve] == [51, 60, 70, 80, 90, 99]
        assert curve[-1]["CE"] == pytest.approx(0.02 + 0.08 * ss.dose_x_of(99))
        fit = ss.susceptibility_slope(pbd, resamples(), 0.10)
        assert fit["slope"] == pytest.approx(0.08) and fit["intercept"] == pytest.approx(0.02)
        assert fit["slope_ci_95_lo"] == pytest.approx(0.08, abs=1e-9)
        expected = 100 * ss.sigmoid((0.10 - 0.02) / 0.08)
        assert fit["threshold_dose"] == pytest.approx(expected) and fit["threshold_dose_ci_95_lo"] == pytest.approx(expected, abs=1e-6)
        assert fit["threshold_extrapolated"] is False and fit["threshold_n_draws_without_solution"] == 0

    def test_threshold_edge_cases(self):
        assert ss.threshold_dose(0.0, 0.0, 0.1) is None and ss.threshold_dose(0.0, -0.1, 0.1) is None
        assert 0 < ss.threshold_dose(0.5, 0.1, 0.1) < 51    # solution below the grid: a valid number that the caller flags as extrapolated
        assert ss.threshold_dose(1000.0, 1.0, 0.1) is None  # overflow-safe
        assert 50 < ss.threshold_dose(0.0, 0.05, 0.10) < 100

    def test_slope_difference_and_flattening(self):
        diff = ss.slope_difference(planted(0.10), planted(0.04), resamples())
        assert diff["suppression_slope"] == pytest.approx(0.06) and diff["flattening_fraction"] == pytest.approx(0.6) and diff["flattening_fraction_reported"]
        flat = ss.slope_difference(planted(0.001), planted(0.0005), resamples())
        assert not flat["flattening_fraction_reported"]

    def test_logistic_recovers_parameters(self):
        rng = random.Random(0)
        a, g, b = -0.3, 0.8, 0.5
        cells = []
        for i in range(400):
            bb, x = rng.gauss(0, 1), ss.dose_x_of(rng.choice([51, 60, 70, 80, 90, 99]))
            p = ss.sigmoid(a + g * bb + b * x)
            cells.append({"b": bb, "x": x, "y": sum(rng.random() < p for _ in range(30)), "n": 30, "story_1_id": STORIES[i % 3], "story_2_id": STORIES[3 + i % 3]})
        fit = ss.fit_logistic(cells)
        assert fit[0] == pytest.approx(a, abs=0.1) and fit[1] == pytest.approx(g, abs=0.1) and fit[2] == pytest.approx(b, abs=0.1)
        rev = ss.logistic_reversal(cells, {"p50": 1.0}, resamples(50))
        assert rev["reversal_dose_p50"] == pytest.approx(100 * ss.sigmoid(-(fit[0] + fit[1] * -1.0) / fit[2]))
        assert rev["reversal_dose_p50_ci_95_lo"] < rev["reversal_dose_p50"] < rev["reversal_dose_p50_ci_95_hi"]

    def test_reversal_dose_formula_and_none_cases(self):
        assert ss.reversal_dose((0.0, 1.0, 0.5), -1.0) == pytest.approx(100 * ss.sigmoid(2.0))
        assert ss.reversal_dose((0.0, 1.0, -0.5), -1.0) is None and ss.reversal_dose((0.0, 1.0, 0.5), -100.0) is None

    def test_sigmoid_is_overflow_safe(self):
        assert ss.sigmoid(-10000) < 1e-100 and ss.sigmoid(10000) == 1.0
