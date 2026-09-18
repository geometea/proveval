"""Pure-Python statistics for the v3 selective-suppression experiment.
No file I/O, no model calls: everything takes plain data structures and is
unit-testable against synthetic fixtures.

Uncertainty everywhere is the STORY-LEVEL bootstrap: each draw resamples
the 12 story identities with replacement and reweights pair (i, j) by
m_i * m_j (v2's story_bootstrap machinery, reused). One set of resamples
(make_story_resamples) is shared by every quantity in an analysis, so any
derived difference (suppression = CE_I0 - CE_I, a headline contrast, a
transfer gap) is computed within the SAME draw for both terms.

Estimands (per intervention I; "pairs" are the 66 story pairs):
  CE_I(cue)         mean over pairs of D_pair under I, context present --
                    D_pair is v2's position-counterbalanced context effect.
  CE_I              pooled: each pair's D_pair averaged over the 5 cues,
                    then averaged over pairs (= grand mean).
  signed suppression     CE_I0 - CE_I
  magnitude suppression  |CE_I0| - |CE_I|      (the frontier's y axis)
  residual context effect CE_I itself
  relative suppression   magnitude suppression / |CE_I0|, reported only when
                    |CE_I0| clears a minimum and its CI excludes zero.
  Drift_I           no-context preferences under I vs under I0:
                    absolute probability shift (frontier x axis), signed
                    shift, A/B disagreement (and its excess over I0's own
                    sampling-noise floor), pair-level preference flips,
                    story-level win-rate shift.
"""

import math
import statistics
from collections import Counter, defaultdict

from controllability_v2_stats import (  # pure helpers, reused verbatim
    baseline_pair_stats,
    bootstrap_ci_from_draws,
    d_pair_from_cell_rates,
    leave_one_story_out,
    percentile,
    predict_at_percentiles,
    weighted_least_squares,
    weighted_mean,
)
from context_analysis_stats import pearson_correlation

__all__ = [
    "baseline_pair_stats", "bootstrap_ci_from_draws", "d_pair_from_cell_rates", "leave_one_story_out",
    "percentile", "predict_at_percentiles", "weighted_least_squares", "weighted_mean", "pearson_correlation",
]


# ---------------------------------------------------------------------------
# Shared story resamples + joint draws
# ---------------------------------------------------------------------------

def make_story_resamples(story_ids, n_draws, rng):
    """List of multiplicity Counters, one per kept draw (draws with fewer
    than 2 distinct stories are skipped)."""
    resamples = []
    for _ in range(n_draws):
        resampled = rng.choices(story_ids, k=len(story_ids))
        if len(set(resampled)) < 2:
            continue
        resamples.append(Counter(resampled))
    return resamples


def joint_draws(pair_value_maps, resamples):
    """[{name: weighted_mean}] per draw, skipping a draw in which any map
    has zero total weight -- so every returned dict has every name."""
    draws = []
    for multiplicities in resamples:
        estimate = {}
        for name, pair_values in pair_value_maps.items():
            value = weighted_mean(pair_values, multiplicities)
            if value is None:
                estimate = None
                break
            estimate[name] = value
        if estimate is not None:
            draws.append(estimate)
    return draws


def story_level_draws(story_values, resamples):
    """Weighted mean of per-STORY values under each resample (weight m_s)."""
    draws = []
    for multiplicities in resamples:
        total = sum(multiplicities.get(s, 0) for s in story_values)
        if total == 0:
            continue
        draws.append(sum(multiplicities.get(s, 0) * v for s, v in story_values.items()) / total)
    return draws


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def ci95(draws):
    lo, hi = bootstrap_ci_from_draws(draws, conf_levels=(0.95,))[0.95]
    return lo, hi


def summarize(name, point, draws):
    lo, hi = ci95(draws)
    return {name: point, f"{name}_ci_95_lo": lo, f"{name}_ci_95_hi": hi}


# ---------------------------------------------------------------------------
# Pair-value helpers
# ---------------------------------------------------------------------------

def pooled_over_cues(pair_values_by_cue):
    """{cue: {(s1, s2): value}} -> {(s1, s2): mean over the cues that have
    the pair}. With every cue present for every pair (the complete design)
    the mean over this map equals the grand mean over (cue, pair)."""
    accum = defaultdict(list)
    for cue, pair_values in pair_values_by_cue.items():
        for pair, value in pair_values.items():
            accum[pair].append(value)
    return {pair: statistics.mean(v) for pair, v in accum.items()}


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def suppression_estimates(control_pairs, treated_pairs, resamples, relative_min_denominator=0.02):
    """Point estimates + story-bootstrap CIs for CE_control, CE_treated,
    signed/magnitude suppression, residual effect, and (guarded) relative
    suppression. Both terms of every difference come from the same draw."""
    common = sorted(set(control_pairs) & set(treated_pairs))
    if not common:
        return None
    control = {p: control_pairs[p] for p in common}
    treated = {p: treated_pairs[p] for p in common}
    ce_control, ce_treated = mean(control.values()), mean(treated.values())
    draws = joint_draws({"control": control, "treated": treated}, resamples)
    d_control = [d["control"] for d in draws]
    d_treated = [d["treated"] for d in draws]
    d_signed = [c - t for c, t in zip(d_control, d_treated)]
    d_magnitude = [abs(c) - abs(t) for c, t in zip(d_control, d_treated)]

    out = {"n_story_pairs": len(common), "n_bootstrap_draws_used": len(draws)}
    out.update(summarize("CE_control", ce_control, d_control))
    out.update(summarize("CE_treated", ce_treated, d_treated))
    out.update(summarize("residual_context_effect", ce_treated, d_treated))
    out.update(summarize("signed_suppression", ce_control - ce_treated, d_signed))
    out.update(summarize("magnitude_suppression", abs(ce_control) - abs(ce_treated), d_magnitude))

    control_ci = out["CE_control_ci_95_lo"], out["CE_control_ci_95_hi"]
    denominator_ok = (
        abs(ce_control) >= relative_min_denominator
        and control_ci[0] is not None and (control_ci[0] > 0 or control_ci[1] < 0)
    )
    if denominator_ok:
        ratio_draws = [m / abs(c) for c, m in zip(d_control, d_magnitude) if abs(c) >= relative_min_denominator]
        out.update(summarize("relative_suppression", (abs(ce_control) - abs(ce_treated)) / abs(ce_control), ratio_draws))
        out["relative_suppression_n_draws_used"] = len(ratio_draws)
        out["relative_suppression_reported"] = True
    else:
        out.update({"relative_suppression": None, "relative_suppression_ci_95_lo": None, "relative_suppression_ci_95_hi": None,
                    "relative_suppression_n_draws_used": 0, "relative_suppression_reported": False})
    return out


# ---------------------------------------------------------------------------
# No-context drift
#
# rates_*: {(s1, s2): {position: {"p": story1-chosen rate, "n": count}}}.
#
# Raw shift metrics (|p_I - p_0|, A/B disagreement, preference flips) all
# have a SAMPLING-NOISE FLOOR: with n observations per cell, two estimates
# of the SAME preference differ by ~0.13 in absolute value at n = 20. The
# frontier therefore uses a noise-CORRECTED quantity as its primary drift
# axis: the root of the unbiased mean squared shift,
#     E[(p_I_hat - p_0_hat)^2] - Var(p_I_hat) - Var(p_0_hat)
# with each variance estimated unbiasedly as p_hat(1-p_hat)/(n-1) per
# display position (a pair rate is the mean of its two position rates).
# Under zero true drift this is ~0 (and can be slightly negative before
# clipping); the raw metrics are reported alongside with their analytic
# null floors.
# ---------------------------------------------------------------------------

def _pair_rate(position_rates):
    """Mean of the two position-specific story1-chosen rates."""
    return mean(v["p"] for v in position_rates.values())


def _pair_rate_variance(position_rates):
    """Unbiased sampling variance of the pair rate (mean over positions):
    sum over positions of p(1-p)/(n-1), divided by k^2. None if any
    position has n < 2 (no variance estimate possible)."""
    variances = []
    for v in position_rates.values():
        if v["n"] < 2:
            return None
        variances.append(v["p"] * (1 - v["p"]) / (v["n"] - 1))
    k = len(variances)
    return sum(variances) / (k * k) if k else None


def drift_pair_values(rates_control, rates_treated, same_sample=False):
    """Per-pair value maps for every drift metric. same_sample=True is the
    degenerate I0-vs-I0 case (one sample compared with itself): every
    drift metric is then exactly zero by definition, never the negative
    value a noise correction would produce for a self-comparison."""
    common = sorted(set(rates_control) & set(rates_treated))
    out = defaultdict(dict)
    if same_sample:
        for pair in common:
            for name in ("abs_prob_shift", "signed_prob_shift", "pair_preference_flip", "squared_shift_noise_corrected",
                         "abs_prob_shift_null_floor", "ab_disagreement_excess"):
                out[name][pair] = 0.0
            out["ab_disagreement"][pair] = mean(2 * v["p"] * (1 - v["p"]) for v in rates_control[pair].values())
            out["ab_disagreement_noise_floor"][pair] = out["ab_disagreement"][pair]
        return dict(out)
    for pair in common:
        r0, r1 = rates_control[pair], rates_treated[pair]
        p0, p1 = _pair_rate(r0), _pair_rate(r1)
        v0, v1 = _pair_rate_variance(r0), _pair_rate_variance(r1)
        out["abs_prob_shift"][pair] = abs(p1 - p0)
        out["signed_prob_shift"][pair] = p1 - p0
        out["pair_preference_flip"][pair] = 1.0 if (p1 - 0.5) * (p0 - 0.5) < 0 else 0.0
        if v0 is not None and v1 is not None:
            out["squared_shift_noise_corrected"][pair] = (p1 - p0) ** 2 - v0 - v1
            out["abs_prob_shift_null_floor"][pair] = math.sqrt(2 / math.pi) * math.sqrt(v0 + v1)
        cell_dis, cell_floor = [], []
        for position, c0 in r0.items():
            c1 = r1.get(position)
            if c1 is None:
                continue
            q0, q1 = c0["p"], c1["p"]
            cell_dis.append(q1 * (1 - q0) + (1 - q1) * q0)
            # unbiased estimate of 2*q0*(1-q0): q0_hat(1-q0_hat) * n/(n-1)
            cell_floor.append(2 * q0 * (1 - q0) * (c0["n"] / (c0["n"] - 1) if c0["n"] > 1 else 1.0))
        out["ab_disagreement"][pair] = mean(cell_dis)
        out["ab_disagreement_noise_floor"][pair] = mean(cell_floor)
        out["ab_disagreement_excess"][pair] = out["ab_disagreement"][pair] - out["ab_disagreement_noise_floor"][pair]
    return dict(out)


def story_win_rates(rates):
    """{story: (mean over its pairs of P(story chosen), summed variance / k^2)}."""
    per_story, per_story_var = defaultdict(list), defaultdict(list)
    for (s1, s2), position_rates in rates.items():
        p = _pair_rate(position_rates)
        v = _pair_rate_variance(position_rates)
        per_story[s1].append(p); per_story[s2].append(1 - p)
        per_story_var[s1].append(v); per_story_var[s2].append(v)
    out = {}
    for s, values in per_story.items():
        variances = per_story_var[s]
        var = None if any(v is None for v in variances) else sum(variances) / (len(variances) ** 2)
        out[s] = (statistics.mean(values), var)
    return out


def spearman_rank_correlation(values_a, values_b):
    """Spearman rho between two equally-keyed dicts (average ranks for ties)."""
    keys = sorted(set(values_a) & set(values_b))
    if len(keys) < 2:
        return None

    def ranks(values):
        ordered = sorted(keys, key=lambda k: values[k])
        r, i = {}, 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
                j += 1
            for k in ordered[i:j + 1]:
                r[k] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = ranks(values_a), ranks(values_b)
    return pearson_correlation([ra[k] for k in keys], [rb[k] for k in keys])


def nocontext_position_effect(rates):
    """Mean over pairs of p(story1 | story1_as_a) - p(story1 | story2_as_a)."""
    return {pair: r["story1_as_a"]["p"] - r["story2_as_a"]["p"] for pair, r in rates.items() if "story1_as_a" in r and "story2_as_a" in r}


def _rms(value):
    return math.sqrt(value) if value is not None and value > 0 else 0.0


def drift_estimates(rates_control, rates_treated, resamples, same_sample=False):
    values = drift_pair_values(rates_control, rates_treated, same_sample)
    if not values.get("abs_prob_shift"):
        return None
    draws = joint_draws(values, resamples)
    out = {"n_story_pairs": len(values["abs_prob_shift"]), "n_bootstrap_draws_used": len(draws)}
    for name, pair_values in values.items():
        out.update(summarize(name, mean(pair_values.values()), [d[name] for d in draws]))
    # primary (noise-corrected) drift: RMS of the unbiased squared shift
    msq = mean(values["squared_shift_noise_corrected"].values()) if values.get("squared_shift_noise_corrected") else None
    out.update(summarize("rms_prob_shift_noise_corrected", _rms(msq), [_rms(d["squared_shift_noise_corrected"]) for d in draws] if msq is not None else []))
    out["abs_prob_shift_excess_over_null_floor"] = (
        out["abs_prob_shift"] - out["abs_prob_shift_null_floor"] if out.get("abs_prob_shift_null_floor") is not None else None
    )

    wr_control, wr_treated = story_win_rates(rates_control), story_win_rates(rates_treated)
    stories = sorted(s for s in wr_control if s in wr_treated)
    story_delta = {s: abs(wr_treated[s][0] - wr_control[s][0]) for s in stories}
    out.update(summarize("story_win_rate_abs_shift", mean(story_delta.values()), story_level_draws(story_delta, resamples)))
    story_sq = {s: (0.0 if same_sample else (wr_treated[s][0] - wr_control[s][0]) ** 2 - (wr_treated[s][1] or 0) - (wr_control[s][1] or 0))
                for s in stories if wr_treated[s][1] is not None and wr_control[s][1] is not None}
    if story_sq:
        out.update(summarize("story_win_rate_rms_shift_noise_corrected", _rms(mean(story_sq.values())), [_rms(d) for d in story_level_draws(story_sq, resamples)]))
    else:
        out.update({"story_win_rate_rms_shift_noise_corrected": None, "story_win_rate_rms_shift_noise_corrected_ci_95_lo": None, "story_win_rate_rms_shift_noise_corrected_ci_95_hi": None})
    out["story_win_rate_spearman"] = spearman_rank_correlation({s: wr_control[s][0] for s in stories}, {s: wr_treated[s][0] for s in stories})
    out["story_win_rate_pearson"] = pearson_correlation([wr_control[s][0] for s in stories], [wr_treated[s][0] for s in stories]) if len(stories) >= 2 else None
    pos = nocontext_position_effect(rates_treated)
    out.update(summarize("nocontext_position_effect", mean(pos.values()), [d["pos"] for d in joint_draws({"pos": pos}, resamples)]))
    return out


# ---------------------------------------------------------------------------
# Suppression-distortion frontier
# ---------------------------------------------------------------------------

def pareto_flags(points):
    """points: {intervention: (drift_x, suppression_y)}. An intervention is
    Pareto-efficient if no other point has y >= its y and x <= its x with at
    least one strict inequality. Returns {intervention: {"pareto_efficient":
    bool, "dominated_by": [...]}}."""
    flags = {}
    for name, (x, y) in points.items():
        dominated_by = [
            other for other, (ox, oy) in points.items()
            if other != name and oy >= y and ox <= x and (oy > y or ox < x)
        ]
        flags[name] = {"pareto_efficient": not dominated_by, "dominated_by": sorted(dominated_by)}
    return flags


# ---------------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------------

def regression_with_story_bootstrap(rows, resamples):
    """rows: [{"story_1_id","story_2_id","x","y"}]. Unweighted point fit;
    per-draw weighted refits for beta's CI. Returns None if underdetermined."""
    if len(rows) < 2:
        return None
    fit = weighted_least_squares([r["x"] for r in rows], [r["y"] for r in rows], [1.0] * len(rows))
    if fit is None:
        return None
    betas = []
    for multiplicities in resamples:
        xs, ys, ws = [], [], []
        for r in rows:
            w = multiplicities.get(r["story_1_id"], 0) * multiplicities.get(r["story_2_id"], 0)
            if w:
                xs.append(r["x"]); ys.append(r["y"]); ws.append(w)
        draw_fit = weighted_least_squares(xs, ys, ws) if ws else None
        if draw_fit is not None:
            betas.append(draw_fit[1])
    intercept, beta = fit
    predictions = predict_at_percentiles(intercept, beta, [r["x"] for r in rows])
    return {
        "n": len(rows), "intercept": intercept, **summarize("beta", beta, betas),
        "predicted_at_p25": predictions[25], "predicted_at_p50": predictions[50], "predicted_at_p75": predictions[75],
    }


def bin_by_strength(pair_strength, n_bins=3):
    """{pair: bin_index} by tertiles (n_bins) of baseline strength; ties
    broken by pair id for determinism."""
    ordered = sorted(pair_strength, key=lambda p: (pair_strength[p], p))
    bins = {}
    for index, pair in enumerate(ordered):
        bins[pair] = min(n_bins - 1, index * n_bins // len(ordered))
    return bins


def binned_means(pair_values, pair_bins, resamples):
    """Per bin: mean of pair_values and a story-bootstrap CI."""
    out = {}
    for b in sorted(set(pair_bins.values())):
        subset = {p: v for p, v in pair_values.items() if pair_bins.get(p) == b}
        if not subset:
            continue
        draws = [d["v"] for d in joint_draws({"v": subset}, resamples)]
        out[b] = {"n_story_pairs": len(subset), **summarize("mean_signed_suppression", mean(subset.values()), draws)}
    return out


# ---------------------------------------------------------------------------
# Held-out cue transfer
# ---------------------------------------------------------------------------

def holdout_transfer(control_pairs, generic_pairs, full_pairs, holdout_pairs, resamples, relative_min_denominator=0.02):
    """For one cue k: control = I0, generic = I1, full = I2 (all five cues
    named), holdout = I2 with k omitted. Magnitude suppression of each
    relative to I0, the transfer gap (full - holdout; positive = naming
    the cue mattered), the guarded transfer fraction, and holdout vs
    generic (does naming the OTHER four cues beat plain 'judge only the
    writing' on the unnamed cue?)."""
    common = sorted(set(control_pairs) & set(generic_pairs) & set(full_pairs) & set(holdout_pairs))
    if not common:
        return None
    maps = {name: {p: m[p] for p in common} for name, m in (("control", control_pairs), ("generic", generic_pairs), ("full", full_pairs), ("holdout", holdout_pairs))}
    point = {name: mean(m.values()) for name, m in maps.items()}
    draws = joint_draws(maps, resamples)

    def magnitude_suppression(c, t):
        return abs(c) - abs(t)

    out = {"n_story_pairs": len(common), "n_bootstrap_draws_used": len(draws)}
    for name in maps:
        out.update(summarize(f"CE_{name}", point[name], [d[name] for d in draws]))
    sup_full_point = magnitude_suppression(point["control"], point["full"])
    sup_hold_point = magnitude_suppression(point["control"], point["holdout"])
    sup_gen_point = magnitude_suppression(point["control"], point["generic"])
    sup_full = [magnitude_suppression(d["control"], d["full"]) for d in draws]
    sup_hold = [magnitude_suppression(d["control"], d["holdout"]) for d in draws]
    sup_gen = [magnitude_suppression(d["control"], d["generic"]) for d in draws]
    out.update(summarize("suppression_full_enumeration", sup_full_point, sup_full))
    out.update(summarize("suppression_holdout", sup_hold_point, sup_hold))
    out.update(summarize("suppression_generic_I1", sup_gen_point, sup_gen))
    out.update(summarize("transfer_gap_full_minus_holdout", sup_full_point - sup_hold_point, [f - h for f, h in zip(sup_full, sup_hold)]))
    out.update(summarize("holdout_minus_generic", sup_hold_point - sup_gen_point, [h - g for h, g in zip(sup_hold, sup_gen)]))
    if abs(sup_full_point) >= relative_min_denominator:
        ratio_draws = [h / f for f, h in zip(sup_full, sup_hold) if abs(f) >= relative_min_denominator]
        out.update(summarize("transfer_fraction", sup_hold_point / sup_full_point, ratio_draws))
        out["transfer_fraction_reported"] = True
    else:
        out.update({"transfer_fraction": None, "transfer_fraction_ci_95_lo": None, "transfer_fraction_ci_95_hi": None, "transfer_fraction_reported": False})
    return out


# ---------------------------------------------------------------------------
# Replicate stability
# ---------------------------------------------------------------------------

def split_half_pair_correlation(half_a, half_b):
    common = sorted(set(half_a) & set(half_b))
    if len(common) < 2:
        return {"r_split_half": None, "n_pairs": len(common)}
    return {"r_split_half": pearson_correlation([half_a[p] for p in common], [half_b[p] for p in common]), "n_pairs": len(common)}
