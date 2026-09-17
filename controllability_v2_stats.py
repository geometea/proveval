"""Pure-Python statistics for the v2 context-controllability experiment:
the story-level bootstrap, the D_pair/ATE estimator, controllability
measures, equivalence classification, and the ambiguity (baseline-strength)
regression. No file I/O and no model calls anywhere in this module --
everything here takes plain Python data structures and is directly
unit-testable against synthetic fixtures.

Story-level bootstrap (item 12): resampling the 66 STORY PAIRS
independently would treat each pair as its own independent unit, which is
wrong -- every pair shares one of only 12 underlying stories with 11 other
pairs, so pairs are not independent. Instead every draw resamples the 12
STORY IDENTITIES with replacement, and each pair (i, j) is reweighted by
that draw's multiplicity product m_i * m_j (0 if either story didn't appear
in the resample) before being folded into a weighted mean/regression -- see
story_bootstrap_joint_draws.
"""

import math
import statistics
from collections import Counter, defaultdict

from context_analysis_stats import pearson_correlation


# ---------------------------------------------------------------------------
# Blind baseline (item 5)
# ---------------------------------------------------------------------------

def baseline_pair_stats(wins_story1, wins_story2):
    """wins_story1/wins_story2 are counts over every valid baseline
    observation of this pair (both display orders, all replicates pooled).
    baseline_p uses a +0.5/+1 (Laplace) correction so a unanimous pair never
    produces an exact 0 or 1 probability or an undefined log-odds."""
    n = wins_story1 + wins_story2
    p = (wins_story1 + 0.5) / (n + 1)
    signed_log_odds = math.log((wins_story1 + 0.5) / (wins_story2 + 0.5))
    return {
        "baseline_wins_story1": wins_story1,
        "baseline_n": n,
        "baseline_p": p,
        "baseline_margin": 2 * abs(p - 0.5),
        "baseline_signed_log_odds": signed_log_odds,
        "baseline_strength": abs(signed_log_odds),
    }


def split_half_baseline_reliability(per_pair_replicate_choices):
    """per_pair_replicate_choices: {(story_1, story_2): [(replicate_number,
    story1_chosen_bool), ...]}. Splits each pair's observations into
    odd/even replicate_number halves, computes each half's story1-win rate,
    and Pearson-correlates the two halves across every pair with data in
    both (reusing context_analysis_stats.pearson_correlation -- not a new
    statistical method). Returns {"r_split_half", "n_pairs"}; r is None
    below 2 pairs."""
    half_a, half_b = {}, {}
    for pair, observations in per_pair_replicate_choices.items():
        odds = [chosen for replicate_number, chosen in observations if replicate_number % 2 == 1]
        evens = [chosen for replicate_number, chosen in observations if replicate_number % 2 == 0]
        if odds:
            half_a[pair] = sum(odds) / len(odds)
        if evens:
            half_b[pair] = sum(evens) / len(evens)

    common = sorted(set(half_a) & set(half_b))
    r = pearson_correlation([half_a[p] for p in common], [half_b[p] for p in common]) if len(common) >= 2 else None
    return {"r_split_half": r, "n_pairs": len(common)}


# ---------------------------------------------------------------------------
# Percentiles / generic story-level bootstrap
# ---------------------------------------------------------------------------

def percentile(sorted_values, p):
    """Linear-interpolation percentile (p in [0, 100]) of an already-sorted list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def weighted_mean(pair_values, multiplicities):
    """sum(m_i*m_j*value) / sum(m_i*m_j) over pair_values={(i,j): value},
    using story multiplicities `multiplicities` (a dict story_id -> count,
    possibly 0/missing). Returns None if the total weight is zero (no pair
    with both endpoints present in this draw)."""
    weighted_sum = 0.0
    weight_total = 0.0
    for (i, j), value in pair_values.items():
        w = multiplicities.get(i, 0) * multiplicities.get(j, 0)
        if w:
            weighted_sum += w * value
            weight_total += w
    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def story_bootstrap_joint_draws(pair_value_maps, story_ids, n_draws, rng):
    """The one story-level bootstrap used everywhere in this module.

    pair_value_maps: {name: {(story_i, story_j): value}} -- one or more
    named per-pair value maps (e.g. {"control": D_pair_control,
    "text_only": D_pair_text_only}), all keyed by the SAME set of pairs, so
    a single resample can be used to compute several jointly-paired
    estimates per draw (needed for derived quantities like
    signed_instruction_difference, which must use ONE resample per draw for
    both terms, not two independently-resampled ones).

    For each of n_draws draws: sample 12 story ids with replacement from
    `story_ids`, compute each story's multiplicity, skip the draw if fewer
    than 2 unique stories were drawn (step 5 of item 12) or if any named
    map's total weight comes out zero, and otherwise weight pair (i, j) by
    m_i * m_j (item 12 steps 2-4).

    `rng` is a random.Random instance (caller-provided so callers can share
    one seeded generator across several bootstrap calls deterministically).
    Returns a list of {name: weighted_estimate} dicts, one per KEPT draw --
    always <= n_draws.
    """
    draws = []
    for _ in range(n_draws):
        resampled = rng.choices(story_ids, k=len(story_ids))
        multiplicities = Counter(resampled)
        if len({sid for sid in resampled}) < 2:
            continue
        estimate = {}
        ok = True
        for name, pair_values in pair_value_maps.items():
            value = weighted_mean(pair_values, multiplicities)
            if value is None:
                ok = False
                break
            estimate[name] = value
        if ok:
            draws.append(estimate)
    return draws


def bootstrap_ci_from_draws(values, conf_levels=(0.95,)):
    """{conf_level: (lo, hi)} from a flat list of scalar bootstrap draws."""
    if not values:
        return {lvl: (None, None) for lvl in conf_levels}
    s = sorted(values)
    cis = {}
    for lvl in conf_levels:
        alpha = 1 - lvl
        cis[lvl] = (percentile(s, 100 * alpha / 2), percentile(s, 100 * (1 - alpha / 2)))
    return cis


# ---------------------------------------------------------------------------
# D_pair / ATE (item 11)
# ---------------------------------------------------------------------------

def d_pair_from_cell_rates(cell_rates):
    """cell_rates: {(assignment, position): p_story1_chosen} for the 4
    required (assignment, position) cells of one story pair. Returns a dict
    with D_pair and the position-decomposed diagnostics (item 16) -- D_pair
    is exactly the mean of context_effect_at_a and context_effect_at_b, so
    every headline context effect is structurally position-counterbalanced.
    Returns None if any of the 4 required cells is missing.
    """
    required = {("forward", "story1_as_a"), ("forward", "story2_as_a"), ("flipped", "story1_as_a"), ("flipped", "story2_as_a")}
    if set(cell_rates) != required:
        return None

    p_forward_a = cell_rates[("forward", "story1_as_a")]
    p_flipped_a = cell_rates[("flipped", "story1_as_a")]
    p_forward_b = cell_rates[("forward", "story2_as_a")]
    p_flipped_b = cell_rates[("flipped", "story2_as_a")]

    context_effect_at_a = p_forward_a - p_flipped_a
    context_effect_at_b = p_forward_b - p_flipped_b
    d_pair = (context_effect_at_a + context_effect_at_b) / 2
    position_effect = ((p_forward_a + p_flipped_a) / 2) - ((p_forward_b + p_flipped_b) / 2)

    return {
        "d_pair": d_pair,
        "context_effect_at_a": context_effect_at_a,
        "context_effect_at_b": context_effect_at_b,
        "context_x_position_interaction": context_effect_at_a - context_effect_at_b,
        "position_effect": position_effect,
    }


def ate_from_pair_values(pair_values):
    """Plain (unweighted) mean of D_pair over however many story pairs have
    a value -- the point estimate; see story_bootstrap_joint_draws for the
    CI."""
    values = list(pair_values.values())
    return statistics.mean(values) if values else None


def leave_one_story_out(pair_values, story_ids):
    """{story_id: ATE recomputed excluding every pair touching story_id}."""
    results = {}
    for excluded in story_ids:
        remaining = [v for (i, j), v in pair_values.items() if i != excluded and j != excluded]
        results[excluded] = statistics.mean(remaining) if remaining else None
    return results


# ---------------------------------------------------------------------------
# Controllability measures (item 13) and equivalence (item 14)
# ---------------------------------------------------------------------------

def controllability_point_estimates(pair_values_control, pair_values_text_only):
    ate_control = ate_from_pair_values(pair_values_control)
    ate_text_only = ate_from_pair_values(pair_values_text_only)
    if ate_control is None or ate_text_only is None:
        return None
    return {
        "ATE_control": ate_control,
        "ATE_text_only": ate_text_only,
        "signed_instruction_difference": ate_text_only - ate_control,
        "magnitude_reduction": abs(ate_control) - abs(ate_text_only),
        "residual_text_only_effect": ate_text_only,
    }


def controllability_bootstrap(pair_values_control, pair_values_text_only, story_ids, n_draws, rng):
    """Joint (paired-per-draw) story bootstrap for ATE_control, ATE_text_only,
    and their two derived quantities. Returns
    {"ATE_control": [...], "ATE_text_only": [...],
     "signed_instruction_difference": [...], "magnitude_reduction": [...]}
    -- each a flat list of per-draw values, ready for bootstrap_ci_from_draws."""
    joint = story_bootstrap_joint_draws(
        {"control": pair_values_control, "text_only": pair_values_text_only}, story_ids, n_draws, rng
    )
    out = {"ATE_control": [], "ATE_text_only": [], "signed_instruction_difference": [], "magnitude_reduction": []}
    for draw in joint:
        control, text_only = draw["control"], draw["text_only"]
        out["ATE_control"].append(control)
        out["ATE_text_only"].append(text_only)
        out["signed_instruction_difference"].append(text_only - control)
        out["magnitude_reduction"].append(abs(control) - abs(text_only))
    return out


def classify_equivalence(text_only_draws, margin, is_frozen):
    """95%/90% percentile CIs for ATE_text_only; a verdict is produced only
    when is_frozen is True (item 14: "If the study config is not frozen,
    output the intervals but do not output a final equivalence verdict").
    "practically_invariant" only when the COMPLETE 90% CI falls inside
    [-margin, +margin]."""
    cis = bootstrap_ci_from_draws(text_only_draws, conf_levels=(0.95, 0.90))
    ci_95, ci_90 = cis[0.95], cis[0.90]
    result = {"ci_95": ci_95, "ci_90": ci_90, "margin": margin}
    if not is_frozen:
        result["verdict"] = None
        return result
    lo, hi = ci_90
    if lo is None or hi is None:
        result["verdict"] = None
    elif -margin <= lo and hi <= margin:
        result["verdict"] = "practically_invariant"
    else:
        result["verdict"] = "not_practically_invariant"
    return result


# ---------------------------------------------------------------------------
# Ambiguity regression: D_pair ~ intercept + beta * baseline_strength (item 15)
# ---------------------------------------------------------------------------

def weighted_least_squares(xs, ys, weights):
    """Simple 1-variable weighted least squares. Returns (intercept, beta),
    or None if fewer than 2 distinct weighted x-values (an undefined slope)."""
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    x_bar = sum(w * x for w, x in zip(weights, xs)) / total_weight
    y_bar = sum(w * y for w, y in zip(weights, ys)) / total_weight
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(weights, xs))
    if sxx == 0:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(weights, xs, ys))
    beta = sxy / sxx
    intercept = y_bar - beta * x_bar
    return intercept, beta


def fit_ambiguity_regression(pair_rows):
    """pair_rows: list of {"story_1_id", "story_2_id", "d_pair", "baseline_strength"}
    for the pairs with both values available. Point estimate is an
    UNWEIGHTED fit (every pair counted once); returns None if underdetermined."""
    if len(pair_rows) < 2:
        return None
    xs = [r["baseline_strength"] for r in pair_rows]
    ys = [r["d_pair"] for r in pair_rows]
    fit = weighted_least_squares(xs, ys, [1.0] * len(pair_rows))
    if fit is None:
        return None
    intercept, beta = fit
    return {"intercept": intercept, "beta": beta}


def ambiguity_regression_bootstrap(pair_rows, story_ids, n_draws, rng):
    """Story-level bootstrap CI for beta: each draw refits the weighted
    regression with weight m_i*m_j per pair (0 excludes it), skipping draws
    with fewer than 2 unique stories or an underdetermined fit. Returns a
    flat list of per-draw beta values."""
    betas = []
    for _ in range(n_draws):
        resampled = rng.choices(story_ids, k=len(story_ids))
        multiplicities = Counter(resampled)
        if len({sid for sid in resampled}) < 2:
            continue
        weights, xs, ys = [], [], []
        for r in pair_rows:
            w = multiplicities.get(r["story_1_id"], 0) * multiplicities.get(r["story_2_id"], 0)
            if w:
                weights.append(w)
                xs.append(r["baseline_strength"])
                ys.append(r["d_pair"])
        fit = weighted_least_squares(xs, ys, weights) if weights else None
        if fit is not None:
            betas.append(fit[1])
    return betas


def predict_at_percentiles(intercept, beta, baseline_strength_values, percentiles=(25, 50, 75)):
    """Estimated context effect (intercept + beta*x) at each named
    percentile of the OBSERVED baseline_strength distribution."""
    s = sorted(baseline_strength_values)
    return {p: intercept + beta * percentile(s, p) for p in percentiles} if s else {p: None for p in percentiles}
