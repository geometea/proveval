"""Simulation-based DESIGN / POWER tool for the v3 families. A
methodological utility: it never touches a manifest or a lock and its
recommendations are for a human to read.

Generative model (hierarchical, all parameters in PARAMS with documented
defaults that can later be replaced by real v2/v3 estimates):
  story quality      q_s ~ N(0, sigma_story)                         12 stories
  pair baseline      b_p = q_i - q_j + N(0, sigma_pair)               logit scale
  position bias      +pi when story 1 is displayed as Passage A
  cue effect         gamma_c  (logit shift toward the cue-favoured story)
  intervention       multiplier m_I on the context effect (I0 = 1)
  intervention x cue interaction  delta_Ic ~ N(0, sigma_ixc)
  ambiguity          context effect x (1 + kappa * exp(-|b_p|))
  no-context drift   intervention I perturbs quality by N(0, drift_I)
  attack recovery    extra logit shift r_I under the strongest attack
  dose response      logit shift beta_I * x(dose), x = logit(dose/100)
  cross-evaluator    profile k scales every context effect by s_k and
                     mixes in an independent quality vector with weight w_k
  replicate noise    Bernoulli sampling of each judgment

Estimators mirror the real analysis (cell rates -> D_pair -> pooled CE,
signed suppression, noise-corrected drift, attack recovery, WLS dose slope
and +10pp threshold, story-level bootstrap CIs) with a reduced draw count.
"""

import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict

import controllability_v3_stats as st
import controllability_v3_stress_stats as ss

PARAMS = {
    "sigma_story": 0.9, "sigma_pair": 0.3, "position_bias": 0.15,
    "cue_effects": {"c1": 1.0, "c2": 0.8, "c3": 0.9, "c4": 0.7, "c5": 0.6},
    "intervention_multipliers": {"I0": 1.0, "I1": 0.7, "I2": 0.4, "I3": 0.5, "I4": 0.15, "I5": 0.2, "I6": 0.55, "I7": 0.3},
    "sigma_ixc": 0.1, "kappa_ambiguity": 0.5,
    "drift": {"I0": 0.0, "I1": 0.1, "I2": 0.1, "I3": 0.1, "I4": 0.15, "I5": 0.6, "I6": 0.1, "I7": 0.4},
    "attack_recovery": {"I0": 0.5, "I1": 0.5, "I2": 0.4, "I3": 0.4, "I4": 0.05, "I5": 0.3, "I6": 0.4, "I7": 0.3},
    "dose_slopes": {"I0": 0.5, "I1": 0.35, "I2": 0.2, "I3": 0.25, "I4": 0.05, "I5": 0.1, "I6": 0.3, "I7": 0.15},
    "profile_context_scale": {"A": 1.0, "B": 0.6}, "profile_quality_mix": {"A": 0.0, "B": 0.4},
    "null_interventions": ["I4"],   # interventions whose TRUE attack recovery / slope is ~0 (for false-positive rates)
}

DEFAULT_DESIGNS = [
    {"name": "primary_r2", "replicates": 2, "n_pairs": 66, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 60, 70, 80, 90, 99], "dev_replicates": 1, "top_k": 3, "profiles": ["A"]},
    {"name": "primary_r3", "replicates": 3, "n_pairs": 66, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 60, 70, 80, 90, 99], "dev_replicates": 1, "top_k": 3, "profiles": ["A"]},
    {"name": "primary_r5", "replicates": 5, "n_pairs": 66, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 60, 70, 80, 90, 99], "dev_replicates": 1, "top_k": 3, "profiles": ["A"]},
    {"name": "primary_r10", "replicates": 10, "n_pairs": 66, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 60, 70, 80, 90, 99], "dev_replicates": 1, "top_k": 3, "profiles": ["A"]},
    {"name": "subset_24pairs_r5", "replicates": 5, "n_pairs": 24, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 60, 70, 80, 90, 99], "dev_replicates": 1, "top_k": 3, "profiles": ["A"]},
    {"name": "dose4_r5", "replicates": 5, "n_pairs": 66, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 70, 90, 99], "dev_replicates": 1, "top_k": 3, "profiles": ["A"]},
    {"name": "five_interventions_r3", "replicates": 3, "n_pairs": 66, "interventions": ["I0", "I1", "I4", "I5", "I7"], "dose_grid": [51, 80, 99], "dev_replicates": 1, "top_k": 2, "profiles": ["A"]},
    {"name": "capability_24pairs_r3_2profiles", "replicates": 3, "n_pairs": 24, "interventions": ["I0", "I1", "I4", "I5", "I7"], "dose_grid": [51, 80, 99], "dev_replicates": 1, "top_k": 1, "profiles": ["A", "B"]},
    {"name": "attack_dev_r2_k2", "replicates": 3, "n_pairs": 66, "interventions": list("I0 I1 I2 I3 I4 I5 I6 I7".split()), "dose_grid": [51, 60, 70, 80, 90, 99], "dev_replicates": 2, "top_k": 2, "profiles": ["A"]},
]

PRECISION_TARGETS = {"suppression_ci_width": 0.15, "drift_ci_width": 0.15, "attack_recovery_ci_width": 0.20, "dose_slope_ci_width": 0.06, "min_coverage": 0.90, "min_ordering_probability": 0.8}

OUTPUT_FILES = ("design_power_summary.csv", "design_precision_summary.csv", "design_cost_runtime_summary.csv", "design_recommendation.json")

STORIES = [f"s{i:02d}" for i in range(12)]
ALL_PAIRS = [(a, b) for i, a in enumerate(STORIES) for b in STORIES[i + 1:]]


def sigmoid(x):
    return 1 / (1 + math.exp(-max(-30, min(30, x))))


class World:
    """One draw of the hierarchical truth."""

    def __init__(self, params, rng):
        self.p = params
        self.q = {s: rng.gauss(0, params["sigma_story"]) for s in STORIES}
        self.q_alt = {s: rng.gauss(0, params["sigma_story"]) for s in STORIES}
        self.pair_noise = {pr: rng.gauss(0, params["sigma_pair"]) for pr in ALL_PAIRS}
        self.ixc = {(i, c): rng.gauss(0, params["sigma_ixc"]) for i in params["intervention_multipliers"] for c in params["cue_effects"]}
        self.drift_perturb = {(i, s): rng.gauss(0, 1) for i in params["drift"] for s in STORIES}

    def quality(self, s, profile, intervention):
        w = self.p["profile_quality_mix"][profile]
        q = (1 - w) * self.q[s] + w * self.q_alt[s]
        return q + self.p["drift"][intervention] * self.drift_perturb[(intervention, s)]

    def baseline_logit(self, pair, profile, intervention):
        return self.quality(pair[0], profile, intervention) - self.quality(pair[1], profile, intervention) + self.pair_noise[pair]

    def context_shift(self, pair, cue, intervention, profile, attack=False, dose=None):
        b = self.baseline_logit(pair, profile, "I0")
        amb = 1 + self.p["kappa_ambiguity"] * math.exp(-abs(b))
        scale = self.p["profile_context_scale"][profile]
        if dose is not None:
            return self.p["dose_slopes"][intervention] * ss.dose_x_of(dose) * scale * amb
        shift = (self.p["cue_effects"][cue] + self.ixc[(intervention, cue)]) * self.p["intervention_multipliers"][intervention]
        if attack:
            shift += self.p["attack_recovery"][intervention]
        return shift * scale * amb

    def p_story1(self, pair, intervention, profile, cue=None, assignment=None, position="story1_as_a", attack=False, dose=None):
        logit = self.baseline_logit(pair, profile, intervention) + (self.p["position_bias"] if position == "story1_as_a" else -self.p["position_bias"])
        if cue is not None or dose is not None:
            sign = 1 if assignment == "forward" else -1
            logit += sign * self.context_shift(pair, cue, intervention, profile, attack, dose)
        return sigmoid(logit)

    # true estimands (large-sample, per world)
    def true_pooled_ce(self, intervention, profile, attack=False):
        vals = []
        for pair in ALL_PAIRS:
            for cue in self.p["cue_effects"]:
                d = [self.p_story1(pair, intervention, profile, cue, "forward", pos, attack) - self.p_story1(pair, intervention, profile, cue, "flipped", pos, attack) for pos in ("story1_as_a", "story2_as_a")]
                vals.append(statistics.mean(d))
        return statistics.mean(vals)


def simulate_rates(world, design, profile, rng, pairs):
    """Return dict of cell rates the estimators consume."""
    R = design["replicates"]

    def draw(p, n=R):
        return sum(rng.random() < p for _ in range(n)) / n

    ctx = {}   # (I, cue, pair) -> {(asg, pos): rate}
    atk = {}
    noctx = {}  # I -> pair -> {pos: {"p", "n"}}
    dose = {}   # I -> dose -> pair -> d_pair
    for I in design["interventions"]:
        noctx[I] = {pair: {pos: {"p": draw(world.p_story1(pair, I, profile, position=pos)), "n": R} for pos in ("story1_as_a", "story2_as_a")} for pair in pairs}
        for pair in pairs:
            for cue in world.p["cue_effects"]:
                ctx[(I, cue, pair)] = {(a, pos): draw(world.p_story1(pair, I, profile, cue, a, pos)) for a in ("forward", "flipped") for pos in ("story1_as_a", "story2_as_a")}
                atk[(I, cue, pair)] = {(a, pos): draw(world.p_story1(pair, I, profile, cue, a, pos, attack=True), max(1, design["replicates"])) for a in ("forward", "flipped") for pos in ("story1_as_a", "story2_as_a")}
            for n in design["dose_grid"]:
                d = {(a, pos): draw(world.p_story1(pair, I, profile, "c1", a, pos, dose=n)) for a in ("forward", "flipped") for pos in ("story1_as_a", "story2_as_a")}
                dose.setdefault(I, {}).setdefault(n, {})[pair] = st.d_pair_from_cell_rates(d)["d_pair"]
    return {"ctx": ctx, "atk": atk, "noctx": noctx, "dose": dose}


def pooled_pairs(cells, I, pairs, cues):
    by_cue = {}
    for cue in cues:
        by_cue[cue] = {pair: st.d_pair_from_cell_rates(cells[(I, cue, pair)])["d_pair"] for pair in pairs}
    return st.pooled_over_cues(by_cue)


def analyse(world, design, rates_by_profile, pairs, resamples, rng):
    """Estimates + CIs mirroring the real analysis, plus truth for scoring."""
    cues = list(world.p["cue_effects"])
    out = {}
    for profile, rates in rates_by_profile.items():
        est = {"ce": {}, "supp": {}, "drift": {}, "attack": {}, "slope": {}, "threshold": {}}
        control = pooled_pairs(rates["ctx"], "I0", pairs, cues)
        for I in design["interventions"]:
            treated = pooled_pairs(rates["ctx"], I, pairs, cues)
            sup = st.suppression_estimates(control, treated, resamples)
            est["ce"][I] = (sup["CE_treated"], sup["CE_treated_ci_95_lo"], sup["CE_treated_ci_95_hi"], world.true_pooled_ce(I, profile))
            est["supp"][I] = (sup["signed_suppression"], sup["signed_suppression_ci_95_lo"], sup["signed_suppression_ci_95_hi"], world.true_pooled_ce("I0", profile) - world.true_pooled_ce(I, profile))
            dr = st.drift_estimates(rates["noctx"]["I0"], rates["noctx"][I], resamples, same_sample=(I == "I0"))
            est["drift"][I] = (dr["rms_prob_shift_noise_corrected"], dr["rms_prob_shift_noise_corrected_ci_95_lo"], dr["rms_prob_shift_noise_corrected_ci_95_hi"], None)
            attacked = pooled_pairs(rates["atk"], I, pairs, cues)
            rec = ss.attack_recovery(treated, attacked, control, resamples)
            est["attack"][I] = (rec["attack_recovery"], rec["attack_recovery_ci_95_lo"], rec["attack_recovery_ci_95_hi"], world.true_pooled_ce(I, profile, attack=True) - world.true_pooled_ce(I, profile))
            sl = ss.susceptibility_slope(rates["dose"][I], resamples, 0.10, tuple(design["dose_grid"]))
            true_slope = _true_slope(world, I, profile, pairs)
            est["slope"][I] = (sl["slope"], sl["slope_ci_95_lo"], sl["slope_ci_95_hi"], true_slope)
            est["threshold"][I] = (sl["threshold_dose"], sl["threshold_dose_ci_95_lo"], sl["threshold_dose_ci_95_hi"], _true_threshold(world, I, profile, pairs))
        out[profile] = est
    return out


def _true_slope(world, I, profile, pairs):
    xs, ys = [], []
    for n in (51, 60, 70, 80, 90, 99):
        for pair in pairs:
            d = statistics.mean(world.p_story1(pair, I, profile, "c1", "forward", pos, dose=n) - world.p_story1(pair, I, profile, "c1", "flipped", pos, dose=n) for pos in ("story1_as_a", "story2_as_a"))
            xs.append(ss.dose_x_of(n)); ys.append(d)
    fit = st.weighted_least_squares(xs, ys, [1.0] * len(xs))
    return fit[1] if fit else None


def _true_threshold(world, I, profile, pairs):
    xs, ys = [], []
    for n in (51, 60, 70, 80, 90, 99):
        for pair in pairs:
            d = statistics.mean(world.p_story1(pair, I, profile, "c1", "forward", pos, dose=n) - world.p_story1(pair, I, profile, "c1", "flipped", pos, dose=n) for pos in ("story1_as_a", "story2_as_a"))
            xs.append(ss.dose_x_of(n)); ys.append(d)
    fit = st.weighted_least_squares(xs, ys, [1.0] * len(xs))
    return ss.threshold_dose(fit[0], fit[1], 0.10) if fit else None


def _metric(records):
    """records: [(est, lo, hi, truth)] -> bias, rmse, ci width, coverage."""
    valid = [r for r in records if r[0] is not None]
    if not valid:
        return {"bias": None, "rmse": None, "ci_width": None, "coverage": None, "n": 0}
    widths = [r[2] - r[1] for r in valid if r[1] is not None and r[2] is not None]
    with_truth = [r for r in valid if r[3] is not None]
    bias = statistics.mean(r[0] - r[3] for r in with_truth) if with_truth else None
    rmse = math.sqrt(statistics.mean((r[0] - r[3]) ** 2 for r in with_truth)) if with_truth else None
    cover = statistics.mean(1.0 if (r[1] is not None and r[2] is not None and r[1] <= r[3] <= r[2]) else 0.0 for r in with_truth) if with_truth else None
    return {"bias": bias, "rmse": rmse, "ci_width": statistics.mean(widths) if widths else None, "coverage": cover, "n": len(valid)}


def evaluate_design(design, params, n_sims, n_draws, seed):
    rng = random.Random(seed)
    records = defaultdict(list)
    ordering_hits, best_hits, vuln_hits, vuln_total, fp_hits, fp_total, fn_hits, fn_total = 0, 0, 0, 0, 0, 0, 0, 0
    cross = defaultdict(list)
    for sim in range(n_sims):
        world = World(params, rng)
        pairs = ALL_PAIRS if design["n_pairs"] >= len(ALL_PAIRS) else rng.sample(ALL_PAIRS, design["n_pairs"])
        story_ids = sorted({s for p in pairs for s in p})
        resamples = st.make_story_resamples(story_ids, n_draws, random.Random(rng.random()))
        rates = {pf: simulate_rates(world, design, pf, rng, pairs) for pf in design["profiles"]}
        est = analyse(world, design, rates, pairs, resamples, rng)
        A = est[design["profiles"][0]]
        for name in ("ce", "supp", "drift", "attack", "slope", "threshold"):
            for I, rec in A[name].items():
                records[(name, I)].append(rec)
        # ordering: interventions (excluding I0) by true suppression vs estimated
        others = [I for I in design["interventions"] if I != "I0"]
        if len(others) >= 2:
            true_order = sorted(others, key=lambda I: -A["supp"][I][3])
            est_order = sorted(others, key=lambda I: -(A["supp"][I][0] or 0))
            ordering_hits += true_order == est_order
            best_hits += true_order[0] == est_order[0]
        for I in design["interventions"]:
            truth = A["attack"][I][3]
            lo = A["attack"][I][1]
            if I in params["null_interventions"]:
                fp_total += 1
                fp_hits += (lo is not None and lo > 0)
            else:
                vuln_total += 1
                detected = (lo is not None and lo > 0)
                vuln_hits += detected
                fn_total += 1
                fn_hits += (not detected)
        if len(design["profiles"]) >= 2:
            B = est[design["profiles"][1]]
            for I in design["interventions"]:
                cross[("delta_supp", I)].append((A["supp"][I][0] - B["supp"][I][0], None, None, A["supp"][I][3] - B["supp"][I][3]))
    summary = {"design": design["name"], "n_sims": n_sims}
    precision = {}
    for (name, I), recs in records.items():
        precision[(name, I)] = _metric(recs)
    power = {"p_correct_full_ordering": ordering_hits / n_sims if n_sims else None, "p_identify_best_suppression": best_hits / n_sims if n_sims else None,
             "p_detect_attack_vulnerability": vuln_hits / vuln_total if vuln_total else None, "false_positive_rate_null_attack": fp_hits / fp_total if fp_total else None,
             "false_negative_rate_attack": fn_hits / fn_total if fn_total else None}
    if cross:
        power["cross_profile_delta_suppression_rmse"] = statistics.mean(_metric(v)["rmse"] or 0 for v in cross.values())
    return summary, precision, power


def cost_of(design, judgments_per_second=None):
    n_pairs = design["n_pairs"]
    nI = len(design["interventions"])
    primary = n_pairs * 5 * nI * 4 * design["replicates"] + n_pairs * nI * 2 * design["replicates"]
    dose = n_pairs * len(design["dose_grid"]) * nI * 4 * design["replicates"]
    attack_dev = 320 * 22 * 4 * design["dev_replicates"] * (nI / 8)
    attack_eval = design["top_k"] * 5 * nI * 44 * 4 * 3
    total = (primary + dose + attack_dev + attack_eval) * len(design["profiles"])
    out = {"primary_judgments": primary, "dose_judgments": dose, "attack_dev_judgments": int(attack_dev), "attack_eval_judgments": attack_eval, "profiles": len(design["profiles"]), "total_judgments": int(total)}
    return out


def run_simulation(designs=None, params=None, n_sims=20, n_draws=200, seed=20260921, out_dir="results/controllability_v3/design_simulation"):
    from controllability_v3_runtime import plan_scenarios
    designs = designs or DEFAULT_DESIGNS
    params = params or PARAMS
    os.makedirs(out_dir, exist_ok=True)
    power_rows, precision_rows, cost_rows = [], [], []
    for design in designs:
        summary, precision, power = evaluate_design(design, params, n_sims, n_draws, seed)
        power_rows.append({**summary, "replicates": design["replicates"], "n_pairs": design["n_pairs"], "n_interventions": len(design["interventions"]),
                           "dose_points": len(design["dose_grid"]), "dev_replicates": design["dev_replicates"], "top_k": design["top_k"], "profiles": len(design["profiles"]), **power})
        for (name, I), m in sorted(precision.items()):
            precision_rows.append({"design": design["name"], "quantity": name, "intervention_id": I, **m})
        cost = cost_of(design)
        plan = plan_scenarios(cost["total_judgments"])
        cost_rows.append({"design": design["name"], **cost, **{f"hours_{k}": v["hours"] for k, v in plan["scenarios"].items()},
                          "throughput_basis": "; ".join(f"{k}={v['judgments_per_second']:.2f} j/s ({v['basis']})" for k, v in plan["scenarios"].items())})
    recommendation = recommend(power_rows, precision_rows, cost_rows)
    _write(power_rows, os.path.join(out_dir, "design_power_summary.csv"))
    _write(precision_rows, os.path.join(out_dir, "design_precision_summary.csv"))
    _write(cost_rows, os.path.join(out_dir, "design_cost_runtime_summary.csv"))
    with open(os.path.join(out_dir, "design_recommendation.json"), "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    try:
        write_plots(power_rows, precision_rows, out_dir)
    except Exception as e:  # plots are a convenience, never a failure
        recommendation["plot_error"] = str(e)
    return {"power": power_rows, "precision": precision_rows, "cost": cost_rows, "recommendation": recommendation}


def recommend(power_rows, precision_rows, cost_rows, targets=PRECISION_TARGETS):
    """Smallest (by total judgments) design meeting every precision target;
    never applied automatically -- a human reads it."""
    by_design = defaultdict(dict)
    for r in precision_rows:
        by_design[r["design"]][(r["quantity"], r["intervention_id"])] = r
    cost = {r["design"]: r["total_judgments"] for r in cost_rows}
    verdicts = []
    for p in power_rows:
        d = p["design"]
        prec = by_design[d]
        def worst(quantity, key):
            vals = [m[key] for (q, I), m in prec.items() if q == quantity and I != "I0" and m[key] is not None]
            return max(vals) if key == "ci_width" and vals else (min(vals) if vals else None)
        checks = {
            "suppression_ci_width": (worst("supp", "ci_width"), targets["suppression_ci_width"], "<="),
            "drift_ci_width": (worst("drift", "ci_width"), targets["drift_ci_width"], "<="),
            "attack_recovery_ci_width": (worst("attack", "ci_width"), targets["attack_recovery_ci_width"], "<="),
            "dose_slope_ci_width": (worst("slope", "ci_width"), targets["dose_slope_ci_width"], "<="),
            "suppression_coverage": (worst("supp", "coverage"), targets["min_coverage"], ">="),
            "ordering_probability": (p["p_identify_best_suppression"], targets["min_ordering_probability"], ">="),
        }
        ok = all(v is not None and ((v <= t) if op == "<=" else (v >= t)) for v, t, op in checks.values())
        verdicts.append({"design": d, "meets_all_targets": ok, "total_judgments": cost.get(d), "checks": {k: {"value": v, "target": t, "op": op} for k, (v, t, op) in checks.items()}})
    meeting = [v for v in verdicts if v["meets_all_targets"]]
    best = min(meeting, key=lambda v: v["total_judgments"]) if meeting else None
    return {"targets": targets, "verdicts": verdicts, "recommended_design": best["design"] if best else None,
            "note": "Recommendation is advisory; no experimental design is changed automatically. Absent a design meeting every target, the verdict table shows which target fails."}


def _write(rows, path):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_plots(power_rows, precision_rows, out_dir):
    from controllability_v2_plots import COLOR_A, COLOR_B, grouped_bar_chart_with_ci
    cats = [r["design"] for r in power_rows]
    grouped_bar_chart_with_ci("Design power: ordering / attack detection", cats,
                              [{"name": "P(identify best suppression)", "color": COLOR_A, "values": [r["p_identify_best_suppression"] for r in power_rows], "ci_lo": [None] * len(cats), "ci_hi": [None] * len(cats)},
                               {"name": "P(detect attack vulnerability)", "color": COLOR_B, "values": [r["p_detect_attack_vulnerability"] for r in power_rows], "ci_lo": [None] * len(cats), "ci_hi": [None] * len(cats)}],
                              "probability", os.path.join(out_dir, "design_power.svg"))
    width = defaultdict(dict)
    for r in precision_rows:
        if r["intervention_id"] == "I1":
            width[r["design"]][r["quantity"]] = r["ci_width"]
    grouped_bar_chart_with_ci("Design precision (I1): 95% CI widths", cats,
                              [{"name": "suppression", "color": COLOR_A, "values": [width[d].get("supp") for d in cats], "ci_lo": [None] * len(cats), "ci_hi": [None] * len(cats)},
                               {"name": "attack recovery", "color": COLOR_B, "values": [width[d].get("attack") for d in cats], "ci_lo": [None] * len(cats), "ci_hi": [None] * len(cats)}],
                              "CI width", os.path.join(out_dir, "design_precision.svg"))
