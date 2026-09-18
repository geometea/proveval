"""v3 experimental family: ADAPTIVE dose response (stress_dose_adaptive).

Distinct from, and additive to, the frozen fixed-grid dose experiment. Same
numeric cue sentence (controllability_v3_stress_design.dose_intro), same
counterbalancing (favoured story x display position = 4 cells), same
frozen intervention wording; only WHICH doses are sampled is decided
adaptively.

Adaptive unit = (intervention, ambiguity stratum). A curve per pair x
intervention would rest on 4 binary observations per round and is
hopelessly noisy; pooling every pair into one curve per intervention hides
the baseline-strength dependence that is the point of the ambiguity
hypothesis. Strata are the tertiles of blind baseline strength (from the
primary no-context I0 rows, a frozen input recorded in the state file);
when no blind baseline exists yet a single pooled stratum is used and the
state records that. Within a unit every round samples every pair of the
stratum at one dose x 4 cells x 1 replicate, so story-level inference is
preserved and each round adds a balanced block.

Model (pre-specified): per unit, logistic regression
    logit P(context-favoured passage chosen) = alpha + beta * x,
    x = logit(dose / (100 - dose))   (x = 0 at a 50/50 split = zero evidence)
fitted by IRLS on all accumulated observations of the unit.
Targets: d10 / d25 = dose where P(x) - P(0) reaches +0.10 / +0.25;
d50 = dose where P = 0.5 (only meaningful when P(0) < 0.5, i.e. the
favoured passage is the baseline-dispreferred one on average in the unit).

Next-dose rule (deterministic, explicit; quantile-sampling design after
Wu 1985): anchors 51, 70, 90, 99 first. Afterwards the targets are
cycled (d10, d25, d10, ...) and the next dose is the ALLOWED dose nearest
to the current point estimate of that target whose per-unit visit cap is
not reached (ties -> lower dose). Sampling at the current quantile
estimate converges to the quantile and keeps every observation in the
transition region; the one-step "minimise threshold variance" criterion
was evaluated and rejected because it prefers the extreme doses (the
threshold variance is dominated by the slope, which extreme leverage pins
best) -- exactly the doses least informative about the transition. Its
value is still recorded per candidate as a diagnostic. When the fit is not
identifiable (beta <= 0) the least-visited anchor is revisited.

Stopping (pre-specified, checked in this order after every completed round):
  1. hard budget: rounds_done >= max_rounds_per_unit
  2. precision: the d10 95% CI width (dose points) <= target_ci_width
  3. flat: after >= min_rounds_for_flat rounds the slope's 95% CI upper
     bound < min_slope_of_interest (no detectable dose response)
  4. out of range: after >= min_rounds_for_flat rounds the d10 point
     estimate and its whole CI lie above the maximum allowed dose
plus a global hard cap on total judgments.

Every decision is appended to the schedule (state before, candidates with
their criterion values, chosen dose, reason, data available). Resume
REPLAYS the raw observations in schedule order and re-derives every
decision; a mismatch with the recorded schedule is a hard error.
"""

import hashlib
import json
import math
import os
import statistics
from collections import defaultdict

import controllability_v3_stress_design as sd
import controllability_v3_stress_stats as ss
from controllability_v3_design import INTERVENTION_IDS
from context_trials import ITEMS_FILE, load_items, story_pairs

FAMILY = "stress_dose_adaptive"
ID_PREFIX = "v3a"
CUE_ID = "reader_consensus_numeric"

ALLOWED_DOSES = (51, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99)
ANCHOR_DOSES = (51, 70, 90, 99)
TARGETS = {"d10": 0.10, "d25": 0.25}
PRIMARY_TARGET = "d10"

DEFAULT_POLICY = {
    "allowed_doses": list(ALLOWED_DOSES),
    "anchor_doses": list(ANCHOR_DOSES),
    "primary_target": PRIMARY_TARGET,
    "targets": dict(TARGETS),
    "max_rounds_per_unit": 12,
    "max_visits_per_dose": 3,
    "target_ci_width_dose_points": 10.0,
    "min_rounds_for_flat": 6,
    "min_slope_of_interest": 0.05,
    "n_strata": 3,
    "stratification": "blind baseline strength tertiles from primary no-context I0 when available, else one pooled stratum",
    "interventions": list(INTERVENTION_IDS),
    "next_dose_rule": "after the anchors, cycle targets (d10, d25) and sample the allowed dose nearest to the current estimate of that target (visit cap respected; ties -> lower dose); variance criterion recorded as a diagnostic only",
    "target_cycle": ["d10", "d25"],
    "global_max_judgments": None,   # filled by design_arithmetic at config time
}


def dose_x(dose):
    return math.log(dose / (100 - dose))


def dose_from_x(x):
    return 100 * ss.sigmoid(x)


# ---------------------------------------------------------------------------
# Ids and cells
# ---------------------------------------------------------------------------

def make_trial_id(pair_id, assignment, position, intervention_id, dose):
    return "::".join([ID_PREFIX, FAMILY, pair_id, CUE_ID, assignment, position, intervention_id, sd.dose_id(dose), "ctx"])


def make_observation_id(trial_id, visit):
    """visit = the k-th time this unit scheduled this dose (1-based); it is
    the replicate index of the cell."""
    return f"{trial_id}::r{visit}"


def build_round_cells(pairs, intervention_id, dose, visit, texts):
    cells = []
    for s1, s2 in pairs:
        for c in sd.build_stress_block(FAMILY, s1, s2, CUE_ID, intervention_id, sd.dose_id(dose),
                                       lambda f, u, n=dose: sd.dose_intro(n, f, u), texts,
                                       {"dose": dose, "dose_logit": dose_x(dose), "cue_family": "reader_consensus", "visit": visit}):
            c["trial_id"] = make_trial_id(c["pair_id"], c["assignment"], c["position"], intervention_id, dose)
            c["block_id"] = "::".join([ID_PREFIX, FAMILY, c["pair_id"], intervention_id, sd.dose_id(dose), f"v{visit}"])
            cells.append(c)
    return cells


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def assign_strata(pair_ids, baseline_strength, n_strata):
    """{pair_id: stratum} by strength tertiles; a single 'all' stratum when
    no baseline is available."""
    if not baseline_strength:
        return {p: "all" for p in pair_ids}
    ordered = sorted(pair_ids, key=lambda p: (baseline_strength.get(p, 0.0), p))
    labels = ["ambiguous", "middle", "strong"] if n_strata == 3 else [f"s{i}" for i in range(n_strata)]
    return {p: labels[min(n_strata - 1, i * n_strata // len(ordered))] for i, p in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_unit(observations):
    """observations: [(x, favoured_won:int, n:int)] aggregated or binary.
    Returns (alpha, beta, cov 2x2) or None."""
    cells = [{"b": 0.0, "x": x, "y": y, "n": n} for x, y, n in observations]
    if not cells or len({round(c["x"], 6) for c in cells}) < 2:
        return None
    return _fit_logistic_2(cells)


def _fit_logistic_2(cells, max_iter=60, tol=1e-9, ridge=1e-6):
    """2-parameter IRLS: logit p = a + b x on binomial cells. Returns
    (a, b, cov) with cov the inverse observed information (with ridge)."""
    a, b = 0.0, 0.0
    for _ in range(max_iter):
        h = [[ridge, 0.0], [0.0, ridge]]
        g = [-ridge * a, -ridge * b]
        for c in cells:
            eta = max(-30.0, min(30.0, a + b * c["x"]))
            mu = ss.sigmoid(eta)
            r = c["y"] - c["n"] * mu
            v = c["n"] * mu * (1 - mu)
            g[0] += r
            g[1] += r * c["x"]
            h[0][0] += v
            h[0][1] += v * c["x"]
            h[1][1] += v * c["x"] ** 2
        h[1][0] = h[0][1]
        det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
        if det <= 1e-12:
            return None
        inv = [[h[1][1] / det, -h[0][1] / det], [-h[1][0] / det, h[0][0] / det]]
        da, db = inv[0][0] * g[0] + inv[0][1] * g[1], inv[1][0] * g[0] + inv[1][1] * g[1]
        a, b = a + da, b + db
        if max(abs(da), abs(db)) < tol:
            break
    h = [[ridge, 0.0], [0.0, ridge]]
    for c in cells:
        mu = ss.sigmoid(max(-30.0, min(30.0, a + b * c["x"])))
        v = c["n"] * mu * (1 - mu)
        h[0][0] += v; h[0][1] += v * c["x"]; h[1][1] += v * c["x"] ** 2
    h[1][0] = h[0][1]
    det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
    cov = [[h[1][1] / det, -h[0][1] / det], [-h[1][0] / det, h[0][0] / det]]
    return a, b, cov


def target_x(a, b, gain):
    """x at which P(x) - P(0) = gain; None if unreachable / beta <= 0."""
    p0 = ss.sigmoid(a)
    p_target = p0 + gain
    if b <= 0 or p_target >= 1:
        return None
    return (math.log(p_target / (1 - p_target)) - a) / b


def target_dose_and_se(a, b, cov, gain):
    """Delta-method SE of the target x, then dose and a 95% CI in dose
    points. Returns dict or None."""
    x = target_x(a, b, gain)
    if x is None:
        return None
    p0 = ss.sigmoid(a)
    pt = p0 + gain
    # x = (logit(pt) - a)/b with pt = sigmoid(a) + gain
    dlogit_dpt = 1 / (pt * (1 - pt))
    dpt_da = p0 * (1 - p0)
    dx_da = (dlogit_dpt * dpt_da - 1) / b
    dx_db = -(math.log(pt / (1 - pt)) - a) / (b * b)
    var = dx_da ** 2 * cov[0][0] + 2 * dx_da * dx_db * cov[0][1] + dx_db ** 2 * cov[1][1]
    se = math.sqrt(max(var, 0.0))
    lo, hi = dose_from_x(x - 1.96 * se), dose_from_x(x + 1.96 * se)
    return {"x": x, "se_x": se, "dose": dose_from_x(x), "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo}


def reversal_dose(a, b, cov):
    """dose at which P = 0.5 (x = -a/b), only when P(0) < 0.5 and b > 0."""
    if b <= 0 or ss.sigmoid(a) >= 0.5:
        return None
    x = -a / b
    dx_da, dx_db = -1 / b, a / (b * b)
    var = dx_da ** 2 * cov[0][0] + 2 * dx_da * dx_db * cov[0][1] + dx_db ** 2 * cov[1][1]
    se = math.sqrt(max(var, 0.0))
    return {"x": x, "se_x": se, "dose": dose_from_x(x), "ci_lo": dose_from_x(x - 1.96 * se), "ci_hi": dose_from_x(x + 1.96 * se)}


def expected_variance_after(a, b, cov, dose, n_round, gain):
    """Delta-method variance of the target x after adding n_round expected
    observations at `dose` (Fisher information update at the current fit)."""
    x = dose_x(dose)
    mu = ss.sigmoid(max(-30.0, min(30.0, a + b * x)))
    v = n_round * mu * (1 - mu)
    det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
    if det <= 0:
        return float("inf")
    info = [[cov[1][1] / det + v, -cov[0][1] / det + v * x], [-cov[1][0] / det + v * x, cov[0][0] / det + v * x * x]]
    d2 = info[0][0] * info[1][1] - info[0][1] * info[1][0]
    if d2 <= 0:
        return float("inf")
    new_cov = [[info[1][1] / d2, -info[0][1] / d2], [-info[1][0] / d2, info[0][0] / d2]]
    t = target_dose_and_se(a, b, new_cov, gain)
    return t["se_x"] ** 2 if t else float("inf")


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def unit_observations(rows):
    """Aggregate valid rows of one unit into [(x, favoured_won, n)] per dose."""
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        chosen = r.get("parsed_choice")
        if chosen not in ("A", "B"):
            continue
        story1 = (r["story_a"] if chosen == "A" else r["story_b"]) == r["story_1_id"]
        fav_won = story1 if r["context_assignment"] == "forward" else (not story1)
        agg[r["dose"]][0 if fav_won else 1] += 1
    return [(dose_x(d), w, w + l) for d, (w, l) in sorted(agg.items())]


def decide_next_dose(policy, rounds_done, visits, unit_rows, n_pairs):
    """Returns a decision dict: {"dose", "reason", "candidates", "fit", "stop"}.
    Deterministic given the accumulated rows."""
    fit = None
    obs = unit_observations(unit_rows)
    if obs:
        fit = fit_unit(obs)
    n_round = n_pairs * 4
    decision = {"rounds_done": rounds_done, "n_observations": sum(n for _, _, n in obs), "fit": None, "candidates": [], "stop": None}
    if fit is not None:
        a, b, cov = fit
        t = target_dose_and_se(a, b, cov, policy["targets"][policy["primary_target"]])
        decision["fit"] = {"alpha": a, "beta": b, "beta_se": math.sqrt(max(cov[1][1], 0.0)), "primary_target": t}
    # --- stopping: the hard budget always applies; every other rule is
    # evaluated only once all anchor rounds have been visited
    anchors_complete = rounds_done >= len(policy["anchor_doses"])
    if rounds_done >= policy["max_rounds_per_unit"]:
        decision["stop"] = "max_rounds_reached"
    elif anchors_complete and fit is not None and decision["fit"]["primary_target"] and decision["fit"]["primary_target"]["ci_width"] <= policy["target_ci_width_dose_points"]:
        decision["stop"] = "threshold_ci_width_below_target"
    elif anchors_complete and fit is not None and rounds_done >= policy["min_rounds_for_flat"] and (fit[1] + 1.96 * math.sqrt(max(fit[2][1][1], 0.0))) < policy["min_slope_of_interest"]:
        decision["stop"] = "slope_confidently_near_zero"
    elif anchors_complete and fit is not None and rounds_done >= policy["min_rounds_for_flat"] and decision["fit"]["primary_target"] and decision["fit"]["primary_target"]["ci_lo"] > max(policy["allowed_doses"]):
        decision["stop"] = "transition_above_dose_range"
    if decision["stop"]:
        decision["dose"], decision["reason"] = None, decision["stop"]
        return decision
    # --- anchors first
    anchors = [d for d in policy["anchor_doses"] if visits.get(d, 0) == 0]
    if rounds_done < len(policy["anchor_doses"]) and anchors:
        decision["dose"], decision["reason"] = anchors[0], "anchor_dose"
        decision["candidates"] = [{"dose": d, "criterion": None} for d in anchors]
        return decision
    if fit is None or decision["fit"]["primary_target"] is None:
        # model not identifiable yet (e.g. beta <= 0): revisit the anchor with fewest visits
        candidates = [d for d in policy["allowed_doses"] if visits.get(d, 0) < policy["max_visits_per_dose"]]
        if not candidates:
            decision["stop"] = decision["dose"] = None
            decision["reason"] = decision["stop"] = "no_dose_available"
            return decision
        chosen = min(policy["anchor_doses"] if any(a in candidates for a in policy["anchor_doses"]) else candidates, key=lambda d: (visits.get(d, 0), d))
        decision["dose"], decision["reason"] = chosen, "model_not_identifiable_revisit_least_visited_anchor"
        return decision
    a, b, cov = fit
    cycle = policy.get("target_cycle", [policy["primary_target"]])
    target_name = cycle[(rounds_done - len(policy["anchor_doses"])) % len(cycle)]
    target = target_dose_and_se(a, b, cov, policy["targets"][target_name])
    available = [d for d in policy["allowed_doses"] if visits.get(d, 0) < policy["max_visits_per_dose"]]
    if not available:
        decision["stop"] = "no_dose_available"
        decision["dose"], decision["reason"] = None, decision["stop"]
        return decision
    if target is None:
        chosen = min(available, key=lambda d: (visits.get(d, 0), d))
        decision["dose"], decision["reason"] = chosen, f"{target_name}_not_identifiable_revisit_least_visited"
        return decision
    cands = [{"dose": d, "distance_to_target": abs(d - target["dose"]), "criterion": expected_variance_after(a, b, cov, d, n_round, policy["targets"][policy["primary_target"]])} for d in available]
    best = min(cands, key=lambda c: (c["distance_to_target"], c["dose"]))
    decision["candidates"] = cands
    decision["target"] = {"name": target_name, "dose": target["dose"], "ci_lo": target["ci_lo"], "ci_hi": target["ci_hi"]}
    decision["dose"], decision["reason"] = best["dose"], f"nearest_allowed_dose_to_{target_name}_estimate"
    return decision


# ---------------------------------------------------------------------------
# State replay
# ---------------------------------------------------------------------------

def units_for(policy, strata):
    stratum_ids = sorted(set(strata.values()))
    return [(i, s) for i in policy["interventions"] for s in stratum_ids]


def unit_key(intervention_id, stratum):
    return f"{intervention_id}|{stratum}"


def replay_state(policy, strata, schedule, valid_rows_by_id):
    """Rebuild per-unit state from the recorded schedule + valid rows.
    Returns {unit_key: {"rounds": [...], "visits": {dose: n}, "stopped": reason|None, "pending_ids": [...]}}."""
    pairs_by_stratum = defaultdict(list)
    for p, s in strata.items():
        pairs_by_stratum[s].append(p)
    state = {unit_key(i, s): {"rounds": [], "visits": {}, "stopped": None, "pending_ids": [], "rows": []} for i, s in units_for(policy, strata)}
    for entry in schedule:
        u = state[entry["unit"]]
        if entry.get("stop"):
            u["stopped"] = entry["stop"]
            continue
        ids = entry["observation_ids"]
        done = [i for i in ids if i in valid_rows_by_id]
        u["rounds"].append({"round": entry["round"], "dose": entry["dose"], "ids": ids, "complete": len(done) == len(ids)})
        u["visits"][entry["dose"]] = u["visits"].get(entry["dose"], 0) + 1
        u["rows"].extend(valid_rows_by_id[i] for i in done)
        if len(done) < len(ids):
            u["pending_ids"].extend(i for i in ids if i not in valid_rows_by_id)
    return state


def next_round_decisions(policy, strata, state, n_pairs_by_stratum):
    """For every unit with no pending observations and not stopped, the
    deterministic next decision (dose or stop)."""
    decisions = {}
    for key, u in state.items():
        if u["stopped"] or u["pending_ids"]:
            continue
        iid, stratum = key.split("|")
        decisions[key] = decide_next_dose(policy, len(u["rounds"]), u["visits"], u["rows"], n_pairs_by_stratum[stratum])
    return decisions


def design_arithmetic(policy, n_pairs=66):
    n_units = len(policy["interventions"]) * policy["n_strata"]
    per_round = n_pairs // policy["n_strata"] * 4
    return {"units": n_units, "judgments_per_round_per_unit": per_round, "anchor_rounds": len(policy["anchor_doses"]),
            "min_judgments_anchors_only": n_units * len(policy["anchor_doses"]) * per_round,
            "max_judgments_hard_budget": n_units * policy["max_rounds_per_unit"] * per_round,
            "fixed_grid_reference_judgments": 63360}


def schedule_sha(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()
