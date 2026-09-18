"""v3 selective-suppression analysis: I/O and orchestration around the pure
estimators in controllability_v3_stats.py. Reads ONLY v3 production result
files (results/controllability_v3/production/), writes ONLY to
results/controllability_v3/analysis/. Never reads or writes any v2 path.

Inclusion rules (never silent -- every exclusion is counted and written):
  - rows whose family is not primary_context / primary_nocontext (primary
    file) or holdout (holdout file) are excluded (pilot rows can never enter);
  - rows whose evaluator identity is not the frozen primary evaluator are
    excluded;
  - one row per planned observation: the valid row if any (the parse is
    re-checked), else the first row written (used for compliance only);
  - a (block, replicate) unit enters an estimator only if EVERY cell of the
    block resolved to a valid answer (4 cells context / 2 cells no-context).

Ambiguity uses baseline strength from the NO-CONTEXT I0 condition (the
blind baseline) -- never from any context-present response.
"""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

from analyze import load_jsonl, write_csv
from controllability_v2_cost import compute_cost_summary
from controllability_v3_design import CUE_ORDER, HEADLINE_CONTRASTS, INTERVENTION_IDS, MATCHED_CONTROL_ID, holdout_instruction_id
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID, profile_id_from_row
from controllability_v3_study_config import STUDY_CONFIG_FILE, load_study_config
from controllability_v3_trials import FAMILY_HOLDOUT, FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT, REQUIRED_CONTEXT_CELLS
from freeze_controllability_v3 import verify_frozen
from run_trial import parse_plain_ab_response
import controllability_v3_stats as st

ANALYSIS_DIR = "results/controllability_v3/analysis"
PRIMARY_FAMILIES = (FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT)

OUTPUT_TABLES = (
    "context_effects_by_intervention.csv",
    "suppression_by_intervention.csv",
    "no_context_drift_by_intervention.csv",
    "suppression_distortion_frontier.csv",
    "suppression_by_cue.csv",
    "suppression_by_ambiguity.csv",
    "held_out_cue_generalization.csv",
    "headline_contrasts.csv",
    "position_diagnostics.csv",
    "leave_one_story_out.csv",
    "intervention_by_cue_interaction.csv",
    "replicate_stability.csv",
    "response_compliance.csv",
    "token_diagnostics.csv",
    "pair_context_effects.csv",
    "pair_nocontext_rates.csv",
    "baseline_pair_strength.csv",
    "cell_counts.csv",
    "incomplete_units.csv",
    "exclusions.csv",
    "cost_summary.csv",
)


# ---------------------------------------------------------------------------
# Loading / inclusion
# ---------------------------------------------------------------------------

def load_rows(path):
    return load_jsonl(path) if path and os.path.exists(path) else []


def _row_is_valid(row):
    if row.get("parsing_status") != "resolved":
        return False
    parsed, _ = parse_plain_ab_response(row.get("raw_response") or "")
    return bool(parsed) and parsed["overall_quality"] == row.get("parsed_choice")


def dedupe_rows(rows):
    """{planned_observation_id: {"first": first-written row, "valid": valid row or None}}."""
    by_id = {}
    for row in rows:
        oid = row["planned_observation_id"]
        entry = by_id.setdefault(oid, {"first": row, "valid": None, "n_rows": 0})
        entry["n_rows"] += 1
        if entry["valid"] is None and _row_is_valid(row):
            entry["valid"] = row
    return by_id


def evaluator_matches(row, study_config):
    primary = study_config["primary_evaluator"]
    ev = row.get("evaluator") or {}
    return (ev.get("provider"), ev.get("requested_model"), ev.get("reasoning_profile")) == (
        primary["provider"], primary["requested_model"], primary["reasoning_profile"]
    )


def include_rows(rows, allowed_families, study_config, profile_id=None, id_prefix="v3::"):
    """Returns (by_id, exclusions) -- exclusions is a list of {reason, n}.
    With profile_id=None (legacy) rows must match the study config's
    primary evaluator; with a profile they must have been judged under
    exactly that evaluator profile (profiles are never pooled)."""
    exclusions = defaultdict(int)
    kept = []
    for row in rows:
        oid = row.get("planned_observation_id", "")
        if not oid.startswith(id_prefix):
            exclusions[f"not_a_{id_prefix.rstrip(':')}_id"] += 1
        elif row.get("family") not in allowed_families:
            exclusions[f"family_not_allowed:{row.get('family')}"] += 1
        elif row.get("collection") == "pilot":
            exclusions["pilot_collection"] += 1
        elif profile_id is not None and profile_id_from_row(row) != profile_id:
            exclusions[f"evaluator_profile_mismatch:{profile_id_from_row(row)}"] += 1
        elif profile_id is None and not evaluator_matches(row, study_config):
            exclusions["evaluator_mismatch"] += 1
        else:
            kept.append(row)
    return dedupe_rows(kept), [{"reason": r, "n_rows": n} for r, n in sorted(exclusions.items())]


def partition_by_profile(rows):
    """{evaluator_profile_id: [rows]} -- profiles are analysed separately, never pooled."""
    parts = defaultdict(list)
    for row in rows:
        parts[profile_id_from_row(row)].append(row)
    return dict(parts)


def story1_chosen(row):
    choice = row.get("parsed_choice")
    if choice not in ("A", "B"):
        return None
    chosen = row["story_a"] if choice == "A" else row["story_b"]
    return chosen == row["story_1_id"]


def complete_units(valid_rows, n_expected, by_id_all):
    """Group valid rows by (block_id, replicate); keep only groups with all
    n_expected cells. Diagnostics distinguish never-attempted cells from
    attempted-but-unresolved ones."""
    groups = defaultdict(list)
    for row in valid_rows:
        groups[(row["block_id"], row["replicate"])].append(row)
    attempted = defaultdict(int)
    for oid, entry in by_id_all.items():
        r = entry["first"]
        attempted[(r["block_id"], r["replicate"])] += 1
    kept, n_complete, diagnostics = [], 0, []
    for key in set(groups) | set(attempted):
        rows = groups.get(key, [])
        if len(rows) == n_expected:
            n_complete += 1
            kept.extend(rows)
        else:
            diagnostics.append({
                "block_id": key[0], "replicate": key[1], "n_valid_cells": len(rows), "n_cells_expected": n_expected,
                "n_cells_attempted": attempted.get(key, 0),
                "reason": "terminally_incomplete" if attempted.get(key, 0) > len(rows) else "temporarily_incomplete",
            })
    return kept, n_complete, diagnostics


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def context_cell_rates(rows, replicate_filter=None):
    lists = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if replicate_filter is not None and not replicate_filter(row["replicate"]):
            continue
        chosen = story1_chosen(row)
        if chosen is None:
            continue
        key = (row["intervention_id"], row["cue"], row["story_1_id"], row["story_2_id"])
        lists[key][(row["context_assignment"], row["display_position"])].append(chosen)
    return {k: {c: sum(v) / len(v) for c, v in cells.items()} for k, cells in lists.items()}


def pair_context_effects(cell_rates):
    out = []
    for (intervention_id, cue_id, s1, s2), cells in cell_rates.items():
        result = st.d_pair_from_cell_rates(cells)
        if result is None:
            continue
        out.append({"intervention_id": intervention_id, "cue_id": cue_id, "story_1_id": s1, "story_2_id": s2, **result})
    return out


def pair_values(pair_rows, intervention_id, cue_id=None, field="d_pair"):
    """{(s1,s2): value} for one intervention; cue_id=None pools over cues
    (per-pair mean over the cues available)."""
    if cue_id is not None:
        return {(r["story_1_id"], r["story_2_id"]): r[field] for r in pair_rows if r["intervention_id"] == intervention_id and r["cue_id"] == cue_id}
    by_cue = defaultdict(dict)
    for r in pair_rows:
        if r["intervention_id"] == intervention_id:
            by_cue[r["cue_id"]][(r["story_1_id"], r["story_2_id"])] = r[field]
    return st.pooled_over_cues(by_cue)


def nocontext_rates(rows, replicate_filter=None):
    """{intervention: {(s1, s2): {position: p_story1_chosen}}}."""
    lists = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if replicate_filter is not None and not replicate_filter(row["replicate"]):
            continue
        chosen = story1_chosen(row)
        if chosen is None:
            continue
        lists[row["intervention_id"]][(row["story_1_id"], row["story_2_id"])][row["display_position"]].append(chosen)
    return {
        i: {pair: {pos: {"p": sum(v) / len(v), "n": len(v)} for pos, v in positions.items()} for pair, positions in pairs.items()}
        for i, pairs in lists.items()
    }


def baseline_strength_table(nocontext_i0_rows):
    counts = defaultdict(lambda: [0, 0])
    for row in nocontext_i0_rows:
        chosen = story1_chosen(row)
        if chosen is None:
            continue
        counts[(row["story_1_id"], row["story_2_id"])][0 if chosen else 1] += 1
    return [{"story_1_id": s1, "story_2_id": s2, **st.baseline_pair_stats(w1, w2)} for (s1, s2), (w1, w2) in sorted(counts.items())]


# ---------------------------------------------------------------------------
# Diagnostics tables
# ---------------------------------------------------------------------------

def response_compliance(by_id):
    groups = defaultdict(lambda: defaultdict(int))
    for entry in by_id.values():
        first = entry["first"]
        key = (first["family"], first["intervention_id"], bool(first["context_present"]))
        g = groups[key]
        g["n_planned_observations_attempted"] += 1
        g[f"n_first_attempt_{first['first_attempt_status']}"] += 1
        g["n_resolved"] += 1 if entry["valid"] is not None else 0
        g["total_rows"] += entry["n_rows"]
        g["total_attempts"] += sum(len(r.get("attempts") or []) for r in [entry["first"]] + ([entry["valid"]] if entry["valid"] is not None and entry["valid"] is not entry["first"] else []))
    out = []
    for (family, intervention_id, context_present), g in sorted(groups.items()):
        n = g["n_planned_observations_attempted"]
        out.append({
            "family": family, "intervention_id": intervention_id, "context_present": context_present,
            "n_planned_observations_attempted": n,
            "n_first_attempt_valid": g["n_first_attempt_valid"], "n_first_attempt_invalid": g["n_first_attempt_invalid"],
            "n_first_attempt_refusal": g["n_first_attempt_refusal"], "n_first_attempt_api_error": g["n_first_attempt_api_error"],
            "first_attempt_valid_rate": g["n_first_attempt_valid"] / n if n else None,
            "n_resolved": g["n_resolved"], "resolved_rate": g["n_resolved"] / n if n else None,
            "mean_attempts_per_observation": g["total_attempts"] / n if n else None,
        })
    return out


def token_diagnostics(by_id, max_output_tokens):
    groups = defaultdict(list)
    for entry in by_id.values():
        if entry["valid"] is not None:
            r = entry["valid"]
            groups[(r["family"], r["intervention_id"], bool(r["context_present"]))].append(r)
    out = []
    for (family, intervention_id, context_present), rows in sorted(groups.items()):
        def col(name):
            return [r.get(name) for r in rows if r.get(name) is not None]
        out_tokens, reasoning, latency = col("output_tokens"), col("reasoning_tokens"), col("latency_seconds")
        out.append({
            "family": family, "intervention_id": intervention_id, "context_present": context_present, "n_resolved": len(rows),
            "mean_input_tokens": st.mean(col("input_tokens")), "mean_cached_input_tokens": st.mean(col("cached_input_tokens")),
            "mean_output_tokens": st.mean(out_tokens), "median_output_tokens": statistics.median(out_tokens) if out_tokens else None,
            "p90_output_tokens": st.percentile(sorted(out_tokens), 90) if out_tokens else None,
            "mean_reasoning_tokens": st.mean(reasoning), "median_reasoning_tokens": statistics.median(reasoning) if reasoning else None,
            "share_at_output_ceiling": (sum(1 for t in out_tokens if t >= max_output_tokens) / len(out_tokens)) if out_tokens else None,
            "mean_latency_seconds": st.mean(latency), "median_latency_seconds": statistics.median(latency) if latency else None,
            "mean_total_attempts": st.mean([r.get("total_attempts") for r in rows]),
        })
    return out


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def analyze(primary_rows, holdout_rows, study_config, is_frozen, n_draws, seed, profile_id=None):
    tag = {"experiment_id": study_config["experiment_id"], "corpus_id": study_config["corpus_id"],
           "evaluator_id": study_config["primary_evaluator"]["evaluator_id"] if profile_id in (None, PRIMARY_PROFILE_ID) else profile_id,
           "evaluator_profile_id": profile_id or PRIMARY_PROFILE_ID, "study_frozen": is_frozen}
    min_denominator = study_config.get("relative_suppression_min_denominator", 0.02)
    results = {}

    by_id, exclusions = include_rows(primary_rows, PRIMARY_FAMILIES, study_config, profile_id)
    holdout_by_id, holdout_exclusions = include_rows(holdout_rows, (FAMILY_HOLDOUT,), study_config, profile_id)
    results["exclusions"] = [{**tag, "file": "primary", **e} for e in exclusions] + [{**tag, "file": "holdout", **e} for e in holdout_exclusions]

    valid_context = [e["valid"] for e in by_id.values() if e["valid"] is not None and e["valid"]["family"] == FAMILY_PRIMARY_CONTEXT]
    valid_nocontext = [e["valid"] for e in by_id.values() if e["valid"] is not None and e["valid"]["family"] == FAMILY_PRIMARY_NOCONTEXT]
    valid_holdout = [e["valid"] for e in holdout_by_id.values() if e["valid"] is not None]
    ctx_by_id = {k: v for k, v in by_id.items() if v["first"]["family"] == FAMILY_PRIMARY_CONTEXT}
    noctx_by_id = {k: v for k, v in by_id.items() if v["first"]["family"] == FAMILY_PRIMARY_NOCONTEXT}

    kept_context, n_complete_ctx, diag_ctx = complete_units(valid_context, len(REQUIRED_CONTEXT_CELLS), ctx_by_id)
    kept_nocontext, n_complete_noctx, diag_noctx = complete_units(valid_nocontext, 2, noctx_by_id)
    kept_holdout, n_complete_hold, diag_hold = complete_units(valid_holdout, len(REQUIRED_CONTEXT_CELLS), holdout_by_id)
    results["incomplete_units"] = (
        [{**tag, "family": FAMILY_PRIMARY_CONTEXT, **d} for d in diag_ctx]
        + [{**tag, "family": FAMILY_PRIMARY_NOCONTEXT, **d} for d in diag_noctx]
        + [{**tag, "family": FAMILY_HOLDOUT, **d} for d in diag_hold]
    )

    cell_rates = context_cell_rates(kept_context)
    pair_rows = pair_context_effects(cell_rates)
    holdout_pair_rows = pair_context_effects(context_cell_rates(kept_holdout))
    rates_by_intervention = nocontext_rates(kept_nocontext)

    i0_nocontext_rows = [r for r in kept_nocontext if r["intervention_id"] == MATCHED_CONTROL_ID]
    baseline_table = baseline_strength_table(i0_nocontext_rows)
    strength = {(r["story_1_id"], r["story_2_id"]): r["baseline_strength"] for r in baseline_table}
    results["baseline_pair_strength"] = [{**tag, **r} for r in baseline_table]

    story_ids = sorted({s for r in pair_rows for s in (r["story_1_id"], r["story_2_id"])}
                       | {s for r in kept_nocontext for s in (r["story_1_id"], r["story_2_id"])}
                       | {s for r in holdout_pair_rows for s in (r["story_1_id"], r["story_2_id"])})
    rng = random.Random(seed)
    resamples = st.make_story_resamples(story_ids, n_draws, rng) if story_ids else []

    interventions = [i for i in INTERVENTION_IDS if any(r["intervention_id"] == i for r in pair_rows)]
    cues = [c for c in CUE_ORDER if any(r["cue_id"] == c for r in pair_rows)]

    # --- context effects by intervention (per cue + pooled), suppression, cue table, interaction, LOSO, position
    ce_rows, sup_rows, sup_cue_rows, interaction_rows, loo_rows, position_rows = [], [], [], [], [], []
    control_pooled = pair_values(pair_rows, MATCHED_CONTROL_ID) if MATCHED_CONTROL_ID in interventions else {}
    pooled_suppression = {}
    for i in interventions:
        for cue in [None] + cues:
            values = pair_values(pair_rows, i, cue)
            if not values:
                continue
            draws = [d["ce"] for d in st.joint_draws({"ce": values}, resamples)]
            point = st.mean(values.values())
            ce_rows.append({**tag, "intervention_id": i, "cue_id": cue or "all", "n_story_pairs": len(values),
                            **st.summarize("context_effect", point, draws), "abs_context_effect": abs(point) if point is not None else None,
                            "n_bootstrap_draws_used": len(draws)})
            group = [r for r in pair_rows if r["intervention_id"] == i and (cue is None or r["cue_id"] == cue)]
            position_rows.append({**tag, "family": FAMILY_PRIMARY_CONTEXT, "intervention_id": i, "cue_id": cue or "all", "n_story_pairs": len(group),
                                  "context_effect_at_story1_as_a": st.mean(r["context_effect_at_a"] for r in group),
                                  "context_effect_at_story1_as_b": st.mean(r["context_effect_at_b"] for r in group),
                                  "context_x_position_interaction": st.mean(r["context_x_position_interaction"] for r in group),
                                  "overall_position_effect": st.mean(r["position_effect"] for r in group)})
            if MATCHED_CONTROL_ID in interventions:
                control = pair_values(pair_rows, MATCHED_CONTROL_ID, cue)
                sup = st.suppression_estimates(control, values, resamples, min_denominator)
                if sup is not None:
                    row = {**tag, "intervention_id": i, "cue_id": cue or "all", **sup}
                    (sup_rows if cue is None else sup_cue_rows).append(row)
                    if cue is None:
                        pooled_suppression[i] = sup
                        for story, loo in st.leave_one_story_out({p: control[p] - values[p] for p in values if p in control}, story_ids).items():
                            loo_ce = st.leave_one_story_out(values, story_ids)[story]
                            loo_rows.append({**tag, "intervention_id": i, "excluded_story_id": story, "context_effect_excluding_story": loo_ce,
                                             "signed_suppression_excluding_story": loo})
    # intervention x cue interaction: cue-specific signed suppression minus pooled signed suppression, same draws
    for i in interventions:
        if i == MATCHED_CONTROL_ID or i not in pooled_suppression:
            continue
        pooled_c, pooled_t = pair_values(pair_rows, MATCHED_CONTROL_ID), pair_values(pair_rows, i)
        for cue in cues:
            c, t = pair_values(pair_rows, MATCHED_CONTROL_ID, cue), pair_values(pair_rows, i, cue)
            common = sorted(set(c) & set(t) & set(pooled_c) & set(pooled_t))
            if not common:
                continue
            maps = {"c": {p: c[p] for p in common}, "t": {p: t[p] for p in common}, "pc": {p: pooled_c[p] for p in common}, "pt": {p: pooled_t[p] for p in common}}
            draws = [(d["c"] - d["t"]) - (d["pc"] - d["pt"]) for d in st.joint_draws(maps, resamples)]
            point = (st.mean(maps["c"].values()) - st.mean(maps["t"].values())) - (st.mean(maps["pc"].values()) - st.mean(maps["pt"].values()))
            interaction_rows.append({**tag, "intervention_id": i, "cue_id": cue, "n_story_pairs": len(common),
                                     "cue_signed_suppression": st.mean(maps["c"].values()) - st.mean(maps["t"].values()),
                                     "pooled_signed_suppression": st.mean(maps["pc"].values()) - st.mean(maps["pt"].values()),
                                     **st.summarize("interaction", point, draws)})
    results["context_effects_by_intervention"] = ce_rows
    results["suppression_by_intervention"] = sup_rows
    results["suppression_by_cue"] = sup_cue_rows
    results["intervention_by_cue_interaction"] = interaction_rows
    results["leave_one_story_out"] = loo_rows

    # --- no-context drift + no-context position effects
    drift_rows = []
    drift_by_intervention = {}
    control_rates = rates_by_intervention.get(MATCHED_CONTROL_ID)
    for i in INTERVENTION_IDS:
        if control_rates is None or i not in rates_by_intervention:
            continue
        drift = st.drift_estimates(control_rates, rates_by_intervention[i], resamples, same_sample=(i == MATCHED_CONTROL_ID))
        if drift is None:
            continue
        drift_by_intervention[i] = drift
        drift_rows.append({**tag, "intervention_id": i, **drift})
        position_rows.append({**tag, "family": FAMILY_PRIMARY_NOCONTEXT, "intervention_id": i, "cue_id": None, "n_story_pairs": drift["n_story_pairs"],
                              "context_effect_at_story1_as_a": None, "context_effect_at_story1_as_b": None, "context_x_position_interaction": None,
                              "overall_position_effect": drift["nocontext_position_effect"]})
    results["no_context_drift_by_intervention"] = drift_rows
    results["position_diagnostics"] = position_rows

    # --- frontier
    frontier_points = {}
    for i in interventions:
        if i in pooled_suppression and i in drift_by_intervention:
            frontier_points[i] = (drift_by_intervention[i]["rms_prob_shift_noise_corrected"], pooled_suppression[i]["magnitude_suppression"])
    flags = st.pareto_flags(frontier_points)
    frontier_rows = []
    for i, (x, y) in frontier_points.items():
        s, d = pooled_suppression[i], drift_by_intervention[i]
        frontier_rows.append({
            **tag, "intervention_id": i,
            "drift_rms_prob_shift_noise_corrected": x, "drift_rms_prob_shift_noise_corrected_ci_95_lo": d["rms_prob_shift_noise_corrected_ci_95_lo"], "drift_rms_prob_shift_noise_corrected_ci_95_hi": d["rms_prob_shift_noise_corrected_ci_95_hi"],
            "drift_abs_prob_shift_raw": d["abs_prob_shift"], "drift_abs_prob_shift_null_floor": d.get("abs_prob_shift_null_floor"),
            "drift_ab_disagreement_excess": d["ab_disagreement_excess"], "drift_ab_disagreement_excess_ci_95_lo": d["ab_disagreement_excess_ci_95_lo"], "drift_ab_disagreement_excess_ci_95_hi": d["ab_disagreement_excess_ci_95_hi"],
            "drift_pair_preference_flip_rate": d["pair_preference_flip"], "drift_story_win_rate_abs_shift_raw": d["story_win_rate_abs_shift"],
            "drift_story_win_rate_rms_shift_noise_corrected": d.get("story_win_rate_rms_shift_noise_corrected"), "drift_story_win_rate_spearman": d["story_win_rate_spearman"],
            "magnitude_suppression": y, "magnitude_suppression_ci_95_lo": s["magnitude_suppression_ci_95_lo"], "magnitude_suppression_ci_95_hi": s["magnitude_suppression_ci_95_hi"],
            "signed_suppression": s["signed_suppression"], "residual_context_effect": s["residual_context_effect"], "CE_control": s["CE_control"],
            "pareto_efficient": flags[i]["pareto_efficient"], "dominated_by": ";".join(flags[i]["dominated_by"]),
        })
    results["suppression_distortion_frontier"] = frontier_rows

    # --- headline contrasts (pooled): residual CE difference, |CE| difference, drift difference
    headline_rows = []
    for a, b in HEADLINE_CONTRASTS:
        if a not in interventions or b not in interventions:
            continue
        va, vb = pair_values(pair_rows, a), pair_values(pair_rows, b)
        common = sorted(set(va) & set(vb))
        if not common:
            continue
        maps = {"a": {p: va[p] for p in common}, "b": {p: vb[p] for p in common}}
        draws = st.joint_draws(maps, resamples)
        pa, pb = st.mean(maps["a"].values()), st.mean(maps["b"].values())
        row = {**tag, "contrast": f"{a}_vs_{b}", "intervention_id": a, "reference_id": b, "n_story_pairs": len(common),
               "CE_intervention": pa, "CE_reference": pb,
               **st.summarize("residual_ce_difference", pa - pb, [d["a"] - d["b"] for d in draws]),
               **st.summarize("abs_ce_difference", abs(pa) - abs(pb), [abs(d["a"]) - abs(d["b"]) for d in draws])}
        if control_rates is not None and a in rates_by_intervention and b in rates_by_intervention:
            da = st.drift_pair_values(control_rates, rates_by_intervention[a], a == MATCHED_CONTROL_ID).get("squared_shift_noise_corrected", {})
            db = st.drift_pair_values(control_rates, rates_by_intervention[b], b == MATCHED_CONTROL_ID).get("squared_shift_noise_corrected", {})
            common_d = sorted(set(da) & set(db))
            if common_d:
                dd = st.joint_draws({"a": {p: da[p] for p in common_d}, "b": {p: db[p] for p in common_d}}, resamples)
                row.update(st.summarize("drift_mean_squared_shift_difference", st.mean(da[p] for p in common_d) - st.mean(db[p] for p in common_d), [d["a"] - d["b"] for d in dd]))
        headline_rows.append(row)
    results["headline_contrasts"] = headline_rows

    # --- ambiguity: suppression vs baseline strength (regression, tertiles, median split)
    ambiguity_rows = []
    if strength and MATCHED_CONTROL_ID in interventions:
        pair_bins = st.bin_by_strength(strength, 3)
        median_strength = statistics.median(strength.values())
        for i in interventions:
            for cue in [None] + cues:
                c, t = pair_values(pair_rows, MATCHED_CONTROL_ID, cue), pair_values(pair_rows, i, cue)
                common = [p for p in sorted(set(c) & set(t)) if p in strength]
                if len(common) < 3:
                    continue
                sup_pairs = {p: c[p] - t[p] for p in common}
                ce_pairs = {p: t[p] for p in common}
                base = {**tag, "intervention_id": i, "cue_id": cue or "all", "n_story_pairs": len(common)}
                fit_sup = st.regression_with_story_bootstrap([{"story_1_id": p[0], "story_2_id": p[1], "x": strength[p], "y": sup_pairs[p]} for p in common], resamples)
                fit_ce = st.regression_with_story_bootstrap([{"story_1_id": p[0], "story_2_id": p[1], "x": strength[p], "y": ce_pairs[p]} for p in common], resamples)
                if fit_ce:
                    ambiguity_rows.append({**base, "analysis": "regression_context_effect_on_baseline_strength", **fit_ce})
                if i == MATCHED_CONTROL_ID:
                    continue  # suppression of I0 relative to itself is identically zero
                if fit_sup:
                    ambiguity_rows.append({**base, "analysis": "regression_signed_suppression_on_baseline_strength", **fit_sup})
                for b, stats_ in st.binned_means(sup_pairs, pair_bins, resamples).items():
                    ambiguity_rows.append({**base, "analysis": f"tertile_{b}_signed_suppression", "bin": b, **stats_,
                                           "mean_context_effect_in_bin": st.mean(ce_pairs[p] for p in common if pair_bins[p] == b),
                                           "mean_control_effect_in_bin": st.mean(c[p] for p in common if pair_bins[p] == b)})
                split = {p: (1 if strength[p] >= median_strength else 0) for p in common}
                for b, label in ((1, "strong_baseline_half"), (0, "ambiguous_half")):
                    stats_ = st.binned_means(sup_pairs, split, resamples).get(b)
                    if stats_:
                        ambiguity_rows.append({**base, "analysis": f"median_split_{label}_signed_suppression", "bin": b, **stats_,
                                               "mean_context_effect_in_bin": st.mean(ce_pairs[p] for p in common if split[p] == b),
                                               "mean_control_effect_in_bin": st.mean(c[p] for p in common if split[p] == b)})
    results["suppression_by_ambiguity"] = ambiguity_rows

    # --- held-out cue generalization
    holdout_rows_out = []
    for cue in CUE_ORDER:
        hid = holdout_instruction_id("I2", cue)
        control = pair_values(pair_rows, MATCHED_CONTROL_ID, cue)
        generic = pair_values(pair_rows, "I1", cue)
        full = pair_values(pair_rows, "I2", cue)
        holdout = pair_values(holdout_pair_rows, hid, cue)
        transfer = st.holdout_transfer(control, generic, full, holdout, resamples, min_denominator) if holdout else None
        if transfer:
            holdout_rows_out.append({**tag, "cue_id": cue, "holdout_instruction_id": hid, **transfer})
    results["held_out_cue_generalization"] = holdout_rows_out

    # --- replicate stability (odd vs even replicates)
    stability_rows = []
    odd, even = (lambda r: r % 2 == 1), (lambda r: r % 2 == 0)
    pairs_odd = pair_context_effects(context_cell_rates(kept_context, odd))
    pairs_even = pair_context_effects(context_cell_rates(kept_context, even))
    for i in interventions:
        a, b = pair_values(pairs_odd, i), pair_values(pairs_even, i)
        corr = st.split_half_pair_correlation(a, b)
        stability_rows.append({**tag, "family": FAMILY_PRIMARY_CONTEXT, "intervention_id": i, "quantity": "pooled_d_pair",
                               "estimate_odd_replicates": st.mean(a.values()), "estimate_even_replicates": st.mean(b.values()), **corr})
    rates_odd, rates_even = nocontext_rates(kept_nocontext, odd), nocontext_rates(kept_nocontext, even)
    for i in INTERVENTION_IDS:
        if i in rates_odd and i in rates_even:
            a = {p: st.mean(c["p"] for c in v.values()) for p, v in rates_odd[i].items()}
            b = {p: st.mean(c["p"] for c in v.values()) for p, v in rates_even[i].items()}
            stability_rows.append({**tag, "family": FAMILY_PRIMARY_NOCONTEXT, "intervention_id": i, "quantity": "pair_story1_rate",
                                   "estimate_odd_replicates": st.mean(a.values()), "estimate_even_replicates": st.mean(b.values()), **st.split_half_pair_correlation(a, b)})
    results["replicate_stability"] = stability_rows

    # --- diagnostics
    all_by_id = {**by_id, **holdout_by_id}
    results["response_compliance"] = [{**tag, **r} for r in response_compliance(all_by_id)]
    results["token_diagnostics"] = [{**tag, **r} for r in token_diagnostics(all_by_id, study_config["max_output_tokens"])]
    results["pair_context_effects"] = [{**tag, "family": FAMILY_PRIMARY_CONTEXT, "baseline_strength": strength.get((r["story_1_id"], r["story_2_id"])), **r} for r in pair_rows] \
        + [{**tag, "family": FAMILY_HOLDOUT, "baseline_strength": strength.get((r["story_1_id"], r["story_2_id"])), **r} for r in holdout_pair_rows]
    results["pair_nocontext_rates"] = [
        {**tag, "intervention_id": i, "story_1_id": p[0], "story_2_id": p[1], "p_story1_story1_as_a": v.get("story1_as_a", {}).get("p"), "n_story1_as_a": v.get("story1_as_a", {}).get("n"), "p_story1_story2_as_a": v.get("story2_as_a", {}).get("p"), "n_story2_as_a": v.get("story2_as_a", {}).get("n"), "p_story1_mean": st.mean(c["p"] for c in v.values())}
        for i, pairs in rates_by_intervention.items() for p, v in sorted(pairs.items())
    ]
    cell_counts = defaultdict(int)
    for r in kept_context:
        cell_counts[(FAMILY_PRIMARY_CONTEXT, r["intervention_id"], r["cue"], r["context_assignment"], r["display_position"])] += 1
    for r in kept_nocontext:
        cell_counts[(FAMILY_PRIMARY_NOCONTEXT, r["intervention_id"], None, None, r["display_position"])] += 1
    for r in kept_holdout:
        cell_counts[(FAMILY_HOLDOUT, r["intervention_id"], r["cue"], r["context_assignment"], r["display_position"])] += 1
    results["cell_counts"] = [{**tag, "family": f, "intervention_id": i, "cue_id": c, "assignment": a, "position": p, "n_resolved_cells_in_complete_units": n}
                              for (f, i, c, a, p), n in sorted(cell_counts.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    all_rows_for_cost = [e["valid"] if e["valid"] is not None else e["first"] for e in all_by_id.values()]
    results["cost_summary"] = [{**tag, **compute_cost_summary(all_rows_for_cost, study_config["primary_evaluator"]["provider"], study_config["primary_evaluator"]["requested_model"])}]

    results["nocontext_rates_by_intervention"] = rates_by_intervention  # kept for cross-evaluator baseline disagreement (not written as a table)
    results["completeness"] = {
        "n_planned_primary": study_config["expected_unique_cells"]["primary_context"] * study_config["replicate_count"] + study_config["expected_unique_cells"]["primary_nocontext"] * study_config["replicate_count"],
        "n_planned_holdout": study_config["expected_unique_cells"]["holdout"] * study_config["holdout_replicate_count"],
        "n_valid_primary": len(valid_context) + len(valid_nocontext), "n_valid_holdout": len(valid_holdout),
        "n_complete_context_units": n_complete_ctx, "n_incomplete_context_units": len(diag_ctx),
        "n_complete_nocontext_units": n_complete_noctx, "n_incomplete_nocontext_units": len(diag_noctx),
        "n_complete_holdout_units": n_complete_hold, "n_incomplete_holdout_units": len(diag_hold),
        "expected_complete_context_units": study_config["expected_unique_cells"]["primary_context"] // 4 * study_config["replicate_count"],
        "expected_complete_nocontext_units": study_config["expected_unique_cells"]["primary_nocontext"] // 2 * study_config["replicate_count"],
        "expected_complete_holdout_units": study_config["expected_unique_cells"]["holdout"] // 4 * study_config["holdout_replicate_count"],
        "n_bootstrap_draws_requested": n_draws, "n_bootstrap_draws_kept": len(resamples), "bootstrap_seed": seed,
        "response_models_observed": sorted({e["valid"].get("response_model") for e in all_by_id.values() if e["valid"] is not None and e["valid"].get("response_model")}),
    }
    c = results["completeness"]
    c["primary_complete"] = (c["n_complete_context_units"] == c["expected_complete_context_units"] and c["n_complete_nocontext_units"] == c["expected_complete_nocontext_units"])
    c["holdout_complete"] = c["n_complete_holdout_units"] == c["expected_complete_holdout_units"]
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(results, study_config, is_frozen, analysis_dir):
    os.makedirs(analysis_dir, exist_ok=True)
    written = []
    for filename in OUTPUT_TABLES:
        key = filename[:-4]
        rows = results.get(key) or []
        if not rows:
            continue
        fieldnames = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        write_csv(rows, fieldnames, os.path.join(analysis_dir, filename))
        written.append(filename)

    headline = {
        "experiment_id": study_config["experiment_id"], "corpus_id": study_config["corpus_id"],
        "primary_evaluator": study_config["primary_evaluator"], "max_output_tokens": study_config["max_output_tokens"],
        "study_frozen": is_frozen, "completeness": results["completeness"],
        "interpretation_note": (
            "Suppression and no-context drift are reported as two separate quantities and are never combined into one score. "
            "A non-significant residual context effect is not evidence of successful suppression; compare effect sizes and CIs directly."
        ),
        "suppression_by_intervention": results["suppression_by_intervention"],
        "no_context_drift_by_intervention": results["no_context_drift_by_intervention"],
        "suppression_distortion_frontier": results["suppression_distortion_frontier"],
        "headline_contrasts": results["headline_contrasts"],
        "held_out_cue_generalization": results["held_out_cue_generalization"],
    }
    with open(os.path.join(analysis_dir, "headline_results.json"), "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)
    with open(os.path.join(analysis_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k != "nocontext_rates_by_intervention"}, f, indent=2)
    written += ["headline_results.json", "summary.json"]

    from controllability_v3_plots import write_all_plots
    written += [os.path.basename(p) for p in write_all_plots(results, analysis_dir)]
    return written


def run_analysis(primary_results_file, holdout_results_file=None, bootstrap_draws=None, analysis_dir=ANALYSIS_DIR,
                 study_config_file=STUDY_CONFIG_FILE, extra_profile_files=None):
    """extra_profile_files: [{"primary": path, "holdout": path}] for result
    files of other evaluator profiles; every profile is analysed separately."""
    study_config = load_study_config(study_config_file)
    is_frozen, reason = verify_frozen(study_config_file)
    if not is_frozen:
        print(f"v3 study design is not frozen or does not match its lock ({reason}); results are labelled study_frozen=false.")
    n_draws = bootstrap_draws or study_config["bootstrap_draws"]
    primary_rows = load_rows(primary_results_file)
    holdout_rows = load_rows(holdout_results_file)
    for extra in extra_profile_files or []:
        primary_rows += load_rows(extra.get("primary"))
        holdout_rows += load_rows(extra.get("holdout"))
    print(f"Raw primary rows: {len(primary_rows)}  Raw holdout rows: {len(holdout_rows)}")

    profiles = sorted(set(partition_by_profile(primary_rows)) | set(partition_by_profile(holdout_rows)) | {PRIMARY_PROFILE_ID})
    per_profile = {}
    for pid in profiles:
        results = analyze(primary_rows, holdout_rows, study_config, is_frozen, n_draws, study_config["random_seed"], profile_id=pid)
        c = results["completeness"]
        print(f"[{pid}] Complete units: context={c['n_complete_context_units']}/{c['expected_complete_context_units']}  "
              f"nocontext={c['n_complete_nocontext_units']}/{c['expected_complete_nocontext_units']}  "
              f"holdout={c['n_complete_holdout_units']}/{c['expected_complete_holdout_units']}  "
              f"primary_complete={c['primary_complete']}  holdout_complete={c['holdout_complete']}")
        for e in results["exclusions"]:
            print(f"[{pid}] Excluded ({e['file']}): {e['reason']}: {e['n_rows']} row(s)")
        out_dir = analysis_dir if pid == PRIMARY_PROFILE_ID else os.path.join(analysis_dir, "profiles", pid)
        written = write_outputs(results, study_config, is_frozen, out_dir)
        print(f"[{pid}] Wrote {len(written)} output file(s) to {out_dir}")
        per_profile[pid] = results

    if len([p for p in per_profile if per_profile[p]["completeness"]["n_valid_primary"] > 0]) > 1:
        write_cross_evaluator_outputs(per_profile, study_config, analysis_dir, n_draws)
    return per_profile[PRIMARY_PROFILE_ID]


def write_cross_evaluator_outputs(per_profile, study_config, analysis_dir, n_draws):
    """Cross-evaluator tables: one row per (profile, intervention) for
    suppression and drift (never pooled), plus baseline disagreement
    between every pair of profiles -- the 'different underlying judge'
    measure that must be read alongside any cross-model suppression
    difference."""
    os.makedirs(analysis_dir, exist_ok=True)
    active = {p: r for p, r in per_profile.items() if r["completeness"]["n_valid_primary"] > 0}
    sup = [row for r in active.values() for row in r["suppression_by_intervention"]]
    drift = [row for r in active.values() for row in r["no_context_drift_by_intervention"]]
    disagreement = []
    story_ids = sorted({s for r in active.values() for rates in r["nocontext_rates_by_intervention"].values() for pair in rates for s in pair})
    rng = random.Random(study_config["random_seed"])
    resamples = st.make_story_resamples(story_ids, n_draws, rng) if story_ids else []
    pids = sorted(active)
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            for iid in INTERVENTION_IDS:
                ra, rb = active[a]["nocontext_rates_by_intervention"].get(iid), active[b]["nocontext_rates_by_intervention"].get(iid)
                if not ra or not rb:
                    continue
                d = st.drift_estimates(ra, rb, resamples)
                if d:
                    disagreement.append({"profile_a": a, "profile_b": b, "intervention_id": iid, **d})
    for name, rows in (("cross_evaluator_suppression.csv", sup), ("cross_evaluator_drift.csv", drift), ("cross_evaluator_baseline_disagreement.csv", disagreement)):
        if rows:
            fieldnames = []
            for row in rows:
                for k in row:
                    if k not in fieldnames:
                        fieldnames.append(k)
            write_csv(rows, fieldnames, os.path.join(analysis_dir, name))
    print(f"Wrote cross-evaluator tables for profiles {pids} to {analysis_dir}")


def main():
    from run_controllability_v3 import HOLDOUT_RESULTS_FILE, PRIMARY_RESULTS_FILE
    parser = argparse.ArgumentParser(description="v3 selective-suppression analysis (no API calls).")
    parser.add_argument("--primary-results-file", default=PRIMARY_RESULTS_FILE)
    parser.add_argument("--holdout-results-file", default=HOLDOUT_RESULTS_FILE)
    parser.add_argument("--study-config-file", default=STUDY_CONFIG_FILE)
    parser.add_argument("--analysis-dir", default=ANALYSIS_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    args = parser.parse_args()
    run_analysis(args.primary_results_file, args.holdout_results_file, args.bootstrap_draws, args.analysis_dir, args.study_config_file)


if __name__ == "__main__":
    main()
