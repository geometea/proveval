"""Analysis for the two v3 STRESS families, per evaluator profile (never
pooled across profiles), plus cross-evaluator tables when several profiles
have data. Reads the stress raw files AND the primary raw file of the same
profile (for the ordinary-wording context effects, the matched control,
and the blind baseline strength). Writes only under the stress analysis
directory. Never reads or writes any v2 path.

Adversarial inclusion rule: the ordinary/attack comparison is made on the
44 held-out evaluation pairs ONLY. Development-half results are reported
solely as the ranking they produced (attack_rankings.csv) and for the
dev-vs-eval generalization table; they never enter a robustness estimate.
"""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

from analyze import load_jsonl, write_csv
import analyze_controllability_v3 as an
import controllability_v3_stats as st
import controllability_v3_stress_design as sd
import controllability_v3_stress_stats as ss
from controllability_v3_design import CUE_ORDER, INTERVENTION_IDS, MATCHED_CONTROL_ID
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID, profile_id_from_row
from controllability_v3_stress_config import DOSE_HEADLINE_CONTRASTS, DOSE_HIGH_DOSES, REVERSAL_TARGET_GAIN_PP, load_stress_config
from controllability_v3_study_config import STUDY_CONFIG_FILE, load_study_config
from controllability_v3_trials import FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT
from freeze_controllability_v3_stress import verify_frozen as verify_stress_frozen

STRESS_OUTPUT_TABLES = (
    "adversarial_attack_recovery.csv", "adversarial_effects_by_intervention.csv", "adversarial_effects_by_cue.csv",
    "attack_rankings.csv", "attack_generalization.csv", "adversarial_leave_one_story_out.csv",
    "dose_response_by_intervention.csv", "dose_response_by_pair_strength.csv", "dose_susceptibility_slopes.csv",
    "dose_reversal_thresholds.csv", "dose_intervention_interactions.csv", "known_random_high_dose_contrasts.csv",
    "stress_response_compliance.csv", "stress_token_diagnostics.csv", "stress_incomplete_units.csv", "stress_exclusions.csv",
)
CROSS_EVALUATOR_TABLES = ("cross_evaluator_attack_robustness.csv", "cross_evaluator_dose_response.csv")


def load_rows(path):
    return load_jsonl(path) if path and os.path.exists(path) else []


# ---------------------------------------------------------------------------
# Rates keyed by variant
# ---------------------------------------------------------------------------

def stress_cell_rates(rows):
    """{(intervention, cue, variant, s1, s2): {(assignment, position): rate}} plus counts."""
    lists = defaultdict(lambda: defaultdict(list))
    for row in rows:
        chosen = an.story1_chosen(row)
        if chosen is None:
            continue
        key = (row["intervention_id"], row["cue"], row["variant"], row["story_1_id"], row["story_2_id"])
        lists[key][(row["context_assignment"], row["display_position"])].append(chosen)
    return {k: {c: sum(v) / len(v) for c, v in cells.items()} for k, cells in lists.items()}


def variant_pair_effects(cell_rates):
    """{(intervention, cue, variant): {(s1, s2): d_pair}} and the same for the position diagnostics."""
    out = defaultdict(dict)
    for (iid, cue, variant, s1, s2), cells in cell_rates.items():
        r = st.d_pair_from_cell_rates(cells)
        if r is not None:
            out[(iid, cue, variant)][(s1, s2)] = r["d_pair"]
    return dict(out)


def ordinary_pair_effects(primary_pair_rows, pair_filter=None):
    """{(intervention, cue): {(s1, s2): d_pair}} from primary pair rows."""
    out = defaultdict(dict)
    for r in primary_pair_rows:
        key = (r["story_1_id"], r["story_2_id"])
        if pair_filter is None or f"{key[0]}_vs_{key[1]}" in pair_filter:
            out[(r["intervention_id"], r["cue_id"])][key] = r["d_pair"]
    return dict(out)


def pooled_over_cues(pairs_by_cue):
    return st.pooled_over_cues(pairs_by_cue)


# ---------------------------------------------------------------------------
# Per-profile analysis
# ---------------------------------------------------------------------------

def analyze_profile(profile_id, primary_rows, dev_rows, eval_rows, dose_rows, study_config, stress_config, n_draws, seed, split, selected):
    tag = {"experiment_id": stress_config["experiment_id"], "evaluator_profile_id": profile_id}
    min_den = stress_config.get("relative_min_denominator", 0.02)
    results = {"exclusions": [], "incomplete_units": []}

    # --- primary (same profile): ordinary effects, control, baseline strength
    by_id, excl = an.include_rows(primary_rows, (FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT), study_config, profile_id)
    results["exclusions"] += [{**tag, "file": "primary", **e} for e in excl]
    valid_ctx = [e["valid"] for e in by_id.values() if e["valid"] is not None and e["valid"]["family"] == FAMILY_PRIMARY_CONTEXT]
    valid_noctx = [e["valid"] for e in by_id.values() if e["valid"] is not None and e["valid"]["family"] == FAMILY_PRIMARY_NOCONTEXT]
    kept_ctx, _, _ = an.complete_units(valid_ctx, 4, {k: v for k, v in by_id.items() if v["first"]["family"] == FAMILY_PRIMARY_CONTEXT})
    kept_noctx, _, _ = an.complete_units(valid_noctx, 2, {k: v for k, v in by_id.items() if v["first"]["family"] == FAMILY_PRIMARY_NOCONTEXT})
    primary_pair_rows = an.pair_context_effects(an.context_cell_rates(kept_ctx))
    primary_cells = an.context_cell_rates(kept_ctx)
    baseline_table = an.baseline_strength_table([r for r in kept_noctx if r["intervention_id"] == MATCHED_CONTROL_ID])
    baseline = {(r["story_1_id"], r["story_2_id"]): r for r in baseline_table}

    # --- stress rows
    def include_stress(rows, family, n_cells=4):
        b, e = an.include_rows(rows, (family,), study_config, profile_id, id_prefix="v3s::")
        results["exclusions"] += [{**tag, "file": family, **x} for x in e]
        valid = [x["valid"] for x in b.values() if x["valid"] is not None]
        kept, n_complete, diag = an.complete_units(valid, n_cells, b)
        results["incomplete_units"] += [{**tag, "family": family, **d} for d in diag]
        return b, kept, n_complete

    dev_by_id, kept_dev, n_dev_units = include_stress(dev_rows, sd.FAMILY_ADV_DEV)
    eval_by_id, kept_eval, n_eval_units = include_stress(eval_rows, sd.FAMILY_ADV_EVAL)
    dose_by_id, kept_dose, n_dose_units = include_stress(dose_rows, sd.FAMILY_DOSE)

    story_ids = sorted({s for rows in (kept_ctx, kept_noctx, kept_dev, kept_eval, kept_dose) for r in rows for s in (r["story_1_id"], r["story_2_id"])})
    rng = random.Random(seed)
    resamples = st.make_story_resamples(story_ids, n_draws, rng) if story_ids else []

    results.update(_adversarial(tag, kept_eval, kept_dev, primary_pair_rows, primary_cells, split, selected, resamples, min_den, story_ids))
    results.update(_dose(tag, kept_dose, baseline, resamples, min_den, stress_config))

    all_by_id = {**dev_by_id, **eval_by_id, **dose_by_id}
    results["response_compliance"] = [{**tag, **r} for r in an.response_compliance(all_by_id)]
    results["token_diagnostics"] = [{**tag, **r} for r in an.token_diagnostics(all_by_id, study_config["max_output_tokens"])]
    results["completeness"] = {
        "evaluator_profile_id": profile_id, "n_primary_context_complete_units": len(kept_ctx) // 4,
        "n_dev_complete_units": n_dev_units, "n_eval_complete_units": n_eval_units, "n_dose_complete_units": n_dose_units,
        "n_dev_rows_valid": len([1 for x in dev_by_id.values() if x["valid"]]), "n_eval_rows_valid": len([1 for x in eval_by_id.values() if x["valid"]]),
        "n_dose_rows_valid": len([1 for x in dose_by_id.values() if x["valid"]]),
        "expected_dose_units": stress_config["dose_response"]["unique_cells"] // 4 * stress_config["dose_response"]["replicates"],
        "n_bootstrap_draws_kept": len(resamples), "bootstrap_seed": seed,
    }
    results["completeness"]["dose_complete"] = results["completeness"]["n_dose_complete_units"] == results["completeness"]["expected_dose_units"]
    return results


def _adversarial(tag, kept_eval, kept_dev, primary_pair_rows, primary_cells, split, selected, resamples, min_den, story_ids):
    out = {"attack_recovery": [], "effects_by_intervention": [], "effects_by_cue": [], "attack_rankings": [], "attack_generalization": [], "adversarial_loso": []}
    eval_pairs = set(split["attack_evaluation_pairs"]) if split else set()
    dev_pairs = set(split["attack_development_pairs"]) if split else set()
    ordinary_eval = ordinary_pair_effects(primary_pair_rows, eval_pairs)
    attack_effects = variant_pair_effects(stress_cell_rates(kept_eval))
    dev_effects = variant_pair_effects(stress_cell_rates(kept_dev))
    attack_cells = {}
    for (iid, cue, variant, s1, s2), cells in stress_cell_rates(kept_eval).items():
        for (asg, pos), p in cells.items():
            attack_cells.setdefault((iid, cue, variant), {})[((s1, s2), asg, pos)] = p
    ordinary_cells = defaultdict(dict)
    for (iid, cue, s1, s2), cells in primary_cells.items():
        if f"{s1}_vs_{s2}" in eval_pairs:
            for (asg, pos), p in cells.items():
                ordinary_cells[(iid, cue)][((s1, s2), asg, pos)] = p
    selected_rows = (selected or {}).get("selected", [])
    rank_of = {r["attack_id"]: r for r in selected_rows}
    for r in (selected or {}).get("ranked", []):
        out["attack_rankings"].append({**tag, **{k: v for k, v in r.items() if k != "template"}, "template_preview": (r.get("template") or "")[:120]})

    per_attack = {}
    for (iid, cue, variant), pairs in sorted(attack_effects.items()):
        # every pair here must be a held-out evaluation pair (asserted, never silently filtered)
        leaked = [p for p in pairs if f"{p[0]}_vs_{p[1]}" not in eval_pairs]
        if leaked:
            raise ValueError(f"attack evaluation rows for {variant} include non-evaluation pairs {leaked[:3]}")
        ordinary = ordinary_eval.get((iid, cue), {})
        control = ordinary_eval.get((MATCHED_CONTROL_ID, cue), {})
        rec = ss.attack_recovery(ordinary, pairs, control, resamples, min_den)
        if rec is None:
            continue
        flips = ss.choice_flip_rate(ordinary_cells.get((iid, cue), {}), attack_cells.get((iid, cue, variant), {}))
        flip_draws = [d["f"] for d in st.joint_draws({"f": flips}, resamples)] if flips else []
        sel = rank_of.get(variant, {})
        row = {**tag, "intervention_id": iid, "cue_id": cue, "attack_id": variant, "dev_rank": sel.get("rank"), "dev_score": sel.get("dev_score"),
               "attack_family": sel.get("attack_family"), **rec, **st.summarize("choice_flip_rate", st.mean(flips.values()) if flips else None, flip_draws)}
        out["attack_recovery"].append(row)
        per_attack[(iid, cue, variant)] = {"pairs": pairs, "rank": sel.get("rank"), "dev_score": sel.get("dev_score"), "recovery": rec["attack_recovery"],
                                          "recovery_lo": rec["attack_recovery_ci_95_lo"], "flips": flips}

    def pooled_attack_pairs(iid, cue=None, top1=False):
        """Per pair: mean over attacks (rank-1 only if top1) and over cues (if cue None)."""
        by_cue = defaultdict(lambda: defaultdict(list))
        for (i, c, v), info in per_attack.items():
            if i != iid or (cue is not None and c != cue) or (top1 and info["rank"] != 1):
                continue
            for p, val in info["pairs"].items():
                by_cue[c][p].append(val)
        return pooled_over_cues({c: {p: statistics.mean(v) for p, v in pairs.items()} for c, pairs in by_cue.items()})

    interventions = sorted({i for i, _, _ in per_attack}, key=INTERVENTION_IDS.index)
    for iid in interventions:
        for cue in [None] + [c for c in CUE_ORDER if any(i == iid and cc == cue_ for (i, cc, _) in per_attack for cue_ in [c])]:
            ord_map = pooled_over_cues({c: m for (i, c), m in ordinary_eval.items() if i == iid and (cue is None or c == cue)})
            ctrl_map = pooled_over_cues({c: m for (i, c), m in ordinary_eval.items() if i == MATCHED_CONTROL_ID and (cue is None or c == cue)})
            for mode, top1 in (("top1_by_dev_score", True), ("mean_of_selected", False)):
                atk_map = pooled_attack_pairs(iid, cue, top1)
                rec = ss.attack_recovery(ord_map, atk_map, ctrl_map, resamples, min_den)
                if rec is None:
                    continue
                attacks_here = [(i, c, v) for (i, c, v) in per_attack if i == iid and (cue is None or c == cue) and (not top1 or per_attack[(i, c, v)]["rank"] == 1)]
                n_atk = len(attacks_here)
                success = sum(1 for k in attacks_here if per_attack[k]["recovery"] > 0) / n_atk if n_atk else None
                success_ci = sum(1 for k in attacks_here if per_attack[k]["recovery_lo"] is not None and per_attack[k]["recovery_lo"] > 0) / n_atk if n_atk else None
                flips = defaultdict(list)
                for k in attacks_here:
                    for p, f in per_attack[k]["flips"].items():
                        flips[p].append(f)
                flips = {p: statistics.mean(v) for p, v in flips.items()}
                flip_draws = [d["f"] for d in st.joint_draws({"f": flips}, resamples)] if flips else []
                row = {**tag, "intervention_id": iid, "cue_id": cue or "all", "attack_set": mode, "n_attacks": n_atk, **rec,
                       "attack_success_rate_point": success, "attack_success_rate_ci_excludes_zero": success_ci,
                       **st.summarize("choice_flip_rate", st.mean(flips.values()) if flips else None, flip_draws)}
                (out["effects_by_intervention"] if cue is None else out["effects_by_cue"]).append(row)
                if cue is None and not top1:
                    rec_pairs = {p: atk_map[p] - ord_map[p] for p in atk_map if p in ord_map}
                    for story, val in st.leave_one_story_out(rec_pairs, story_ids).items():
                        out["adversarial_loso"].append({**tag, "intervention_id": iid, "excluded_story_id": story, "attack_recovery_excluding_story": val,
                                                        "CE_attack_excluding_story": st.leave_one_story_out(atk_map, story_ids)[story]})

    # generalization: dev score vs held-out recovery, per intervention across selected attacks
    for iid in interventions:
        pts = [(per_attack[k]["dev_score"], per_attack[k]["recovery"]) for k in per_attack if k[0] == iid and per_attack[k]["dev_score"] is not None]
        for k in per_attack:
            if k[0] == iid:
                dev_ce = st.mean(dev_effects.get(k, {}).values()) if dev_effects.get(k) else None
                out["attack_generalization"].append({**tag, "intervention_id": iid, "cue_id": k[1], "attack_id": k[2], "dev_rank": per_attack[k]["rank"],
                                                     "dev_score": per_attack[k]["dev_score"], "dev_ce_attack": dev_ce,
                                                     "eval_attack_recovery": per_attack[k]["recovery"], "eval_ce_attack": st.mean(per_attack[k]["pairs"].values())})
        if len(pts) >= 3:
            r = st.pearson_correlation([a for a, _ in pts], [b for _, b in pts])
            out["attack_generalization"].append({**tag, "intervention_id": iid, "cue_id": "all", "attack_id": "SUMMARY", "n_attacks": len(pts),
                                                 "dev_score_mean": st.mean(a for a, _ in pts), "eval_attack_recovery_mean": st.mean(b for _, b in pts),
                                                 "dev_eval_pearson_r": r, "shrinkage": st.mean(a for a, _ in pts) - st.mean(b for _, b in pts)})
    return {"adversarial_attack_recovery": out["attack_recovery"], "adversarial_effects_by_intervention": out["effects_by_intervention"],
            "adversarial_effects_by_cue": out["effects_by_cue"], "attack_rankings": out["attack_rankings"],
            "attack_generalization": out["attack_generalization"], "adversarial_leave_one_story_out": out["adversarial_loso"]}


def _dose(tag, kept_dose, baseline, resamples, min_den, stress_config):
    out = {k: [] for k in ("by_intervention", "by_strength", "slopes", "thresholds", "interactions", "known_random")}
    effects = variant_pair_effects(stress_cell_rates(kept_dose))   # (iid, cue, dNN) -> pairs
    by_int = defaultdict(dict)
    for (iid, cue, variant), pairs in effects.items():
        by_int[iid][int(variant[1:])] = pairs
    target = stress_config["dose_response"]["reversal_target_gain_pp"] / 100
    grid = tuple(stress_config["dose_response"]["grid"])
    strength = {p: r["baseline_strength"] for p, r in baseline.items()}
    median_strength = statistics.median(strength.values()) if strength else None
    halves = {p: ("strong_baseline_half" if strength[p] >= median_strength else "ambiguous_half") for p in strength} if strength else {}
    # favoured-wins counts for the logistic model
    counts = defaultdict(lambda: [0, 0])
    for r in kept_dose:
        chosen = an.story1_chosen(r)
        if chosen is None:
            continue
        fav_won = chosen if r["context_assignment"] == "forward" else (not chosen)
        counts[(r["intervention_id"], r["dose"], r["story_1_id"], r["story_2_id"], r["context_assignment"], r["display_position"])][0 if fav_won else 1] += 1
    quantiles = {}
    if strength:
        abs_lo = sorted(abs(r["baseline_signed_log_odds"]) for r in baseline.values())
        quantiles = {"p25": st.percentile(abs_lo, 25), "p50": st.percentile(abs_lo, 50), "p75": st.percentile(abs_lo, 75)}

    curves = {}
    for iid in [i for i in INTERVENTION_IDS if i in by_int]:
        pbd = by_int[iid]
        curve = ss.dose_curve(pbd, resamples)
        curves[iid] = {c["dose"]: c["CE"] for c in curve}
        for c in curve:
            dis = [(k, v) for k, v in counts.items() if k[0] == iid and k[1] == c["dose"] and (k[2], k[3]) in baseline
                   and (baseline[(k[2], k[3])]["baseline_signed_log_odds"] * (1 if k[4] == "forward" else -1)) < 0]
            won, n = sum(v[0] for _, v in dis), sum(v[0] + v[1] for _, v in dis)
            out["by_intervention"].append({**tag, "intervention_id": iid, **c, "p_favoured_wins_when_baseline_dispreferred": (won / n) if n else None,
                                           "n_obs_baseline_dispreferred": n})
        slope = ss.susceptibility_slope(pbd, resamples, target, grid)
        if slope:
            out["slopes"].append({**tag, "intervention_id": iid, **slope})
        cells = [{"story_1_id": k[2], "story_2_id": k[3], "b": baseline[(k[2], k[3])]["baseline_signed_log_odds"] * (1 if k[4] == "forward" else -1),
                  "x": ss.dose_x_of(k[1]), "y": v[0], "n": v[0] + v[1]} for k, v in counts.items() if k[0] == iid and (k[2], k[3]) in baseline]
        rev = ss.logistic_reversal(cells, quantiles, resamples) if cells and quantiles else None
        if rev:
            out["thresholds"].append({**tag, "intervention_id": iid, "plus_gain_threshold_dose": slope["threshold_dose"] if slope else None,
                                      "plus_gain_threshold_ci_95_lo": slope["threshold_dose_ci_95_lo"] if slope else None,
                                      "plus_gain_threshold_ci_95_hi": slope["threshold_dose_ci_95_hi"] if slope else None, **rev})
        if halves:
            half_slopes = {}
            for label in ("strong_baseline_half", "ambiguous_half"):
                sub = {d: {p: v for p, v in pairs.items() if halves.get(p) == label} for d, pairs in pbd.items()}
                sub = {d: pairs for d, pairs in sub.items() if pairs}
                if not sub:
                    continue
                sl = ss.susceptibility_slope(sub, resamples, target, grid)
                if sl:
                    half_slopes[label] = sl
                    out["by_strength"].append({**tag, "intervention_id": iid, "pair_strength_group": label, "n_story_pairs": len(next(iter(sub.values()))),
                                               **{f"CE_at_{c['dose']}": c["CE"] for c in ss.dose_curve(sub, resamples)}, **sl})
            if len(half_slopes) == 2:
                diff = ss.slope_difference({d: {p: v for p, v in pairs.items() if halves.get(p) == "strong_baseline_half"} for d, pairs in pbd.items()},
                                           {d: {p: v for p, v in pairs.items() if halves.get(p) == "ambiguous_half"} for d, pairs in pbd.items()}, resamples, min_den)
                if diff:
                    out["interactions"].append({**tag, "intervention_id": iid, "interaction": "ambiguous_minus_strong_slope",
                                                "slope_strong": diff["slope_control"], "slope_ambiguous": diff["slope_treated"],
                                                "difference": -diff["suppression_slope"], "difference_ci_95_lo": -diff["suppression_slope_ci_95_hi"],
                                                "difference_ci_95_hi": -diff["suppression_slope_ci_95_lo"]})
        if iid != MATCHED_CONTROL_ID and MATCHED_CONTROL_ID in by_int:
            diff = ss.slope_difference(by_int[MATCHED_CONTROL_ID], pbd, resamples, min_den)
            if diff:
                out["interactions"].append({**tag, "intervention_id": iid, "interaction": "suppression_slope_vs_I0", **diff})
    for a, b in DOSE_HEADLINE_CONTRASTS:
        if a in by_int and b in by_int:
            for dose in sorted(set(by_int[a]) & set(by_int[b])):
                ma, mb = by_int[a][dose], by_int[b][dose]
                common = sorted(set(ma) & set(mb))
                if not common:
                    continue
                draws = st.joint_draws({"a": {p: ma[p] for p in common}, "b": {p: mb[p] for p in common}}, resamples)
                out["known_random"].append({**tag, "contrast": f"{a}_vs_{b}", "intervention_id": a, "reference_id": b, "dose": dose,
                                            "high_dose": dose in DOSE_HIGH_DOSES, "n_story_pairs": len(common), "CE_intervention": st.mean(ma[p] for p in common),
                                            "CE_reference": st.mean(mb[p] for p in common),
                                            **st.summarize("CE_difference", st.mean(ma[p] for p in common) - st.mean(mb[p] for p in common), [d["a"] - d["b"] for d in draws])})
    return {"dose_response_by_intervention": out["by_intervention"], "dose_response_by_pair_strength": out["by_strength"],
            "dose_susceptibility_slopes": out["slopes"], "dose_reversal_thresholds": out["thresholds"],
            "dose_intervention_interactions": out["interactions"], "known_random_high_dose_contrasts": out["known_random"]}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_TABLE_KEYS = {
    "stress_response_compliance.csv": "response_compliance", "stress_token_diagnostics.csv": "token_diagnostics",
    "stress_incomplete_units.csv": "incomplete_units", "stress_exclusions.csv": "exclusions",
}


def _write_rows(rows, path):
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    write_csv(rows, fieldnames, path)


def write_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for filename in STRESS_OUTPUT_TABLES:
        rows = results.get(_TABLE_KEYS.get(filename, filename[:-4])) or []
        if rows:
            _write_rows(rows, os.path.join(out_dir, filename))
            written.append(filename)
    with open(os.path.join(out_dir, "stress_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    written.append("stress_summary.json")
    return written


def write_cross_evaluator_outputs(per_profile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    robustness = [row for r in per_profile.values() for row in r["adversarial_effects_by_intervention"]]
    dose = []
    for pid, r in per_profile.items():
        slopes = {x["intervention_id"]: x for x in r["dose_susceptibility_slopes"]}
        thresholds = {x["intervention_id"]: x for x in r["dose_reversal_thresholds"]}
        for iid in slopes:
            t = thresholds.get(iid, {})
            dose.append({"evaluator_profile_id": pid, "intervention_id": iid, "slope": slopes[iid]["slope"], "slope_ci_95_lo": slopes[iid]["slope_ci_95_lo"],
                         "slope_ci_95_hi": slopes[iid]["slope_ci_95_hi"], "threshold_dose": slopes[iid]["threshold_dose"],
                         "reversal_dose_p50": t.get("reversal_dose_p50"), "reversal_dose_p50_ci_95_lo": t.get("reversal_dose_p50_ci_95_lo"),
                         "reversal_dose_p50_ci_95_hi": t.get("reversal_dose_p50_ci_95_hi")})
    written = []
    for name, rows in (("cross_evaluator_attack_robustness.csv", robustness), ("cross_evaluator_dose_response.csv", dose)):
        if rows:
            _write_rows(rows, os.path.join(out_dir, name))
            written.append(name)
    return written


def run_stress_analysis(primary_results_file, dev_results_file, eval_results_file, dose_results_file, analysis_dir, bootstrap_draws=None,
                        study_config_file=STUDY_CONFIG_FILE, extra_profile_files=None):
    study_config = load_study_config(study_config_file)
    stress_config = load_stress_config()
    ok, reason = verify_stress_frozen("design")
    if not ok:
        print(f"Stress design is not frozen or does not match its lock ({reason}); outputs are labelled accordingly.")
    n_draws = bootstrap_draws or stress_config["bootstrap_draws"]
    split = sd.load_split(stress_config["adversarial"]["split_file"]) if os.path.exists(stress_config["adversarial"]["split_file"]) else None
    selected = json.load(open(stress_config["adversarial"]["selected_attacks_file"])) if os.path.exists(stress_config["adversarial"]["selected_attacks_file"]) else None
    files = [{"primary": primary_results_file, "dev": dev_results_file, "eval": eval_results_file, "dose": dose_results_file}] + list(extra_profile_files or [])
    rows = {k: [] for k in ("primary", "dev", "eval", "dose")}
    for f in files:
        for k in rows:
            rows[k] += load_rows(f.get(k))
    print(f"Raw rows: primary={len(rows['primary'])} dev={len(rows['dev'])} eval={len(rows['eval'])} dose={len(rows['dose'])}")
    profiles = sorted({profile_id_from_row(r) for k in ("dev", "eval", "dose") for r in rows[k]} | {PRIMARY_PROFILE_ID})
    per_profile = {}
    for pid in profiles:
        res = analyze_profile(pid, rows["primary"], rows["dev"], rows["eval"], rows["dose"], study_config, stress_config, n_draws, stress_config["random_seed"], split, selected)
        res["stress_design_frozen"] = ok
        out_dir = analysis_dir if pid == PRIMARY_PROFILE_ID else os.path.join(analysis_dir, "profiles", pid)
        written = write_outputs(res, out_dir)
        c = res["completeness"]
        print(f"[{pid}] complete units: dev={c['n_dev_complete_units']} eval={c['n_eval_complete_units']} dose={c['n_dose_complete_units']}/{c['expected_dose_units']}  wrote {len(written)} file(s) to {out_dir}")
        per_profile[pid] = res
    active = {p: r for p, r in per_profile.items() if r["completeness"]["n_dose_rows_valid"] + r["completeness"]["n_eval_rows_valid"] > 0}
    if len(active) > 1:
        print(f"Wrote cross-evaluator tables: {write_cross_evaluator_outputs(active, analysis_dir)}")
    return per_profile


def main():
    from run_controllability_v3 import ADV_DEV_RESULTS_FILE, ADV_EVAL_RESULTS_FILE, DOSE_RESULTS_FILE, PRIMARY_RESULTS_FILE, STRESS_ANALYSIS_DIR
    parser = argparse.ArgumentParser(description="v3 stress analysis (no API calls).")
    parser.add_argument("--primary-results-file", default=PRIMARY_RESULTS_FILE)
    parser.add_argument("--dev-results-file", default=ADV_DEV_RESULTS_FILE)
    parser.add_argument("--eval-results-file", default=ADV_EVAL_RESULTS_FILE)
    parser.add_argument("--dose-results-file", default=DOSE_RESULTS_FILE)
    parser.add_argument("--analysis-dir", default=STRESS_ANALYSIS_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    args = parser.parse_args()
    run_stress_analysis(args.primary_results_file, args.dev_results_file, args.eval_results_file, args.dose_results_file, args.analysis_dir, args.bootstrap_draws)


if __name__ == "__main__":
    main()
