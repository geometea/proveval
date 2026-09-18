"""Analyses for the three experimental families (per evaluator profile,
never pooled): adaptive dose, iterative adversary, paired capability sweep.
Writes only under results/controllability_v3/analysis/experimental/."""

import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict

from analyze import load_jsonl, write_csv
import analyze_controllability_v3 as an
import analyze_controllability_v3_stress as ast
import controllability_v3_adaptive_dose as ad
import controllability_v3_capability_sweep as cap
import controllability_v3_iterative_attack as it
import controllability_v3_stats as st
import controllability_v3_stress_design as sd
import controllability_v3_stress_stats as ss
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID, profile_id_from_row
from controllability_v3_trials import FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT

EXPERIMENTAL_ANALYSIS_DIR = "results/controllability_v3/analysis/experimental"


def load_rows(path):
    return load_jsonl(path) if path and os.path.exists(path) else []


def _write(rows, path):
    if not rows:
        return False
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    write_csv(rows, fields, path)
    return True


def _valid_rows(rows, family, profile_id, study_config, id_prefix):
    by_id, excl = an.include_rows(rows, (family,), study_config, profile_id, id_prefix=id_prefix)
    return [e["valid"] for e in by_id.values() if e["valid"] is not None], excl, by_id


# ---------------------------------------------------------------------------
# Adaptive dose
# ---------------------------------------------------------------------------

def analyze_adaptive(rows, schedule, state, policy, profile_id, study_config, fixed_rows=None, n_draws=500, seed=1):
    valid, excl, _ = _valid_rows(rows, ad.FAMILY, profile_id, study_config, "v3a::")
    by_unit = defaultdict(list)
    for r in valid:
        by_unit[(r["intervention_id"], state["strata"].get(r["story_pair"], "all"))].append(r)
    thresholds, slopes, stops, decisions = [], [], [], []
    stop_by_unit = {e["unit"]: e for e in schedule if e.get("stop")}
    rounds_by_unit = defaultdict(int)
    for e in schedule:
        if not e.get("stop"):
            rounds_by_unit[e["unit"]] += 1
    story_ids = sorted({s for r in valid for s in (r["story_1_id"], r["story_2_id"])})
    resamples = st.make_story_resamples(story_ids, n_draws, random.Random(seed)) if story_ids else []
    for (iid, stratum), unit_rows in sorted(by_unit.items()):
        key = ad.unit_key(iid, stratum)
        obs = ad.unit_observations(unit_rows)
        fit = ad.fit_unit(obs)
        base = {"evaluator_profile_id": profile_id, "intervention_id": iid, "stratum": stratum, "n_judgments": len(unit_rows), "rounds": rounds_by_unit.get(key, 0),
                "doses_visited": sorted({r["dose"] for r in unit_rows})}
        if fit is None:
            thresholds.append({**base, "fit": "not identifiable"})
            continue
        a, b, cov = fit
        row = {**base, "alpha": a, "beta": b, "beta_se": math.sqrt(max(cov[1][1], 0.0)), "p_at_zero_evidence": ss.sigmoid(a)}
        for name, gain in policy["targets"].items():
            t = ad.target_dose_and_se(a, b, cov, gain)
            row.update({f"{name}_dose": t["dose"] if t else None, f"{name}_ci_lo": t["ci_lo"] if t else None, f"{name}_ci_hi": t["ci_hi"] if t else None,
                        f"{name}_extrapolated": (None if not t else not (min(policy["allowed_doses"]) <= t["dose"] <= max(policy["allowed_doses"])))})
        rev = ad.reversal_dose(a, b, cov)
        row.update({"d50_dose": rev["dose"] if rev else None, "d50_ci_lo": rev["ci_lo"] if rev else None, "d50_ci_hi": rev["ci_hi"] if rev else None})
        # story-bootstrap check of the primary target (pairs resampled by story)
        pair_rows = defaultdict(list)
        for r in unit_rows:
            pair_rows[(r["story_1_id"], r["story_2_id"])].append(r)
        draws = []
        for m in resamples:
            weighted = []
            for pair, prs in pair_rows.items():
                w = m.get(pair[0], 0) * m.get(pair[1], 0)
                if w:
                    weighted.extend(prs * w)
            f = ad.fit_unit(ad.unit_observations(weighted)) if weighted else None
            if f:
                t = ad.target_dose_and_se(f[0], f[1], f[2], policy["targets"][policy["primary_target"]])
                if t:
                    draws.append(t["dose"])
        lo, hi = st.ci95(draws)
        row.update({f"{policy['primary_target']}_story_bootstrap_ci_lo": lo, f"{policy['primary_target']}_story_bootstrap_ci_hi": hi, "n_bootstrap_draws_used": len(draws)})
        thresholds.append(row)
        slopes.append({**base, "beta": b, "beta_se": row["beta_se"], "beta_ci_lo": b - 1.96 * row["beta_se"], "beta_ci_hi": b + 1.96 * row["beta_se"]})
    for key in sorted(set(rounds_by_unit) | set(stop_by_unit)):
        iid, stratum = key.split("|")
        stops.append({"evaluator_profile_id": profile_id, "intervention_id": iid, "stratum": stratum, "rounds": rounds_by_unit.get(key, 0),
                      "stopping_reason": stop_by_unit.get(key, {}).get("stop"), "stopped": key in stop_by_unit})
    for e in schedule:
        decisions.append({"evaluator_profile_id": profile_id, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in e.items() if k != "observation_ids"}, "n_observation_ids": len(e.get("observation_ids", []))})
    comparison = []
    if fixed_rows:
        fixed_valid, _, _ = _valid_rows(fixed_rows, sd.FAMILY_DOSE, profile_id, study_config, "v3s::")
        fixed_by_int = defaultdict(list)
        for r in fixed_valid:
            fixed_by_int[r["intervention_id"]].append(r)
        for iid, frs in sorted(fixed_by_int.items()):
            adaptive_rows = [r for (i, _), rs in by_unit.items() if i == iid for r in rs]
            for label, rs in (("fixed_grid", frs), ("adaptive_pooled_over_strata", adaptive_rows)):
                f = ad.fit_unit(ad.unit_observations(rs)) if rs else None
                t = ad.target_dose_and_se(f[0], f[1], f[2], policy["targets"][policy["primary_target"]]) if f else None
                comparison.append({"evaluator_profile_id": profile_id, "intervention_id": iid, "design": label, "n_judgments": len(rs),
                                   "beta": f[1] if f else None, "d10_dose": t["dose"] if t else None, "d10_ci_width": t["ci_width"] if t else None})
    return {"adaptive_thresholds": thresholds, "adaptive_slopes": slopes, "adaptive_stopping_reasons": stops, "adaptive_decisions": decisions,
            "adaptive_vs_fixed_grid": comparison, "exclusions": [{"family": ad.FAMILY, **e} for e in excl]}


# ---------------------------------------------------------------------------
# Iterative adversary
# ---------------------------------------------------------------------------

def iterative_dev_effects(dev_rows, profile_id, study_config):
    valid, excl, by_id = _valid_rows(dev_rows, it.FAMILY_DEV, profile_id, study_config, "v3i::")
    kept, _, _ = an.complete_units(valid, 4, by_id)
    return {k[2]: v for k, v in ast.variant_pair_effects(ast.stress_cell_rates(kept)).items()}, excl


def analyze_iterative(records, dev_rows, eval_rows, primary_rows, policy, split, profile_id, study_config, selected=None, n_draws=500, seed=1):
    dev_effects, excl = iterative_dev_effects(dev_rows, profile_id, study_config)
    dev_pairs, eval_pairs = set(split["attack_development_pairs"]), set(split["attack_evaluation_pairs"])
    leaked = {p for pairs in dev_effects.values() for p in pairs if f"{p[0]}_vs_{p[1]}" in eval_pairs}
    if leaked:
        raise ValueError(f"development results contain held-out evaluation pairs: {sorted(leaked)[:3]}")
    ordinary_dev, ordinary_eval, control_dev, control_eval = {}, {}, {}, {}
    if primary_rows:
        by_id, _ = an.include_rows(primary_rows, (FAMILY_PRIMARY_CONTEXT,), study_config, profile_id)
        pv = [e["valid"] for e in by_id.values() if e["valid"] is not None]
        kept, _, _ = an.complete_units(pv, 4, by_id)
        pr = an.pair_context_effects(an.context_cell_rates(kept))
        ordinary_dev, ordinary_eval = ast.ordinary_pair_effects(pr, dev_pairs), ast.ordinary_pair_effects(pr, eval_pairs)
    scores = it.score_members(it.members(records), dev_effects, ordinary_dev, policy["min_dev_pairs_for_score"])
    per_gen, curve = [], []
    for key in it.keys(policy):
        cue, iid = key
        control = ordinary_dev.get(("I0", cue))
        control_ce = statistics.mean(control.values()) if control else None
        cumulative, broken_at, broken_queries = 0, None, None
        for g in sorted({m["generation"] for m in it.members(records, key)}):
            gen_members = [m for m in it.members(records, key) if m["generation"] == g]
            scored = [scores[m["attack_id"]] for m in gen_members if scores[m["attack_id"]]["scored"]]
            cumulative += sum(scores[m["attack_id"]]["n_dev_pairs"] for m in gen_members) * 4 * policy["dev_replicates"]
            best = max((s["dev_score"] for s in scored), default=None)
            best_ce = max((s["dev_ce_attack"] for s in scored), default=None)
            if broken_at is None and any(it.is_broken(s, control_ce, policy) for s in scored):
                broken_at, broken_queries = g, cumulative
            per_gen.append({"evaluator_profile_id": profile_id, "cue_id": cue, "intervention_id": iid, "generation": g, "n_members": len(gen_members),
                            "n_valid": sum(1 for m in gen_members if m.get("validation_ok")), "n_scored": len(scored), "best_dev_score": best,
                            "mean_dev_score": statistics.mean(s["dev_score"] for s in scored) if scored else None, "best_dev_ce_attack": best_ce,
                            "dev_ce_ordinary": next((s["dev_ce_ordinary"] for s in scored if s["dev_ce_ordinary"] is not None), None),
                            "cumulative_dev_judgments": cumulative, "n_duplicates_rejected": sum(1 for m in gen_members if any("duplicate" in r for r in m.get("validation_reasons", [])))})
        best_ever = max((scores[m["attack_id"]]["dev_score"] for m in it.members(records, key) if scores.get(m["attack_id"], {}).get("scored")), default=None)
        curve.append({"evaluator_profile_id": profile_id, "cue_id": cue, "intervention_id": iid, "generations_completed": it.current_generation(records, key),
                      "best_dev_score_ever": best_ever, "broken": broken_at is not None, "broken_at_generation": broken_at, "queries_to_break": broken_queries,
                      "break_rule": f"recovery >= {policy['break_recovery_pp']} and CE_attack >= {policy['break_fraction_of_control']} x CE_I0_ordinary_dev (when available)",
                      "total_dev_judgments": cumulative, "control_ce_dev": control_ce})
    generalization = []
    if eval_rows:
        valid, excl_e, by_id = _valid_rows(eval_rows, it.FAMILY_EVAL, profile_id, study_config, "v3i::")
        kept, _, _ = an.complete_units(valid, 4, by_id)
        eval_effects = ast.variant_pair_effects(ast.stress_cell_rates(kept))
        story_ids = sorted({s for r in kept for s in (r["story_1_id"], r["story_2_id"])})
        resamples = st.make_story_resamples(story_ids, n_draws, random.Random(seed)) if story_ids else []
        for (iid, cue, aid), pairs in sorted(eval_effects.items()):
            if any(f"{p[0]}_vs_{p[1]}" not in eval_pairs for p in pairs):
                raise ValueError(f"held-out results for {aid} include non-evaluation pairs")
            rec = ss.attack_recovery(ordinary_eval.get((iid, cue), {}), pairs, ordinary_eval.get(("I0", cue), {}), resamples) if ordinary_eval else None
            s = scores.get(aid, {})
            generalization.append({"evaluator_profile_id": profile_id, "cue_id": cue, "intervention_id": iid, "attack_id": aid, "generation": next((m["generation"] for m in it.members(records) if m["attack_id"] == aid), None),
                                   "dev_score": s.get("dev_score"), "dev_ce_attack": s.get("dev_ce_attack"), "heldout_ce_attack": statistics.mean(pairs.values()),
                                   **({k: rec[k] for k in ("CE_ordinary", "attack_recovery", "attack_recovery_ci_95_lo", "attack_recovery_ci_95_hi", "robust_suppression")} if rec else {}),
                                   "dev_to_heldout_shrinkage": (s.get("dev_score") - rec["attack_recovery"]) if (rec and s.get("dev_score") is not None) else None})
    return {"iterative_attack_generation_summary": per_gen, "iterative_attack_learning_curve": curve, "iterative_attack_generalization": generalization,
            "exclusions": [{"family": it.FAMILY_DEV, **e} for e in excl]}


# ---------------------------------------------------------------------------
# Capability sweep
# ---------------------------------------------------------------------------

def _cap_rates(rows):
    ctx = defaultdict(lambda: defaultdict(list))
    noctx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        chosen = an.story1_chosen(r)
        if chosen is None:
            continue
        pair = (r["story_1_id"], r["story_2_id"])
        if r["condition"] == "nocontext":
            noctx[r["intervention_id"]][pair][r["display_position"]].append(chosen)
        else:
            ctx[(r["intervention_id"], r["condition"], r["cue"], r["variant"], pair)][(r["context_assignment"], r["display_position"])].append(chosen)
    ctx_rates = {k: {c: sum(v) / len(v) for c, v in cells.items()} for k, cells in ctx.items()}
    noctx_rates = {i: {p: {pos: {"p": sum(v) / len(v), "n": len(v)} for pos, v in positions.items()} for p, positions in pairs.items()} for i, pairs in noctx.items()}
    return ctx_rates, noctx_rates


def _cap_pairs(ctx_rates, iid, condition, variant_filter=None):
    by_cue = defaultdict(dict)
    for (i, cond, cue, variant, pair), cells in ctx_rates.items():
        if i == iid and cond == condition and (variant_filter is None or variant_filter(variant)):
            d = st.d_pair_from_cell_rates(cells)
            if d is not None:
                by_cue[(cue, variant)][pair] = d["d_pair"]
    return st.pooled_over_cues(by_cue) if by_cue else {}


def analyze_capability(rows_by_profile, design, study_config, n_draws=500, seed=1):
    per_profile, metrics = [], {}
    story_ids = sorted({s for rows in rows_by_profile.values() for r in rows for s in (r["story_1_id"], r["story_2_id"])})
    resamples = st.make_story_resamples(story_ids, n_draws, random.Random(seed)) if story_ids else []
    rates = {}
    for pid, rows in rows_by_profile.items():
        valid, excl, by_id = _valid_rows(rows, cap.FAMILY, pid, study_config, "v3c::")
        ctx_rates, noctx_rates = _cap_rates(valid)
        rates[pid] = (ctx_rates, noctx_rates)
        control = _cap_pairs(ctx_rates, "I0", "ordinary")
        for iid in design["interventions"]:
            ordinary = _cap_pairs(ctx_rates, iid, "ordinary")
            sup = st.suppression_estimates(control, ordinary, resamples) if control and ordinary else None
            drift = st.drift_estimates(noctx_rates.get("I0", {}), noctx_rates.get(iid, {}), resamples, same_sample=(iid == "I0")) if noctx_rates.get(iid) else None
            attacked = _cap_pairs(ctx_rates, iid, "attack")
            rec = ss.attack_recovery(ordinary, attacked, control, resamples) if attacked else None
            dose_pairs = {int(v[1:]): _cap_pairs(ctx_rates, iid, "dose", lambda var, v=v: var == v) for v in {k[3] for k in ctx_rates if k[0] == iid and k[1] == "dose"}}
            slope = ss.susceptibility_slope(dose_pairs, resamples, 0.10, tuple(sorted(dose_pairs))) if len(dose_pairs) >= 2 else None
            high = dose_pairs.get(max(dose_pairs)) if dose_pairs else None
            high_draws = [d["h"] for d in st.joint_draws({"h": high}, resamples)] if high else []
            blind = noctx_rates.get(iid, {})
            row = {"evaluator_profile_id": pid, "intervention_id": iid, "n_pairs": len(ordinary),
                   "blind_p_story1_mean": st.mean(st.mean(v["p"] for v in pr.values()) for pr in blind.values()) if blind else None,
                   **({k: sup[k] for k in ("CE_control", "CE_treated", "signed_suppression", "signed_suppression_ci_95_lo", "signed_suppression_ci_95_hi", "magnitude_suppression")} if sup else {}),
                   **({"drift_rms": drift["rms_prob_shift_noise_corrected"], "drift_rms_ci_lo": drift["rms_prob_shift_noise_corrected_ci_95_lo"], "drift_rms_ci_hi": drift["rms_prob_shift_noise_corrected_ci_95_hi"]} if drift else {}),
                   **({"attack_recovery": rec["attack_recovery"], "attack_recovery_ci_95_lo": rec["attack_recovery_ci_95_lo"], "attack_recovery_ci_95_hi": rec["attack_recovery_ci_95_hi"]} if rec else {}),
                   **({"dose_slope": slope["slope"], "dose_slope_ci_95_lo": slope["slope_ci_95_lo"], "dose_slope_ci_95_hi": slope["slope_ci_95_hi"]} if slope else {}),
                   **(st.summarize("high_dose_residual_ce", st.mean(high.values()), high_draws) if high else {}), "high_dose": max(dose_pairs) if dose_pairs else None}
            per_profile.append(row)
            metrics[(pid, iid)] = {"supp": (control, ordinary), "attack": (ordinary, attacked, control), "dose": dose_pairs, "blind": blind, "ce": ordinary}
    between = []
    pids = sorted(rows_by_profile)
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            for iid in design["interventions"]:
                ma, mb = metrics.get((a, iid)), metrics.get((b, iid))
                if not ma or not mb:
                    continue
                row = {"profile_a": a, "profile_b": b, "intervention_id": iid}
                if ma["blind"] and mb["blind"]:
                    d = st.drift_estimates(ma["blind"], mb["blind"], resamples)
                    row.update({"blind_preference_disagreement_rms": d["rms_prob_shift_noise_corrected"], "blind_preference_disagreement_rms_ci_lo": d["rms_prob_shift_noise_corrected_ci_95_lo"],
                                "blind_preference_disagreement_rms_ci_hi": d["rms_prob_shift_noise_corrected_ci_95_hi"], "blind_spearman": d["story_win_rate_spearman"]})
                for name, key_a, key_b in (("context_susceptibility", ma["ce"], mb["ce"]),):
                    common = sorted(set(key_a) & set(key_b))
                    if common:
                        dr = st.joint_draws({"a": {p: key_a[p] for p in common}, "b": {p: key_b[p] for p in common}}, resamples)
                        row.update(st.summarize(f"delta_{name}", st.mean(key_a[p] for p in common) - st.mean(key_b[p] for p in common), [x["a"] - x["b"] for x in dr]))
                ca, ta = ma["supp"]; cb, tb = mb["supp"]
                common = sorted(set(ca) & set(ta) & set(cb) & set(tb))
                if common:
                    dr = st.joint_draws({"ca": {p: ca[p] for p in common}, "ta": {p: ta[p] for p in common}, "cb": {p: cb[p] for p in common}, "tb": {p: tb[p] for p in common}}, resamples)
                    point = (st.mean(ca[p] for p in common) - st.mean(ta[p] for p in common)) - (st.mean(cb[p] for p in common) - st.mean(tb[p] for p in common))
                    row.update(st.summarize("delta_suppression", point, [(x["ca"] - x["ta"]) - (x["cb"] - x["tb"]) for x in dr]))
                oa, aa, _ = ma["attack"]; ob, ab, _ = mb["attack"]
                common = sorted(set(oa) & set(aa) & set(ob) & set(ab)) if aa and ab else []
                if common:
                    dr = st.joint_draws({"oa": {p: oa[p] for p in common}, "aa": {p: aa[p] for p in common}, "ob": {p: ob[p] for p in common}, "ab": {p: ab[p] for p in common}}, resamples)
                    point = (st.mean(aa[p] for p in common) - st.mean(oa[p] for p in common)) - (st.mean(ab[p] for p in common) - st.mean(ob[p] for p in common))
                    row.update(st.summarize("delta_attack_recovery", point, [(x["aa"] - x["oa"]) - (x["ab"] - x["ob"]) for x in dr]))
                if ma["dose"] and mb["dose"]:
                    diff = ss.slope_difference(ma["dose"], mb["dose"], resamples)
                    if diff:
                        row.update({"delta_dose_slope": diff["suppression_slope"], "delta_dose_slope_ci_lo": diff["suppression_slope_ci_95_lo"], "delta_dose_slope_ci_hi": diff["suppression_slope_ci_95_hi"]})
                between.append(row)
    return {"capability_per_profile": per_profile, "capability_between_profiles": between}


def write_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key, rows in results.items():
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            if _write(rows, os.path.join(out_dir, f"{key}.csv")):
                written.append(f"{key}.csv")
    with open(os.path.join(out_dir, "experimental_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return written + ["experimental_summary.json"]
