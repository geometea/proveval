"""v2 context-controllability analysis: loading, evaluator partitioning,
complete-superblock/unit filtering, the full CSV/JSON/plot output set, and
the CLI entry point. All the actual statistics (story bootstrap, D_pair/ATE,
controllability measures, equivalence, ambiguity regression) live in
controllability_v2_stats.py as pure functions -- this module is the I/O and
orchestration layer around them, mirroring the repo's existing split between
context_analysis_stats.py (pure math) and analyze_controllability.py
(orchestration) for v1.

Evaluator separation (item 9/18): every treatment/baseline result row
carries a full "evaluator" identity dict (see
controllability_v2_execution.build_evaluator_identity). partition_by_resolved_evaluator()
matches rows against the frozen study config's primary_evaluator/
replication_evaluators by (provider, requested_model, reasoning_profile),
and further splits any one configured evaluator into separate resolved
partitions if it turns out to have returned more than one distinct
response_model -- so two different actual model versions served under what
was requested as one evaluator are NEVER pooled into one inferential
sample. Analysis runs separately per resolved evaluator partition; a
cross-model summary table is produced afterward, never a pooled one.
"""

import argparse
import json
import os
import random
from collections import defaultdict

from analyze import load_jsonl, write_csv
from controllability_v2_corpus import CORPUS_FILE
from controllability_v2_study_config import STUDY_CONFIG_FILE, load_study_config
from controllability_v2_trials import BASELINE_TRIALS_FILE, TRIALS_FILE
from freeze_controllability_v2 import verify_frozen
from run_controllability_v2 import BASELINE_RESULTS_FILE, TREATMENT_RESULTS_FILE
import controllability_v2_stats as st

ANALYSIS_DIR = "results/controllability_v2/analysis"

REQUIRED_CELLS = {("forward", "story1_as_a"), ("forward", "story2_as_a"), ("flipped", "story1_as_a"), ("flipped", "story2_as_a")}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(path):
    if not os.path.exists(path):
        return []
    return load_jsonl(path)


def resolved_response_model(row):
    """The response_model of the attempt that actually resolved this
    observation (the last attempt, since run_one_observation stops
    retrying as soon as one is valid) -- None if never resolved."""
    if row.get("parsing_status") != "resolved":
        return None
    attempts = row.get("attempts") or []
    return attempts[-1].get("response_model") if attempts else None


def evaluator_key(row):
    ev = row.get("evaluator") or {}
    return (ev.get("provider"), ev.get("requested_model"), ev.get("reasoning_profile"))


def partition_by_resolved_evaluator(rows, study_config, strict=False):
    """Returns (partitions, unmatched, warnings).

    partitions: {resolved_evaluator_id: {"config": {...}, "rows": [...]}}.
    A configured evaluator that resolved to more than one response_model
    version is split into "{evaluator_id}#{response_model}" sub-partitions
    (never pooled) unless strict=True, in which case that's a hard error
    (item 9: "split them into separate evaluators or fail analysis").
    Rows whose (provider, requested_model, reasoning_profile) doesn't match
    ANY configured evaluator are returned separately as `unmatched`, never
    silently included in any partition.
    """
    configured = [study_config["primary_evaluator"], *study_config.get("replication_evaluators", [])]
    id_to_config = {e["evaluator_id"]: e for e in configured}
    key_to_id = {(e["provider"], e["requested_model"], e["reasoning_profile"]): e["evaluator_id"] for e in configured}

    matched = defaultdict(list)
    unmatched = []
    for row in rows:
        eid = key_to_id.get(evaluator_key(row))
        (matched[eid] if eid is not None else unmatched).append(row)
    if None in matched:
        unmatched.extend(matched.pop(None))

    partitions, warnings = {}, []
    for evaluator_id, ev_rows in matched.items():
        by_version = defaultdict(list)
        for row in ev_rows:
            by_version[resolved_response_model(row)].append(row)
        versions_seen = sorted(v for v in by_version if v is not None)
        cfg = id_to_config[evaluator_id]

        if len(versions_seen) > 1:
            if strict:
                raise ValueError(
                    f"Evaluator {evaluator_id!r} resolved to multiple response_model versions "
                    f"{versions_seen} -- refusing to analyze (strict mode). Split into separate "
                    f"evaluators in the study config, or re-run without --strict-model-version."
                )
            warnings.append(
                f"Evaluator {evaluator_id!r} resolved to multiple response_model versions {versions_seen} "
                f"-- split into separate evaluators, never pooled."
            )
            for version, version_rows in by_version.items():
                sub_id = f"{evaluator_id}#{version}" if version is not None else f"{evaluator_id}#unresolved"
                partitions[sub_id] = {"config": {**cfg, "response_model": version}, "rows": version_rows}
        else:
            version = versions_seen[0] if versions_seen else None
            partitions[evaluator_id] = {"config": {**cfg, "response_model": version}, "rows": ev_rows}

    return partitions, unmatched, warnings


# ---------------------------------------------------------------------------
# Complete-unit filtering (item 4/17: primary analysis uses only complete
# planned superblocks/baseline units)
# ---------------------------------------------------------------------------

def filter_complete_superblocks(rows):
    """Group by (superblock_id, replicate_number); a group counts as
    complete only if it has all 8 planned cells AND every one resolved to a
    valid response. Returns (kept_rows, n_complete, n_incomplete)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["superblock_id"], row["replicate_number"])].append(row)

    kept, n_complete, n_incomplete = [], 0, 0
    for group_rows in groups.values():
        if len(group_rows) == 8 and all(r["parsing_status"] == "resolved" for r in group_rows):
            n_complete += 1
            kept.extend(group_rows)
        else:
            n_incomplete += 1
    return kept, n_complete, n_incomplete


def filter_complete_baseline_units(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["block_id"], row["replicate_number"])].append(row)

    kept, n_complete, n_incomplete = [], 0, 0
    for group_rows in groups.values():
        if len(group_rows) == 2 and all(r["parsing_status"] == "resolved" for r in group_rows):
            n_complete += 1
            kept.extend(group_rows)
        else:
            n_incomplete += 1
    return kept, n_complete, n_incomplete


# ---------------------------------------------------------------------------
# Story identity (never raw A/B letter identity) -- item "estimators operate
# on story identity rather than raw A/B identity"
# ---------------------------------------------------------------------------

def story1_chosen(row):
    """True/False, or None if this observation never resolved. Maps the
    raw "A"/"B" choice back to which STORY was displayed as that letter
    (story_a_id/story_b_id), then compares to story_1_id -- the raw letter
    is never treated as the estimand by itself."""
    response = row.get("first_valid_response")
    if response is None:
        return None
    meta = row["trial_meta"]
    chosen_id = meta["story_a_id"] if response.get("overall_quality") == "A" else meta["story_b_id"]
    return chosen_id == meta["story_1_id"]


# ---------------------------------------------------------------------------
# Treatment: cell rates -> pair-level D_pair/position diagnostics
# ---------------------------------------------------------------------------

def build_treatment_cell_rates(rows):
    """{(contrast_id, instruction_condition, story_1_id, story_2_id): {(assignment, position): rate}}."""
    cell_lists = defaultdict(lambda: defaultdict(list))
    for row in rows:
        chosen = story1_chosen(row)
        if chosen is None:
            continue
        meta = row["trial_meta"]
        group_key = (meta["contrast_id"], meta["instruction_condition"], meta["story_1_id"], meta["story_2_id"])
        cell_lists[group_key][(meta["assignment"], meta["position"])].append(chosen)

    return {
        group_key: {cell_key: sum(vals) / len(vals) for cell_key, vals in cells.items()}
        for group_key, cells in cell_lists.items()
    }


def build_pair_context_effects(cell_rates_by_group):
    """One row per (contrast, instruction_condition, story pair) with a
    complete 4-cell rate set -- see controllability_v2_stats.d_pair_from_cell_rates."""
    rows = []
    for (contrast_id, instruction_condition, s1, s2), cells in cell_rates_by_group.items():
        result = st.d_pair_from_cell_rates(cells)
        if result is None:
            continue
        rows.append({"contrast_id": contrast_id, "instruction_condition": instruction_condition, "story_1_id": s1, "story_2_id": s2, **result})
    return rows


def pair_values_for(pair_rows, contrast_id, instruction_condition):
    return {
        (r["story_1_id"], r["story_2_id"]): r["d_pair"]
        for r in pair_rows
        if r["contrast_id"] == contrast_id and r["instruction_condition"] == instruction_condition
    }


# ---------------------------------------------------------------------------
# Baseline: independent pair stats + split-half reliability
# ---------------------------------------------------------------------------

def build_baseline_pair_table(rows):
    counts = defaultdict(lambda: [0, 0])
    per_pair_replicate_choices = defaultdict(list)
    for row in rows:
        chosen = story1_chosen(row)
        if chosen is None:
            continue
        meta = row["trial_meta"]
        key = (meta["story_1_id"], meta["story_2_id"])
        counts[key][0 if chosen else 1] += 1
        per_pair_replicate_choices[key].append((row["replicate_number"], chosen))

    table = [{"story_1_id": s1, "story_2_id": s2, **st.baseline_pair_stats(w1, w2)} for (s1, s2), (w1, w2) in counts.items()]
    reliability = st.split_half_baseline_reliability(per_pair_replicate_choices)
    return table, reliability


# ---------------------------------------------------------------------------
# Response compliance (item 8) -- first-attempt status rates, from the RAW
# (unfiltered, not-necessarily-resolved) result rows
# ---------------------------------------------------------------------------

def response_compliance_table(rows):
    grouped = defaultdict(lambda: {
        "n_attempted": 0, "n_valid_first_attempt": 0, "n_invalid_first_attempt": 0,
        "n_refusal_first_attempt": 0, "n_api_error_first_attempt": 0,
    })
    for row in rows:
        meta = row.get("trial_meta") or {}
        key = (meta.get("contrast_id"), meta.get("instruction_condition"))
        g = grouped[key]
        g["n_attempted"] += 1
        status = row.get("first_attempt_status")
        g[f"n_{status}_first_attempt"] = g.get(f"n_{status}_first_attempt", 0) + 1

    rows_out = []
    for (contrast_id, instruction_condition), g in grouped.items():
        rows_out.append({
            "contrast_id": contrast_id, "instruction_condition": instruction_condition, **g,
            "valid_rate": g["n_valid_first_attempt"] / g["n_attempted"] if g["n_attempted"] else None,
        })
    return rows_out


# ---------------------------------------------------------------------------
# Per-evaluator analysis
# ---------------------------------------------------------------------------

CONTRAST_ORDER = [
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
]


def evaluator_tag(evaluator_id, config, study_config):
    """Identifier fields stamped onto every output row -- item 20: "Every
    row must include sufficient experiment/corpus/evaluator/version
    identifiers to prevent accidental pooling."""
    return {
        "experiment_id": study_config["experiment_id"],
        "corpus_id": study_config["corpus_id"],
        "evaluator_id": evaluator_id,
        "role": config.get("role"),
        "provider": config.get("provider"),
        "requested_model": config.get("requested_model"),
        "response_model": config.get("response_model"),
        "reasoning_profile": config.get("reasoning_profile"),
    }


def analyze_one_evaluator(evaluator_id, partition, study_config, is_frozen, n_bootstrap_draws, bootstrap_seed):
    config = partition["config"]
    tag = evaluator_tag(evaluator_id, config, study_config)

    treatment_rows = [r for r in partition["rows"] if "superblock_id" in r]
    baseline_rows = [r for r in partition["rows"] if "block_id" in r and "superblock_id" not in r]

    kept_treatment, n_complete_sb, n_incomplete_sb = filter_complete_superblocks(treatment_rows)
    kept_baseline, n_complete_bu, n_incomplete_bu = filter_complete_baseline_units(baseline_rows)

    cell_rates = build_treatment_cell_rates(kept_treatment)
    pair_rows = build_pair_context_effects(cell_rates)

    baseline_table, reliability = build_baseline_pair_table(kept_baseline)
    baseline_strength_by_pair = {(r["story_1_id"], r["story_2_id"]): r["baseline_strength"] for r in baseline_table}
    for row in pair_rows:
        row["baseline_strength"] = baseline_strength_by_pair.get((row["story_1_id"], row["story_2_id"]))

    story_ids = sorted({s for r in pair_rows for s in (r["story_1_id"], r["story_2_id"])})
    rng = random.Random(bootstrap_seed)

    cue_ate_rows, controllability_rows, equivalence_rows, ambiguity_rows, position_rows, loo_rows = [], [], [], [], [], []

    contrast_ids = sorted({r["contrast_id"] for r in pair_rows}, key=lambda c: CONTRAST_ORDER.index(c) if c in CONTRAST_ORDER else 99)
    instruction_conditions = sorted({r["instruction_condition"] for r in pair_rows})

    for contrast_id in contrast_ids:
        for instruction_condition in instruction_conditions:
            values = pair_values_for(pair_rows, contrast_id, instruction_condition)
            if not values:
                continue
            ate_point = st.ate_from_pair_values(values)
            draws = [d["ate"] for d in st.story_bootstrap_joint_draws({"ate": values}, story_ids, n_bootstrap_draws, rng)]
            ci = st.bootstrap_ci_from_draws(draws, conf_levels=(0.95,))[0.95]
            cue_ate_rows.append({
                **tag, "contrast_id": contrast_id, "instruction_condition": instruction_condition,
                "n_story_pairs": len(values), "ATE_probability_points": ate_point,
                "ATE_percentage_points": ate_point * 100 if ate_point is not None else None,
                "ci_95_lo": ci[0], "ci_95_hi": ci[1], "n_bootstrap_draws_used": len(draws),
            })

            loo = st.leave_one_story_out(values, story_ids)
            for story_id, loo_ate in loo.items():
                loo_rows.append({**tag, "contrast_id": contrast_id, "instruction_condition": instruction_condition,
                                  "excluded_story_id": story_id, "ATE_excluding_story": loo_ate})

            group_rows = [r for r in pair_rows if r["contrast_id"] == contrast_id and r["instruction_condition"] == instruction_condition]
            position_rows.append({
                **tag, "contrast_id": contrast_id, "instruction_condition": instruction_condition,
                "n_story_pairs": len(group_rows),
                "context_effect_at_story1_as_a": _mean(r["context_effect_at_a"] for r in group_rows),
                "context_effect_at_story1_as_b": _mean(r["context_effect_at_b"] for r in group_rows),
                "context_x_position_interaction": _mean(r["context_x_position_interaction"] for r in group_rows),
                "overall_position_effect": _mean(r["position_effect"] for r in group_rows),
            })

        # --- controllability measures + equivalence (needs both instruction conditions) ---
        control_values = pair_values_for(pair_rows, contrast_id, "matched_control")
        text_only_values = pair_values_for(pair_rows, contrast_id, "text_only")
        if control_values and text_only_values:
            point = st.controllability_point_estimates(control_values, text_only_values)
            boot = st.controllability_bootstrap(control_values, text_only_values, story_ids, n_bootstrap_draws, rng)
            cis = {name: st.bootstrap_ci_from_draws(draws, conf_levels=(0.95,))[0.95] for name, draws in boot.items()}
            # residual_text_only_effect is defined as exactly ATE_text_only (item 13) -- same bootstrap draws, same CI.
            cis["residual_text_only_effect"] = cis["ATE_text_only"]
            controllability_rows.append({
                **tag, "contrast_id": contrast_id,
                **point,
                **{f"{name}_ci_95_lo": cis[name][0] for name in point},
                **{f"{name}_ci_95_hi": cis[name][1] for name in point},
            })

            equivalence = st.classify_equivalence(boot["ATE_text_only"], study_config["equivalence_margin"], is_frozen)
            equivalence_rows.append({
                **tag, "contrast_id": contrast_id,
                "equivalence_margin": equivalence["margin"],
                "ci_95_lo": equivalence["ci_95"][0], "ci_95_hi": equivalence["ci_95"][1],
                "ci_90_lo": equivalence["ci_90"][0], "ci_90_hi": equivalence["ci_90"][1],
                "verdict": equivalence["verdict"], "study_frozen": is_frozen,
            })

        # --- ambiguity: D_pair ~ baseline_strength, independent baseline data ---
        for instruction_condition in instruction_conditions:
            joined = [r for r in pair_rows if r["contrast_id"] == contrast_id and r["instruction_condition"] == instruction_condition and r["baseline_strength"] is not None]
            fit = st.fit_ambiguity_regression(joined)
            if fit is None:
                continue
            betas = st.ambiguity_regression_bootstrap(joined, story_ids, n_bootstrap_draws, rng)
            beta_ci = st.bootstrap_ci_from_draws(betas, conf_levels=(0.95,))[0.95]
            predictions = st.predict_at_percentiles(fit["intercept"], fit["beta"], [r["baseline_strength"] for r in joined])
            ambiguity_rows.append({
                **tag, "contrast_id": contrast_id, "instruction_condition": instruction_condition,
                "n_story_pairs": len(joined), "intercept": fit["intercept"], "beta": fit["beta"],
                "beta_ci_95_lo": beta_ci[0], "beta_ci_95_hi": beta_ci[1],
                "predicted_effect_at_p25": predictions[25], "predicted_effect_at_p50": predictions[50],
                "predicted_effect_at_p75": predictions[75],
            })

    tagged_pair_rows = [{**tag, **row} for row in pair_rows]
    tagged_baseline_rows = [{**tag, **row} for row in baseline_table]
    reliability_row = {**tag, **reliability}

    cell_counts_rows = _cell_counts(tag, kept_treatment, kept_baseline)
    compliance_rows = [{**tag, **row} for row in response_compliance_table(treatment_rows + baseline_rows)]

    return {
        "evaluator_id": evaluator_id,
        "config": config,
        "n_complete_treatment_superblocks": n_complete_sb,
        "n_incomplete_treatment_superblocks": n_incomplete_sb,
        "n_complete_baseline_units": n_complete_bu,
        "n_incomplete_baseline_units": n_incomplete_bu,
        "pair_context_effects": tagged_pair_rows,
        "baseline_pair_strength": tagged_baseline_rows,
        "baseline_reliability": [reliability_row],
        "cue_ates": cue_ate_rows,
        "controllability_effects": controllability_rows,
        "equivalence_results": equivalence_rows,
        "ambiguity_interactions": ambiguity_rows,
        "position_effects": position_rows,
        "leave_one_story_out": loo_rows,
        "response_compliance": compliance_rows,
        "cell_counts": cell_counts_rows,
    }


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _cell_counts(tag, kept_treatment, kept_baseline):
    counts = defaultdict(int)
    for row in kept_treatment:
        meta = row["trial_meta"]
        counts[(meta["contrast_id"], meta["instruction_condition"], meta["assignment"], meta["position"])] += 1
    for row in kept_baseline:
        meta = row["trial_meta"]
        counts[(None, meta["instruction_condition"], None, meta["position"])] += 1

    return [
        {**tag, "contrast_id": contrast_id, "instruction_condition": instruction_condition,
         "assignment": assignment, "position": position, "n_resolved_cells": n}
        for (contrast_id, instruction_condition, assignment, position), n in counts.items()
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_all_csvs(per_evaluator_results, evaluator_inventory_rows):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    def collect(key):
        return [row for result in per_evaluator_results for row in result[key]]

    exports = {
        "pair_context_effects.csv": collect("pair_context_effects"),
        "cue_ates.csv": collect("cue_ates"),
        "controllability_effects.csv": collect("controllability_effects"),
        "equivalence_results.csv": collect("equivalence_results"),
        "ambiguity_interactions.csv": collect("ambiguity_interactions"),
        "position_effects.csv": collect("position_effects"),
        "leave_one_story_out.csv": collect("leave_one_story_out"),
        "response_compliance.csv": collect("response_compliance"),
        "cell_counts.csv": collect("cell_counts"),
        "baseline_pair_strength.csv": collect("baseline_pair_strength"),
        "baseline_reliability.csv": collect("baseline_reliability"),
        "evaluator_inventory.csv": evaluator_inventory_rows,
    }
    for filename, rows in exports.items():
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        write_csv(rows, fieldnames, os.path.join(ANALYSIS_DIR, filename))
    return exports


def build_headline_results(per_evaluator_results, study_config, is_frozen):
    headline = {"experiment_id": study_config["experiment_id"], "corpus_id": study_config["corpus_id"],
                "study_frozen": is_frozen, "evaluators": {}}
    for result in per_evaluator_results:
        headline["evaluators"][result["evaluator_id"]] = {
            "config": result["config"],
            "n_complete_treatment_superblocks": result["n_complete_treatment_superblocks"],
            "n_complete_baseline_units": result["n_complete_baseline_units"],
            "controllability_effects": result["controllability_effects"],
            "equivalence_results": result["equivalence_results"],
        }

    # cross-model summary: one row per (evaluator, contrast) -- descriptive
    # only, still never pooled into a single inferential estimate.
    headline["cross_model_summary"] = [
        {"evaluator_id": result["evaluator_id"], **row}
        for result in per_evaluator_results
        for row in result["controllability_effects"]
    ]
    return headline


def main():
    parser = argparse.ArgumentParser(description="v2 context-controllability analysis.")
    parser.add_argument("--treatment-results-file", default=TREATMENT_RESULTS_FILE)
    parser.add_argument("--baseline-results-file", default=BASELINE_RESULTS_FILE)
    parser.add_argument("--study-config-file", default=STUDY_CONFIG_FILE)
    parser.add_argument("--bootstrap-draws", type=int, default=None, help="Default: bootstrap_draws from the study config")
    parser.add_argument("--strict-model-version", action="store_true",
                         help="Fail instead of splitting when one configured evaluator resolves to multiple response_model versions")
    args = parser.parse_args()

    study_config = load_study_config(args.study_config_file)
    is_frozen, reason = verify_frozen(args.study_config_file, CORPUS_FILE)
    if not is_frozen:
        print(f"Study design is not frozen or does not match its lock ({reason}). "
              f"Bootstrap intervals will be reported; no equivalence verdict will be produced.")

    n_bootstrap_draws = args.bootstrap_draws or study_config["bootstrap_draws"]

    treatment_rows = load_rows(args.treatment_results_file)
    baseline_rows = load_rows(args.baseline_results_file)
    print(f"Raw treatment rows: {len(treatment_rows)}  Raw baseline rows: {len(baseline_rows)}")

    all_rows = treatment_rows + baseline_rows
    partitions, unmatched, warnings = partition_by_resolved_evaluator(all_rows, study_config, strict=args.strict_model_version)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if unmatched:
        print(f"Excluded {len(unmatched)} row(s) whose evaluator identity matched no configured evaluator.")

    per_evaluator_results = []
    evaluator_inventory_rows = []
    for evaluator_id, partition in partitions.items():
        result = analyze_one_evaluator(evaluator_id, partition, study_config, is_frozen, n_bootstrap_draws, study_config["random_seed"])
        per_evaluator_results.append(result)
        config = partition["config"]
        evaluator_inventory_rows.append({
            "evaluator_id": evaluator_id, "role": config.get("role"), "provider": config.get("provider"),
            "requested_model": config.get("requested_model"), "response_model": config.get("response_model"),
            "reasoning_profile": config.get("reasoning_profile"), "n_rows": len(partition["rows"]),
            "n_complete_treatment_superblocks": result["n_complete_treatment_superblocks"],
            "n_complete_baseline_units": result["n_complete_baseline_units"],
        })
        print(f"Evaluator {evaluator_id}: complete treatment superblocks={result['n_complete_treatment_superblocks']} "
              f"(incomplete={result['n_incomplete_treatment_superblocks']}), "
              f"complete baseline units={result['n_complete_baseline_units']} (incomplete={result['n_incomplete_baseline_units']})")

    write_all_csvs(per_evaluator_results, evaluator_inventory_rows)
    headline = build_headline_results(per_evaluator_results, study_config, is_frozen)
    with open(os.path.join(ANALYSIS_DIR, "headline_results.json"), "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)

    from controllability_v2_plots import write_all_plots
    write_all_plots(per_evaluator_results, ANALYSIS_DIR)

    print(f"Wrote analysis outputs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
