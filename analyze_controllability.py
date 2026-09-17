"""Focused analyser for the standalone context-controllability experiment
(controllability_trials.py).

Thin composition layer: every effect-size calculation here is reused,
unmodified, from context_analysis_io.py and context_analysis_pairwise.py --
including compare_pairwise_regimes(), which already computes
"effect under naturalistic" / "effect under text_only_invariance" per
(model, contrast, category). This module adds only:

  - complete-unit filtering (see filter_complete_treatment_units/
    filter_complete_baseline_units): a (block_id, model, replicate_id,
    sampling_regime) unit is used for the PRIMARY analysis only if every
    one of its required cells has a valid response. Incomplete units are
    reported, never silently dropped without a count, and the raw
    observations are never deleted -- this only filters a copy used for
    the primary calculations.
  - the derived suppression_magnitude = abs(naturalistic_effect) -
    abs(text_only_effect) column (positive = the instruction reduced the
    SIZE of the context effect; see build_suppression_table), alongside
    the retained signed_regime_difference = naturalistic_effect -
    text_only_effect and the residual text_only_effect itself.
  - human-readable contrast names, attached from each row's own
    contrast_id (never parsed from it).
  - the blind-baseline analysis (baseline_margin, and its correlation with
    abs(context_effect), stratified by (model, evaluation_regime,
    contrast_id) -- never pooled across the 5 contrasts, so each
    correlation is computed over at most the 66 underlying text pairs).
  - first-attempt response-compliance reporting (diagnostic only -- an
    invalid response is never treated as an A/B judgment).

The `categories` argument to every reused context_analysis_pairwise.py
function is always [PRIMARY_CATEGORY] here (this experiment's single plain
A/B outcome; see controllability_trials.RESPONSE_FORMAT/PRIMARY_CATEGORY),
never RATING_FIELDS -- there is no second rubric to analyze.

Distinct from analyze_context.py by design: this script never runs,
prints, or writes anything for the single-text, context_prompt,
tie-allowed, or researcher-reference-agreement families -- only this
experiment's own directional pairwise effect and the diagnostics that
belong with it. analyze_context.py itself is completely unmodified.

Interpreting baseline_margin = 2 * abs(p_text1_wins - 0.5): 0 means the
blind (no-context) evaluator was close to indifferent between the two
texts; 1 means it was maximally decisive. A NEGATIVE correlation between
baseline_margin and abs(context_effect) means context effects are LARGER
on pairs the evaluator was closer to indifferent about (i.e. context moves
close calls more than it overrides a strong pre-existing preference); a
positive correlation would mean the opposite. This is a plain Pearson
correlation (context_analysis_stats.pearson_correlation, reused
unmodified) -- never a new statistical method, and never a
Bradley-Terry/Elo ranking.

Run this file directly: python3 analyze_controllability.py
No API calls are made.
"""

import argparse
import os
import statistics
from collections import defaultdict

from analyze import attempt_number, is_successful, load_jsonl, write_csv
from context_analysis_io import collapse_attempts, filter_by_sampling_regime, get_trial_meta, load_trials_by_id
from context_analysis_pairwise import (
    analyze_directional_pairwise_effects,
    analyze_leave_one_story_out,
    analyze_pairwise_cell_rates,
    analyze_position_and_interaction_effects,
    choice_to_story_id,
    compare_pairwise_regimes,
    summarize_directional_effects_by_contrast,
)
from context_analysis_stats import pearson_correlation
from controllability_trials import (
    BASELINE_EXPERIMENT_ID,
    BASELINE_POSITIONS,
    BASELINE_TRIALS_FILE,
    CONTRAST_DISPLAY_NAMES,
    EXPERIMENT_ID,
    PRIMARY_CATEGORY,
    REQUIRED_CELLS,
    TRIALS_FILE,
)
from run_trial import SAMPLING_REGIMES

RESULTS_FILE = "results/controllability/raw.jsonl"
BASELINE_RESULTS_FILE = "results/controllability/baseline_raw.jsonl"
ANALYSIS_DIR = "results/controllability/analysis"
DEFAULT_SAMPLING_REGIME = "low_variance_primary"

CATEGORIES = [PRIMARY_CATEGORY]


def context_name(contrast_id):
    return CONTRAST_DISPLAY_NAMES.get(contrast_id, contrast_id)


def attach_context_name(rows):
    for row in rows:
        row["context"] = context_name(row["contrast_id"])
    return rows


# ---------------------------------------------------------------------------
# Never analyse partial counterbalanced units silently
# ---------------------------------------------------------------------------

def filter_complete_treatment_units(observations):
    """Retain observations belonging to a (block_id, model, replicate_id,
    sampling_regime) unit only if all 4 required cells (forward/
    story1_as_a, forward/story2_as_a, flipped/story1_as_a, flipped/
    story2_as_a) have a valid response. Returns (complete_observations,
    n_complete_units, n_incomplete_units). Never mutates or drops anything
    from the raw `observations` passed in -- returns a new dict.
    """
    by_unit = defaultdict(dict)
    for key, obs in observations.items():
        if obs.get("type") != "context_pairwise" or "block_id" not in obs or "assignment" not in obs:
            continue
        unit_key = (obs["block_id"], obs["model"], obs["replicate_id"], obs.get("sampling_regime"))
        by_unit[unit_key][(obs["assignment"], obs["position"])] = key

    complete_keys = set()
    n_complete = n_incomplete = 0
    for unit_key, cells in by_unit.items():
        if set(cells) == REQUIRED_CELLS and all(
            observations[k]["parsed_response"] is not None for k in cells.values()
        ):
            n_complete += 1
            complete_keys.update(cells.values())
        else:
            n_incomplete += 1

    return {k: v for k, v in observations.items() if k in complete_keys}, n_complete, n_incomplete


def filter_complete_baseline_units(observations):
    """Retain observations belonging to a (block_id, model, replicate_id,
    sampling_regime) unit only if both display positions have a valid
    response. Returns (complete_observations, n_complete_units,
    n_incomplete_units)."""
    by_unit = defaultdict(dict)
    for key, obs in observations.items():
        if obs.get("type") != "context_pairwise" or "block_id" not in obs:
            continue
        unit_key = (obs["block_id"], obs["model"], obs["replicate_id"], obs.get("sampling_regime"))
        by_unit[unit_key][obs["position"]] = key

    complete_keys = set()
    n_complete = n_incomplete = 0
    for unit_key, cells in by_unit.items():
        if set(cells) == BASELINE_POSITIONS and all(
            observations[k]["parsed_response"] is not None for k in cells.values()
        ):
            n_complete += 1
            complete_keys.update(cells.values())
        else:
            n_incomplete += 1

    return {k: v for k, v in observations.items() if k in complete_keys}, n_complete, n_incomplete


# ---------------------------------------------------------------------------
# Suppression table: naturalistic_effect / text_only_effect /
# suppression_magnitude (primary) / signed_regime_difference (retained)
# ---------------------------------------------------------------------------

def build_suppression_table(by_contrast_rows):
    """Reuses compare_pairwise_regimes() unmodified (it already restricts to
    rows present under BOTH regimes).

    suppression_magnitude = abs(naturalistic_effect) - abs(text_only_effect):
    positive means the instruction reduced the SIZE of the context effect;
    zero means no reduction; negative means the effect got LARGER under the
    text-only instruction. This is the primary "suppression" quantity.

    signed_regime_difference = naturalistic_effect - text_only_effect is
    also retained (a signed comparison of the two raw effects, not of their
    magnitudes) -- not the primary measure, but not discarded either.
    """
    rows = []
    for row in compare_pairwise_regimes(by_contrast_rows):
        nat, txt = row["effect_naturalistic"], row["effect_text_only_invariance"]
        rows.append(
            {
                "model": row["model"],
                "contrast_id": row["contrast_id"],
                "context": context_name(row["contrast_id"]),
                "category": row["category"],
                "naturalistic_effect": nat,
                "text_only_effect": txt,
                "suppression_magnitude": round(abs(nat) - abs(txt), 3),
                "signed_regime_difference": round(nat - txt, 3),
                "attenuation": row["attenuation"],
            }
        )
    return rows


CONTRAST_ORDER = [
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
]


def print_suppression_table(rows, model, category=PRIMARY_CATEGORY):
    group = [r for r in rows if r["model"] == model and r["category"] == category]
    print(f"\nPRIMARY: {category.replace('_', ' ')} -- suppression_magnitude = abs(natural) - abs(text_only)")
    print("(positive = instruction shrank the effect; negative = the effect grew under the instruction)")
    print(f"{'context':<32}{'natural':>10}{'text_only':>12}{'suppression':>14}")
    for row in sorted(group, key=lambda r: CONTRAST_ORDER.index(r["contrast_id"]) if r["contrast_id"] in CONTRAST_ORDER else 99):
        print(f"{row['context']:<32}{row['naturalistic_effect']:>+10.3f}{row['text_only_effect']:>+12.3f}{row['suppression_magnitude']:>+14.3f}")


# ---------------------------------------------------------------------------
# Blind baseline: p_text1_wins with no context, and baseline_margin
# ---------------------------------------------------------------------------

def analyze_baseline_margin(baseline_observations, categories=CATEGORIES):
    """For each (model, story_1_id, story_2_id, category), pool the two
    no-context display positions (reusing choice_to_story_id, never a second
    A/B-to-story-identity mapping) into p_text1_wins, then
    baseline_margin = 2 * abs(p_text1_wins - 0.5): 0 means the blind
    evaluator was close to indifferent, 1 means it was maximally decisive.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "tie": 0, "n": 0})
    for obs in baseline_observations.values():
        if obs["type"] != "context_pairwise":
            continue
        for category in categories:
            chosen = choice_to_story_id(obs, category)
            key = (obs["model"], obs["story_1_id"], obs["story_2_id"], category)
            t = tallies[key]
            t["n"] += 1
            if chosen is None:
                t["tie"] += 1
            elif chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            else:
                t["story_2_preferred"] += 1

    rows = []
    for (model, s1, s2, category), t in tallies.items():
        if not t["n"]:
            continue
        p1 = t["story_1_preferred"] / t["n"]
        rows.append(
            {
                "model": model,
                "story_1_id": s1,
                "story_2_id": s2,
                "category": category,
                "p_text1_wins": round(p1, 3),
                "n": t["n"],
                "baseline_margin": round(2 * abs(p1 - 0.5), 3),
            }
        )
    return rows


def join_effect_to_baseline_margin(directional_rows, baseline_rows):
    """Pair each treatment story-pair's directional context effect with that
    same pair's blind baseline_margin (both keyed by
    (model, story_1_id, story_2_id, category) -- the 66 pairs are identical
    across the treatment and baseline manifests)."""
    baseline_by_key = {(r["model"], r["story_1_id"], r["story_2_id"], r["category"]): r for r in baseline_rows}
    rows = []
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        key = (row["model"], row["story_1_id"], row["story_2_id"], row["category"])
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        rows.append(
            {
                "model": row["model"],
                "evaluation_regime": row["evaluation_regime"],
                "contrast_id": row["contrast_id"],
                "context": context_name(row["contrast_id"]),
                "story_1_id": row["story_1_id"],
                "story_2_id": row["story_2_id"],
                "category": row["category"],
                "context_effect": row["directional_effect_a_minus_b"],
                "abs_context_effect": abs(row["directional_effect_a_minus_b"]),
                "baseline_margin": baseline["baseline_margin"],
                "baseline_p_text1_wins": baseline["p_text1_wins"],
            }
        )
    return rows


def correlate_effect_and_baseline_margin(joined_rows, category=PRIMARY_CATEGORY):
    """Plain Pearson correlation (reused, not reimplemented) between
    baseline_margin and abs(context_effect), computed SEPARATELY for each
    (model, evaluation_regime, contrast_id) -- never pooled across the 5
    contrasts, so each row's n_story_pairs is at most the 66 underlying
    text pairs. A negative r means context effects are larger on pairs
    where the blind evaluator was closer to indifferent (baseline_margin
    near 0); a positive r means the opposite.
    """
    grouped = defaultdict(list)
    for row in joined_rows:
        if row["category"] != category:
            continue
        grouped[(row["model"], row["evaluation_regime"], row["contrast_id"])].append(row)

    rows = []
    for (model, evaluation_regime, contrast_id), group in grouped.items():
        xs = [r["baseline_margin"] for r in group]
        ys = [r["abs_context_effect"] for r in group]
        r = pearson_correlation(xs, ys)
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "contrast_id": contrast_id,
                "context": context_name(contrast_id),
                "category": category,
                "n_story_pairs": len(group),
                "r_baseline_margin_vs_abs_context_effect": round(r, 3) if r is not None else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# First-attempt response-compliance reporting (diagnostic only)
# ---------------------------------------------------------------------------

def first_attempt_compliance(raw_rows, trials_by_id, experiment_ids):
    """By (model, evaluation_regime, contrast_id): n_attempted,
    n_valid_first_attempt, n_invalid_first_attempt,
    n_api_error_first_attempt, valid_rate. Restricted to attempt_id==1 so a
    retry never double-counts an observation. API-call failures (validation_
    error starting with "API call failed") are kept distinguishable from a
    malformed/refused model response. This is diagnostic only -- it never
    treats an invalid response as an A/B judgment, and doesn't feed the
    effect calculations above.

    Metadata is read the same way collapse_attempts does (prefer the row's
    own embedded trial_meta, else join the trials manifest by trial_id --
    see context_analysis_io.get_trial_meta), so this works whether or not a
    given row happens to carry trial_meta directly.
    """
    grouped = defaultdict(lambda: {"n_attempted": 0, "n_valid_first_attempt": 0, "n_invalid_first_attempt": 0, "n_api_error_first_attempt": 0})
    for row in raw_rows:
        if attempt_number(row) != 1:
            continue
        meta = get_trial_meta(row, trials_by_id) or {}
        if meta.get("experiment_id") not in experiment_ids:
            continue
        key = (row["model"], meta.get("evaluation_regime"), meta.get("contrast_id"))
        g = grouped[key]
        g["n_attempted"] += 1
        if is_successful(row):
            g["n_valid_first_attempt"] += 1
        elif (row.get("validation_error") or "").startswith("API call failed"):
            g["n_api_error_first_attempt"] += 1
        else:
            g["n_invalid_first_attempt"] += 1

    rows = []
    for (model, evaluation_regime, contrast_id), g in grouped.items():
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "contrast_id": contrast_id,
                "context": context_name(contrast_id) if contrast_id != "no_context_baseline" else "no-context baseline",
                **g,
                "valid_rate": round(g["n_valid_first_attempt"] / g["n_attempted"], 3) if g["n_attempted"] else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(model, sampling_regime, suppression_rows, position_rows, loo_rows, correlation_rows,
                   n_complete_units, n_incomplete_units, n_complete_baseline_units, n_incomplete_baseline_units):
    print("\n=== Context controllability experiment ===")
    print(f"Evaluator: {model}")
    print(f"Sampling regime: {sampling_regime}")
    print(f"Complete treatment blocks used in primary analysis: {n_complete_units} (incomplete/excluded: {n_incomplete_units})")
    print(f"Complete baseline pairs used in primary analysis: {n_complete_baseline_units} (incomplete/excluded: {n_incomplete_baseline_units})")

    print_suppression_table(suppression_rows, model)

    position_for_model = [r for r in position_rows if r["model"] == model and r["category"] == PRIMARY_CATEGORY]
    if position_for_model:
        interactions = [r["context_x_position_interaction"] for r in position_for_model]
        positions = [r["position_effect_pooled"] for r in position_for_model if r["position_effect_pooled"] != ""]
        print("\nPosition-effect diagnostic (apparent context effect is not just A/B display-position preference):")
        print(f"  context x position interaction: mean={statistics.mean(interactions):+.3f} "
              f"range=[{min(interactions):+.3f}, {max(interactions):+.3f}]")
        if positions:
            print(f"  position effect (pooled over assignment): mean={statistics.mean(positions):+.3f} "
                  f"range=[{min(positions):+.3f}, {max(positions):+.3f}]")

    loo_for_model = [
        r for r in loo_rows
        if r["model"] == model and r["category"] == PRIMARY_CATEGORY and r["excluded_story_id"] != "(none -- full aggregate)"
    ]
    if loo_for_model:
        loo_values = [r["mean_directional_effect"] for r in loo_for_model]
        print(f"\nLeave-one-story-out sensitivity (overall_quality, aggregate recomputed excluding each story):")
        print(f"  range=[{min(loo_values):+.3f}, {max(loo_values):+.3f}]")

    corr_for_model = [r for r in correlation_rows if r["model"] == model]
    if corr_for_model:
        print("\nBaseline margin vs. context effect, per contrast (negative r = context moves close calls more):")
        for row in sorted(corr_for_model, key=lambda r: (r["evaluation_regime"], r["contrast_id"])):
            r_display = row["r_baseline_margin_vs_abs_context_effect"]
            r_str = f"{r_display:+.3f}" if r_display != "" else "n/a"
            print(f"  {row['evaluation_regime']} | {row['context']}: r={r_str}  (n={row['n_story_pairs']} story pairs)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_raw_and_observations(results_file, trials_file, sampling_regime, experiment_id, label):
    if not os.path.exists(results_file):
        print(f"No results file found at {results_file} ({label}). Skipping.")
        return [], None

    raw_rows = load_jsonl(results_file)
    trials_by_id = load_trials_by_id(trials_file)
    all_observations, unresolved, absorbed, missing_metadata = collapse_attempts(raw_rows, trials_by_id)
    observations, excluded_by_regime = filter_by_sampling_regime(all_observations, sampling_regime)

    experiment_observations = {k: v for k, v in observations.items() if v.get("experiment_id") == experiment_id}
    excluded_other_experiment = len(observations) - len(experiment_observations)

    print(f"Raw API attempts ({label}): {len(raw_rows)}")
    print(f"Completed observations ({label}, {sampling_regime}): {len(experiment_observations)}")
    if unresolved:
        print(f"Unresolved failures ({label}): {len(unresolved)}")
    if excluded_by_regime:
        print(f"Excluded {excluded_by_regime} {label} observation(s) recorded under a different sampling regime.")
    if excluded_other_experiment:
        print(f"Excluded {excluded_other_experiment} {label} observation(s) belonging to a different experiment_id.")

    return raw_rows, experiment_observations


def warn_if_pooling_response_models(observations, label):
    distinct = {o.get("response_model") for o in observations.values() if o.get("response_model")}
    if len(distinct) > 1:
        print(f"WARNING: {label} observations were served by more than one response_model: {sorted(distinct)} "
              f"-- results below still pool by requested `model`; treat this as a flag to split them manually.")


def main():
    parser = argparse.ArgumentParser(description="Focused analysis for the standalone context-controllability experiment.")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Treatment trials manifest (default: {TRIALS_FILE})")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Treatment results file (default: {RESULTS_FILE})")
    parser.add_argument("--baseline-trials-file", default=BASELINE_TRIALS_FILE,
                         help=f"Blind baseline trials manifest (default: {BASELINE_TRIALS_FILE})")
    parser.add_argument("--baseline-results-file", default=BASELINE_RESULTS_FILE,
                         help=f"Blind baseline results file (default: {BASELINE_RESULTS_FILE})")
    parser.add_argument(
        "--sampling-regime",
        choices=list(SAMPLING_REGIMES),
        default=DEFAULT_SAMPLING_REGIME,
        help=f"Only observations recorded under this regime are analyzed (default: {DEFAULT_SAMPLING_REGIME}).",
    )
    args = parser.parse_args()

    raw_treatment_rows, observations = load_raw_and_observations(
        args.results_file, args.trials_file, args.sampling_regime, EXPERIMENT_ID, "treatment"
    )
    raw_baseline_rows, baseline_observations = load_raw_and_observations(
        args.baseline_results_file, args.baseline_trials_file, args.sampling_regime, BASELINE_EXPERIMENT_ID, "baseline"
    )

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    trials_by_id = {**load_trials_by_id(args.trials_file), **load_trials_by_id(args.baseline_trials_file)}
    compliance_rows = first_attempt_compliance(
        raw_treatment_rows + raw_baseline_rows, trials_by_id, {EXPERIMENT_ID, BASELINE_EXPERIMENT_ID}
    )
    if compliance_rows:
        write_csv(
            compliance_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "n_attempted", "n_valid_first_attempt",
             "n_invalid_first_attempt", "n_api_error_first_attempt", "valid_rate"],
            os.path.join(ANALYSIS_DIR, "response_compliance.csv"),
        )

    directional_rows = []
    suppression_rows = []
    position_rows = []
    loo_rows = []
    n_complete_units = n_incomplete_units = 0
    if observations:
        warn_if_pooling_response_models(observations, "treatment")
        complete_observations, n_complete_units, n_incomplete_units = filter_complete_treatment_units(observations)
        print(f"Complete treatment blocks (all 4 cells valid): {n_complete_units}  incomplete/excluded: {n_incomplete_units}")

        directional_rows = analyze_directional_pairwise_effects(complete_observations, categories=CATEGORIES)
        attach_context_name(directional_rows)
        by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
        attach_context_name(by_contrast_rows)
        suppression_rows = build_suppression_table(by_contrast_rows)

        cell_rows = analyze_pairwise_cell_rates(complete_observations, categories=CATEGORIES)
        position_rows = analyze_position_and_interaction_effects(cell_rows)
        attach_context_name(position_rows)

        loo_rows = analyze_leave_one_story_out(directional_rows)
        attach_context_name(loo_rows)

        write_csv(
            by_contrast_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "category",
             "mean_directional_effect", "min_directional_effect", "max_directional_effect",
             "range_directional_effect", "n_story_pairs"],
            os.path.join(ANALYSIS_DIR, "pairwise_effects.csv"),
        )
        write_csv(
            directional_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "story_1_id", "story_2_id", "category",
             "p_story1_preferred_given_value_a", "n_value_a", "tie_n_value_a",
             "p_story1_preferred_given_value_b", "n_value_b", "tie_n_value_b",
             "directional_effect_a_minus_b"],
            os.path.join(ANALYSIS_DIR, "per_story_pair_effects.csv"),
        )
        write_csv(
            suppression_rows,
            ["model", "contrast_id", "context", "category", "naturalistic_effect", "text_only_effect",
             "suppression_magnitude", "signed_regime_difference", "attenuation"],
            os.path.join(ANALYSIS_DIR, "suppression_effects.csv"),
        )
        write_csv(
            position_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "story_1_id", "story_2_id", "category",
             "context_effect_at_position_a", "context_effect_at_position_b", "context_x_position_interaction",
             "position_effect_pooled", "position_effect_under_forward", "position_effect_under_flipped"],
            os.path.join(ANALYSIS_DIR, "position_effects.csv"),
        )
        write_csv(
            loo_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "category",
             "excluded_story_id", "mean_directional_effect", "n_story_pairs"],
            os.path.join(ANALYSIS_DIR, "leave_one_story_out.csv"),
        )

    baseline_rows = []
    joined_rows = []
    correlation_rows = []
    n_complete_baseline_units = n_incomplete_baseline_units = 0
    if baseline_observations:
        warn_if_pooling_response_models(baseline_observations, "baseline")
        complete_baseline_observations, n_complete_baseline_units, n_incomplete_baseline_units = filter_complete_baseline_units(baseline_observations)
        print(f"Complete baseline pairs (both positions valid): {n_complete_baseline_units}  incomplete/excluded: {n_incomplete_baseline_units}")

        baseline_rows = analyze_baseline_margin(complete_baseline_observations)
        write_csv(
            baseline_rows,
            ["model", "story_1_id", "story_2_id", "category", "p_text1_wins", "n", "baseline_margin"],
            os.path.join(ANALYSIS_DIR, "baseline_margin.csv"),
        )

    if directional_rows and baseline_rows:
        joined_rows = join_effect_to_baseline_margin(directional_rows, baseline_rows)
        correlation_rows = correlate_effect_and_baseline_margin(joined_rows)
        write_csv(
            joined_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "story_1_id", "story_2_id", "category",
             "context_effect", "abs_context_effect", "baseline_margin", "baseline_p_text1_wins"],
            os.path.join(ANALYSIS_DIR, "context_effect_vs_baseline_margin.csv"),
        )
        write_csv(
            correlation_rows,
            ["model", "evaluation_regime", "contrast_id", "context", "category", "n_story_pairs",
             "r_baseline_margin_vs_abs_context_effect"],
            os.path.join(ANALYSIS_DIR, "baseline_margin_correlation.csv"),
        )

    if directional_rows:
        for model in sorted({r["model"] for r in directional_rows}):
            print_summary(model, args.sampling_regime, suppression_rows, position_rows, loo_rows, correlation_rows,
                           n_complete_units, n_incomplete_units, n_complete_baseline_units, n_incomplete_baseline_units)
        print(f"\nWrote tidy tables to {ANALYSIS_DIR}/")
    else:
        print("No complete treatment observations yet. Nothing to analyze.")


if __name__ == "__main__":
    main()
