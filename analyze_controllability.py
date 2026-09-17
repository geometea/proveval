"""Focused analyser for the standalone context-controllability experiment
(controllability_trials.py).

Thin composition layer: every effect-size calculation here is reused,
unmodified, from context_analysis_io.py and context_analysis_pairwise.py --
including compare_pairwise_regimes(), which already computes exactly
"effect under naturalistic" / "effect under text_only_invariance" per
(model, contrast, category); this module only adds the derived
suppression_effect = naturalistic_effect - text_only_effect column, attaches
human-readable contrast names, and presents the result (console summary
plus CSVs). The excerpt rubric's rating categories
(context_analysis_common.EXCERPT_RATING_FIELDS) are passed into the shared
analysis functions via their existing `categories` parameter -- no
duplicated statistics for a second rubric.

Distinct from analyze_context.py by design: this script never runs, prints,
or writes anything for the single-text, context_prompt, tie-allowed, or
researcher-reference-agreement families -- only this experiment's own
directional pairwise effect and the diagnostics that belong with it.
analyze_context.py itself is completely unmodified.

Primary estimand (unchanged from the general benchmark, see
context_analysis_pairwise.analyze_directional_pairwise_effects):

    P(text_1 preferred | text_1 receives contrast value "a")
  - P(text_1 preferred | text_1 receives contrast value "b")

naturalistic_effect / text_only_effect are this same estimand computed
separately per evaluation_regime (context_analysis_pairwise.
compare_pairwise_regimes); suppression_effect is how much of the
naturalistic effect the "ignore the context" instruction removes. This is
never a significance test, and no Bradley-Terry/Elo ranking is computed
anywhere in this file.

The blind, no-context baseline (controllability_trials.build_baseline_trials)
is analyzed separately: for each text pair, baseline_margin =
2 * abs(p_text1_wins - 0.5) measures how close to indifferent the evaluator
was with no context at all. Relating abs(context_effect) to baseline_margin
(a plain Pearson correlation -- context_analysis_stats.pearson_correlation,
reused unmodified) answers whether context mainly moves close calls or can
override a strong pre-existing textual preference.

Run this file directly: python3 analyze_controllability.py
No API calls are made.
"""

import argparse
import os
import statistics
from collections import defaultdict

from analyze import load_jsonl, write_csv
from context_analysis_common import EXCERPT_RATING_FIELDS
from context_analysis_io import collapse_attempts, filter_by_sampling_regime, load_trials_by_id
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
    BASELINE_TRIALS_FILE,
    CONTRAST_DISPLAY_NAMES,
    EXPERIMENT_ID,
    TRIALS_FILE,
)
from run_trial import SAMPLING_REGIMES

RESULTS_FILE = "results/controllability/raw.jsonl"
BASELINE_RESULTS_FILE = "results/controllability/baseline_raw.jsonl"
ANALYSIS_DIR = "results/controllability/analysis"
DEFAULT_SAMPLING_REGIME = "low_variance_primary"

# overall_quality is reported first/primary; the excerpt rubric's other 4
# categories are still fully retained everywhere.
PRIMARY_CATEGORY = "overall_quality"


def attach_context_name(rows):
    """Add a human-readable "context" display name from each row's own
    contrast_id, via CONTRAST_DISPLAY_NAMES -- never parsed from contrast_id
    text."""
    for row in rows:
        row["context"] = CONTRAST_DISPLAY_NAMES[row["contrast_id"]]
    return rows


# ---------------------------------------------------------------------------
# Suppression table: naturalistic_effect / text_only_effect / suppression_effect
# ---------------------------------------------------------------------------

def build_suppression_table(by_contrast_rows):
    """Reuses compare_pairwise_regimes() unmodified (it already restricts to
    rows present under BOTH regimes) and adds the one derived column this
    experiment is about: suppression_effect = naturalistic_effect -
    text_only_effect. Positive means the instruction reduced (suppressed)
    the naturalistic effect; negative means the effect got larger; zero
    means the instruction had no measurable effect on this contrast/category.
    """
    rows = []
    for row in compare_pairwise_regimes(by_contrast_rows):
        rows.append(
            {
                "model": row["model"],
                "contrast_id": row["contrast_id"],
                "context": CONTRAST_DISPLAY_NAMES[row["contrast_id"]],
                "category": row["category"],
                "naturalistic_effect": row["effect_naturalistic"],
                "text_only_effect": row["effect_text_only_invariance"],
                "suppression_effect": round(row["effect_naturalistic"] - row["effect_text_only_invariance"], 3),
                "attenuation": row["attenuation"],
            }
        )
    return rows


def print_suppression_table(rows, model, category=PRIMARY_CATEGORY):
    group = [r for r in rows if r["model"] == model and r["category"] == category]
    print(f"\nPRIMARY: {category.replace('_', ' ')} -- how much the 'ignore context' instruction suppresses each context effect")
    print(f"{'context':<32}{'natural':>10}{'text_only':>12}{'suppression':>14}")
    for row in sorted(group, key=lambda r: CONTRAST_ORDER.index(r["contrast_id"]) if r["contrast_id"] in CONTRAST_ORDER else 99):
        print(f"{row['context']:<32}{row['naturalistic_effect']:>+10.3f}{row['text_only_effect']:>+12.3f}{row['suppression_effect']:>+14.3f}")


CONTRAST_ORDER = [
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
]


# ---------------------------------------------------------------------------
# Blind baseline: p_text1_wins with no context, and baseline_margin
# ---------------------------------------------------------------------------

def analyze_baseline_margin(baseline_observations, categories=EXCERPT_RATING_FIELDS):
    """For each (model, story_1_id, story_2_id, category), pool the two
    no-context display positions (reusing choice_to_story_id, never a second
    A/B-to-story-identity mapping) into p_text1_wins, then
    baseline_margin = 2 * abs(p_text1_wins - 0.5): 1.0 means the evaluator
    was unanimous with no context at all, 0.0 means an exact toss-up.
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
                "context": CONTRAST_DISPLAY_NAMES[row["contrast_id"]],
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
    abs(context_effect) and baseline_margin, per (model, evaluation_regime),
    restricted to `category`. A positive r means larger context effects
    tend to occur on pairs where the evaluator was closer to indifferent to
    begin with; near zero (or negative) means context can move judgments
    even on pairs with a strong pre-existing textual preference."""
    grouped = defaultdict(list)
    for row in joined_rows:
        if row["category"] != category:
            continue
        grouped[(row["model"], row["evaluation_regime"])].append(row)

    rows = []
    for (model, evaluation_regime), group in grouped.items():
        xs = [r["baseline_margin"] for r in group]
        ys = [r["abs_context_effect"] for r in group]
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "category": category,
                "n_story_pairs": len(group),
                "r_baseline_margin_vs_abs_context_effect": (
                    round(r, 3) if (r := pearson_correlation(xs, ys)) is not None else ""
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(model, sampling_regime, suppression_rows, position_rows, loo_rows, correlation_rows):
    print("\n=== Context controllability experiment ===")
    print(f"Evaluator: {model}")
    print(f"Sampling regime: {sampling_regime}")

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
        print("\nBaseline margin vs. context effect (does context mainly move close calls, or override strong preferences?):")
        for row in sorted(corr_for_model, key=lambda r: r["evaluation_regime"]):
            r_display = row["r_baseline_margin_vs_abs_context_effect"]
            r_str = f"{r_display:+.3f}" if r_display != "" else "n/a"
            print(f"  {row['evaluation_regime']}: r={r_str}  (n={row['n_story_pairs']} story-pair observations)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_experiment_observations(results_file, trials_file, sampling_regime, experiment_id, label):
    if not os.path.exists(results_file):
        print(f"No results file found at {results_file} ({label}). Skipping.")
        return None

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

    return experiment_observations


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

    observations = load_experiment_observations(
        args.results_file, args.trials_file, args.sampling_regime, EXPERIMENT_ID, "treatment"
    )
    baseline_observations = load_experiment_observations(
        args.baseline_results_file, args.baseline_trials_file, args.sampling_regime, BASELINE_EXPERIMENT_ID, "baseline"
    )

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    directional_rows = []
    suppression_rows = []
    position_rows = []
    loo_rows = []
    if observations:
        directional_rows = analyze_directional_pairwise_effects(observations, categories=EXCERPT_RATING_FIELDS)
        attach_context_name(directional_rows)
        by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
        attach_context_name(by_contrast_rows)
        suppression_rows = build_suppression_table(by_contrast_rows)

        cell_rows = analyze_pairwise_cell_rates(observations, categories=EXCERPT_RATING_FIELDS)
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
             "suppression_effect", "attenuation"],
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
    if baseline_observations:
        baseline_rows = analyze_baseline_margin(baseline_observations)
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
            ["model", "evaluation_regime", "category", "n_story_pairs", "r_baseline_margin_vs_abs_context_effect"],
            os.path.join(ANALYSIS_DIR, "baseline_margin_correlation.csv"),
        )

    if directional_rows:
        for model in sorted({r["model"] for r in directional_rows}):
            print_summary(model, args.sampling_regime, suppression_rows, position_rows, loo_rows, correlation_rows)
        print(f"\nWrote tidy tables to {ANALYSIS_DIR}/")
    else:
        print("No completed treatment observations yet. Nothing to analyze.")


if __name__ == "__main__":
    main()
