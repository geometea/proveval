"""Focused analyser for the standalone named-LLM provenance pairwise
experiment (llm_provenance_trials.py).

Thin composition layer: every effect-size calculation here is reused,
unmodified, from context_analysis_io.py and context_analysis_pairwise.py.
This file only selects the experiment's own observations, attaches
human-readable provenance labels (from the contrast's own structured a/b
fields, never by parsing contrast_id text), and presents the result --
console summary plus CSVs -- for the six-pair provenance-effect matrix and
its position/heterogeneity/leave-one-out companions.

Distinct from analyze_context.py by design: this script never runs,
prints, or writes anything for the single-text, context_prompt,
tie-allowed, or researcher-reference-agreement families -- only the
directional pairwise provenance effect and the diagnostics that belong
with it. analyze_context.py itself is completely unmodified.

The primary estimand is unchanged from the general benchmark:

    P(story_1 preferred | story_1 receives provenance value a)
  - P(story_1 preferred | story_1 receives provenance value b)

so for provenance_claude_vs_chatgpt, positive means the Claude label
increased the probability the underlying story was preferred, relative to
the ChatGPT label -- not a raw "Story A wins more" count, and never
conflated with the counterbalanced display-position effect (see
context_analysis_pairwise.analyze_position_and_interaction_effects).

No Bradley-Terry/Elo ranking is computed here on purpose: the six pairwise
effects are not assumed transitive, and forcing them into a scalar ranking
would hide exactly the kind of non-transitive pattern (Claude > ChatGPT >
Gemini > Claude) this experiment exists to be able to show. The raw
pairwise matrix is the primary output.

Run this file directly: python3 analyze_llm_provenance.py
No API calls are made, and no non-Anthropic provider is contacted --
"Claude"/"ChatGPT"/"Gemini"/"DeepSeek" here are claimed-authorship labels
attached to fixed prose, not evaluator backends (see README.md's
"Standalone named-LLM provenance experiment" section).
"""

import argparse
import os
import statistics

from analyze import load_jsonl, write_csv
from context_contrasts import load_contrasts
from context_analysis_io import collapse_attempts, filter_by_sampling_regime, load_trials_by_id
from context_analysis_pairwise import (
    analyze_directional_pairwise_effects,
    analyze_leave_one_story_out,
    analyze_pairwise_cell_rates,
    analyze_per_story_context_effects,
    analyze_position_and_interaction_effects,
    summarize_directional_effects_by_contrast,
)
from llm_provenance_trials import (
    CONTRASTS_FILE,
    EXPERIMENT_ID,
    PROVENANCE_LABEL_NAMES,
    PROVENANCE_LABELS,
    TRIALS_FILE,
)
from run_trial import SAMPLING_REGIMES

RESULTS_FILE = "results/llm_provenance/raw.jsonl"
ANALYSIS_DIR = "results/llm_provenance/analysis"
DEFAULT_SAMPLING_REGIME = "low_variance_primary"

# overall_quality is reported first/primary; the other four categories are
# still fully retained everywhere (the response schema is unchanged).
PRIMARY_CATEGORY = "overall_quality"

LABEL_NAMES = [PROVENANCE_LABEL_NAMES[value_id] for value_id in PROVENANCE_LABELS]


def contrast_labels(contrasts):
    """{contrast_id: (label_a, label_b)}, from each contrast's own
    structured "a"/"b" fields -- never by parsing contrast_id text."""
    return {c["id"]: (PROVENANCE_LABEL_NAMES[c["a"]], PROVENANCE_LABEL_NAMES[c["b"]]) for c in contrasts}


def attach_labels(rows, labels_by_contrast_id):
    """Add label_a/label_b to each row in place, from its contrast_id."""
    for row in rows:
        row["label_a"], row["label_b"] = labels_by_contrast_id[row["contrast_id"]]
    return rows


def build_provenance_matrix(by_contrast_rows, category=PRIMARY_CATEGORY):
    """{(model, evaluation_regime): {label: {label: effect}}} for one
    category. Diagonal is 0.0; matrix[row][col] is the measured directional
    effect of assigning `row`'s label rather than `col`'s label, so
    matrix[col][row] == -matrix[row][col] by construction (antisymmetric).

    A contrast missing from the input (e.g. incomplete results) leaves its
    two off-diagonal cells at the default 0.0, indistinguishable here from a
    genuinely measured null effect -- pairwise_effects.csv (which also
    carries n_story_pairs) is the source of truth for whether a cell was
    actually measured.
    """
    matrices = {}
    for row in by_contrast_rows:
        if row["category"] != category:
            continue
        key = (row["model"], row["evaluation_regime"])
        matrix = matrices.setdefault(key, {name: {other: 0.0 for other in LABEL_NAMES} for name in LABEL_NAMES})
        matrix[row["label_a"]][row["label_b"]] = row["mean_directional_effect"]
        matrix[row["label_b"]][row["label_a"]] = -row["mean_directional_effect"]
    return matrices


def matrix_csv_rows(matrices):
    rows = []
    for (model, evaluation_regime), matrix in matrices.items():
        for row_label in LABEL_NAMES:
            for column_label in LABEL_NAMES:
                rows.append(
                    {
                        "model": model,
                        "evaluation_regime": evaluation_regime,
                        "row_label": row_label,
                        "column_label": column_label,
                        "directional_effect": round(matrix[row_label][column_label], 3),
                    }
                )
    return rows


def format_matrix(matrix):
    header = " " * 13 + "".join(f"{name:>10}" for name in LABEL_NAMES)
    lines = [header]
    for row_label in LABEL_NAMES:
        cells = "".join(f"{matrix[row_label][column_label]:>10.3f}" for column_label in LABEL_NAMES)
        lines.append(f"{row_label:<13}{cells}")
    return "\n".join(lines)


def model_regime_keys(rows):
    """Distinct (model, evaluation_regime) pairs present in `rows`, sorted for stable output."""
    return sorted({(r["model"], r["evaluation_regime"]) for r in rows})


def print_summary(model, evaluation_regime, sampling_regime, by_contrast_rows, matrices, position_rows, per_story_rows, loo_rows):
    print("\n=== Named-LLM provenance experiment ===")
    print(f"Evaluator: {model}")
    print(f"Evaluation regime: {evaluation_regime}")
    print(f"Sampling regime: {sampling_regime}")

    print(f"\nPRIMARY: {PRIMARY_CATEGORY.replace('_', ' ')}")
    primary_rows = [
        r for r in by_contrast_rows
        if r["model"] == model and r["evaluation_regime"] == evaluation_regime and r["category"] == PRIMARY_CATEGORY
    ]
    for row in sorted(primary_rows, key=lambda r: (r["label_a"], r["label_b"])):
        print(f"\n{row['label_a']} vs {row['label_b']}:")
        print(f"  mean directional effect: {row['mean_directional_effect']:+.3f}")
        print(f"  range: [{row['min_directional_effect']:+.3f}, {row['max_directional_effect']:+.3f}]")
        print(f"  n story pairs: {row['n_story_pairs']}")

    matrix = matrices.get((model, evaluation_regime))
    if matrix:
        print("\nPairwise provenance-effect matrix (overall_quality; row label assigned rather than column label):")
        print(format_matrix(matrix))

    position_for_group = [r for r in position_rows if r["model"] == model and r["evaluation_regime"] == evaluation_regime]
    if position_for_group:
        interactions = [r["context_x_position_interaction"] for r in position_for_group]
        positions = [r["position_effect_pooled"] for r in position_for_group if r["position_effect_pooled"] != ""]
        print("\nPosition-effect diagnostic (apparent label preference is not just A/B display-position preference):")
        print(f"  context x position interaction: mean={statistics.mean(interactions):+.3f} "
              f"range=[{min(interactions):+.3f}, {max(interactions):+.3f}]")
        if positions:
            print(f"  position effect (pooled over assignment): mean={statistics.mean(positions):+.3f} "
                  f"range=[{min(positions):+.3f}, {max(positions):+.3f}]")

    per_story_for_group = [
        r for r in per_story_rows
        if r["model"] == model and r["evaluation_regime"] == evaluation_regime and r["category"] == PRIMARY_CATEGORY
    ]
    if per_story_for_group:
        ranges = [r["range_effect"] for r in per_story_for_group]
        print(f"\nHeterogeneity across stories (overall_quality, range of each story's effect across its opponents):")
        print(f"  mean range={statistics.mean(ranges):.3f}  max range={max(ranges):.3f}")

    loo_for_group = [
        r for r in loo_rows
        if r["model"] == model and r["evaluation_regime"] == evaluation_regime and r["category"] == PRIMARY_CATEGORY
        and r["excluded_story_id"] != "(none -- full aggregate)"
    ]
    if loo_for_group:
        loo_values = [r["mean_directional_effect"] for r in loo_for_group]
        print(f"\nLeave-one-story-out sensitivity (overall_quality, aggregate recomputed excluding each story):")
        print(f"  range=[{min(loo_values):+.3f}, {max(loo_values):+.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Focused analysis for the standalone named-LLM provenance pairwise experiment.")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Trials manifest for metadata fallback (default: {TRIALS_FILE})")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Results file to analyze (default: {RESULTS_FILE})")
    parser.add_argument(
        "--sampling-regime",
        choices=list(SAMPLING_REGIMES),
        default=DEFAULT_SAMPLING_REGIME,
        help=f"Only observations recorded under this regime are analyzed (default: {DEFAULT_SAMPLING_REGIME}).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"No results file found at {args.results_file}. Nothing to analyze.")
        return

    raw_rows = load_jsonl(args.results_file)
    trials_by_id = load_trials_by_id(args.trials_file)
    all_observations, unresolved, absorbed, missing_metadata = collapse_attempts(raw_rows, trials_by_id)
    observations, excluded_by_regime = filter_by_sampling_regime(all_observations, args.sampling_regime)

    # Restrict to this experiment's own observations -- a defensive filter
    # in case --results-file is ever shared with the general benchmark
    # (trial_meta/experiment_id is embedded on every saved result row
    # regardless of which trials file produced it; see run_trial.trial_metadata).
    experiment_observations = {k: v for k, v in observations.items() if v.get("experiment_id") == EXPERIMENT_ID}
    excluded_other_experiment = len(observations) - len(experiment_observations)

    print(f"Raw API attempts: {len(raw_rows)}")
    print(f"Completed observations ({args.sampling_regime}): {len(experiment_observations)}")
    if unresolved:
        print(f"Unresolved failures: {len(unresolved)}")
    if excluded_by_regime:
        print(f"Excluded {excluded_by_regime} observation(s) recorded under a different sampling regime.")
    if excluded_other_experiment:
        print(f"Excluded {excluded_other_experiment} observation(s) belonging to a different experiment_id.")

    if not experiment_observations:
        print("No completed observations for this experiment yet. Nothing to analyze.")
        return

    contrasts = load_contrasts(CONTRASTS_FILE)
    labels_by_contrast_id = contrast_labels(contrasts)

    directional_rows = attach_labels(analyze_directional_pairwise_effects(experiment_observations), labels_by_contrast_id)
    by_contrast_rows = attach_labels(summarize_directional_effects_by_contrast(directional_rows), labels_by_contrast_id)

    cell_rows = analyze_pairwise_cell_rates(experiment_observations)
    position_rows = attach_labels(analyze_position_and_interaction_effects(cell_rows), labels_by_contrast_id)

    per_story_rows = attach_labels(analyze_per_story_context_effects(directional_rows), labels_by_contrast_id)
    loo_rows = attach_labels(analyze_leave_one_story_out(directional_rows), labels_by_contrast_id)

    matrices = build_provenance_matrix(by_contrast_rows)

    for model, evaluation_regime in model_regime_keys(directional_rows):
        print_summary(model, evaluation_regime, args.sampling_regime, by_contrast_rows, matrices, position_rows, per_story_rows, loo_rows)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    write_csv(
        by_contrast_rows,
        ["model", "evaluation_regime", "contrast_id", "label_a", "label_b", "category",
         "mean_directional_effect", "min_directional_effect", "max_directional_effect",
         "range_directional_effect", "n_story_pairs"],
        os.path.join(ANALYSIS_DIR, "pairwise_effects.csv"),
    )
    write_csv(
        directional_rows,
        ["model", "evaluation_regime", "contrast_id", "label_a", "label_b", "story_1_id", "story_2_id", "category",
         "p_story1_preferred_given_value_a", "n_value_a", "tie_n_value_a",
         "p_story1_preferred_given_value_b", "n_value_b", "tie_n_value_b",
         "directional_effect_a_minus_b"],
        os.path.join(ANALYSIS_DIR, "per_story_pair_effects.csv"),
    )
    write_csv(
        matrix_csv_rows(matrices),
        ["model", "evaluation_regime", "row_label", "column_label", "directional_effect"],
        os.path.join(ANALYSIS_DIR, "overall_quality_matrix.csv"),
    )
    write_csv(
        position_rows,
        ["model", "evaluation_regime", "contrast_id", "label_a", "label_b", "story_1_id", "story_2_id", "category",
         "context_effect_at_position_a", "context_effect_at_position_b", "context_x_position_interaction",
         "position_effect_pooled", "position_effect_under_forward", "position_effect_under_flipped"],
        os.path.join(ANALYSIS_DIR, "position_effects.csv"),
    )
    write_csv(
        per_story_rows,
        ["model", "evaluation_regime", "contrast_id", "label_a", "label_b", "category", "story_id",
         "n_opponents", "mean_effect", "min_effect", "max_effect", "range_effect", "per_opponent_effects"],
        os.path.join(ANALYSIS_DIR, "per_story_context_effects.csv"),
    )
    write_csv(
        loo_rows,
        ["model", "evaluation_regime", "contrast_id", "label_a", "label_b", "category",
         "excluded_story_id", "mean_directional_effect", "n_story_pairs"],
        os.path.join(ANALYSIS_DIR, "leave_one_story_out.csv"),
    )

    print(f"\nWrote tidy tables to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
