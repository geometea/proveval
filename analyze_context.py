"""Offline analysis for the v0.2 context benchmark (context_trials.py results).

Reads a results file (default results/context_raw.jsonl) produced by
run_trial.py/run_batch.py against data/context_trials.jsonl, collapses retry
attempts, and runs analyses specific to the new trial types (context_single,
context_pairwise, context_prompt). Completely separate from analyze.py, which
keeps analyzing the v0.1 pilot pipeline unchanged.

This file is the thin orchestrator: argument parsing, the fixed print/CSV
order in main(), and print_naturalistic_vs_invariance_comparison (which
genuinely spans both PRIMARY effects, so it doesn't belong to either format's
own module). The actual analysis logic lives in, and is re-exported here
from, five focused modules -- read whichever matches what you're looking at:

  context_analysis_common.py    -- shared file-path/schema constants
  context_analysis_io.py        -- loading results, collapsing retry attempts,
                                    sampling_regime filtering, dataset inventory
  context_analysis_stats.py     -- tie-aware rank statistics (average_ranks,
                                    kendall_tau_b, spearman_tie_aware,
                                    tied_groups) and the descriptive
                                    naturalistic-vs-invariance attenuation label
  context_analysis_single_text.py -- PRIMARY single-text treatment-vs-baseline
  context_analysis_pairwise.py  -- PRIMARY pairwise directional effect, the
                                    position/interaction/heterogeneity/
                                    leave-one-out decompositions, the optional
                                    tie-allowed diagnostic, and context_prompt
  context_analysis_reference.py -- SECONDARY researcher-reference agreement
                                    (single-text rankings and direct pairwise
                                    comparison against the researcher's judgments)

Every name these modules define is imported (not wildcarded) into this
module's namespace below, so `import analyze_context as ac; ac.whatever(...)`
keeps working exactly as it did when this was one 1,800-line file.

Two independent factors are recorded on every v0.2 trial/result and must
never be silently pooled:

- sampling_regime (run_trial.SAMPLING_REGIMES): API sampling settings.
  Selected once per analysis run via --sampling-regime (default
  low_variance_primary); the excluded regime's count is reported.
- evaluation_regime (context_trials.EVALUATION_REGIMES): "naturalistic"
  (ordinary evaluation framing) or "text_only_invariance" (explicitly
  instructed to judge only the prose). This is the substantive research
  variable, so unlike sampling_regime it is NOT filtered down to one value --
  every analysis below stratifies by it and reports both regimes side by
  side, so naturalistic context sensitivity and invariance-instructed
  effects can be compared directly rather than averaged together.

This project's primary question is how external contextual cues change LLM
evaluations of fixed prose, and how robust those evaluations are to
contextual perturbation. The analyses below are organized around that,
PRIMARY first, SECONDARY after -- never collapsed into one "ranking"
analysis (see FINAL_DESIGN.md's "Primary and secondary empirical questions"):

  PRIMARY -- context sensitivity and invariance:
    Single-text: does context move the score of the SAME story relative to
      its own uncontextualized behavioral baseline (see FINAL_DESIGN.md's
      "What 'model preference' means")?       -> analyze_treatment_vs_neutral
    Pairwise (choice_mode="forced" ONLY -- see is_forced_choice_pairwise):
      does assigning context to a story change its probability of
      being preferred?                        -> analyze_directional_pairwise_effects
      Position effect / context x position interaction (a pooled context
      effect must not hide a large or context-dependent position effect):
                                                -> analyze_pairwise_cell_rates,
                                                   analyze_position_and_interaction_effects
      Heterogeneity across the 66 non-independent story pairs (12 stories,
      11 pairs each -- see FINAL_DESIGN.md): per-story summaries and
      leave-one-story-out sensitivity, never just a bare mean:
                                                -> analyze_per_story_context_effects,
                                                   analyze_leave_one_story_out
    Invariance: how much of either effect survives an explicit
      text-only-judge instruction?            -> compare_single_text_regimes /
                                                   compare_pairwise_regimes
                                                   (evaluation_regime stratification;
                                                   see FINAL_DESIGN.md's demand-
                                                   characteristics caveat)
    Stochastic robustness: repeated identical cells are kept as separate
      observations throughout (never collapsed to a majority vote), so
      effect sizes above can be read against ordinary response variation --
      see collapse_attempts and "Replication" in FINAL_DESIGN.md.

  SECONDARY, optional -- tie-allowed hedging/indifference diagnostic
  (choice_mode="tie_allowed" ONLY, never pooled with the forced-choice
  PRIMARY results above): how often does the model decline a strict
  preference, and does context shift that rate? -> analyze_tie_allowed_diagnostic

  SECONDARY -- agreement with the researcher reference ordering (a
  single-researcher, ORDINAL-only, personalized preference ranking over the
  12-story corpus -- not population ground truth, not a cardinal utility,
  and not the organizing goal of this benchmark; see FINAL_DESIGN.md's
  "Researcher reference ranking"). This is never the headline result:
    Single-text: does the (possibly tied) score ordering under a condition
      agree with the researcher's ordering?   -> rank_from_single_text_by_condition
                                                  + Kendall tau-b / tie-aware Spearman
    Pairwise: do direct model pairwise choices agree with the researcher's
      direct pairwise judgments?              -> analyze_pairwise_vs_human_reference

main() prints/writes all of the above in one fixed scientific order:
stochasticity, single-text PRIMARY, pairwise PRIMARY (directional, then
position/interaction, then heterogeneity, then leave-one-out), the
naturalistic-vs-invariance comparison for both PRIMARY effects, the optional
tie-allowed diagnostic, and finally the SECONDARY researcher-reference
agreement -- reference-agreement is deliberately last and is never the
headline output.

The two PRIMARY effect analyses are additionally compared across
evaluation_regime (compare_single_text_regimes / compare_pairwise_regimes)
with a purely descriptive "attenuation" label -- not a new statistical
model. A table or CSV ranking conditions by reference-agreement is
descriptive only, not evidence that the top-ranked condition is genuinely
"best" -- see the multiple-comparisons caution in FINAL_DESIGN.md.

The neutral/no-context condition and the researcher reference ordering are
two DIFFERENT reference concepts and neither is ground truth: the neutral
baseline is this model's own no-context answer, used only to measure
within-story context-induced change; the researcher reference is one
person's fixed ordinal preference, used only for the secondary,
personalized agreement analysis. A naturalistic context effect is not
automatically "bias"; an effect that survives text_only_invariance
instructions is described as an invariance effect, never automatically as
bias/sycophancy/irrationality (see FINAL_DESIGN.md's terminology section).

Model rating ties are legitimate and are never broken by story ID, filename,
alphabetical order, or insertion order -- see context_analysis_stats.py's
average_ranks/kendall_tau_b/spearman_tie_aware/tied_groups.

Run this file directly: python3 analyze_context.py
No API calls are made, and ANTHROPIC_API_KEY is never read.
"""

import argparse
import os

from analyze import load_jsonl, write_csv
from run_trial import SAMPLING_REGIMES

from context_analysis_common import (
    ANALYSIS_DIR,
    DEFAULT_SAMPLING_REGIME,
    HUMAN_PAIRWISE_FILE,
    HUMAN_REFERENCE_FILE,
    RATING_FIELDS,
    RESULTS_FILE,
    TRIALS_FILE,
)
from context_analysis_io import (
    collapse_attempts,
    count_by,
    filter_by_sampling_regime,
    get_trial_meta,
    load_trials_by_id,
    print_counts,
    print_inventory,
)
from context_analysis_stats import (
    average_ranks,
    describe_attenuation,
    human_reference_scores,
    kendall_tau_b,
    pairwise_diagnostics_from_scores,
    pearson_correlation,
    spearman_tie_aware,
    tied_groups,
)
from context_analysis_single_text import (
    analyze_treatment_vs_neutral,
    compare_single_text_regimes,
    neutral_baseline_means,
    print_treatment_vs_neutral,
    summarize_delta_by_model_dimension_value,
    summarize_delta_by_story_condition,
)
from context_analysis_pairwise import (
    analyze_directional_pairwise_effects,
    analyze_leave_one_story_out,
    analyze_pairwise_cell_rates,
    analyze_pairwise_changed_diagnostic,
    analyze_per_story_context_effects,
    analyze_position_and_interaction_effects,
    analyze_prompt_context_effects,
    analyze_tie_allowed_diagnostic,
    choice_to_story_id,
    compare_pairwise_regimes,
    index_prompt_by_value,
    index_pairwise_by_assignment_fixed_position,
    is_forced_choice_pairwise,
    summarize_directional_effects_by_contrast,
    summarize_directional_pairwise_effects,
    summarize_grouped_change_rate,
    summarize_leave_one_story_out,
    summarize_pairwise_changed_diagnostic,
    summarize_per_story_context_effects,
    summarize_position_and_interaction_effects,
    summarize_prompt_context_effects,
    summarize_tie_allowed_diagnostic,
)
from context_analysis_reference import (
    analyze_pairwise_vs_human_reference,
    compute_pairwise_wins,
    human_comparison_row,
    load_human_pairwise,
    load_human_reference,
    print_ranking_with_ties,
    rank_from_pairwise_wins,
    rank_from_single_text,
    rank_from_single_text_by_condition,
    ranking_csv_rows,
    summarize_pairwise_vs_human_reference,
)

# Re-export everything above under analyze_context's own namespace (flake8:
# these imports are used via `analyze_context.<name>`, not by name in this
# file) so external callers -- notably tests/test_analyze_context.py -- keep
# working unchanged after the module split.
__all__ = [
    "ANALYSIS_DIR", "DEFAULT_SAMPLING_REGIME", "HUMAN_PAIRWISE_FILE", "HUMAN_REFERENCE_FILE",
    "RATING_FIELDS", "RESULTS_FILE", "TRIALS_FILE",
    "collapse_attempts", "count_by", "filter_by_sampling_regime", "get_trial_meta",
    "load_trials_by_id", "print_counts", "print_inventory",
    "average_ranks", "describe_attenuation", "human_reference_scores", "kendall_tau_b",
    "pairwise_diagnostics_from_scores", "pearson_correlation", "spearman_tie_aware", "tied_groups",
    "analyze_treatment_vs_neutral", "compare_single_text_regimes", "neutral_baseline_means",
    "print_treatment_vs_neutral", "summarize_delta_by_model_dimension_value", "summarize_delta_by_story_condition",
    "analyze_directional_pairwise_effects", "analyze_leave_one_story_out", "analyze_pairwise_cell_rates",
    "analyze_pairwise_changed_diagnostic", "analyze_per_story_context_effects",
    "analyze_position_and_interaction_effects", "analyze_prompt_context_effects",
    "analyze_tie_allowed_diagnostic", "choice_to_story_id", "compare_pairwise_regimes",
    "index_prompt_by_value", "index_pairwise_by_assignment_fixed_position", "is_forced_choice_pairwise",
    "summarize_directional_effects_by_contrast", "summarize_directional_pairwise_effects",
    "summarize_grouped_change_rate", "summarize_leave_one_story_out", "summarize_pairwise_changed_diagnostic",
    "summarize_per_story_context_effects", "summarize_position_and_interaction_effects",
    "summarize_prompt_context_effects", "summarize_tie_allowed_diagnostic",
    "analyze_pairwise_vs_human_reference", "compute_pairwise_wins", "human_comparison_row",
    "load_human_pairwise", "load_human_reference", "print_ranking_with_ties", "rank_from_pairwise_wins",
    "rank_from_single_text", "rank_from_single_text_by_condition", "ranking_csv_rows",
    "summarize_pairwise_vs_human_reference",
]


def print_naturalistic_vs_invariance_comparison(single_regime_comparison_rows, pairwise_regime_comparison_rows):
    """Both PRIMARY effects' naturalistic-vs-invariance comparison, printed
    together at one point in the output (see module docstring's ordering).
    Purely descriptive attenuation labels, never a significance test --
    see describe_attenuation and FINAL_DESIGN.md's demand-characteristics
    caveat: a reduced invariance-regime effect shows the model responding to
    that instruction's wording, not proof that a context-free preference was
    recovered, since the instruction is itself an intervention the model may
    react to simply by inferring it is being evaluated.
    """
    print("\n=== Naturalistic vs. text-only-invariance: does the effect survive an explicit judge-the-prose-only instruction? ===")
    print("(descriptive attenuation labels only, not a significance test; a reduced or absent invariance effect")
    print(" shows responsiveness to that instruction's wording, not proof that a context-free 'true preference'")
    print(" was recovered -- the instruction is itself an intervention the model may react to just by inferring")
    print(" it is being evaluated. See FINAL_DESIGN.md's demand-characteristics caveat.)")
    if single_regime_comparison_rows:
        print("\n  Single-text (overall_quality only):")
        for row in sorted(single_regime_comparison_rows, key=lambda r: (r["model"], r["dimension"], r["value"])):
            if row["category"] != "overall_quality":
                continue
            print(
                f"    {row['model']} | {row['dimension']}={row['value']}: "
                f"naturalistic={row['delta_naturalistic']:+.3f}  invariance={row['delta_text_only_invariance']:+.3f}  "
                f"({row['attenuation']})"
            )
    if pairwise_regime_comparison_rows:
        print("\n  Pairwise directional effect:")
        for row in sorted(pairwise_regime_comparison_rows, key=lambda r: (r["model"], r["contrast_id"], r["category"])):
            print(
                f"    {row['model']} | {row['contrast_id']} | {row['category']}: "
                f"naturalistic={row['effect_naturalistic']:+.3f}  invariance={row['effect_text_only_invariance']:+.3f}  "
                f"({row['attenuation']})"
            )
    if not single_regime_comparison_rows and not pairwise_regime_comparison_rows:
        print("  No (model, dimension/contrast, category) present under both evaluation regimes yet.")


def main():
    parser = argparse.ArgumentParser(description="Offline analysis for the v0.2 context benchmark.")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Results file to analyze (default: {RESULTS_FILE})")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Trials manifest for metadata fallback (default: {TRIALS_FILE})")
    parser.add_argument(
        "--sampling-regime",
        choices=list(SAMPLING_REGIMES),
        default=DEFAULT_SAMPLING_REGIME,
        help="Only observations recorded under this regime are analyzed (default: low_variance_primary). "
        "The primary and secondary regimes are never pooled automatically. Unlike sampling_regime, "
        "evaluation_regime (naturalistic / text_only_invariance) is not filtered here -- it's stratified "
        "throughout instead, since it's the substantive variable under study.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"No results file found at {args.results_file}. Nothing to analyze.")
        return

    raw_rows = load_jsonl(args.results_file)
    trials_by_id = load_trials_by_id(args.trials_file)
    all_observations, unresolved, absorbed, missing_metadata = collapse_attempts(raw_rows, trials_by_id)
    observations, excluded_by_regime = filter_by_sampling_regime(all_observations, args.sampling_regime)

    print_inventory(raw_rows, all_observations, unresolved, absorbed, missing_metadata)
    print(f"\nSampling regime selected for this analysis run: {args.sampling_regime}")
    print(f"  Observations analyzed under this regime: {len(observations)}")
    if excluded_by_regime:
        print(
            f"  Excluded {excluded_by_regime} observation(s) recorded under a different sampling regime "
            f"(pass --sampling-regime to analyze them instead; regimes are never pooled automatically)."
        )
    evaluation_regime_counts = count_by(observations.values(), lambda o: o.get("evaluation_regime", "n/a"))
    print(f"  Evaluation regimes present in this run (stratified below, never pooled): {evaluation_regime_counts}")

    # (1) Stochasticity / repeated-call behavior: every replicate of a cell is
    # kept as its own observation, never collapsed to a majority vote (see
    # collapse_attempts and "Replication" in FINAL_DESIGN.md) -- the "by
    # replicate" breakdown above is exactly this; a dedicated effect-size-vs-
    # replicate-variance statistic is an acknowledged gap, not computed here.
    print("\n(1) Stochastic robustness: replicates are preserved as separate observations above, never")
    print("    collapsed to a majority vote, so effects below can be read against ordinary response variation.")

    # (2) PRIMARY, single-text: treatment vs. an uncontextualized behavioral baseline
    delta_obs_rows = analyze_treatment_vs_neutral(observations)
    delta_by_story_condition_rows = summarize_delta_by_story_condition(delta_obs_rows)
    delta_by_model_dim_value_rows = summarize_delta_by_model_dimension_value(delta_obs_rows)
    print_treatment_vs_neutral(delta_by_model_dim_value_rows)

    # (3) PRIMARY, pairwise: directional context effect, plus its immediate
    # SECONDARY raw-choice aside and the related context_prompt family
    directional_rows = analyze_directional_pairwise_effects(observations)
    directional_by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
    summarize_directional_pairwise_effects(directional_rows, directional_by_contrast_rows)

    changed_rows = analyze_pairwise_changed_diagnostic(observations)
    summarize_pairwise_changed_diagnostic(changed_rows)

    prompt_effect_rows = analyze_prompt_context_effects(observations)
    summarize_prompt_context_effects(prompt_effect_rows)

    # (4) A/B display-position effect and context x position interaction
    cell_rows = analyze_pairwise_cell_rates(observations)
    position_interaction_rows = analyze_position_and_interaction_effects(cell_rows)
    summarize_position_and_interaction_effects(position_interaction_rows)

    # (5) Heterogeneity across story pairs and stories (not 66 independent units)
    per_story_rows = analyze_per_story_context_effects(directional_rows)
    summarize_per_story_context_effects(per_story_rows)

    # (6) Leave-one-story-out sensitivity (descriptive robustness check only)
    leave_one_out_rows = analyze_leave_one_story_out(directional_rows)
    summarize_leave_one_story_out(leave_one_out_rows)

    # (7) Naturalistic vs. text-only-invariance comparison, both PRIMARY effects together
    single_regime_comparison_rows = compare_single_text_regimes(delta_by_model_dim_value_rows)
    pairwise_regime_comparison_rows = compare_pairwise_regimes(directional_by_contrast_rows)
    print_naturalistic_vs_invariance_comparison(single_regime_comparison_rows, pairwise_regime_comparison_rows)

    # (8) SECONDARY, optional: tie-allowed hedging/indifference diagnostic
    tie_allowed_rows = analyze_tie_allowed_diagnostic(observations)
    summarize_tie_allowed_diagnostic(tie_allowed_rows)

    # (9) SECONDARY: direct pairwise choices vs the researcher's judgments
    human_pairs = load_human_pairwise()
    pairwise_vs_human_rows, pairwise_vs_human_summary_rows = analyze_pairwise_vs_human_reference(observations, human_pairs)
    summarize_pairwise_vs_human_reference(pairwise_vs_human_summary_rows)

    # --- SECONDARY: tie-aware ranking agreement with the researcher reference
    # (per-condition is preferred over the pooled diagnostic further below) ---
    human_reference = load_human_reference()

    single_by_condition = rank_from_single_text_by_condition(observations)
    single_by_condition_rows = []
    human_comparison_rows = []
    print("\n=== SECONDARY: tie-aware ranking per (model, evaluation_regime, dimension, value) vs researcher reference ===")
    print("(personalized, ordinal-only agreement analysis -- not the benchmark's primary question; see FINAL_DESIGN.md.")
    print(" Includes the neutral baseline as its own condition; ties are never broken artificially.)")
    if not single_by_condition:
        print("  No context_single observations found.")
    for (model, evaluation_regime, dimension, value), scores in sorted(single_by_condition.items(), key=lambda kv: str(kv[0])):
        print(f"  Model: {model}  evaluation_regime={evaluation_regime}  dimension={dimension}  value={value}")
        print_ranking_with_ties(scores)
        single_by_condition_rows.extend(
            ranking_csv_rows(scores, {"model": model, "evaluation_regime": evaluation_regime, "dimension": dimension, "value": value})
        )
        row = human_comparison_row(model, evaluation_regime, dimension, value, "single_text_per_condition", scores, human_reference, human_pairs)
        human_comparison_rows.append(row)
        tau_text = row["kendall_tau_b"] if row["kendall_tau_b"] != "" else "unavailable"
        spearman_text = row["spearman_vs_human"] if row["spearman_vs_human"] != "" else "unavailable"
        print(f"    vs researcher reference (n_common={row['n_common_with_human_reference']}): Kendall tau-b={tau_text}  Spearman(tie-aware)={spearman_text}")

    print("\n=== SECONDARY diagnostic only: pooled ranking across ALL conditions (per evaluation_regime) ===")
    print("(averages over every context_single trial regardless of dimension/value, within one evaluation_regime;")
    print(" a rough sanity check, NOT the main result -- never pooled across evaluation_regime)")
    single_pooled_by_key = rank_from_single_text(observations)
    single_ranking_rows = []
    for (model, evaluation_regime), scores in sorted(single_pooled_by_key.items()):
        print(f"  Model: {model}  evaluation_regime={evaluation_regime}")
        print_ranking_with_ties(scores)
        single_ranking_rows.extend(ranking_csv_rows(scores, {"model": model, "evaluation_regime": evaluation_regime}))
        human_comparison_rows.append(
            human_comparison_row(model, evaluation_regime, "ALL", "ALL", "single_text_pooled_diagnostic", scores, human_reference, human_pairs)
        )

    print("\n=== SECONDARY diagnostic only: pooled Copeland ranking (all contrasts/assignments/positions, per evaluation_regime) ===")
    print("(not a ranking under any one context condition -- see rank_from_pairwise_wins docstring)")
    pairwise_wins = compute_pairwise_wins(observations)
    pairwise_ranking_rows = []
    for (model, evaluation_regime), tally_by_story in sorted(pairwise_wins.items()):
        scores = rank_from_pairwise_wins(tally_by_story)  # {story_id: (copeland, win_rate, games)}
        print(f"  Model: {model}  evaluation_regime={evaluation_regime}")
        groups = tied_groups({sid: v[0] for sid, v in scores.items()}, list(scores.keys()))
        for g in groups:
            ids = ", ".join(g["story_ids"])
            print(f"    rank {g['rank_position']}: {ids}  (copeland={g['score']})")
        rank_by_story = {s: g["rank_position"] for g in groups for s in g["story_ids"]}
        size_by_story = {s: len(g["story_ids"]) for g in groups for s in g["story_ids"]}
        for sid in sorted(scores):
            copeland, win_rate, games = scores[sid]
            pairwise_ranking_rows.append(
                {
                    "model": model, "evaluation_regime": evaluation_regime, "story_id": sid, "copeland": copeland,
                    "win_rate": round(win_rate, 3), "games": games,
                    "rank_position": rank_by_story[sid], "tie_group_size": size_by_story[sid],
                }
            )
        scores_for_comparison = {sid: (v[0], v[2]) for sid, v in scores.items()}  # (copeland, games) as (score, n)
        human_comparison_rows.append(
            human_comparison_row(model, evaluation_regime, "ALL", "ALL", "pairwise_pooled_diagnostic", scores_for_comparison, human_reference, human_pairs)
        )

    print("\n=== SECONDARY: agreement with the researcher reference ordering -- availability ===")
    print("(personalized, ordinal, exploratory -- a descriptive comparison, not evidence of an objectively 'best' context;")
    print(" see the multiple-comparisons caution in FINAL_DESIGN.md before reading too much into any single top result)")
    if human_reference is None:
        print("  data/human_reference.json does not exist yet (run human_ranking.py once enough")
        print("  pairwise judgments are collected). Kendall tau-b / Spearman are unavailable for now.")
    if not human_pairs:
        print("  data/human_pairwise.jsonl has no judgments yet. Pairwise concordance diagnostics are unavailable.")
    print(f"  Full model/dimension/value breakdown written to {ANALYSIS_DIR}/human_comparison.csv")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    write_csv(
        changed_rows,
        ["model", "evaluation_regime", "contrast_id", "dimension", "story_1_id", "story_2_id", "position", "replicate_id", "category",
         "story1_context_forward", "story1_context_flipped", "forward_choice", "forward_chosen_story_id",
         "flipped_choice", "flipped_chosen_story_id", "changed"],
        os.path.join(ANALYSIS_DIR, "pairwise_changed_diagnostic.csv"),
    )
    write_csv(
        directional_rows,
        ["model", "evaluation_regime", "contrast_id", "story_1_id", "story_2_id", "category", "p_story1_preferred_given_value_a",
         "n_value_a", "tie_n_value_a", "p_story1_preferred_given_value_b", "n_value_b", "tie_n_value_b",
         "directional_effect_a_minus_b"],
        os.path.join(ANALYSIS_DIR, "pairwise_directional_effects.csv"),
    )
    write_csv(
        directional_by_contrast_rows,
        ["model", "evaluation_regime", "contrast_id", "category", "mean_directional_effect",
         "min_directional_effect", "max_directional_effect", "range_directional_effect", "n_story_pairs"],
        os.path.join(ANALYSIS_DIR, "pairwise_directional_effects_by_contrast.csv"),
    )
    write_csv(
        pairwise_regime_comparison_rows,
        ["model", "contrast_id", "category", "effect_naturalistic", "effect_text_only_invariance", "attenuation"],
        os.path.join(ANALYSIS_DIR, "pairwise_regime_comparison.csv"),
    )
    write_csv(
        cell_rows,
        ["model", "evaluation_regime", "contrast_id", "story_1_id", "story_2_id", "category",
         "forward_story1_as_a_story_1_wins", "forward_story1_as_a_story_2_wins", "forward_story1_as_a_n",
         "forward_story2_as_a_story_1_wins", "forward_story2_as_a_story_2_wins", "forward_story2_as_a_n",
         "flipped_story1_as_a_story_1_wins", "flipped_story1_as_a_story_2_wins", "flipped_story1_as_a_n",
         "flipped_story2_as_a_story_1_wins", "flipped_story2_as_a_story_2_wins", "flipped_story2_as_a_n"],
        os.path.join(ANALYSIS_DIR, "pairwise_four_cell_rates.csv"),
    )
    write_csv(
        position_interaction_rows,
        ["model", "evaluation_regime", "contrast_id", "story_1_id", "story_2_id", "category",
         "context_effect_at_position_a", "context_effect_at_position_b", "context_x_position_interaction",
         "position_effect_pooled", "position_effect_under_forward", "position_effect_under_flipped"],
        os.path.join(ANALYSIS_DIR, "pairwise_position_and_interaction_effects.csv"),
    )
    write_csv(
        per_story_rows,
        ["model", "evaluation_regime", "contrast_id", "category", "story_id", "n_opponents",
         "mean_effect", "min_effect", "max_effect", "range_effect", "per_opponent_effects"],
        os.path.join(ANALYSIS_DIR, "pairwise_per_story_context_effects.csv"),
    )
    write_csv(
        leave_one_out_rows,
        ["model", "evaluation_regime", "contrast_id", "category", "excluded_story_id",
         "mean_directional_effect", "n_story_pairs"],
        os.path.join(ANALYSIS_DIR, "pairwise_leave_one_story_out.csv"),
    )
    write_csv(
        tie_allowed_rows,
        ["model", "evaluation_regime", "contrast_id", "story_1_id", "story_2_id", "category", "assignment",
         "p_story_1_chosen", "p_story_2_chosen", "p_tie", "n"],
        os.path.join(ANALYSIS_DIR, "pairwise_tie_allowed_diagnostic.csv"),
    )
    write_csv(
        prompt_effect_rows,
        ["model", "evaluation_regime", "contrast_id", "story_a_id", "story_b_id", "replicate_id", "category", "baseline_value",
         "treatment_value", "baseline_choice", "treatment_choice", "changed"],
        os.path.join(ANALYSIS_DIR, "prompt_context_effects.csv"),
    )
    write_csv(
        pairwise_vs_human_rows,
        ["model", "evaluation_regime", "story_a_id", "story_b_id", "contrast_id", "assignment", "position", "replicate_id",
         "human_winner", "model_choice", "model_chosen_story_id", "outcome"],
        os.path.join(ANALYSIS_DIR, "pairwise_vs_human_reference_observations.csv"),
    )
    write_csv(
        pairwise_vs_human_summary_rows,
        ["model", "evaluation_regime", "concordant", "discordant", "model_tied", "n", "concordant_rate"],
        os.path.join(ANALYSIS_DIR, "pairwise_vs_human_reference_summary.csv"),
    )
    write_csv(
        delta_obs_rows,
        ["model", "evaluation_regime", "story_id", "dimension", "value", "replicate_id", "category", "treatment_rating",
         "neutral_baseline_mean", "delta"],
        os.path.join(ANALYSIS_DIR, "treatment_vs_neutral_observations.csv"),
    )
    write_csv(
        delta_by_story_condition_rows,
        ["model", "evaluation_regime", "story_id", "dimension", "value", "category", "mean_delta", "n"],
        os.path.join(ANALYSIS_DIR, "treatment_vs_neutral_by_story_condition.csv"),
    )
    write_csv(
        delta_by_model_dim_value_rows,
        ["model", "evaluation_regime", "dimension", "value", "category", "mean_delta", "n"],
        os.path.join(ANALYSIS_DIR, "treatment_vs_neutral_by_model_dimension_value.csv"),
    )
    write_csv(
        single_regime_comparison_rows,
        ["model", "dimension", "value", "category", "delta_naturalistic", "delta_text_only_invariance", "attenuation"],
        os.path.join(ANALYSIS_DIR, "single_text_regime_comparison.csv"),
    )
    write_csv(
        single_by_condition_rows,
        ["model", "evaluation_regime", "dimension", "value", "story_id", "mean_overall_quality", "n", "rank_position", "tie_group_size"],
        os.path.join(ANALYSIS_DIR, "single_text_rankings_by_condition.csv"),
    )
    write_csv(
        single_ranking_rows,
        ["model", "evaluation_regime", "story_id", "mean_overall_quality", "n", "rank_position", "tie_group_size"],
        os.path.join(ANALYSIS_DIR, "single_text_rankings_pooled_diagnostic.csv"),
    )
    write_csv(
        pairwise_ranking_rows,
        ["model", "evaluation_regime", "story_id", "copeland", "win_rate", "games", "rank_position", "tie_group_size"],
        os.path.join(ANALYSIS_DIR, "pairwise_rankings_pooled_diagnostic.csv"),
    )
    write_csv(
        human_comparison_rows,
        ["model", "evaluation_regime", "dimension", "value", "ranking_source", "n_common_with_human_reference", "kendall_tau_b",
         "spearman_vs_human", "pairwise_concordant", "pairwise_discordant", "pairwise_model_tied",
         "pairwise_concordant_rate", "pairwise_checked_n"],
        os.path.join(ANALYSIS_DIR, "human_comparison.csv"),
    )

    print(f"\nWrote tidy tables to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
