"""PRIMARY: single-text treatment vs. an uncontextualized behavioral
baseline, stratified by evaluation_regime.

A naturalistic treatment observation is only ever compared against the
naturalistic neutral baseline for that story, never against the
text_only_invariance baseline. This is this project's main question
(context sensitivity), not the researcher-reference comparisons in
context_analysis_reference.py, which are secondary.
"""

import statistics
from collections import defaultdict

from context_analysis_common import RATING_FIELDS
from context_analysis_stats import describe_attenuation


def neutral_baseline_means(observations):
    """Per (model, story_id, evaluation_regime): mean rating for each
    RATING_FIELDS category, from the neutral no-context context_single
    trial's replicate(s). A REFERENCE BASELINE for computing deltas -- not a
    ground-truth score. Keyed by evaluation_regime so a naturalistic
    treatment is never compared against an invariance-regime baseline."""
    sums = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single" or obs["dimension"] != "neutral":
            continue
        key = (obs["model"], obs["story_id"], obs["evaluation_regime"])
        for field in RATING_FIELDS:
            sums[key][field].append(obs["parsed_response"][field])
    return {key: {field: statistics.mean(vals) for field, vals in fields.items()} for key, fields in sums.items()}


def analyze_treatment_vs_neutral(observations):
    """Observation-level delta = treatment rating - neutral baseline mean,
    holding story/model/evaluation_regime fixed, for every context_single
    treatment observation and every rating category. Answers: holding the
    prose fixed, how does adding context change the rating relative to the
    same model's no-context baseline, under this evaluation regime?"""
    baselines = neutral_baseline_means(observations)
    rows = []
    for obs in observations.values():
        if obs["type"] != "context_single" or obs["dimension"] == "neutral":
            continue
        baseline = baselines.get((obs["model"], obs["story_id"], obs["evaluation_regime"]))
        if baseline is None:
            continue
        for field in RATING_FIELDS:
            rows.append(
                {
                    "model": obs["model"],
                    "evaluation_regime": obs["evaluation_regime"],
                    "story_id": obs["story_id"],
                    "dimension": obs["dimension"],
                    "value": obs["value"],
                    "replicate_id": obs["replicate_id"],
                    "category": field,
                    "treatment_rating": obs["parsed_response"][field],
                    "neutral_baseline_mean": round(baseline[field], 3),
                    "delta": round(obs["parsed_response"][field] - baseline[field], 3),
                }
            )
    return rows


def summarize_delta_by_story_condition(delta_rows):
    """story x treatment condition x evaluation_regime: mean delta across replicates."""
    grouped = defaultdict(list)
    for row in delta_rows:
        key = (row["model"], row["evaluation_regime"], row["story_id"], row["dimension"], row["value"], row["category"])
        grouped[key].append(row["delta"])
    return [
        {"model": m, "evaluation_regime": er, "story_id": s, "dimension": d, "value": v, "category": c,
         "mean_delta": round(statistics.mean(deltas), 3), "n": len(deltas)}
        for (m, er, s, d, v, c), deltas in grouped.items()
    ]


def summarize_delta_by_model_dimension_value(delta_rows):
    """aggregated model x evaluation_regime x dimension x value: mean delta across all stories/replicates."""
    grouped = defaultdict(list)
    for row in delta_rows:
        key = (row["model"], row["evaluation_regime"], row["dimension"], row["value"], row["category"])
        grouped[key].append(row["delta"])
    return [
        {"model": m, "evaluation_regime": er, "dimension": d, "value": v, "category": c,
         "mean_delta": round(statistics.mean(deltas), 3), "n": len(deltas)}
        for (m, er, d, v, c), deltas in grouped.items()
    ]


def compare_single_text_regimes(by_model_dim_value_rows):
    """For each (model, dimension, value, category) present under BOTH
    evaluation regimes, report delta_naturalistic, delta_text_only_invariance,
    and a descriptive attenuation label. Rows missing one regime are skipped."""
    by_key = defaultdict(dict)
    for row in by_model_dim_value_rows:
        key = (row["model"], row["dimension"], row["value"], row["category"])
        by_key[key][row["evaluation_regime"]] = row["mean_delta"]

    rows = []
    for (model, dimension, value, category), by_regime in by_key.items():
        if "naturalistic" not in by_regime or "text_only_invariance" not in by_regime:
            continue
        nat, inv = by_regime["naturalistic"], by_regime["text_only_invariance"]
        rows.append(
            {
                "model": model,
                "dimension": dimension,
                "value": value,
                "category": category,
                "delta_naturalistic": nat,
                "delta_text_only_invariance": inv,
                "attenuation": describe_attenuation(nat, inv),
            }
        )
    return rows


def print_treatment_vs_neutral(by_model_dim_value_rows):
    """PRIMARY single-text effect only. The naturalistic-vs-invariance
    comparison for this effect is printed later, together with the pairwise
    one -- see analyze_context.print_naturalistic_vs_invariance_comparison --
    so both regime comparisons appear at the same point in the output, per
    the documented scientific ordering (see analyze_context.py's module
    docstring)."""
    print("\n=== PRIMARY: single-text treatment vs an UNCONTEXTUALIZED BEHAVIORAL BASELINE ===")
    print("(the neutral condition is this model's own no-context rating, a reference point for measuring")
    print(" context-induced change -- not a ground-truth or 'true preference' score; stratified by")
    print(" evaluation_regime -- naturalistic and text_only_invariance are never pooled)")
    if not by_model_dim_value_rows:
        print("  No context_single treatment observations with a matching neutral baseline found.")
        return
    print("\n  By (model, evaluation_regime, dimension, value), averaged across stories/replicates, overall_quality only:")
    for row in sorted(by_model_dim_value_rows, key=lambda r: (r["model"], r["evaluation_regime"], r["dimension"], r["value"])):
        if row["category"] != "overall_quality":
            continue
        print(
            f"    {row['model']} | {row['evaluation_regime']} | {row['dimension']}={row['value']}: "
            f"mean_delta={row['mean_delta']:+.3f} (n={row['n']})"
        )
