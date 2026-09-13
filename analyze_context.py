"""Offline analysis for the v0.2 context benchmark (context_trials.py results).

Reads a results file (default results/context_raw.jsonl) produced by
run_trial.py/run_batch.py against data/context_trials.jsonl, collapses retry
attempts, and runs analyses specific to the new trial types (context_single,
context_pairwise, context_prompt). Completely separate from analyze.py, which
keeps analyzing the v0.1 pilot pipeline unchanged.

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
alphabetical order, or insertion order -- see average_ranks/kendall_tau_b/
spearman_tie_aware/tied_groups below.

Run this file directly: python3 analyze_context.py
No API calls are made, and ANTHROPIC_API_KEY is never read.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

from analyze import load_jsonl, is_successful, attempt_number, write_csv
from run_trial import SAMPLING_REGIMES

RESULTS_FILE = "results/context_raw.jsonl"
TRIALS_FILE = "data/context_trials.jsonl"
HUMAN_REFERENCE_FILE = "data/human_reference.json"
HUMAN_PAIRWISE_FILE = "data/human_pairwise.jsonl"
ANALYSIS_DIR = "results/context_analysis"
DEFAULT_SAMPLING_REGIME = "low_variance_primary"

RATING_FIELDS = ["plot_structure", "prose_style", "characterization", "originality", "overall_quality"]


# ---------------------------------------------------------------------------
# Loading trial metadata: prefer a result row's own embedded trial_meta
# (see run_trial.trial_metadata), fall back to joining the trials manifest by
# trial_id if it's missing. Either way, no trial_id parsing is involved.
# ---------------------------------------------------------------------------

def load_trials_by_id(path=TRIALS_FILE):
    if not os.path.exists(path):
        return {}
    return {t["trial_id"]: t for t in load_jsonl(path)}


def get_trial_meta(row, trials_by_id):
    if row.get("trial_meta"):
        return row["trial_meta"]
    trial = trials_by_id.get(row["trial_id"])
    if trial is None:
        return None
    return {k: v for k, v in trial.items() if k != "prompt"}


# ---------------------------------------------------------------------------
# Collapsing retries (same semantics as analyze.py: group by
# (trial_id, model, replicate_id, sampling_regime), keep the most recent
# success). sampling_regime is part of the key so a retry under a different
# regime is never silently merged into the same observation; evaluation_regime
# doesn't need to be, since it's already baked into a distinct trial_id (see
# context_trials.py), so a naturalistic and an invariance trial can never
# collide here. This collapses ATTEMPTS (retries of the same cell after a
# parse/validation failure), never REPLICATES -- every replicate_id remains
# its own observation; repeated identical cells are preserved as separate
# data points, not reduced to a majority vote (see
# analyze_directional_pairwise_effects and neutral_baseline_means, which both
# estimate frequencies/means across them).
# ---------------------------------------------------------------------------

def collapse_attempts(raw_rows, trials_by_id):
    """Returns (observations, unresolved, absorbed_failed_attempts, missing_metadata)."""
    attempts_by_key = defaultdict(list)
    for row in raw_rows:
        key = (row["trial_id"], row["model"], row["replicate_id"], row.get("sampling_regime"))
        attempts_by_key[key].append(row)

    observations = {}
    unresolved = {}
    absorbed_failed_attempts = 0
    missing_metadata = []

    for key, attempts in attempts_by_key.items():
        attempts = sorted(attempts, key=attempt_number)
        successes = [a for a in attempts if is_successful(a)]
        if not successes:
            unresolved[key] = attempts
            continue

        row = successes[-1]
        meta = get_trial_meta(row, trials_by_id)
        if meta is None:
            missing_metadata.append(key)
            continue

        trial_id, model, replicate_id, sampling_regime = key
        observations[key] = {
            **meta,
            "model": model,
            "replicate_id": replicate_id,
            "attempt_id": attempt_number(row),
            "parsed_response": row["parsed_response"],
            "sampling_regime": sampling_regime,
            "sampling_params": row.get("sampling_params"),
        }
        absorbed_failed_attempts += len(attempts) - 1

    return observations, unresolved, absorbed_failed_attempts, missing_metadata


def filter_by_sampling_regime(observations, regime):
    """Keep only observations recorded under `regime`. Returns (kept, excluded_count).
    This is the one place sampling regimes get selected -- nothing downstream
    ever pools across sampling regimes, since everything after this operates
    on `kept`. evaluation_regime is NOT filtered here: it's the substantive
    variable under study, so it stays as a live stratification key in every
    analysis below instead (see module docstring)."""
    kept = {k: v for k, v in observations.items() if v.get("sampling_regime") == regime}
    return kept, len(observations) - len(kept)


# ---------------------------------------------------------------------------
# Dataset inventory
# ---------------------------------------------------------------------------

def count_by(items, key_fn):
    counts = defaultdict(int)
    for item in items:
        counts[key_fn(item)] += 1
    return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))


def print_counts(counts):
    for key, count in counts.items():
        print(f"  {key}: {count}")


def print_inventory(raw_rows, observations, unresolved, absorbed, missing_metadata):
    print("=== Dataset inventory (context benchmark, all sampling regimes) ===")
    print(f"Raw API attempts: {len(raw_rows)}")
    print(f"Completed observations: {len(observations)}")
    print(f"Unresolved failures: {len(unresolved)}")
    print(f"Failed attempts absorbed by a later success: {absorbed}")
    if missing_metadata:
        print(f"Completed but missing trial metadata (excluded from analysis): {len(missing_metadata)}")

    print("\nCompleted observations by type:")
    print_counts(count_by(observations.values(), lambda o: o["type"]))

    print("\nCompleted observations by model:")
    print_counts(count_by(observations.values(), lambda o: o["model"]))

    print("\nCompleted observations by sampling regime:")
    print_counts(count_by(observations.values(), lambda o: o.get("sampling_regime", "n/a")))

    print("\nCompleted observations by evaluation regime:")
    print_counts(count_by(observations.values(), lambda o: o.get("evaluation_regime", "n/a")))

    print("\nCompleted observations by dimension:")
    print_counts(count_by(observations.values(), lambda o: o.get("dimension", "n/a")))

    contrast_obs = [o for o in observations.values() if "contrast_id" in o]
    if contrast_obs:
        print("\nCompleted observations by contrast:")
        print_counts(count_by(contrast_obs, lambda o: o["contrast_id"]))

    print("\nCompleted observations by replicate:")
    print_counts(count_by(observations.values(), lambda o: o["replicate_id"]))


# ---------------------------------------------------------------------------
# Pure-Python tie-aware rank statistics. None of these ever use story ID,
# filename, insertion order, or any other arbitrary field to break a tie --
# order of the `items` argument never affects the result, only the scores/
# ranks looked up by item identity.
# ---------------------------------------------------------------------------

def average_ranks(values, items):
    """1-based average rank per item (rank 1 = highest score). Tied items
    share the mean of the rank positions their group spans."""
    ordered = sorted(items, key=lambda it: -values[it])
    ranks = {}
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        while j + 1 < n and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = avg_rank
        i = j + 1
    return ranks


def tied_groups(values, items):
    """Best-first list of tie groups: [{"rank_position", "story_ids", "score"}, ...].
    story_ids within one group are sorted alphabetically for STABLE DISPLAY
    ONLY -- every member of a group shares the same rank_position and score,
    so this ordering never affects any statistic."""
    items = list(items)
    ranks = average_ranks(values, items)
    by_score = defaultdict(list)
    for it in items:
        by_score[values[it]].append(it)
    groups = []
    for score in sorted(by_score, reverse=True):
        members = sorted(by_score[score])
        groups.append({"rank_position": ranks[members[0]], "story_ids": members, "score": score})
    return groups


def pearson_correlation(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def spearman_tie_aware(values_a, values_b, items):
    """SECONDARY rank statistic. Correct under ties: Pearson correlation of
    the two sides' average ranks (the standard tie-corrected formula) --
    never the tie-free shortcut formula."""
    items = list(items)
    if len(items) < 2:
        return None
    ranks_a = average_ranks(values_a, items)
    ranks_b = average_ranks(values_b, items)
    return pearson_correlation([ranks_a[it] for it in items], [ranks_b[it] for it in items])


def kendall_tau_b(values_a, values_b, items):
    """PRIMARY rank statistic. Kendall's tau-b: a pair tied on either side is
    excluded from the concordant/discordant count and folded into that
    side's tie term, rather than forced into an arbitrary order (tau-a would
    require that). Returns None if undefined (fewer than 2 items, or one
    side has every pair tied)."""
    items = list(items)
    n = len(items)
    if n < 2:
        return None
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_diff = values_a[items[i]] - values_a[items[j]]
            b_diff = values_b[items[i]] - values_b[items[j]]
            if a_diff == 0 and b_diff == 0:
                ties_a += 1
                ties_b += 1
            elif a_diff == 0:
                ties_a += 1
            elif b_diff == 0:
                ties_b += 1
            elif (a_diff > 0) == (b_diff > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    denom = ((n0 - ties_a) * (n0 - ties_b)) ** 0.5
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def human_reference_scores(ranking):
    """Turn the researcher's best-first ordinal reference list into
    {item: score} (higher = more preferred), so it can feed the same
    tie-aware functions above as a possibly-tied model score dict. The
    reference ordering itself has no ties -- it's an ordinal relation only
    (A > B > C...), not a cardinal preference intensity."""
    n = len(ranking)
    return {story_id: n - i for i, story_id in enumerate(ranking)}


def pairwise_diagnostics_from_scores(values, human_pairs):
    """For every known human judgment (winner, loser) with both stories
    present in `values`, classify using the model's raw scores: concordant
    (model score agrees with the human winner), discordant (disagrees), or
    model_tied (the model gave both stories the exact same score -- there is
    no strict model preference to compare, and this is never converted into
    a fabricated win/loss). Returns (concordant, discordant, model_tied, checked).
    """
    concordant = discordant = model_tied = 0
    for winner, loser in human_pairs:
        if winner not in values or loser not in values:
            continue
        if values[winner] == values[loser]:
            model_tied += 1
        elif values[winner] > values[loser]:
            concordant += 1
        else:
            discordant += 1
    return concordant, discordant, model_tied, concordant + discordant + model_tied


# ---------------------------------------------------------------------------
# Descriptive-only naturalistic-vs-invariance comparison. No new statistical
# model: a plain magnitude/sign comparison, used only to label an effect as
# attenuated/amplified/reversed/unchanged under text-only instructions.
# ---------------------------------------------------------------------------

def describe_attenuation(naturalistic_effect, invariance_effect):
    """Purely descriptive label comparing an invariance-regime effect to the
    same effect under naturalistic framing. Not a significance test."""
    if naturalistic_effect == 0:
        return "no_naturalistic_effect" if invariance_effect == 0 else "invariance_effect_only"
    same_sign = (naturalistic_effect > 0) == (invariance_effect > 0)
    if invariance_effect != 0 and not same_sign:
        return "reversed"
    ratio = abs(invariance_effect) / abs(naturalistic_effect)
    if ratio < 0.5:
        return "attenuated"
    if ratio > 1.5:
        return "amplified"
    return "unchanged"


# ---------------------------------------------------------------------------
# Pairwise: SECONDARY diagnostic (raw A/B/tie "changed") and PRIMARY
# directional effect, both operating in story identity, both stratified by
# evaluation_regime.
# ---------------------------------------------------------------------------

def choice_to_story_id(obs, category):
    """Map a raw A/B/tie choice to the actual story id that was chosen, or
    None for a tie. This is what lets every analysis below operate in story
    identity instead of raw A/B letters."""
    choice = obs["parsed_response"][category]
    if choice == "A":
        return obs["story_a_id"]
    if choice == "B":
        return obs["story_b_id"]
    return None


def is_forced_choice_pairwise(obs):
    """True for a context_pairwise/context_prompt observation from the
    PRIMARY forced-choice task. choice_mode defaults to "forced" for
    observations that predate the field (there are none in this repo's own
    data, but this keeps the check total) -- never for "tie_allowed", which
    must always be excluded from every primary forced-choice analysis below
    so the two choice modes are never silently pooled into one estimate.
    """
    return obs.get("choice_mode", "forced") == "forced"


def index_pairwise_by_assignment_fixed_position(observations, position="story1_as_a"):
    index = {}
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or obs.get("position") != position or not is_forced_choice_pairwise(obs):
            continue
        key = (obs["model"], obs["evaluation_regime"], obs["story_1_id"], obs["story_2_id"], obs["contrast_id"], obs["replicate_id"])
        index.setdefault(key, {})[obs["assignment"]] = obs
    return index


def analyze_pairwise_changed_diagnostic(observations, position="story1_as_a"):
    """SECONDARY diagnostic only -- see analyze_directional_pairwise_effects
    for the PRIMARY result. Restricted to one fixed display position (default:
    story_1 shown as Story A) so this reproduces the original "did the raw
    A/B/tie choice change between forward and flipped" comparison without
    conflating display position with context assignment (see
    context_contrasts.build_contrast_block). A bare changed=True/False does
    NOT say which story or which context was preferred -- chosen story ids
    are preserved here for that, but the primary answer is the function below.
    Stratified by evaluation_regime -- naturalistic and text_only_invariance
    cells are never compared against each other here.
    """
    index = index_pairwise_by_assignment_fixed_position(observations, position)
    rows = []
    for (model, evaluation_regime, s1, s2, contrast_id, replicate_id), pair in index.items():
        forward, flipped = pair.get("forward"), pair.get("flipped")
        if forward is None or flipped is None:
            continue
        for category in RATING_FIELDS:
            forward_choice = forward["parsed_response"][category]
            flipped_choice = flipped["parsed_response"][category]
            rows.append(
                {
                    "model": model,
                    "evaluation_regime": evaluation_regime,
                    "contrast_id": contrast_id,
                    "dimension": forward["dimension"],
                    "story_1_id": s1,
                    "story_2_id": s2,
                    "position": position,
                    "replicate_id": replicate_id,
                    "category": category,
                    "story1_context_forward": forward["context_a"]["value"],
                    "story1_context_flipped": flipped["context_a"]["value"],
                    "forward_choice": forward_choice,
                    "forward_chosen_story_id": choice_to_story_id(forward, category) or "tie",
                    "flipped_choice": flipped_choice,
                    "flipped_chosen_story_id": choice_to_story_id(flipped, category) or "tie",
                    "changed": forward_choice != flipped_choice,
                }
            )
    return rows


def summarize_grouped_change_rate(rows, group_key, label):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row["changed"])
    print(f"  By {label}:")
    for key, changes in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        rate = 100 * sum(changes) / len(changes)
        print(f"    {key}: n={len(changes)} changed={sum(changes)} ({rate:.1f}%)")


def summarize_pairwise_changed_diagnostic(rows):
    print("\n=== SECONDARY diagnostic: raw A/B/tie changed under forward/flipped (position=story1_as_a only) ===")
    print("(Boolean 'changed' only -- does not say which story/context was preferred. See the PRIMARY")
    print(" directional-effect analysis below for that.)")
    if not rows:
        print("  No matched forward/flipped pairs found.")
        return
    overall_changed = sum(1 for r in rows if r["changed"])
    print(f"  n={len(rows)} matched (category, forward/flipped) comparisons")
    print(f"  Overall: {overall_changed}/{len(rows)} ({100 * overall_changed / len(rows):.1f}%) changed")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["model"], "model")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["evaluation_regime"], "evaluation regime")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["contrast_id"], "contrast")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["category"], "category")


def analyze_directional_pairwise_effects(observations):
    """PRIMARY pairwise context-effect analysis, in story identity, pooling
    over the counterbalanced display position (see
    context_contrasts.build_contrast_block) and over replicates, stratified
    by evaluation_regime. For each
    (model, evaluation_regime, contrast_id, story_1_id, story_2_id, category),
    estimates:

        P(story_1 preferred | story_1 receives contrast value "a")
      - P(story_1 preferred | story_1 receives contrast value "b")

    Positive means story_1 is favored more often when it carries value "a";
    negative means the opposite. This is directional and never collapses to
    a single changed=True/False boolean -- two scenarios with opposite signs
    are never conflated (see offline verification item D). Naturalistic and
    text_only_invariance observations are never pooled into one estimate.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "tie": 0, "n": 0})
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        for category in RATING_FIELDS:
            chosen = choice_to_story_id(obs, category)
            key = (obs["model"], obs["evaluation_regime"], obs["contrast_id"], obs["story_1_id"], obs["story_2_id"], category, obs["assignment"])
            t = tallies[key]
            t["n"] += 1
            if chosen is None:
                t["tie"] += 1
            elif chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            else:
                t["story_2_preferred"] += 1

    by_pair = defaultdict(dict)
    for (model, evaluation_regime, contrast_id, s1, s2, category, assignment), t in tallies.items():
        by_pair[(model, evaluation_regime, contrast_id, s1, s2, category)][assignment] = t

    rows = []
    for (model, evaluation_regime, contrast_id, s1, s2, category), by_assignment in by_pair.items():
        fwd, flp = by_assignment.get("forward"), by_assignment.get("flipped")
        if fwd is None or flp is None:
            continue
        p_a = fwd["story_1_preferred"] / fwd["n"] if fwd["n"] else None
        p_b = flp["story_1_preferred"] / flp["n"] if flp["n"] else None
        effect = (p_a - p_b) if None not in (p_a, p_b) else None
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "contrast_id": contrast_id,
                "story_1_id": s1,
                "story_2_id": s2,
                "category": category,
                "p_story1_preferred_given_value_a": round(p_a, 3) if p_a is not None else "",
                "n_value_a": fwd["n"],
                "tie_n_value_a": fwd["tie"],
                "p_story1_preferred_given_value_b": round(p_b, 3) if p_b is not None else "",
                "n_value_b": flp["n"],
                "tie_n_value_b": flp["tie"],
                "directional_effect_a_minus_b": round(effect, 3) if effect is not None else "",
            }
        )
    return rows


def summarize_directional_effects_by_contrast(directional_rows):
    """Aggregate the per-story-pair directional effect across story pairs,
    per (model, evaluation_regime, contrast_id, category) -- the pairwise
    analogue of the single-text model x dimension x value aggregate.

    Reports heterogeneity alongside the mean (min/max/range across the
    underlying per-pair effects), not the mean alone -- with only 12
    stories/66 pairs, a single mean can obscure whether every pair actually
    behaves similarly (see analyze_per_story_context_effects and
    analyze_leave_one_story_out for finer-grained heterogeneity diagnostics).
    """
    grouped = defaultdict(list)
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        grouped[key].append(row["directional_effect_a_minus_b"])
    return [
        {
            "model": m,
            "evaluation_regime": er,
            "contrast_id": c,
            "category": cat,
            "mean_directional_effect": round(statistics.mean(effects), 3),
            "min_directional_effect": round(min(effects), 3),
            "max_directional_effect": round(max(effects), 3),
            "range_directional_effect": round(max(effects) - min(effects), 3),
            "n_story_pairs": len(effects),
        }
        for (m, er, c, cat), effects in grouped.items()
    ]


def summarize_directional_pairwise_effects(rows, by_contrast_rows):
    print("\n=== PRIMARY pairwise context-effect analysis: directional, in story identity ===")
    print('(P(story_1 preferred | value "a") - P(story_1 preferred | value "b"), position counterbalanced,')
    print(" replicates pooled into the estimate, not discarded; stratified by evaluation_regime)")
    if not rows:
        print("  No complete forward+flipped story-pair blocks found.")
        return
    print(f"  n={len(rows)} (model, evaluation_regime, contrast, story pair, category) directional estimates")
    print("\n  By (model, evaluation_regime, contrast, category), averaged across story pairs:")
    for row in sorted(by_contrast_rows, key=lambda r: (r["model"], r["evaluation_regime"], r["contrast_id"], r["category"])):
        print(
            f"    {row['model']} | {row['evaluation_regime']} | {row['contrast_id']} | {row['category']}: "
            f"mean_effect={row['mean_directional_effect']:+.3f} "
            f"range=[{row['min_directional_effect']:+.3f}, {row['max_directional_effect']:+.3f}] "
            f"(n_story_pairs={row['n_story_pairs']})"
        )


def analyze_pairwise_cell_rates(observations):
    """Expose the four raw counterbalanced cells separately, never collapsed
    into the pooled directional estimate above.

    For each (model, evaluation_regime, contrast_id, story_1_id, story_2_id,
    category) block, reports story_1_wins/story_2_wins/n for each of the 4
    cells (forward x story1_as_a, forward x story2_as_a, flipped x
    story1_as_a, flipped x story2_as_a) -- "story2_as_a" means story_1 is
    displayed as Story B. Restricted to forced-choice observations (see
    is_forced_choice_pairwise), since ties make a "wins" count ambiguous.

    This is the shared basis for both the raw four-cell-rates output (an
    aggregate CSV is not the only way to see pairwise results) and the
    position/context-x-position analyses below, which are derived from
    exactly these four numbers per block.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "n": 0})
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        for category in RATING_FIELDS:
            chosen = choice_to_story_id(obs, category)
            key = (
                obs["model"], obs["evaluation_regime"], obs["contrast_id"],
                obs["story_1_id"], obs["story_2_id"], category, obs["assignment"], obs["position"],
            )
            t = tallies[key]
            t["n"] += 1
            if chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            elif chosen == obs["story_2_id"]:
                t["story_2_preferred"] += 1

    by_block = defaultdict(dict)
    for (model, evaluation_regime, contrast_id, s1, s2, category, assignment, position), t in tallies.items():
        by_block[(model, evaluation_regime, contrast_id, s1, s2, category)][(assignment, position)] = t

    cell_labels = [
        ("forward", "story1_as_a"), ("forward", "story2_as_a"),
        ("flipped", "story1_as_a"), ("flipped", "story2_as_a"),
    ]
    rows = []
    for (model, evaluation_regime, contrast_id, s1, s2, category), cells in by_block.items():
        if not all(label in cells for label in cell_labels):
            continue  # incomplete block (e.g. a partial/failed run); skip rather than guess
        row = {"model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id,
               "story_1_id": s1, "story_2_id": s2, "category": category}
        for assignment, position in cell_labels:
            t = cells[(assignment, position)]
            row[f"{assignment}_{position}_story_1_wins"] = t["story_1_preferred"]
            row[f"{assignment}_{position}_story_2_wins"] = t["story_2_preferred"]
            row[f"{assignment}_{position}_n"] = t["n"]
        rows.append(row)
    return rows


def analyze_position_and_interaction_effects(cell_rows):
    """Decompose the pooled directional context effect into a position
    effect and a context x position interaction diagnostic, from the four
    raw cells (analyze_pairwise_cell_rates) -- so a pooled context effect
    can never silently hide a large or context-dependent position effect.

    For each block, using p(cell) = story_1_wins / n:
      - context_effect_at_position_a  = p(forward, story1_as_a) - p(flipped, story1_as_a)
      - context_effect_at_position_b  = p(forward, story2_as_a) - p(flipped, story2_as_a)
      - context_x_position_interaction = context_effect_at_position_a - context_effect_at_position_b
      - position_effect_pooled = p(story_1 chosen | displayed as A) - p(story_1 chosen | displayed as B),
          pooling story_1_wins/n across assignment within each position
      - position_effect_under_forward  = p(forward, story1_as_a) - p(forward, story2_as_a)
      - position_effect_under_flipped  = p(flipped, story1_as_a) - p(flipped, story2_as_a)

    This is a descriptive decomposition of one 2x2 (assignment x position)
    table per block, not a new inferential/hierarchical model.
    """
    rows = []
    for row in cell_rows:
        def p(assignment, position):
            n = row[f"{assignment}_{position}_n"]
            return row[f"{assignment}_{position}_story_1_wins"] / n if n else None

        p_fwd_a, p_fwd_b = p("forward", "story1_as_a"), p("forward", "story2_as_a")
        p_flp_a, p_flp_b = p("flipped", "story1_as_a"), p("flipped", "story2_as_a")
        if None in (p_fwd_a, p_fwd_b, p_flp_a, p_flp_b):
            continue

        n_a = row["forward_story1_as_a_n"] + row["flipped_story1_as_a_n"]
        n_b = row["forward_story2_as_a_n"] + row["flipped_story2_as_a_n"]
        wins_a = row["forward_story1_as_a_story_1_wins"] + row["flipped_story1_as_a_story_1_wins"]
        wins_b = row["forward_story2_as_a_story_1_wins"] + row["flipped_story2_as_a_story_1_wins"]
        p_position_a = wins_a / n_a if n_a else None
        p_position_b = wins_b / n_b if n_b else None

        context_effect_at_a = p_fwd_a - p_flp_a
        context_effect_at_b = p_fwd_b - p_flp_b
        rows.append(
            {
                "model": row["model"],
                "evaluation_regime": row["evaluation_regime"],
                "contrast_id": row["contrast_id"],
                "story_1_id": row["story_1_id"],
                "story_2_id": row["story_2_id"],
                "category": row["category"],
                "context_effect_at_position_a": round(context_effect_at_a, 3),
                "context_effect_at_position_b": round(context_effect_at_b, 3),
                "context_x_position_interaction": round(context_effect_at_a - context_effect_at_b, 3),
                "position_effect_pooled": round(p_position_a - p_position_b, 3) if None not in (p_position_a, p_position_b) else "",
                "position_effect_under_forward": round(p_fwd_a - p_fwd_b, 3),
                "position_effect_under_flipped": round(p_flp_a - p_flp_b, 3),
            }
        )
    return rows


def summarize_position_and_interaction_effects(rows):
    print("\n=== PRIMARY: A/B display-position effect and context x position interaction ===")
    print("(a pooled context effect can hide a large or context-dependent position effect; this decomposes")
    print(" one 2x2 assignment x position table per block into both -- descriptive, not a new inferential model)")
    if not rows:
        print("  No complete 4-cell blocks found.")
        return
    print(f"  n={len(rows)} (model, evaluation_regime, contrast, story pair, category) blocks")
    interactions = [r["context_x_position_interaction"] for r in rows]
    positions = [r["position_effect_pooled"] for r in rows if r["position_effect_pooled"] != ""]
    print(f"  context x position interaction: mean={statistics.mean(interactions):+.3f}  "
          f"range=[{min(interactions):+.3f}, {max(interactions):+.3f}]")
    if positions:
        print(f"  position effect (pooled over assignment): mean={statistics.mean(positions):+.3f}  "
              f"range=[{min(positions):+.3f}, {max(positions):+.3f}]")


def analyze_per_story_context_effects(directional_rows):
    """Per-story summary of the directional context effect across a story's
    opponents, preserving every underlying per-pair value.

    66 unordered story pairs are NOT 66 independent samples: each of the 12
    stories appears in 11 of them, so one unusual story can create the
    appearance of a repeated effect across many pairs. This reorganizes the
    existing per-pair directional_effect_a_minus_b (analyze_directional_pairwise_effects)
    by story identity so that question is directly answerable: does a
    context value generally help whichever story holds it, or is one
    specific story (e.g. it responds unusually across most/all of its 11
    opponents) driving an apparently repeated effect? The per-pair effect is
    identical from either story's point of view in a two-outcome forced
    choice (holding value "a" either helps or doesn't, symmetrically for
    whichever story holds it) -- so grouping by story here reveals whether
    that "holding value a helps" pattern is uniform across a story's
    opponents or concentrated/absent for a particular one.
    """
    grouped = defaultdict(list)
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        base_key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        grouped[(*base_key, row["story_1_id"])].append((row["story_2_id"], row["directional_effect_a_minus_b"]))
        grouped[(*base_key, row["story_2_id"])].append((row["story_1_id"], row["directional_effect_a_minus_b"]))

    rows = []
    for (model, evaluation_regime, contrast_id, category, story_id), pair_effects in grouped.items():
        effects = [e for _, e in pair_effects]
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "contrast_id": contrast_id,
                "category": category,
                "story_id": story_id,
                "n_opponents": len(pair_effects),
                "mean_effect": round(statistics.mean(effects), 3),
                "min_effect": round(min(effects), 3),
                "max_effect": round(max(effects), 3),
                "range_effect": round(max(effects) - min(effects), 3),
                "per_opponent_effects": "; ".join(f"{opp}={e:+.3f}" for opp, e in sorted(pair_effects)),
            }
        )
    return rows


def summarize_per_story_context_effects(rows):
    print("\n=== Per-story context-effect summary (heterogeneity across a story's opponents) ===")
    print("(66 pairs are not 66 independent units -- each of 12 stories appears in 11 pairs; this checks")
    print(" whether the aggregate effect is uniform across stories or concentrated in one unusual story)")
    if not rows:
        print("  No per-story effects available.")
        return
    by_widest_range = sorted(rows, key=lambda r: -r["range_effect"])[:5]
    print("  Widest within-story range across opponents (top 5, largest heterogeneity first):")
    for row in by_widest_range:
        print(
            f"    {row['model']} | {row['evaluation_regime']} | {row['contrast_id']} | {row['category']} | "
            f"{row['story_id']}: mean={row['mean_effect']:+.3f} range=[{row['min_effect']:+.3f}, {row['max_effect']:+.3f}] "
            f"(n_opponents={row['n_opponents']})"
        )


def analyze_leave_one_story_out(directional_rows):
    """Sensitivity/robustness diagnostic: recompute each aggregate context
    effect after excluding every pair involving each story in turn.

    Descriptive only -- NOT a formal correction for the non-independence of
    story pairs (see analyze_per_story_context_effects docstring). Answers:
    is the headline aggregate (mean directional effect across all pairs for
    a model/evaluation_regime/contrast/category) carried largely by one
    unusual story, or does it hold up under removing any single story?
    """
    grouped = defaultdict(list)
    all_stories = defaultdict(set)
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        grouped[key].append(row)
        all_stories[key].add(row["story_1_id"])
        all_stories[key].add(row["story_2_id"])

    rows = []
    for key, pair_rows in grouped.items():
        model, evaluation_regime, contrast_id, category = key
        full_mean = statistics.mean(r["directional_effect_a_minus_b"] for r in pair_rows)
        rows.append(
            {
                "model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id, "category": category,
                "excluded_story_id": "(none -- full aggregate)",
                "mean_directional_effect": round(full_mean, 3), "n_story_pairs": len(pair_rows),
            }
        )
        for excluded in sorted(all_stories[key]):
            remaining = [r for r in pair_rows if r["story_1_id"] != excluded and r["story_2_id"] != excluded]
            if not remaining:
                continue
            rows.append(
                {
                    "model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id, "category": category,
                    "excluded_story_id": excluded,
                    "mean_directional_effect": round(statistics.mean(r["directional_effect_a_minus_b"] for r in remaining), 3),
                    "n_story_pairs": len(remaining),
                }
            )
    return rows


def summarize_leave_one_story_out(rows):
    print("\n=== Leave-one-story-out sensitivity (descriptive robustness check, not a formal correction) ===")
    if not rows:
        print("  No leave-one-story-out results available.")
        return
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        by_key[key][row["excluded_story_id"]] = row["mean_directional_effect"]
    for key, by_excluded in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        full = by_excluded.get("(none -- full aggregate)")
        loo_values = [v for k, v in by_excluded.items() if k != "(none -- full aggregate)"]
        if full is None or not loo_values:
            continue
        print(
            f"  {key[0]} | {key[1]} | {key[2]} | {key[3]}: full_aggregate={full:+.3f}  "
            f"leave-one-out range=[{min(loo_values):+.3f}, {max(loo_values):+.3f}]"
        )


def compare_pairwise_regimes(directional_by_contrast_rows):
    """For each (model, contrast_id, category) present under BOTH evaluation
    regimes, report the naturalistic effect, the text_only_invariance
    effect, and a descriptive attenuation label. Rows missing one regime are
    skipped (nothing to compare)."""
    by_key = defaultdict(dict)
    for row in directional_by_contrast_rows:
        key = (row["model"], row["contrast_id"], row["category"])
        by_key[key][row["evaluation_regime"]] = row["mean_directional_effect"]

    rows = []
    for (model, contrast_id, category), by_regime in by_key.items():
        if "naturalistic" not in by_regime or "text_only_invariance" not in by_regime:
            continue
        nat, inv = by_regime["naturalistic"], by_regime["text_only_invariance"]
        rows.append(
            {
                "model": model,
                "contrast_id": contrast_id,
                "category": category,
                "effect_naturalistic": nat,
                "effect_text_only_invariance": inv,
                "attenuation": describe_attenuation(nat, inv),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# SECONDARY, optional hedging/indifference diagnostic: choice_mode="tie_allowed"
# observations only (see context_trials.build_tie_allowed_pairwise_trials).
# Never pooled with the PRIMARY forced-choice observations above -- this
# section only ever reads obs with choice_mode == "tie_allowed", the primary
# analyses above only ever read is_forced_choice_pairwise(obs). This
# measures how often the model declines to state a strict preference, and
# whether context shifts that tendency -- deliberately called "tie rate" /
# "hedging rate", never "uncertainty" in a strong psychological sense.
# ---------------------------------------------------------------------------

def analyze_tie_allowed_diagnostic(observations):
    """P(story_1 chosen) / P(story_2 chosen) / P(tie), in story identity, per
    (model, evaluation_regime, contrast_id, story_1_id, story_2_id, category,
    assignment) -- the tie-allowed analogue of analyze_pairwise_cell_rates,
    but reporting rates (incl. tie) rather than forced win counts, since ties
    are a real, informative outcome here rather than an excluded case.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "tie": 0, "n": 0})
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or obs.get("choice_mode") != "tie_allowed":
            continue
        for category in RATING_FIELDS:
            chosen = choice_to_story_id(obs, category)
            key = (obs["model"], obs["evaluation_regime"], obs["contrast_id"],
                   obs["story_1_id"], obs["story_2_id"], category, obs["assignment"])
            t = tallies[key]
            t["n"] += 1
            if chosen is None:
                t["tie"] += 1
            elif chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            else:
                t["story_2_preferred"] += 1

    rows = []
    for (model, evaluation_regime, contrast_id, s1, s2, category, assignment), t in tallies.items():
        n = t["n"]
        rows.append(
            {
                "model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id,
                "story_1_id": s1, "story_2_id": s2, "category": category, "assignment": assignment,
                "p_story_1_chosen": round(t["story_1_preferred"] / n, 3) if n else "",
                "p_story_2_chosen": round(t["story_2_preferred"] / n, 3) if n else "",
                "p_tie": round(t["tie"] / n, 3) if n else "",
                "n": n,
            }
        )
    return rows


def summarize_tie_allowed_diagnostic(rows):
    print("\n=== SECONDARY, optional: tie-allowed hedging/indifference diagnostic ===")
    print("(choice_mode=\"tie_allowed\" observations only, never pooled with the primary forced-choice results;")
    print(" reports how often the model declines a strict preference, and whether context shifts that rate --")
    print(" \"tie rate\"/\"hedging rate\", not a claim about a psychological state of uncertainty)")
    if not rows:
        print("  No tie-allowed observations found (this diagnostic is optional and not run by default).")
        return
    tie_rates = [r["p_tie"] for r in rows if r["p_tie"] != ""]
    by_assignment = defaultdict(list)
    for r in rows:
        if r["p_tie"] != "":
            by_assignment[r["assignment"]].append(r["p_tie"])
    print(f"  n={len(rows)} (model, evaluation_regime, contrast, story pair, category, assignment) cells")
    print(f"  Overall tie rate: mean={statistics.mean(tie_rates):.3f}  range=[{min(tie_rates):.3f}, {max(tie_rates):.3f}]")
    for assignment, rates in sorted(by_assignment.items()):
        print(f"    {assignment}: mean tie rate={statistics.mean(rates):.3f} (n={len(rates)})")


# ---------------------------------------------------------------------------
# context_prompt: does a genuinely extraneous prompt-level sentence change
# the decision, holding story identity/position/story-level-context fixed?
# (Prompt-scope context is not attributed to either story, so the
# assignment x position counterbalance above doesn't apply here.) Stratified
# by evaluation_regime like everything else.
# ---------------------------------------------------------------------------

def index_prompt_by_value(observations):
    index = {}
    for obs in observations.values():
        if obs["type"] != "context_prompt":
            continue
        key = (obs["model"], obs["evaluation_regime"], obs["story_a_id"], obs["story_b_id"], obs["contrast_id"], obs["replicate_id"])
        index.setdefault(key, {})[obs["value"]] = obs
    return index


def analyze_prompt_context_effects(observations):
    index = index_prompt_by_value(observations)
    rows = []

    for (model, evaluation_regime, story_a, story_b, contrast_id, replicate_id), values in index.items():
        if len(values) < 2:
            continue
        baseline_id = next((v for v, obs in values.items() if not obs.get("prompt_context_text")), sorted(values)[0])
        baseline_obs = values[baseline_id]
        for value_id, obs in values.items():
            if value_id == baseline_id:
                continue
            for category in RATING_FIELDS:
                baseline_choice = baseline_obs["parsed_response"][category]
                treatment_choice = obs["parsed_response"][category]
                rows.append(
                    {
                        "model": model,
                        "evaluation_regime": evaluation_regime,
                        "contrast_id": contrast_id,
                        "story_a_id": story_a,
                        "story_b_id": story_b,
                        "replicate_id": replicate_id,
                        "category": category,
                        "baseline_value": baseline_id,
                        "treatment_value": value_id,
                        "baseline_choice": baseline_choice,
                        "treatment_choice": treatment_choice,
                        "changed": baseline_choice != treatment_choice,
                    }
                )
    return rows


def summarize_prompt_context_effects(rows):
    print("\n=== Context effect from extraneous prompt-level sentences (baseline vs treatment) ===")
    if not rows:
        print("  No matched baseline/treatment prompt-context pairs found.")
        return

    overall_changed = sum(1 for r in rows if r["changed"])
    print(f"  n={len(rows)} matched (category, baseline/treatment) comparisons")
    print(f"  Overall: {overall_changed}/{len(rows)} ({100 * overall_changed / len(rows):.1f}%) changed")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["model"], "model")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["evaluation_regime"], "evaluation regime")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["contrast_id"], "contrast")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["category"], "category")


# ---------------------------------------------------------------------------
# SECONDARY: pairwise agreement with the researcher reference. DIRECT
# comparison in story identity, no derived ranking involved -- for every
# context_pairwise observation whose two displayed stories exactly match a
# judgment the researcher made, does the model's choice agree? A secondary,
# personalized comparison, not evidence about which context is "better" --
# stratified by evaluation_regime, so the more interesting reading is
# whether contextual perturbation makes this agreement more or less robust,
# not "which context wins" (see FINAL_DESIGN.md).
# ---------------------------------------------------------------------------

def analyze_pairwise_vs_human_reference(observations, human_pairs, category="overall_quality"):
    """Restricted to `category` (default overall_quality, the closest
    analogue to a single preference judgment). Reports concordant/
    discordant/model_tied counts -- never converts a model tie into a
    fabricated win or loss. Stratified by (model, evaluation_regime).
    """
    human_winner_by_pair = {frozenset((w, l)): w for w, l in human_pairs}
    rows = []
    tally = defaultdict(lambda: {"concordant": 0, "discordant": 0, "model_tied": 0})

    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        pair_key = frozenset((obs["story_a_id"], obs["story_b_id"]))
        human_winner = human_winner_by_pair.get(pair_key)
        if human_winner is None:
            continue
        chosen = choice_to_story_id(obs, category)
        if chosen is None:
            outcome = "model_tied"
        elif chosen == human_winner:
            outcome = "concordant"
        else:
            outcome = "discordant"
        tally[(obs["model"], obs["evaluation_regime"])][outcome] += 1
        rows.append(
            {
                "model": obs["model"],
                "evaluation_regime": obs["evaluation_regime"],
                "story_a_id": obs["story_a_id"],
                "story_b_id": obs["story_b_id"],
                "contrast_id": obs["contrast_id"],
                "assignment": obs["assignment"],
                "position": obs["position"],
                "replicate_id": obs["replicate_id"],
                "human_winner": human_winner,
                "model_choice": obs["parsed_response"][category],
                "model_chosen_story_id": chosen or "tie",
                "outcome": outcome,
            }
        )

    summary_rows = []
    for (model, evaluation_regime), counts in tally.items():
        n = counts["concordant"] + counts["discordant"] + counts["model_tied"]
        decided = counts["concordant"] + counts["discordant"]
        summary_rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "concordant": counts["concordant"],
                "discordant": counts["discordant"],
                "model_tied": counts["model_tied"],
                "n": n,
                "concordant_rate": round(counts["concordant"] / decided, 3) if decided else "",
            }
        )
    return rows, summary_rows


def summarize_pairwise_vs_human_reference(summary_rows):
    print("\n=== SECONDARY: direct model choices vs the researcher's pairwise judgments (no derived ranking) ===")
    print("(restricted to overall_quality, and to story pairs the researcher judged; personalized, ordinal, exploratory)")
    if not summary_rows:
        print("  No context_pairwise observations matched a researcher judgment.")
        return
    for row in sorted(summary_rows, key=lambda r: (r["model"], r["evaluation_regime"])):
        rate_text = f"{row['concordant_rate']:.3f}" if row["concordant_rate"] != "" else "unavailable"
        print(
            f"  {row['model']} | {row['evaluation_regime']}: concordant={row['concordant']} discordant={row['discordant']} "
            f"model_tied={row['model_tied']} (n={row['n']}, concordant_rate={rate_text})"
        )


# ---------------------------------------------------------------------------
# PRIMARY: single-text treatment vs neutral baseline deltas, stratified by
# evaluation_regime -- a naturalistic treatment observation is only ever
# compared against the naturalistic neutral baseline for that story, never
# against the text_only_invariance baseline. This is this project's main
# question (context sensitivity), not the researcher-reference comparisons
# further below, which are secondary.
# ---------------------------------------------------------------------------

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
    one -- see print_naturalistic_vs_invariance_comparison -- so both
    regime comparisons appear at the same point in the output, per the
    documented scientific ordering (see module docstring)."""
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


# ---------------------------------------------------------------------------
# SECONDARY: single-text rankings (tie-aware) and agreement with the
# researcher reference ordering. This is a personalized, ordinal-only
# reference-agreement analysis -- not the benchmark's organizing goal, and
# not evidence that any one condition is objectively "better" (see
# FINAL_DESIGN.md's "Researcher reference ranking" and its multiple-
# comparisons caution). Scores are always kept as {story_id: (mean, n)}
# dicts -- never pre-sorted with an arbitrary tie-break -- until
# presentation time, where tied_groups() reports rank_position/tie_group_size
# without breaking ties. Stratified by evaluation_regime throughout.
# ---------------------------------------------------------------------------

def rank_from_single_text_by_condition(observations):
    """SECONDARY reference-agreement ranking: one score dict per
    (model, evaluation_regime, dimension, value), including the neutral
    baseline as its own condition. Returns
    {(model, evaluation_regime, dimension, value): {story_id: (mean_score, n)}}.
    Feeds into the researcher-reference agreement statistics below -- this
    is a secondary, personalized analysis, not the benchmark's primary
    context-sensitivity question (see analyze_treatment_vs_neutral for that).
    """
    scores = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single":
            continue
        key = (obs["model"], obs["evaluation_regime"], obs["dimension"], obs["value"])
        scores[key][obs["story_id"]].append(obs["parsed_response"]["overall_quality"])
    return {
        key: {story_id: (statistics.mean(vals), len(vals)) for story_id, vals in by_story.items()}
        for key, by_story in scores.items()
    }


def rank_from_single_text(observations):
    """DIAGNOSTIC ONLY: per (model, evaluation_regime), pooling every
    context_single condition together (still never pooling across
    evaluation_regime). Returns {(model, evaluation_regime): {story_id: (mean_score, n)}}."""
    scores = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single":
            continue
        scores[(obs["model"], obs["evaluation_regime"])][obs["story_id"]].append(obs["parsed_response"]["overall_quality"])
    return {
        key: {story_id: (statistics.mean(vals), len(vals)) for story_id, vals in by_story.items()}
        for key, by_story in scores.items()
    }


def compute_pairwise_wins(observations):
    """DIAGNOSTIC ONLY: tally wins/losses/ties per story per
    (model, evaluation_regime), pooling ALL context_pairwise observations
    regardless of contrast/assignment/position (but never across
    evaluation_regime). See rank_from_pairwise_wins for why this isn't a
    per-condition ranking."""
    records = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        key = (obs["model"], obs["evaluation_regime"])
        a, b = obs["story_a_id"], obs["story_b_id"]
        choice = obs["parsed_response"]["overall_quality"]
        if choice == "A":
            records[key][a]["wins"] += 1
            records[key][b]["losses"] += 1
        elif choice == "B":
            records[key][b]["wins"] += 1
            records[key][a]["losses"] += 1
        else:
            records[key][a]["ties"] += 1
            records[key][b]["ties"] += 1
    return records


def rank_from_pairwise_wins(tally_by_story):
    """DIAGNOSTIC ONLY. Returns {story_id: (copeland, win_rate, games)} --
    Copeland score (wins - losses) is the scalar used for ranking/rank
    statistics; win_rate/games are informational only. Pools every
    context_pairwise observation regardless of context condition, so this
    can't be read as a ranking "under" any particular context -- see
    analyze_directional_pairwise_effects for the primary pairwise result.
    """
    return {
        story_id: (tally["wins"] - tally["losses"], (tally["wins"] + 0.5 * tally["ties"]) / max(1, tally["wins"] + tally["losses"] + tally["ties"]), tally["wins"] + tally["losses"] + tally["ties"])
        for story_id, tally in tally_by_story.items()
    }


def load_human_reference(path=HUMAN_REFERENCE_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)["ranking"]


def load_human_pairwise(path=HUMAN_PAIRWISE_FILE):
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                pairs.append((row["winner"], row["loser"]))
    return pairs


def human_comparison_row(model, evaluation_regime, dimension, value, ranking_source, scores_dict, human_reference, human_pairs):
    """scores_dict: {story_id: (mean_score, n)} or {story_id: (scalar, ..., ...)}
    where the FIRST tuple element is always the scalar used for ranking."""
    values = {sid: v[0] for sid, v in scores_dict.items()}
    items = list(values.keys())

    tau = spearman = None
    n_common = 0
    if human_reference is not None:
        common = [it for it in items if it in human_reference]
        n_common = len(common)
        if n_common >= 2:
            human_scores = human_reference_scores(human_reference)
            tau = kendall_tau_b(values, human_scores, common)
            spearman = spearman_tie_aware(values, human_scores, common)

    concordant, discordant, model_tied, checked = pairwise_diagnostics_from_scores(values, human_pairs)
    decided = concordant + discordant
    return {
        "model": model,
        "evaluation_regime": evaluation_regime,
        "dimension": dimension,
        "value": value,
        "ranking_source": ranking_source,
        "n_common_with_human_reference": n_common,
        "kendall_tau_b": round(tau, 3) if tau is not None else "",
        "spearman_vs_human": round(spearman, 3) if spearman is not None else "",
        "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "pairwise_model_tied": model_tied,
        "pairwise_concordant_rate": round(concordant / decided, 3) if decided else "",
        "pairwise_checked_n": checked,
    }


def ranking_csv_rows(scores_with_n, extra_fields, score_field_name="mean_overall_quality"):
    """scores_with_n: {story_id: (score, n)}. rank_position/tie_group_size
    come from tied_groups() -- ties share the identical rank_position, never
    broken by story ID or any other arbitrary field. Row order (sorted by
    story_id) is for stable display only."""
    values = {sid: v[0] for sid, v in scores_with_n.items()}
    items = list(scores_with_n.keys())
    rank_by_story, group_size_by_story = {}, {}
    for g in tied_groups(values, items):
        for sid in g["story_ids"]:
            rank_by_story[sid] = g["rank_position"]
            group_size_by_story[sid] = len(g["story_ids"])
    rows = []
    for sid in sorted(items):
        score, n = scores_with_n[sid]
        rows.append(
            {
                **extra_fields,
                "story_id": sid,
                score_field_name: round(score, 3),
                "n": n,
                "rank_position": rank_by_story[sid],
                "tie_group_size": group_size_by_story[sid],
            }
        )
    return rows


def print_ranking_with_ties(scores_with_n, indent="    "):
    values = {sid: v[0] for sid, v in scores_with_n.items()}
    for g in tied_groups(values, list(scores_with_n.keys())):
        ids = ", ".join(g["story_ids"])
        tie_note = f"  [tied group of {len(g['story_ids'])}]" if len(g["story_ids"]) > 1 else ""
        print(f"{indent}rank {g['rank_position']}: {ids}  (score={g['score']:.2f}){tie_note}")


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
