"""Offline analysis for the v0.2 context benchmark (context_trials.py results).

Reads a results file (default results/context_raw.jsonl) produced by
run_trial.py/run_batch.py against data/context_trials.jsonl, collapses retry
attempts, and runs analyses specific to the new trial types (context_single,
context_pairwise, context_prompt). Completely separate from analyze.py, which
keeps analyzing the v0.1 pilot pipeline unchanged.

Run this file directly: python3 analyze_context.py
No API calls are made, and ANTHROPIC_API_KEY is never read.
"""

import json
import os
import statistics
from collections import defaultdict

from analyze import load_jsonl, is_successful, attempt_number, write_csv

RESULTS_FILE = "results/context_raw.jsonl"
TRIALS_FILE = "data/context_trials.jsonl"
HUMAN_REFERENCE_FILE = "data/human_reference.json"
HUMAN_PAIRWISE_FILE = "data/human_pairwise.jsonl"
ANALYSIS_DIR = "results/context_analysis"

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
# (trial_id, model, replicate_id), keep the most recent success)
# ---------------------------------------------------------------------------

def collapse_attempts(raw_rows, trials_by_id):
    """Returns (observations, unresolved, absorbed_failed_attempts, missing_metadata).

    missing_metadata: keys with a successful attempt but no way to recover
    trial metadata (neither trial_meta on the row nor a matching trials-file
    entry) -- these are reported, not silently dropped or guessed at.
    """
    attempts_by_key = defaultdict(list)
    for row in raw_rows:
        key = (row["trial_id"], row["model"], row["replicate_id"])
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

        trial_id, model, replicate_id = key
        observations[key] = {
            **meta,
            "model": model,
            "replicate_id": replicate_id,
            "attempt_id": attempt_number(row),
            "parsed_response": row["parsed_response"],
        }
        absorbed_failed_attempts += len(attempts) - 1

    return observations, unresolved, absorbed_failed_attempts, missing_metadata


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
    print("=== Dataset inventory (context benchmark) ===")
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

    print("\nCompleted observations by dimension:")
    print_counts(count_by(observations.values(), lambda o: o.get("dimension", "n/a")))

    contrast_obs = [o for o in observations.values() if "contrast_id" in o]
    if contrast_obs:
        print("\nCompleted observations by contrast:")
        print_counts(count_by(contrast_obs, lambda o: o["contrast_id"]))

    print("\nCompleted observations by replicate:")
    print_counts(count_by(observations.values(), lambda o: o["replicate_id"]))


# ---------------------------------------------------------------------------
# D. Context effect on pairwise A/B/tie decisions (forward vs flipped)
# ---------------------------------------------------------------------------

def index_pairwise_by_assignment(observations):
    """Group context_pairwise observations by everything except "assignment"."""
    index = {}
    for obs in observations.values():
        if obs["type"] != "context_pairwise":
            continue
        key = (obs["model"], obs["story_a_id"], obs["story_b_id"], obs["contrast_id"], obs["replicate_id"])
        index.setdefault(key, {})[obs["assignment"]] = obs
    return index


def analyze_pairwise_context_effects(observations):
    """For each matched forward/flipped pair, per category: did the A/B/tie
    choice change even though the underlying texts and A/B positions were
    fixed? A change is evidence of a context effect; no change is consistent
    with a story-quality-driven judgment (see context_contrasts.py)."""
    index = index_pairwise_by_assignment(observations)
    rows = []

    for (model, story_a, story_b, contrast_id, replicate_id), pair in index.items():
        forward, flipped = pair.get("forward"), pair.get("flipped")
        if forward is None or flipped is None:
            continue
        for category in RATING_FIELDS:
            forward_choice = forward["parsed_response"][category]
            flipped_choice = flipped["parsed_response"][category]
            rows.append(
                {
                    "model": model,
                    "contrast_id": contrast_id,
                    "dimension": forward["dimension"],
                    "story_a_id": story_a,
                    "story_b_id": story_b,
                    "replicate_id": replicate_id,
                    "category": category,
                    "forward_choice": forward_choice,
                    "flipped_choice": flipped_choice,
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


def summarize_pairwise_context_effects(rows):
    print("\n=== Context effect on pairwise A/B/tie decisions (forward vs flipped) ===")
    if not rows:
        print("  No matched forward/flipped pairs found.")
        return

    overall_changed = sum(1 for r in rows if r["changed"])
    print(f"  n={len(rows)} matched (category, forward/flipped) comparisons")
    print(f"  Overall: {overall_changed}/{len(rows)} ({100 * overall_changed / len(rows):.1f}%) changed when context was swapped")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["model"], "model")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["contrast_id"], "contrast")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["category"], "category")
    print()
    summarize_grouped_change_rate(
        rows, lambda r: " vs ".join(sorted([r["story_a_id"], r["story_b_id"]])), "story pair"
    )
    print()
    summarize_grouped_change_rate(rows, lambda r: r["replicate_id"], "replicate")


# ---------------------------------------------------------------------------
# context_prompt: does a genuinely extraneous prompt-level sentence change
# the decision, holding story identity/position/story-level-context fixed?
# ---------------------------------------------------------------------------

def index_prompt_by_value(observations):
    index = {}
    for obs in observations.values():
        if obs["type"] != "context_prompt":
            continue
        key = (obs["model"], obs["story_a_id"], obs["story_b_id"], obs["contrast_id"], obs["replicate_id"])
        index.setdefault(key, {})[obs["value"]] = obs
    return index


def analyze_prompt_context_effects(observations):
    index = index_prompt_by_value(observations)
    rows = []

    for (model, story_a, story_b, contrast_id, replicate_id), values in index.items():
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
    summarize_grouped_change_rate(rows, lambda r: r["contrast_id"], "contrast")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["category"], "category")


# ---------------------------------------------------------------------------
# E. Provisional ranking from pairwise overall_quality judgments
# ---------------------------------------------------------------------------

def compute_pairwise_wins(observations):
    """Tally wins/losses/ties per story per model, pooling ALL context_pairwise
    trials' overall_quality choice regardless of which context condition was
    applied. This deliberately treats context as noise to average over, to
    get one baseline "how does this model rank the corpus" estimate per model.
    DIAGNOSTIC ONLY -- see rank_from_pairwise_wins for why this can't be read
    as a ranking "under" any particular context condition.
    """
    records = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))
    for obs in observations.values():
        if obs["type"] != "context_pairwise":
            continue
        model = obs["model"]
        a, b = obs["story_a_id"], obs["story_b_id"]
        choice = obs["parsed_response"]["overall_quality"]
        if choice == "A":
            records[model][a]["wins"] += 1
            records[model][b]["losses"] += 1
        elif choice == "B":
            records[model][b]["wins"] += 1
            records[model][a]["losses"] += 1
        else:
            records[model][a]["ties"] += 1
            records[model][b]["ties"] += 1
    return records


def rank_from_pairwise_wins(tally_by_story):
    """Copeland score (wins - losses) as the primary sort key, win_rate as a
    tiebreaker. This is ONE simple, transparent ranking estimator -- not the
    only possible method (e.g. it ignores strength of opponent).

    DIAGNOSTIC ONLY, not a per-condition ranking: it pools every
    context_pairwise observation regardless of contrast/assignment. The
    forward/flipped contrast trials are built to measure *causal context
    sensitivity* (see analyze_pairwise_context_effects above) -- each
    observation only ever pits two *different* claimed context values against
    each other, so there is no single context condition this pooled ranking
    can honestly be said to be "under". Do not present it as a per-condition
    human-alignment result; use it only as a rough sanity-check baseline. A
    true per-condition pairwise ranking would need the OPTIONAL same-context
    trial family (context_trials.build_context_pairwise_same_trials), which
    is not part of any required run -- see FINAL_DESIGN.md.
    """
    rows = []
    for story_id, tally in tally_by_story.items():
        games = tally["wins"] + tally["losses"] + tally["ties"]
        win_rate = (tally["wins"] + 0.5 * tally["ties"]) / games if games else 0.0
        copeland = tally["wins"] - tally["losses"]
        rows.append((story_id, copeland, win_rate, games))
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    return rows


# ---------------------------------------------------------------------------
# F. Rankings from single-text overall_quality scores
# ---------------------------------------------------------------------------

def rank_from_single_text(observations):
    """DIAGNOSTIC ONLY: per model, mean overall_quality across ALL
    context_single trials for each story, pooling every dimension/value
    condition (including the neutral baseline) together. This answers "how
    does this model rank the corpus on average, across every context this
    benchmark happened to try" -- a rough sanity check, not a per-condition
    human-alignment result. Use rank_from_single_text_by_condition for that.
    """
    scores = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single":
            continue
        scores[obs["model"]][obs["story_id"]].append(obs["parsed_response"]["overall_quality"])

    rankings = {}
    for model, by_story in scores.items():
        rows = [(story_id, statistics.mean(values), len(values)) for story_id, values in by_story.items()]
        rows.sort(key=lambda r: (-r[1], r[0]))
        rankings[model] = rows
    return rankings


def rank_from_single_text_by_condition(observations):
    """PRIMARY human-alignment ranking: one story ranking per
    (model, dimension, value), built from mean overall_quality across
    context_single trials -- kept separate rather than pooled across
    conditions. This includes the neutral baseline as its own condition
    (dimension="neutral", value="neutral"; see context_packets.neutral_condition
    and context_trials.build_context_single_trials).

    This is what can actually answer "which model + context setup best
    matches the fixed human preference ranking?" -- a pooled ranking cannot,
    since it averages away exactly the context distinction the benchmark
    exists to measure.

    Returns {(model, dimension, value): [(story_id, mean_score, n), ...]}.
    """
    scores = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single":
            continue
        key = (obs["model"], obs["dimension"], obs["value"])
        scores[key][obs["story_id"]].append(obs["parsed_response"]["overall_quality"])

    rankings = {}
    for key, by_story in scores.items():
        rows = [(story_id, statistics.mean(values), len(values)) for story_id, values in by_story.items()]
        rows.sort(key=lambda r: (-r[1], r[0]))
        rankings[key] = rows
    return rankings


# ---------------------------------------------------------------------------
# G/H. Comparison against the human reference ranking
# ---------------------------------------------------------------------------

def load_human_reference(path=HUMAN_REFERENCE_FILE):
    """Returns the ranking (best-first list of story ids), or None if the
    file doesn't exist -- i.e. no complete/unique human ranking yet."""
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


def spearman_correlation(ranking_a, ranking_b):
    """Pure-Python Spearman rank correlation between two full orderings of the
    same item set (best-first lists). None if the item sets don't match."""
    if set(ranking_a) != set(ranking_b):
        return None
    rank_a = {item: i for i, item in enumerate(ranking_a)}
    rank_b = {item: i for i, item in enumerate(ranking_b)}
    n = len(ranking_a)
    d_squared_sum = sum((rank_a[item] - rank_b[item]) ** 2 for item in ranking_a)
    return 1 - (6 * d_squared_sum) / (n * (n**2 - 1))


def pairwise_agreement_with_human(ranking, human_pairs):
    """Fraction of the known human winner/loser judgments a ranking agrees
    with (ranking places winner ahead of loser). Works even with an
    incomplete human_pairwise.jsonl -- unlike Spearman, it doesn't need a
    full reference ranking. Returns (agreement_rate_or_None, num_checked)."""
    if not human_pairs:
        return None, 0
    rank_index = {item: i for i, item in enumerate(ranking)}
    checked = 0
    agreements = 0
    for winner, loser in human_pairs:
        if winner not in rank_index or loser not in rank_index:
            continue
        checked += 1
        if rank_index[winner] < rank_index[loser]:
            agreements += 1
    if checked == 0:
        return None, 0
    return agreements / checked, checked


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}. Nothing to analyze.")
        return

    raw_rows = load_jsonl(RESULTS_FILE)
    trials_by_id = load_trials_by_id(TRIALS_FILE)
    observations, unresolved, absorbed, missing_metadata = collapse_attempts(raw_rows, trials_by_id)

    print_inventory(raw_rows, observations, unresolved, absorbed, missing_metadata)

    pairwise_effect_rows = analyze_pairwise_context_effects(observations)
    summarize_pairwise_context_effects(pairwise_effect_rows)

    prompt_effect_rows = analyze_prompt_context_effects(observations)
    summarize_prompt_context_effects(prompt_effect_rows)

    human_reference = load_human_reference()
    human_pairs = load_human_pairwise()

    def human_comparison_row(model, dimension, value, ranking_source, ranking):
        spearman = spearman_correlation(ranking, human_reference) if human_reference is not None else None
        agreement, checked = pairwise_agreement_with_human(ranking, human_pairs)
        return {
            "model": model,
            "dimension": dimension,
            "value": value,
            "ranking_source": ranking_source,
            "spearman_vs_human": round(spearman, 3) if spearman is not None else "",
            "pairwise_agreement": round(agreement, 3) if agreement is not None else "",
            "pairwise_agreement_n": checked,
        }, spearman, agreement, checked

    human_comparison_rows = []

    # --- PRIMARY: single-text ranking vs human reference, kept separate per
    # (model, dimension, value) -- this is what actually answers "which model
    # + context setup best matches the human preference ranking?" ---
    single_by_condition = rank_from_single_text_by_condition(observations)
    single_by_condition_rows = []
    print("\n=== F1. PRIMARY human-alignment analysis: single-text ranking per (model, dimension, value) ===")
    print("(includes the neutral no-context baseline as its own condition; NOT pooled across conditions)")
    if not single_by_condition:
        print("  No context_single observations found.")
    for (model, dimension, value), ranked in sorted(single_by_condition.items(), key=lambda kv: str(kv[0])):
        ranking = [row[0] for row in ranked]
        print(f"  Model: {model}  dimension={dimension}  value={value}")
        for i, (story_id, mean_score, n) in enumerate(ranked, start=1):
            print(f"    {i}. {story_id}  (mean_overall_quality={mean_score:.2f}, n={n})")
            single_by_condition_rows.append(
                {
                    "model": model, "dimension": dimension, "value": value,
                    "rank": i, "story_id": story_id,
                    "mean_overall_quality": round(mean_score, 3), "n": n,
                }
            )
        row, spearman, agreement, checked = human_comparison_row(
            model, dimension, value, "single_text_per_condition", ranking
        )
        human_comparison_rows.append(row)
        spearman_text = f"{spearman:.3f}" if spearman is not None else "unavailable"
        agreement_text = f"{agreement:.2f} ({checked} known pair(s))" if agreement is not None else "unavailable"
        print(f"    vs human reference: Spearman={spearman_text}  pairwise_agreement={agreement_text}")

    # --- DIAGNOSTIC ONLY below: pooled rankings, not per-condition results ---
    print("\n=== F2. Diagnostic only: single-text ranking pooled across ALL conditions ===")
    print("(averages over every context_single trial regardless of dimension/value; a rough sanity check, NOT the main result)")
    single_pooled_by_model = rank_from_single_text(observations)
    single_ranking_rows = []
    for model, ranked in sorted(single_pooled_by_model.items()):
        ranking = [row[0] for row in ranked]
        print(f"  Model: {model}")
        for i, (story_id, mean_score, n) in enumerate(ranked, start=1):
            print(f"    {i}. {story_id}  (mean_overall_quality={mean_score:.2f}, n={n})")
            single_ranking_rows.append(
                {"model": model, "rank": i, "story_id": story_id, "mean_overall_quality": round(mean_score, 3), "n": n}
            )
        row, spearman, agreement, checked = human_comparison_row(
            model, "ALL", "ALL", "single_text_pooled_diagnostic", ranking
        )
        human_comparison_rows.append(row)

    print("\n=== E. Diagnostic only: pooled pairwise (overall_quality) ranking ===")
    print("(pools all context_pairwise trials/contrasts/forward+flipped; these trials measure causal")
    print(" context SENSITIVITY, not a ranking under any one context condition -- see")
    print(" rank_from_pairwise_wins docstring. Not the main human-alignment result.)")
    pairwise_wins = compute_pairwise_wins(observations)
    pairwise_ranking_rows = []
    for model, tally_by_story in sorted(pairwise_wins.items()):
        ranked = rank_from_pairwise_wins(tally_by_story)
        ranking = [row[0] for row in ranked]
        print(f"  Model: {model}")
        for i, (story_id, copeland, win_rate, games) in enumerate(ranked, start=1):
            print(f"    {i}. {story_id}  (copeland={copeland}, win_rate={win_rate:.2f}, games={games})")
            pairwise_ranking_rows.append(
                {"model": model, "rank": i, "story_id": story_id, "copeland": copeland, "win_rate": round(win_rate, 3), "games": games}
            )
        row, spearman, agreement, checked = human_comparison_row(
            model, "ALL", "ALL", "pairwise_pooled_diagnostic", ranking
        )
        human_comparison_rows.append(row)

    print("\n=== G/H. Comparison against human reference: availability ===")
    if human_reference is None:
        print("  data/human_reference.json does not exist yet (run human_ranking.py once enough")
        print("  pairwise judgments are collected). Spearman correlation is unavailable for now.")
    if not human_pairs:
        print("  data/human_pairwise.jsonl has no judgments yet. Pairwise agreement is unavailable.")
    print(f"  Full model/dimension/value breakdown written to {ANALYSIS_DIR}/human_comparison.csv")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    write_csv(
        pairwise_effect_rows,
        ["model", "contrast_id", "dimension", "story_a_id", "story_b_id", "replicate_id", "category", "forward_choice", "flipped_choice", "changed"],
        os.path.join(ANALYSIS_DIR, "pairwise_context_effects.csv"),
    )
    write_csv(
        prompt_effect_rows,
        ["model", "contrast_id", "story_a_id", "story_b_id", "replicate_id", "category", "baseline_value", "treatment_value", "baseline_choice", "treatment_choice", "changed"],
        os.path.join(ANALYSIS_DIR, "prompt_context_effects.csv"),
    )
    write_csv(
        pairwise_ranking_rows,
        ["model", "rank", "story_id", "copeland", "win_rate", "games"],
        os.path.join(ANALYSIS_DIR, "pairwise_rankings_pooled_diagnostic.csv"),
    )
    write_csv(
        single_ranking_rows,
        ["model", "rank", "story_id", "mean_overall_quality", "n"],
        os.path.join(ANALYSIS_DIR, "single_text_rankings_pooled_diagnostic.csv"),
    )
    write_csv(
        single_by_condition_rows,
        ["model", "dimension", "value", "rank", "story_id", "mean_overall_quality", "n"],
        os.path.join(ANALYSIS_DIR, "single_text_rankings_by_condition.csv"),
    )
    write_csv(
        human_comparison_rows,
        ["model", "dimension", "value", "ranking_source", "spearman_vs_human", "pairwise_agreement", "pairwise_agreement_n"],
        os.path.join(ANALYSIS_DIR, "human_comparison.csv"),
    )

    print(f"\nWrote tidy tables to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
