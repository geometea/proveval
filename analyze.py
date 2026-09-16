"""Analyze results/raw.jsonl for the Proveval v0.1 experiment.

Run this file directly: python3 analyze.py

Reads results/raw.jsonl, collapses retry attempts into one record per
experimental observation, prints a dataset inventory and the
provenance-swap / control analyses, and writes tidy derived tables to
results/analysis/. No API calls are made and no input files are changed.

Trial structure (story ids, condition, task) is recovered by parsing each
result's trial_id (see parse_trial_id) rather than by joining against
data/trials.jsonl -- this file never reads the trials manifest.
"""

import csv
import json
import os
import statistics
from collections import defaultdict

RESULTS_FILE = "results/raw.jsonl"
ANALYSIS_DIR = "results/analysis"

RATING_FIELDS = ["plot_structure", "prose_style", "characterization", "originality", "overall_quality"]

# Which side (A or B) carries the "AI-generated" label under each comparative
# provenance condition. Used to compare the same physical story/position
# across a condition pair that only swaps which side is AI-labelled.
CONDITION_AI_SIDE = {
    "self_vs_ai": "b",
    "ai_vs_self": "a",
    "ai_vs_journal": "a",
    "journal_vs_ai": "b",
}


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Read a .jsonl file into a list of dicts."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_trial_id(trial_id):
    """Break a trial_id into its structural parts.

    Fields are separated by "__"; story/condition ids may themselves contain
    single underscores (e.g. "dunnest_smoke", "neutral_metadata"), which is
    safe because none of them contain "__".

    Formats:
      single__{story}__{condition}__{task}
      comparison__{story_a}__{story_b}__{condition}
      comparison_control__{story}__{condition}
    """
    parts = trial_id.split("__")
    prefix = parts[0]

    if prefix == "single" and len(parts) == 4:
        _, story, condition, task = parts
        return {"trial_type": "single", "story_a": story, "story_b": None, "condition": condition, "task": task}

    if prefix == "comparison_control" and len(parts) == 3:
        _, story, condition = parts
        return {"trial_type": "comparison_control", "story_a": story, "story_b": story, "condition": condition, "task": None}

    if prefix == "comparison" and len(parts) == 4:
        _, story_a, story_b, condition = parts
        return {"trial_type": "comparison", "story_a": story_a, "story_b": story_b, "condition": condition, "task": None}

    raise ValueError(f"Unrecognized trial_id format: {trial_id}")


def is_successful(row):
    """A raw attempt counts as successful if it parsed and validated cleanly."""
    return row.get("parsed_response") is not None and row.get("validation_error") is None


def attempt_number(row):
    """Existing rows may not have attempt_id; treat that as attempt 1."""
    return row.get("attempt_id", 1)


# ---------------------------------------------------------------------------
# Collapsing retries into observations
# ---------------------------------------------------------------------------

def collapse_attempts(raw_rows):
    """Group raw attempts by (trial_id, model, replicate_id).

    Returns (observations, unresolved, absorbed_failed_attempts):
      observations: {key: observation dict}, one per key with >=1 successful
                     attempt (the most recent successful attempt is used)
      unresolved:   {key: [attempt rows]} for keys where every attempt failed
      absorbed_failed_attempts: count of failed attempts that happened before
                     an eventual success -- preserved in this count, but not
                     counted as failures anywhere else in the output
    """
    attempts_by_key = defaultdict(list)
    for row in raw_rows:
        # replicate_id defaults to 1, matching attempt_number's own default
        # just below: a row saved by run_trial.py's single-trial CLI (see
        # README.md's `python3 run_trial.py <trial_id>`) carries no
        # replicate_id at all, since replication is a run_batch.py concept.
        key = (row["trial_id"], row["model"], row.get("replicate_id", 1))
        attempts_by_key[key].append(row)

    observations = {}
    unresolved = {}
    absorbed_failed_attempts = 0

    for key, attempts in attempts_by_key.items():
        attempts = sorted(attempts, key=attempt_number)
        successes = [a for a in attempts if is_successful(a)]
        if successes:
            observations[key] = build_observation(key, successes[-1])
            # Only the failed attempts before the eventual success count as
            # "absorbed" -- if more than one attempt for this key happened
            # to succeed (run_trial.py has no completed-check, unlike
            # run_batch.py's is_completed guard, so re-running it can
            # produce two successes for the same key), the extra success
            # must never be miscounted as an absorbed failure.
            absorbed_failed_attempts += len(attempts) - len(successes)
        else:
            unresolved[key] = attempts

    return observations, unresolved, absorbed_failed_attempts


def build_observation(key, row):
    """Combine the parsed trial_id with the row's own fields into one flat record.

    model and replicate_id come straight from the row (the trial_id text
    doesn't encode them); trial_type/story_a/story_b/condition/task come
    from parsing the trial_id.
    """
    trial_id, model, replicate_id = key
    parsed = parse_trial_id(trial_id)
    return {
        "trial_id": trial_id,
        "model": model,
        "replicate_id": replicate_id,
        "attempt_id": attempt_number(row),
        "trial_type": parsed["trial_type"],
        "story_a": parsed["story_a"],
        "story_b": parsed["story_b"],
        "condition": parsed["condition"],
        "task": parsed["task"],
        "parsed_response": row["parsed_response"],
    }


def single_ratings(observation):
    """Return the 5 rating values for a single-story observation."""
    return observation["parsed_response"]


def comparison_ratings(observation, side):
    """Return the 5 rating values for one side ('story_a' or 'story_b') of a comparison."""
    return observation["parsed_response"][side]


def preference(observation):
    """Return the -2..+2 preference score for a comparison observation."""
    return observation["parsed_response"]["preference"]


# ---------------------------------------------------------------------------
# Dataset inventory
# ---------------------------------------------------------------------------

def count_by(items, key_fn):
    """Count items into a dict keyed by key_fn(item), sorted by key."""
    counts = defaultdict(int)
    for item in items:
        counts[key_fn(item)] += 1
    return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))


def print_counts(counts):
    for key, count in counts.items():
        print(f"  {key}: {count}")


def print_inventory(raw_rows, observations, unresolved, absorbed_failed_attempts):
    print("=== Dataset inventory ===")
    print(f"Raw API attempts: {len(raw_rows)}")
    print(f"Completed observations: {len(observations)}")
    print(f"Unresolved failures: {len(unresolved)}")
    print(f"Failed attempts absorbed by a later success: {absorbed_failed_attempts}")

    print("\nCompleted observations by trial type:")
    print_counts(count_by(observations.values(), lambda o: o["trial_type"]))

    print("\nCompleted observations by condition:")
    print_counts(count_by(observations.values(), lambda o: o["condition"]))

    print("\nCompleted observations by replicate:")
    print_counts(count_by(observations.values(), lambda o: o["replicate_id"]))


# ---------------------------------------------------------------------------
# Identical-text control analysis
# ---------------------------------------------------------------------------

def preference_distribution(prefs):
    dist = {v: 0 for v in (-2, -1, 0, 1, 2)}
    for p in prefs:
        dist[p] += 1
    return dist


def analyze_controls(observations):
    """Preference distribution and tie rate for identical-text control trials."""
    controls = [o for o in observations.values() if o["trial_type"] == "comparison_control"]

    print("\n=== Identical-text control analysis ===")
    by_condition = defaultdict(list)
    for obs in controls:
        by_condition[obs["condition"]].append(preference(obs))

    rows = []
    total_ties = 0
    for condition, prefs in sorted(by_condition.items()):
        dist = preference_distribution(prefs)
        ties = dist[0]
        total_ties += ties
        pct = 100 * ties / len(prefs) if prefs else 0
        print(f"  {condition}: n={len(prefs)} ties={ties} ({pct:.1f}%) distribution={dist}")
        rows.append(
            {
                "condition": condition,
                "n": len(prefs),
                "mean_preference": round(statistics.mean(prefs), 3) if prefs else "",
                "tie_count": ties,
                "tie_pct": round(pct, 1),
                "pref_neg2": dist[-2],
                "pref_neg1": dist[-1],
                "pref_0": dist[0],
                "pref_pos1": dist[1],
                "pref_pos2": dist[2],
            }
        )

    overall_n = len(controls)
    overall_pct = 100 * total_ties / overall_n if overall_n else 0
    print(f"  TOTAL: n={overall_n} ties={total_ties} ({overall_pct:.1f}%)")

    return rows


# ---------------------------------------------------------------------------
# Provenance-swap analysis (preference shift)
# ---------------------------------------------------------------------------

def index_comparisons(observations):
    """Group completed comparison observations by (model, story_a, story_b, replicate_id, condition)."""
    index = {}
    for obs in observations.values():
        if obs["trial_type"] != "comparison":
            continue
        key = (obs["model"], obs["story_a"], obs["story_b"], obs["replicate_id"], obs["condition"])
        index[key] = obs
    return index


def analyze_provenance_swap(observations, condition_a, condition_b, swap_name):
    """Pair comparison observations that differ only in condition_a vs condition_b.

    shift = pref(condition_a) - pref(condition_b)

    Call with condition_a/condition_b ordered so that a positive shift means
    "moved toward the AI label" (see the two call sites in main()).
    """
    index = index_comparisons(observations)
    pairs = []
    for (model, story_a, story_b, replicate_id, condition), obs_a in index.items():
        if condition != condition_a:
            continue
        obs_b = index.get((model, story_a, story_b, replicate_id, condition_b))
        if obs_b is None:
            continue

        pref_a = preference(obs_a)
        pref_b = preference(obs_b)
        shift = pref_a - pref_b
        pairs.append(
            {
                "swap": swap_name,
                "model": model,
                "story_a": story_a,
                "story_b": story_b,
                "replicate_id": replicate_id,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "pref_condition_a": pref_a,
                "pref_condition_b": pref_b,
                "shift": shift,
                "advantage": shift / 2,
            }
        )
    return pairs


def summarize_shifts(pairs, label):
    print(f"\n=== {label} ===")
    if not pairs:
        print("  No matched pairs found.")
        return

    shifts = [p["shift"] for p in pairs]
    n_pos = sum(1 for s in shifts if s > 0)
    n_zero = sum(1 for s in shifts if s == 0)
    n_neg = sum(1 for s in shifts if s < 0)
    mean_shift = statistics.mean(shifts)
    print(f"  n={len(shifts)}  positive={n_pos}  zero={n_zero}  negative={n_neg}")
    print(f"  mean shift={mean_shift:.3f}  estimated AI-label advantage={mean_shift / 2:.3f}")

    print("  By replicate:")
    by_replicate = defaultdict(list)
    for p in pairs:
        by_replicate[p["replicate_id"]].append(p["shift"])
    for replicate_id, s in sorted(by_replicate.items()):
        print(f"    replicate {replicate_id}: n={len(s)}  mean shift={statistics.mean(s):.3f}")

    print("  By unordered story pair:")
    by_pair = defaultdict(list)
    for p in pairs:
        pair_key = tuple(sorted([p["story_a"], p["story_b"]]))
        by_pair[pair_key].append(p["shift"])
    for pair_key, s in sorted(by_pair.items()):
        print(f"    {pair_key[0]} vs {pair_key[1]}: n={len(s)}  mean shift={statistics.mean(s):.3f}")


# ---------------------------------------------------------------------------
# Rating-dimension effects (same story/position, AI-labelled vs other-labelled)
# ---------------------------------------------------------------------------

def analyze_rating_effects(observations, condition_x, condition_y, swap_name):
    """For matched condition_x/condition_y pairs, compare the same physical
    story at the same position (A or B) when it is AI-labelled versus when it
    carries the other label (self, or literary_journal), for each rating
    dimension. condition_x and condition_y must put the AI label on opposite
    sides (see CONDITION_AI_SIDE).
    """
    index = index_comparisons(observations)
    diffs = {dim: [] for dim in RATING_FIELDS}
    detail_rows = []

    for (model, story_a, story_b, replicate_id, condition), obs_x in index.items():
        if condition != condition_x:
            continue
        obs_y = index.get((model, story_a, story_b, replicate_id, condition_y))
        if obs_y is None:
            continue

        for position, story_id in (("a", story_a), ("b", story_b)):
            side = "story_" + position
            ai_obs, other_obs = (obs_x, obs_y) if CONDITION_AI_SIDE[condition_x] == position else (obs_y, obs_x)
            ai_ratings = comparison_ratings(ai_obs, side)
            other_ratings = comparison_ratings(other_obs, side)
            for dim in RATING_FIELDS:
                diff = ai_ratings[dim] - other_ratings[dim]
                diffs[dim].append(diff)
                detail_rows.append(
                    {
                        "swap": swap_name,
                        "model": model,
                        "story": story_id,
                        "position": position.upper(),
                        "replicate_id": replicate_id,
                        "dimension": dim,
                        "ai_minus_other": diff,
                    }
                )

    print(f"\n=== Rating effects: {swap_name} (AI-labelled minus other-labelled) ===")
    for dim in RATING_FIELDS:
        values = diffs[dim]
        if values:
            print(f"  {dim}: n={len(values)}  mean={statistics.mean(values):.3f}")
        else:
            print(f"  {dim}: n=0")

    return detail_rows


# ---------------------------------------------------------------------------
# Writing tidy CSVs
# ---------------------------------------------------------------------------

def write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_observation_summary(observations, path):
    fieldnames = [
        "trial_id", "model", "replicate_id", "attempt_id", "trial_type",
        "story_a", "story_b", "condition", "task", "preference",
        "a_plot_structure", "a_prose_style", "a_characterization", "a_originality", "a_overall_quality",
        "b_plot_structure", "b_prose_style", "b_characterization", "b_originality", "b_overall_quality",
    ]
    rows = []
    for obs in observations.values():
        row = {k: "" for k in fieldnames}
        row.update(
            {
                "trial_id": obs["trial_id"],
                "model": obs["model"],
                "replicate_id": obs["replicate_id"],
                "attempt_id": obs["attempt_id"],
                "trial_type": obs["trial_type"],
                "story_a": obs["story_a"] or "",
                "story_b": obs["story_b"] or "",
                "condition": obs["condition"],
                "task": obs["task"] or "",
            }
        )
        if obs["trial_type"] == "single":
            ratings = single_ratings(obs)
            for dim in RATING_FIELDS:
                row[f"a_{dim}"] = ratings[dim]
        else:
            row["preference"] = preference(obs)
            for dim in RATING_FIELDS:
                row[f"a_{dim}"] = comparison_ratings(obs, "story_a")[dim]
                row[f"b_{dim}"] = comparison_ratings(obs, "story_b")[dim]
        rows.append(row)
    write_csv(rows, fieldnames, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}. Nothing to analyze.")
        return

    raw_rows = load_jsonl(RESULTS_FILE)
    observations, unresolved, absorbed_failed_attempts = collapse_attempts(raw_rows)

    print_inventory(raw_rows, observations, unresolved, absorbed_failed_attempts)

    control_rows = analyze_controls(observations)

    # self vs AI: "self_vs_ai" puts AI on Story B, "ai_vs_self" puts AI on
    # Story A, so pref(self_vs_ai) - pref(ai_vs_self) is positive when
    # preference moves toward whichever story is AI-labelled.
    self_ai_pairs = analyze_provenance_swap(observations, "self_vs_ai", "ai_vs_self", "self_vs_ai vs ai_vs_self")
    summarize_shifts(self_ai_pairs, "Provenance swap: self vs AI")
    self_ai_rating_rows = analyze_rating_effects(observations, "self_vs_ai", "ai_vs_self", "self_vs_ai vs ai_vs_self")

    # literary journal vs AI: "journal_vs_ai" puts AI on Story B, so the same
    # sign convention needs condition_a="journal_vs_ai", condition_b="ai_vs_journal".
    journal_ai_pairs = analyze_provenance_swap(
        observations, "journal_vs_ai", "ai_vs_journal", "journal_vs_ai vs ai_vs_journal"
    )
    summarize_shifts(journal_ai_pairs, "Provenance swap: literary journal vs AI")
    journal_ai_rating_rows = analyze_rating_effects(
        observations, "ai_vs_journal", "journal_vs_ai", "ai_vs_journal vs journal_vs_ai"
    )

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    write_observation_summary(observations, os.path.join(ANALYSIS_DIR, "observation_summary.csv"))

    write_csv(
        self_ai_pairs + journal_ai_pairs,
        [
            "swap", "model", "story_a", "story_b", "replicate_id",
            "condition_a", "condition_b", "pref_condition_a", "pref_condition_b",
            "shift", "advantage",
        ],
        os.path.join(ANALYSIS_DIR, "provenance_swaps.csv"),
    )

    write_csv(
        self_ai_rating_rows + journal_ai_rating_rows,
        ["swap", "model", "story", "position", "replicate_id", "dimension", "ai_minus_other"],
        os.path.join(ANALYSIS_DIR, "rating_effects.csv"),
    )

    write_csv(
        control_rows,
        [
            "condition", "n", "mean_preference", "tie_count", "tie_pct",
            "pref_neg2", "pref_neg1", "pref_0", "pref_pos1", "pref_pos2",
        ],
        os.path.join(ANALYSIS_DIR, "control_summary.csv"),
    )

    print(f"\nWrote tidy tables to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
