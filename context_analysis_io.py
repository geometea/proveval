"""Loading raw results into collapsed observations, and the dataset-inventory
report -- the shared entry point every other context_analysis_*.py module
consumes (they all operate on the `observations` dict this module builds).

Collapsing retries (same semantics as analyze.py: group by
(trial_id, model, replicate_id, sampling_regime), keep the most recent
success). sampling_regime is part of the key so a retry under a different
regime is never silently merged into the same observation; evaluation_regime
doesn't need to be, since it's already baked into a distinct trial_id (see
context_trials.py), so a naturalistic and an invariance trial can never
collide here. This collapses ATTEMPTS (retries of the same cell after a
parse/validation failure), never REPLICATES -- every replicate_id remains
its own observation; repeated identical cells are preserved as separate
data points, not reduced to a majority vote (see
context_analysis_pairwise.analyze_directional_pairwise_effects and
context_analysis_single_text.neutral_baseline_means, which both estimate
frequencies/means across them).
"""

import os
from collections import defaultdict

from analyze import load_jsonl, is_successful, attempt_number
from context_analysis_common import TRIALS_FILE


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


def collapse_attempts(raw_rows, trials_by_id):
    """Returns (observations, unresolved, absorbed_failed_attempts, missing_metadata)."""
    attempts_by_key = defaultdict(list)
    for row in raw_rows:
        # replicate_id defaults to 1, matching attempt_number's own default
        # just below: a row saved by run_trial.py's single-trial CLI carries
        # no replicate_id at all, since replication is a run_batch.py concept.
        key = (row["trial_id"], row["model"], row.get("replicate_id", 1), row.get("sampling_regime"))
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
            # Execution identity (see model_providers.py / run_batch.py):
            # preserved on every observation, never collapsed away, so
            # downstream analysis can detect (and refuse to silently pool)
            # results from different providers/models/reasoning profiles/
            # execution modes that happen to share a trial_id+replicate_id.
            "provider": row.get("provider"),
            "requested_model": row.get("requested_model"),
            "response_model": row.get("response_model"),
            "reasoning_profile": row.get("reasoning_profile"),
            "provider_reasoning_settings": row.get("provider_reasoning_settings"),
            "execution_mode": row.get("execution_mode"),
            "request_id": row.get("request_id"),
            "reasoning_tokens": row.get("reasoning_tokens"),
        }
        # Only the failed attempts before the eventual success count as
        # "absorbed" -- if more than one attempt for this key happened to
        # succeed, the extra success must never be miscounted as a failure.
        absorbed_failed_attempts += len(attempts) - len(successes)

    return observations, unresolved, absorbed_failed_attempts, missing_metadata


def filter_by_sampling_regime(observations, regime):
    """Keep only observations recorded under `regime`. Returns (kept, excluded_count).
    This is the one place sampling regimes get selected -- nothing downstream
    ever pools across sampling regimes, since everything after this operates
    on `kept`. evaluation_regime is NOT filtered here: it's the substantive
    variable under study, so it stays as a live stratification key in every
    analysis instead (see analyze_context.py's module docstring)."""
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
