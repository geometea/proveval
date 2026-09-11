"""Run many trials through the Claude API in one batch.

Usage:
    python3 run_batch.py --replicates 3                     # run everything, 3x each
    python3 run_batch.py --type single --limit 10            # first 10 single trials
    python3 run_batch.py --replicates 2 --limit 3 --dry-run  # show selection only

Reuses the same API-call and validation logic as run_trial.py.
Trial order is shuffled with a fixed seed so runs are reproducible.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from random import Random

from run_trial import (
    DEFAULT_MODEL,
    RESULTS_FILE,
    TRIALS_FILE,
    call_claude,
    load_trials,
    parse_and_validate,
    save_result,
)

RANDOM_SEED = 42


def select_trials(trials, only_type, id_prefix, limit):
    """Filter by type and trial_id prefix (if given), shuffle with a fixed seed, then cut to limit."""
    if only_type:
        trials = [t for t in trials if t["type"] == only_type]

    if id_prefix:
        trials = [t for t in trials if t["trial_id"].startswith(id_prefix)]

    trials = list(trials)
    Random(RANDOM_SEED).shuffle(trials)

    if limit is not None:
        trials = trials[:limit]
    return trials


def build_observations(trials, replicates):
    """Expand each trial into one observation per replicate_id (1..replicates)."""
    observations = []
    for trial in trials:
        for replicate_id in range(1, replicates + 1):
            observations.append((trial, replicate_id))
    return observations


def load_existing_results(path):
    """Return {(trial_id, model, replicate_id): [existing result records]}.

    A key can have more than one record if earlier attempts failed and were
    retried; failures are never deleted, only added to.
    """
    existing = {}
    if not os.path.exists(path):
        return existing
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                result = json.loads(line)
                key = (result["trial_id"], result["model"], result["replicate_id"])
                existing.setdefault(key, []).append(result)
    return existing


def is_completed(records):
    """A key is done only if one of its records has a valid parsed response."""
    return any(r["parsed_response"] is not None and r["validation_error"] is None for r in records)


def next_attempt_id(records):
    """Return the next attempt number for an observation. A missing attempt_id counts as 1."""
    if not records:
        return 1
    return max(r.get("attempt_id", 1) for r in records) + 1


def run_one(trial, replicate_id, model, attempt_id):
    """Call the API for one trial, validate it, save a result row, and return a status word.

    Always saves a new row, even on failure, so a failed call can still be diagnosed
    later; existing rows (from earlier attempts) are never modified or removed.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        api_result = call_claude(trial["prompt"], model)
    except Exception as e:
        print(f"Error calling the API for {trial['trial_id']} (replicate {replicate_id}): {e}")
        save_result(
            {
                "trial_id": trial["trial_id"],
                "model": model,
                "replicate_id": replicate_id,
                "attempt_id": attempt_id,
                "timestamp": timestamp,
                "stop_reason": None,
                "input_tokens": None,
                "output_tokens": None,
                "response_text": None,
                "parsed_response": None,
                "validation_error": f"API call failed: {e}",
            }
        )
        return "error"

    response_text = api_result["response_text"]
    parsed_response, validation_error = parse_and_validate(response_text, trial["type"])

    save_result(
        {
            "trial_id": trial["trial_id"],
            "model": model,
            "replicate_id": replicate_id,
            "attempt_id": attempt_id,
            "timestamp": timestamp,
            "stop_reason": api_result["stop_reason"],
            "input_tokens": api_result["input_tokens"],
            "output_tokens": api_result["output_tokens"],
            "response_text": response_text,
            "parsed_response": parsed_response,
            "validation_error": validation_error,
        }
    )
    return "invalid" if validation_error else "valid"


def select_failed_observations(existing_results, trials_by_id, model, only_type, id_prefix, limit):
    """Find (trial, replicate_id) pairs that have failed attempts and no successful one.

    Only observations for the current model are considered, since a different
    model's failure can't be retried without also re-running everything else.
    """
    candidates = []
    for (trial_id, result_model, replicate_id), records in existing_results.items():
        if result_model != model or is_completed(records):
            continue
        trial = trials_by_id.get(trial_id)
        if trial is None:
            continue  # trial no longer exists in data/trials.jsonl
        if only_type and trial["type"] != only_type:
            continue
        if id_prefix and not trial_id.startswith(id_prefix):
            continue
        candidates.append((trial, replicate_id))

    Random(RANDOM_SEED).shuffle(candidates)
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Run many trials through the Claude API.")
    parser.add_argument("--replicates", type=int, default=1, help="How many times to run each trial")
    parser.add_argument("--type", choices=["single", "comparison"], help="Only run this trial type")
    parser.add_argument("--id-prefix", help="Only run trials whose trial_id starts with this prefix")
    parser.add_argument("--limit", type=int, help="Only run the first N selected trials (for testing)")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only select observations with a failed attempt and no successful one (ignores --replicates)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without calling the API")
    args = parser.parse_args()

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    existing_results = load_existing_results(RESULTS_FILE)

    if args.retry_failed:
        trials_by_id = {t["trial_id"]: t for t in load_trials(TRIALS_FILE)}
        observations = select_failed_observations(
            existing_results, trials_by_id, model, args.type, args.id_prefix, args.limit
        )
    else:
        trials = load_trials(TRIALS_FILE)
        trials = select_trials(trials, args.type, args.id_prefix, args.limit)
        observations = build_observations(trials, args.replicates)

    total = len(observations)
    for i, (trial, replicate_id) in enumerate(observations, start=1):
        key = (trial["trial_id"], model, replicate_id)
        label = f"{i} / {total} — {trial['trial_id']} — replicate {replicate_id}"
        records = existing_results.get(key, [])

        if records and is_completed(records):
            print(f"{label} — skipped-valid")
            continue

        is_retry = bool(records)  # records exist, but none of them are valid
        attempt_id = next_attempt_id(records)

        if args.dry_run:
            status = f"retrying-failed (attempt {attempt_id})" if is_retry else "planned"
            print(f"{label} — {status}")
            continue

        if is_retry:
            print(f"{label} — retrying-failed (attempt {attempt_id})")

        try:
            status = run_one(trial, replicate_id, model, attempt_id)
            print(f"{label} — {status}")
        except Exception as e:
            print(f"{label} — error: {e}")


if __name__ == "__main__":
    main()
