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


def select_trials(trials, only_type, limit):
    """Filter by type, shuffle with a fixed seed, then cut to limit."""
    if only_type:
        trials = [t for t in trials if t["type"] == only_type]

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


def load_existing_observations(path):
    """Return the set of (trial_id, model, replicate_id) already saved."""
    existing = set()
    if not os.path.exists(path):
        return existing
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                result = json.loads(line)
                existing.add((result["trial_id"], result["model"], result["replicate_id"]))
    return existing


def run_one(trial, replicate_id, model):
    """Call the API for one trial, validate the response, and save the result."""
    api_result = call_claude(trial["prompt"], model)
    response_text = api_result["response_text"]
    parsed_response, validation_error = parse_and_validate(response_text, trial["type"])

    save_result(
        {
            "trial_id": trial["trial_id"],
            "model": model,
            "replicate_id": replicate_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stop_reason": api_result["stop_reason"],
            "input_tokens": api_result["input_tokens"],
            "output_tokens": api_result["output_tokens"],
            "response_text": response_text,
            "parsed_response": parsed_response,
            "validation_error": validation_error,
        }
    )
    return validation_error


def main():
    parser = argparse.ArgumentParser(description="Run many trials through the Claude API.")
    parser.add_argument("--replicates", type=int, default=1, help="How many times to run each trial")
    parser.add_argument("--type", choices=["single", "comparison"], help="Only run this trial type")
    parser.add_argument("--limit", type=int, help="Only run the first N selected trials (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without calling the API")
    args = parser.parse_args()

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    trials = load_trials(TRIALS_FILE)
    trials = select_trials(trials, args.type, args.limit)
    observations = build_observations(trials, args.replicates)
    existing = load_existing_observations(RESULTS_FILE)

    total = len(observations)
    for i, (trial, replicate_id) in enumerate(observations, start=1):
        key = (trial["trial_id"], model, replicate_id)
        label = f"{i} / {total} | {trial['trial_id']} | replicate {replicate_id}"

        if key in existing:
            print(f"{label} -- skipped (already have this result)")
            continue

        if args.dry_run:
            print(f"{label} -- would call the API (model: {model})")
            continue

        try:
            validation_error = run_one(trial, replicate_id, model)
            existing.add(key)
            status = "invalid" if validation_error else "valid"
            print(f"{label} -- {status}")
        except Exception as e:
            print(f"{label} -- FAILED: {e}")


if __name__ == "__main__":
    main()
