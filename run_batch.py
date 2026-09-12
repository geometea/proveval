"""Run many trials through the Claude API in one batch.

Usage:
    python3 run_batch.py --replicates 3                     # run everything, 3x each
    python3 run_batch.py --type single --limit 10            # first 10 single trials
    python3 run_batch.py --replicates 2 --limit 3 --dry-run  # show selection only

    # v0.2 context_single: sample the neutral baseline more than treatment
    python3 run_batch.py --trials-file data/context_trials.jsonl --type context_single \
        --replicates-treatment 3 --replicates-neutral 6 --dry-run

Reuses the same API-call and validation logic as run_trial.py.
Trial order is shuffled with a fixed seed so runs are reproducible.

Every v0.2 context trial type's result row records which sampling regime
produced it (--sampling-regime; see run_trial.SAMPLING_REGIMES) so the
primary low-variance regime and the secondary provider-default regime can
never be silently pooled by analysis code -- v0.1 trial types are
unaffected and never carry this field, regardless of the flag's default.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from random import Random

from run_trial import (
    CONTEXT_TRIAL_TYPES,
    DEFAULT_MODEL,
    RESULTS_FILE,
    SAMPLING_REGIMES,
    TRIALS_FILE,
    call_claude,
    load_trials,
    parse_and_validate,
    resolve_model,
    resolve_sampling_params,
    save_result,
    trial_metadata,
)

RANDOM_SEED = 42


def is_neutral_single_trial(trial):
    """The v0.2 no-context baseline (context_trials.build_context_single_trials
    adds one per story via context_packets.neutral_condition) -- the only
    trial kind --replicates-neutral applies to."""
    return trial["type"] == "context_single" and trial.get("condition_id") == "neutral"


def resolve_replicate_counts(replicates, replicates_treatment, replicates_neutral):
    """Treatment replicate count defaults to --replicates; neutral defaults to
    the treatment count (never fewer) and can be set higher, since the same
    neutral estimate is reused as a baseline for every treatment comparison
    on that story. Raises ValueError if neutral is explicitly set below the
    treatment count -- that would under-sample the one condition every
    treatment-vs-neutral delta depends on.
    """
    treatment = replicates_treatment if replicates_treatment is not None else replicates
    neutral = replicates_neutral if replicates_neutral is not None else treatment
    if neutral < treatment:
        raise ValueError(
            f"--replicates-neutral ({neutral}) is less than the treatment replicate "
            f"count ({treatment}). The neutral baseline is reused across every "
            f"treatment comparison for a story, so it should never be sampled less "
            f"than each individual treatment condition."
        )
    return treatment, neutral


def select_trials(trials, only_type, id_prefix, conditions, contrasts, evaluation_regimes, limit):
    """Filter by type, trial_id prefix, condition_id, contrast_id, and
    evaluation_regime (each if given), shuffle with a fixed seed, then cut
    to limit.

    condition_id (old single/comparison trials, and context_single),
    contrast_id (context_pairwise/context_prompt), and evaluation_regime
    (all v0.2 context trial types) are read with .get() since not every
    trial type has all of these fields.
    """
    if only_type:
        trials = [t for t in trials if t["type"] == only_type]

    if id_prefix:
        trials = [t for t in trials if t["trial_id"].startswith(id_prefix)]

    if conditions:
        trials = [t for t in trials if t.get("condition_id") in conditions]

    if contrasts:
        trials = [t for t in trials if t.get("contrast_id") in contrasts]

    if evaluation_regimes:
        trials = [t for t in trials if t.get("evaluation_regime") in evaluation_regimes]

    trials = list(trials)
    Random(RANDOM_SEED).shuffle(trials)

    if limit is not None:
        trials = trials[:limit]
    return trials


def build_observations(trials, replicates_treatment, replicates_neutral):
    """Expand each trial into one observation per replicate_id (1..N).

    The neutral no-context baseline (see is_neutral_single_trial) uses
    replicates_neutral; every other trial uses replicates_treatment. This
    lets neutral be sampled more precisely than any individual treatment
    condition without inflating the cost of every treatment cell to match --
    see resolve_replicate_counts for the >= treatment default/floor.
    """
    observations = []
    for trial in trials:
        n = replicates_neutral if is_neutral_single_trial(trial) else replicates_treatment
        for replicate_id in range(1, n + 1):
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


def run_one(trial, replicate_id, model, attempt_id, results_file, sampling_regime):
    """Call the API for one trial, validate it, save a result row, and return a status word.

    Always saves a new row, even on failure, so a failed call can still be diagnosed
    later; existing rows (from earlier attempts) are never modified or removed.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    trial_meta = trial_metadata(trial) if trial["type"] in CONTEXT_TRIAL_TYPES else None
    sampling_params = resolve_sampling_params(trial["type"], sampling_regime)

    try:
        api_result = call_claude(trial["prompt"], model, sampling_params)
    except Exception as e:
        print(f"Error calling the API for {trial['trial_id']} (replicate {replicate_id}): {e}")
        result = {
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
        if trial_meta is not None:
            result["trial_meta"] = trial_meta
        if sampling_params is not None:
            result["sampling_regime"] = sampling_regime
            result["sampling_params"] = sampling_params
        save_result(result, results_file)
        return "error"

    response_text = api_result["response_text"]
    parsed_response, validation_error = parse_and_validate(response_text, trial["type"])

    result = {
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
    if trial_meta is not None:
        result["trial_meta"] = trial_meta
    if sampling_params is not None:
        result["sampling_regime"] = sampling_regime
        result["sampling_params"] = sampling_params
    save_result(result, results_file)
    return "invalid" if validation_error else "valid"


def select_failed_observations(existing_results, trials_by_id, model, only_type, id_prefix, conditions, contrasts, evaluation_regimes, limit):
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
            continue  # trial no longer exists in the trials file
        if only_type and trial["type"] != only_type:
            continue
        if id_prefix and not trial_id.startswith(id_prefix):
            continue
        if conditions and trial.get("condition_id") not in conditions:
            continue
        if contrasts and trial.get("contrast_id") not in contrasts:
            continue
        if evaluation_regimes and trial.get("evaluation_regime") not in evaluation_regimes:
            continue
        candidates.append((trial, replicate_id))

    Random(RANDOM_SEED).shuffle(candidates)
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Run many trials through the Claude API.")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Trials manifest to read (default: {TRIALS_FILE})")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Results file to append to (default: {RESULTS_FILE})")
    parser.add_argument("--model", help="Override the model (default: $ANTHROPIC_MODEL, else claude-sonnet-5)")
    parser.add_argument(
        "--sampling-regime",
        choices=list(SAMPLING_REGIMES),
        default="low_variance_primary",
        help="v0.2 context trial types only (ignored, and not recorded, for v0.1 trial types): "
        "low_variance_primary (default; temperature=0) or provider_default_secondary",
    )
    parser.add_argument("--replicates", type=int, default=1, help="Replicate count for all trials (fallback for --replicates-treatment)")
    parser.add_argument(
        "--replicates-treatment",
        type=int,
        help="Replicate count for non-neutral trials (default: --replicates). "
        "Applies to every trial except the context_single neutral baseline.",
    )
    parser.add_argument(
        "--replicates-neutral",
        type=int,
        help="Replicate count for the context_single neutral no-context baseline "
        "(default: same as the treatment count; must be >= it -- see resolve_replicate_counts)",
    )
    parser.add_argument("--type", help="Only run this trial type (e.g. single, comparison, context_pairwise, ...)")
    parser.add_argument("--id-prefix", help="Only run trials whose trial_id starts with this prefix")
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        help="Only run trials with this condition_id (repeatable to allow several)",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        dest="contrasts",
        help="Only run trials with this contrast_id (repeatable; for context_pairwise/context_prompt trials)",
    )
    parser.add_argument(
        "--evaluation-regime",
        action="append",
        dest="evaluation_regimes",
        choices=["naturalistic", "text_only_invariance"],
        help="Only run trials with this evaluation_regime (repeatable; all v0.2 context trial types have one -- "
        "see context_trials.EVALUATION_REGIMES). Independent of --sampling-regime.",
    )
    parser.add_argument("--limit", type=int, help="Only run the first N selected trials (for testing)")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only select observations with a failed attempt and no successful one (ignores --replicates)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without calling the API")
    args = parser.parse_args()

    model = resolve_model(args.model)
    existing_results = load_existing_results(args.results_file)

    conditions = set(args.conditions) if args.conditions else None
    contrasts = set(args.contrasts) if args.contrasts else None
    evaluation_regimes = set(args.evaluation_regimes) if args.evaluation_regimes else None

    try:
        replicates_treatment, replicates_neutral = resolve_replicate_counts(
            args.replicates, args.replicates_treatment, args.replicates_neutral
        )
    except ValueError as e:
        parser.error(str(e))

    if args.retry_failed:
        trials_by_id = {t["trial_id"]: t for t in load_trials(args.trials_file)}
        observations = select_failed_observations(
            existing_results, trials_by_id, model, args.type, args.id_prefix, conditions, contrasts, evaluation_regimes, args.limit
        )
    else:
        trials = load_trials(args.trials_file)
        trials = select_trials(trials, args.type, args.id_prefix, conditions, contrasts, evaluation_regimes, args.limit)
        observations = build_observations(trials, replicates_treatment, replicates_neutral)

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
            status = run_one(trial, replicate_id, model, attempt_id, args.results_file, args.sampling_regime)
            print(f"{label} — {status}")
        except Exception as e:
            print(f"{label} — error: {e}")


if __name__ == "__main__":
    main()
