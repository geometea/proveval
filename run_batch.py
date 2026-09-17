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
sampling_regime is also part of an observation's identity for the runner's
own "already completed" / "already failed" bookkeeping (see
load_existing_results/result_sampling_regime), matching
analyze_context.collapse_attempts: a valid result under one sampling regime
never causes the same trial/replicate to be skipped when subsequently run
under the other regime into the same results file.

--limit operates on complete experimental units, not raw trial rows: a
context_pairwise 4-cell counterbalanced block (forward/flipped x
A/B-position) is always selected or skipped as a whole, so a budget cap can
never buy an orphan cell and leave the rest of its block unrun -- see
group_trials_into_units/group_failed_candidates_into_units. Every other
trial type's replicate logic (including the neutral-vs-treatment single-text
counts from --replicates-treatment/--replicates-neutral) is unaffected.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from random import Random
from threading import Lock

import model_providers
from run_trial import (
    CONTEXT_TRIAL_TYPES,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    RESULTS_FILE,
    SAMPLING_REGIMES,
    TRIALS_FILE,
    default_sampling_regime_for_trials,
    load_trials,
    max_tokens_for,
    parse_and_validate,
    resolve_model,
    resolve_provider,
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


def group_trials_into_units(trials):
    """Group trials into selection units so a budget/--limit can't orphan a
    pairwise counterbalance block.

    context_pairwise cells carry a "block_id" shared by all 4 cells of one
    story-pair/contrast/evaluation_regime/choice_mode block (see
    context_contrasts.build_contrast_block). Cells sharing a block_id are
    grouped into a single unit here, so shuffling and slicing to --limit
    below operates on whole blocks, never on individual forward/flipped or
    A/B-position cells -- a block is always selected or skipped as a whole.

    Trials without a block_id (context_single, context_prompt, and every
    v0.1 trial type) have no such grouping concern and are each their own
    singleton unit, so --limit continues to mean exactly what it did before
    for those trial types (a count of individual trials).

    Returns a list of units (each unit a list of 1+ trials), in first-seen
    order; the caller shuffles/slices this list, not the raw trials.
    """
    units = []
    block_index = {}
    for trial in trials:
        block_id = trial.get("block_id")
        if block_id is None:
            units.append([trial])
            continue
        if block_id not in block_index:
            block_index[block_id] = []
            units.append(block_index[block_id])
        block_index[block_id].append(trial)
    return units


def select_trials(trials, only_type, id_prefix, conditions, contrasts, evaluation_regimes, limit):
    """Filter by type, trial_id prefix, condition_id, contrast_id, and
    evaluation_regime (each if given), shuffle with a fixed seed, then cut
    to limit.

    condition_id (old single/comparison trials, and context_single),
    contrast_id (context_pairwise/context_prompt), and evaluation_regime
    (all v0.2 context trial types) are read with .get() since not every
    trial type has all of these fields.

    Shuffling and --limit operate on whole selection units (see
    group_trials_into_units), not on raw trial rows: for context_pairwise,
    a unit is a complete 4-cell block, so --limit counts blocks, never
    individual cells, and can never leave a block partially selected. For
    every other trial type, a unit is just that one trial, so --limit means
    exactly what it always has.
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

    units = group_trials_into_units(list(trials))
    Random(RANDOM_SEED).shuffle(units)

    if limit is not None:
        units = units[:limit]
    return [trial for unit in units for trial in unit]


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


def result_sampling_regime(trial_type, sampling_regime):
    """The sampling_regime value that will actually be recorded on a result
    row for this trial type, given the --sampling-regime a run is using.

    v0.1 trial types never record a sampling_regime at all (see
    run_trial.resolve_sampling_params, which returns None for any type not
    in CONTEXT_TRIAL_TYPES) -- their identity key's regime slot is always
    None, regardless of what --sampling-regime was passed. Every v0.2
    context trial type records exactly the regime it was run under.
    """
    return sampling_regime if trial_type in CONTEXT_TRIAL_TYPES else None


def load_existing_results(path):
    """Return {(trial_id, model, replicate_id, sampling_regime): [existing result records]}.

    sampling_regime is part of the identity key -- matching
    analyze_context.collapse_attempts -- so a result recorded under
    low_variance_primary is never mistaken for a completed observation of
    the same trial/model/replicate under provider_default_secondary (or
    vice versa). Without this, running the same trial under a second
    sampling regime into the same results file would see the first
    regime's valid result and skip the second regime's call entirely.

    A key can have more than one record if earlier attempts failed and were
    retried; failures are never deleted, only added to.
    """
    existing = {}
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result = json.loads(line)
                key = (result["trial_id"], result["model"], result["replicate_id"], result.get("sampling_regime"))
                existing.setdefault(key, []).append(result)
    return existing


def is_completed(records, expected_prompt_sha256=None, expected_provider=None, expected_model=None, expected_reasoning_profile=None):
    """A key is done only if one of its records has a valid parsed response.

    expected_prompt_sha256, when given, additionally requires that record's
    own trial_meta.prompt_sha256 to match -- so a result saved against an
    earlier version of a trial's prompt (e.g. after wording changed) is
    never mistaken for satisfying the CURRENT trial. Trials that don't carry
    a prompt_sha256 (every trial type except the standalone
    context-controllability experiment -- see controllability_trials.py)
    pass None here and get the exact old behavior, unaffected.

    expected_provider/expected_model/expected_reasoning_profile, when
    given, similarly require the record's own "provider"/"requested_model"/
    "reasoning_profile" fields to match -- but ONLY when the record actually
    carries that field. Every result saved before multi-provider support
    was added has none of these fields at all; treating an absent field as
    "unknown, assume compatible" (rather than "mismatch") means those
    pre-existing completed results still count as done, exactly as before,
    while a genuine provider/model/reasoning-profile change on results that
    DO carry this metadata is correctly treated as incomplete.
    """
    for r in records:
        if r["parsed_response"] is None or r["validation_error"] is not None:
            continue
        if expected_prompt_sha256 is not None:
            recorded = (r.get("trial_meta") or {}).get("prompt_sha256")
            if recorded != expected_prompt_sha256:
                continue
        if expected_provider is not None and r.get("provider") not in (None, expected_provider):
            continue
        if expected_model is not None and r.get("requested_model") not in (None, expected_model):
            continue
        if expected_reasoning_profile is not None and r.get("reasoning_profile") not in (None, expected_reasoning_profile):
            continue
        return True
    return False


def next_attempt_id(records):
    """Return the next attempt number for an observation. A missing attempt_id counts as 1."""
    if not records:
        return 1
    return max(r.get("attempt_id", 1) for r in records) + 1


RETRYABLE_ERROR_HINTS = ("429", "500", "502", "503", "504", "timeout", "timed out", "connection")


def is_retryable_error(exc):
    """429 / 5xx / network-timeout-shaped errors -- a conservative substring
    check against the exception's own message, since each provider SDK
    raises its own exception types. Never retries a malformed-response
    situation (that isn't an exception at all -- see parse_and_validate)."""
    text = str(exc).lower()
    return any(hint in text for hint in RETRYABLE_ERROR_HINTS)


def run_one(trial, replicate_id, model, attempt_id, results_file, sampling_regime, provider=DEFAULT_PROVIDER,
            reasoning_profile=model_providers.DEFAULT_REASONING_PROFILE, execution_mode="direct", max_transient_retries=0):
    """Call the provider for one trial, validate it, save a result row, and
    return a status word.

    Always saves a new row, even on failure, so a failed call can still be diagnosed
    later; existing rows (from earlier attempts) are never modified or removed.

    max_transient_retries (used by --concurrency execution, e.g. for
    DeepSeek, which has no native batch endpoint) retries a 429/5xx/timeout
    -shaped failure in place, with exponential backoff, resending the exact
    same prompt/reasoning_profile every time -- never a "please answer
    correctly" corrective reprompt. A non-retryable failure (or exhausting
    the retries) still saves exactly one failed result row, same as
    max_transient_retries=0.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    trial_meta = trial_metadata(trial) if trial["type"] in CONTEXT_TRIAL_TYPES else None
    sampling_params = resolve_sampling_params(trial["type"], sampling_regime)
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, reasoning_profile)
    max_output_tokens = max_tokens_for(provider, trial.get("response_format"))

    base_result = {
        "trial_id": trial["trial_id"],
        "provider": provider,
        "requested_model": model,
        "model": model,  # kept for backward compatibility with every existing analysis/test that reads "model"
        "reasoning_profile": reasoning_profile,
        "provider_reasoning_settings": reasoning_settings,
        "execution_mode": execution_mode,
        "replicate_id": replicate_id,
        "attempt_id": attempt_id,
        "timestamp": timestamp,
    }

    api_result = None
    call_error = None
    delay = 2
    for retry in range(max_transient_retries + 1):
        try:
            api_result = model_providers.call_model(provider, model, trial["prompt"], max_output_tokens, reasoning_profile, sampling_params)
            call_error = None
            break
        except Exception as e:
            call_error = e
            if retry < max_transient_retries and is_retryable_error(e):
                time.sleep(delay)
                delay *= 2
                continue
            break

    if call_error is not None:
        e = call_error
        print(f"Error calling {provider} for {trial['trial_id']} (replicate {replicate_id}): {e}")
        result = {
            **base_result,
            "response_model": None,
            "stop_reason": None,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "request_id": None,
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
    parsed_response, validation_error = parse_and_validate(
        response_text, trial["type"], trial.get("choice_mode"), trial.get("rubric", "full"), trial.get("response_format")
    )

    result = {
        **base_result,
        "response_model": api_result.get("response_model"),
        "stop_reason": api_result["stop_reason"],
        "input_tokens": api_result["input_tokens"],
        "output_tokens": api_result["output_tokens"],
        "reasoning_tokens": api_result.get("reasoning_tokens"),
        "request_id": api_result.get("request_id"),
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


def group_failed_candidates_into_units(candidates):
    """Analogous to group_trials_into_units, but for (trial, replicate_id)
    retry candidates.

    Cells of the same context_pairwise block AND the same replicate_id are
    grouped into one unit, so a --limit-bounded retry can't retry some
    failed cells of a block+replicate while leaving sibling failed cells of
    that same block+replicate un-retried. Candidates without a block_id are
    each their own singleton unit, unchanged from before.
    """
    units = []
    block_index = {}
    for trial, replicate_id in candidates:
        block_id = trial.get("block_id")
        if block_id is None:
            units.append([(trial, replicate_id)])
            continue
        key = (block_id, replicate_id)
        if key not in block_index:
            block_index[key] = []
            units.append(block_index[key])
        block_index[key].append((trial, replicate_id))
    return units


def select_failed_observations(existing_results, trials_by_id, model, sampling_regime, only_type, id_prefix, conditions, contrasts, evaluation_regimes, limit,
                                provider=None, reasoning_profile=None):
    """Find (trial, replicate_id) pairs that have failed attempts and no successful one.

    Only observations for the current model AND the current --sampling-regime
    are considered: a failure recorded under one sampling regime is never
    retried as if it belonged to the other, since a retry re-runs the API
    call under sampling_regime and would otherwise silently record a second
    regime's result under what looks like the first regime's failed slot.
    provider/reasoning_profile, when given, are passed through to
    is_completed()'s backward-compatible identity check.

    Shuffling and --limit operate on whole units (see
    group_failed_candidates_into_units) so a retry batch can't orphan part
    of a failed context_pairwise block+replicate the way raw-candidate
    slicing could.
    """
    candidates = []
    for (trial_id, result_model, replicate_id, result_regime), records in existing_results.items():
        if result_model != model:
            continue
        trial = trials_by_id.get(trial_id)
        if trial is None:
            continue  # trial no longer exists in the trials file
        if is_completed(records, trial.get("prompt_sha256"), provider, model, reasoning_profile):
            continue
        if result_regime != result_sampling_regime(trial["type"], sampling_regime):
            continue
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

    units = group_failed_candidates_into_units(candidates)
    Random(RANDOM_SEED).shuffle(units)
    if limit is not None:
        units = units[:limit]
    return [candidate for unit in units for candidate in unit]


def main():
    parser = argparse.ArgumentParser(description="Run many trials through the Claude API.")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Trials manifest to read (default: {TRIALS_FILE})")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Results file to append to (default: {RESULTS_FILE})")
    parser.add_argument("--model", help="Override the model (default: $ANTHROPIC_MODEL, else claude-sonnet-5)")
    parser.add_argument(
        "--sampling-regime",
        choices=list(SAMPLING_REGIMES),
        default=None,
        help="v0.2 context trial types only (ignored, and not recorded, for v0.1 trial types): "
        "low_variance_primary (temperature=0) or provider_default_secondary. Default: "
        "provider_default_secondary when every selected trial is a plain_ab (controllability) "
        "trial, low_variance_primary otherwise -- see default_sampling_regime_for_trials.",
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run the first N selected units (for testing). A unit is one complete "
        "4-cell context_pairwise block, or one trial for every other type -- see "
        "group_trials_into_units; a block is never partially selected.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only select observations with a failed attempt and no successful one (ignores --replicates)",
    )
    parser.add_argument("--provider", choices=list(model_providers.PROVIDERS),
                         help=f"Model provider (default: {DEFAULT_PROVIDER}, for backward compatibility)")
    parser.add_argument(
        "--reasoning-profile",
        choices=list(model_providers.REASONING_PROFILES_LOGICAL),
        default=model_providers.DEFAULT_REASONING_PROFILE,
        help=f"Provider-neutral reasoning level (default: {model_providers.DEFAULT_REASONING_PROFILE}); "
        "mapped to each provider's own native setting -- see model_providers.REASONING_PROFILES. "
        "The same profile is used for every experimental condition; never alters the prompt text.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Bounded concurrent execution with retry/backoff on 429/5xx/timeout (default: 1, sequential). "
        "Intended for providers with no native batch endpoint (e.g. DeepSeek).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without calling the API")
    args = parser.parse_args()

    provider = resolve_provider(args.provider)
    model = resolve_model(provider, args.model)
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
        sampling_regime = args.sampling_regime or default_sampling_regime_for_trials(list(trials_by_id.values()))
        observations = select_failed_observations(
            existing_results, trials_by_id, model, sampling_regime,
            args.type, args.id_prefix, conditions, contrasts, evaluation_regimes, args.limit,
            provider, args.reasoning_profile,
        )
    else:
        trials = load_trials(args.trials_file)
        trials = select_trials(trials, args.type, args.id_prefix, conditions, contrasts, evaluation_regimes, args.limit)
        sampling_regime = args.sampling_regime or default_sampling_regime_for_trials(trials)
        observations = build_observations(trials, replicates_treatment, replicates_neutral)

    total = len(observations)
    counts = {"valid": 0, "invalid": 0, "error": 0, "skipped-valid": 0}
    counts_lock = Lock()
    to_run = []  # (label, trial, replicate_id, attempt_id) -- decided sequentially, executed per --concurrency

    for i, (trial, replicate_id) in enumerate(observations, start=1):
        key = (trial["trial_id"], model, replicate_id, result_sampling_regime(trial["type"], sampling_regime))
        label = f"{i} / {total} — {trial['trial_id']} — replicate {replicate_id}"
        records = existing_results.get(key, [])

        # prompt_sha256/provider/model/reasoning_profile (see
        # controllability_trials.py and model_providers.py) are all part of
        # the completion check: a result saved against a different prompt
        # wording, provider, requested model, or reasoning profile never
        # counts as satisfying the current trial (backward-compatible: a
        # pre-existing result that carries none of this metadata still
        # counts, exactly as before -- see is_completed).
        if records and is_completed(records, trial.get("prompt_sha256"), provider, model, args.reasoning_profile):
            print(f"{label} — skipped-valid")
            counts["skipped-valid"] += 1
            continue

        is_retry = bool(records)  # records exist, but none of them satisfy this trial
        attempt_id = next_attempt_id(records)

        if args.dry_run:
            status = f"retrying-failed (attempt {attempt_id})" if is_retry else "planned"
            print(f"{label} — {status}")
            continue

        if is_retry:
            print(f"{label} — retrying-failed (attempt {attempt_id})")

        to_run.append((label, trial, replicate_id, attempt_id))

    if args.dry_run:
        return

    def execute(label, trial, replicate_id, attempt_id):
        try:
            status = run_one(
                trial, replicate_id, model, attempt_id, args.results_file, sampling_regime,
                provider, args.reasoning_profile,
                execution_mode="concurrent" if args.concurrency > 1 else "direct",
                max_transient_retries=4 if args.concurrency > 1 else 0,
            )
            print(f"{label} — {status}")
            with counts_lock:
                counts[status] = counts.get(status, 0) + 1
        except Exception as e:
            print(f"{label} — error: {e}")
            with counts_lock:
                counts["error"] += 1

    if args.concurrency > 1:
        # Bounded concurrent execution (e.g. for providers with no native
        # batch endpoint, such as DeepSeek -- see model_providers.py). Each
        # request is still an independent call of run_one with the exact
        # same trial/prompt/reasoning_profile as sequential execution would
        # use; only the scheduling is concurrent.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(execute, *task) for task in to_run]
            for future in as_completed(futures):
                future.result()  # re-raise anything execute() itself didn't catch
    else:
        for task in to_run:
            execute(*task)

    if total:
        print(
            f"\nDone. valid={counts['valid']} invalid={counts['invalid']} "
            f"error={counts['error']} skipped-valid={counts['skipped-valid']}"
        )


if __name__ == "__main__":
    main()
