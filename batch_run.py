"""Native provider batch execution: submit / status / collect.

Reuses run_batch.py's trial selection, block-aware --limit, replicate
expansion, and completion-identity check verbatim -- this file adds no
second experiment-selection implementation. All provider-specific request
construction/submission/collection lives in model_providers.py; this file
only orchestrates: pick trials, build one job file, submit, and later
collect normalized results into the ordinary results JSONL.

Usage:
    python3 batch_run.py submit --provider anthropic --model claude-sonnet-5 \
        --reasoning-profile low --trials-file data/controllability_trials.jsonl \
        --results-file results/controllability/claude.jsonl --limit 2 --dry-run

    python3 batch_run.py status --job-file results/batch_jobs/<job>.json

    python3 batch_run.py collect --job-file results/batch_jobs/<job>.json

DeepSeek has no native batch endpoint (see model_providers.BATCH_SUPPORTED_
PROVIDERS): `submit --provider deepseek` fails clearly and points at
run_batch.py's bounded concurrent execution instead.

No API calls are made by `submit --dry-run`, `status`, or anything in this
file except the network calls `submit` (without --dry-run), `status`, and
`collect` make to the chosen provider.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import model_providers
from run_batch import (
    build_observations,
    is_completed,
    load_existing_results,
    resolve_replicate_counts,
    result_sampling_regime,
    select_trials,
)
from run_trial import (
    CONTEXT_TRIAL_TYPES,
    DEFAULT_PROVIDER,
    SAMPLING_REGIMES,
    load_trials,
    max_tokens_for,
    parse_and_validate,
    resolve_model,
    resolve_provider,
    save_result,
    trial_metadata,
)

JOB_DIR = "results/batch_jobs"


def request_key(trial_id, replicate_id):
    """Stable, unique per (trial_id, replicate_id) -- never derived from
    output position, and used as the provider's custom_id/key so collected
    results are always matched back by this, never by batch result order."""
    return f"{trial_id}::{replicate_id}"


def select_eligible_observations(args, provider, model):
    """Exactly the same selection run_batch.py would use: trial filters,
    block-aware --limit, replicate expansion -- then drops anything already
    completed under the current (prompt_sha256, provider, model,
    reasoning_profile) identity. Returns (observations, existing_results)."""
    trials = load_trials(args.trials_file)
    trials = select_trials(trials, args.type, args.id_prefix,
                            set(args.conditions) if args.conditions else None,
                            set(args.contrasts) if args.contrasts else None,
                            set(args.evaluation_regimes) if args.evaluation_regimes else None,
                            args.limit)
    replicates_treatment, replicates_neutral = resolve_replicate_counts(args.replicates, None, None)
    observations = build_observations(trials, replicates_treatment, replicates_neutral)

    existing_results = load_existing_results(args.results_file)
    eligible = []
    for trial, replicate_id in observations:
        key = (trial["trial_id"], model, replicate_id, result_sampling_regime(trial["type"], args.sampling_regime))
        records = existing_results.get(key, [])
        if records and is_completed(records, trial.get("prompt_sha256"), provider, model, args.reasoning_profile):
            continue
        eligible.append((trial, replicate_id))
    return eligible, existing_results


def add_common_selection_arguments(parser):
    parser.add_argument("--trials-file", required=True, help="Trials manifest to read")
    parser.add_argument("--results-file", required=True, help="Results file already-completed observations are checked against, and normalized results are appended to on collect")
    parser.add_argument("--provider", choices=list(model_providers.PROVIDERS), help=f"Model provider (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", help="Override the model (default: the provider's own default -- see model_providers.DEFAULT_MODELS)")
    parser.add_argument("--reasoning-profile", choices=list(model_providers.REASONING_PROFILES_LOGICAL),
                         default=model_providers.DEFAULT_REASONING_PROFILE,
                         help=f"Provider-neutral reasoning level (default: {model_providers.DEFAULT_REASONING_PROFILE})")
    parser.add_argument("--sampling-regime", choices=list(SAMPLING_REGIMES), default="low_variance_primary")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--type", help="Only select this trial type")
    parser.add_argument("--id-prefix", help="Only select trials whose trial_id starts with this prefix")
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument("--contrast", action="append", dest="contrasts")
    parser.add_argument("--evaluation-regime", action="append", dest="evaluation_regimes",
                         choices=["naturalistic", "text_only_invariance"])
    parser.add_argument("--limit", type=int, help="Only select the first N units (whole 4-cell context_pairwise blocks, or one trial for every other type)")
    parser.add_argument("--dry-run", action="store_true", help="Select and build requests, print counts, make no network calls")


def cmd_submit(args):
    provider = resolve_provider(args.provider)
    model = resolve_model(provider, args.model)

    if provider not in model_providers.BATCH_SUPPORTED_PROVIDERS:
        raise SystemExit(
            f"'{provider}' has no native batch endpoint (supported: {list(model_providers.BATCH_SUPPORTED_PROVIDERS)}). "
            f"Use bounded concurrent direct execution instead, e.g.:\n"
            f"    python3 run_batch.py --provider {provider} --model {model} --reasoning-profile {args.reasoning_profile} "
            f"--concurrency 20 --trials-file {args.trials_file} --results-file {args.results_file}"
        )

    eligible, _ = select_eligible_observations(args, provider, model)
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, args.reasoning_profile)
    max_output_tokens = max_tokens_for(provider, eligible[0][0].get("response_format") if eligible else None)

    requests = []
    request_meta = {}
    for trial, replicate_id in eligible:
        key = request_key(trial["trial_id"], replicate_id)
        if key in request_meta:
            raise ValueError(f"Duplicate request key: {key!r}")  # (trial_id, replicate_id) must be unique per submission
        requests.append({"request_key": key, "prompt": trial["prompt"]})
        request_meta[key] = {
            "trial_id": trial["trial_id"],
            "replicate_id": replicate_id,
            "prompt_sha256": trial.get("prompt_sha256"),
        }

    print(f"Provider: {provider}  Model: {model}  Reasoning profile: {args.reasoning_profile} (settings: {reasoning_settings})")
    print(f"Eligible requests (not already completed): {len(requests)}")

    if args.dry_run:
        payloads = [
            model_providers.build_batch_request_payload(provider, model, r["request_key"], r["prompt"], max_output_tokens, reasoning_settings)
            for r in requests
        ]
        assert len({p.get("custom_id") or p.get("key") for p in payloads}) == len(payloads), "duplicate request id in batch payload"
        print(f"Constructed {len(payloads)} provider-native request payload(s). No network calls made.")
        return

    if not requests:
        print("Nothing to submit.")
        return

    submitted = model_providers.submit_batch(provider, model, requests, args.reasoning_profile, max_output_tokens)

    os.makedirs(JOB_DIR, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    job_id = f"{provider}_{submitted['provider_batch_id']}".replace("/", "_")
    job_path = os.path.join(JOB_DIR, f"{job_id}.json")
    job = {
        "provider": provider,
        "provider_batch_id": submitted["provider_batch_id"],
        "requested_model": model,
        "reasoning_profile": args.reasoning_profile,
        "provider_reasoning_settings": reasoning_settings,
        "created_at": created_at,
        "trials_file": args.trials_file,
        "results_file": args.results_file,
        "sampling_regime": args.sampling_regime,
        "request_count": len(requests),
        "requests": request_meta,
        "collected_request_keys": [],
    }
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)

    print(f"Submitted batch job: {submitted['provider_batch_id']}")
    print(f"Job file: {job_path}")


def load_job(job_file):
    with open(job_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_job(job_file, job):
    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)


def cmd_status(args):
    job = load_job(args.job_file)
    status = model_providers.batch_status(job["provider"], job["provider_batch_id"])
    print(f"Provider: {job['provider']}  Batch: {job['provider_batch_id']}")
    print(f"Status: {status['status']}")
    if status.get("raw"):
        print(f"Details: {status['raw']}")
    collected = len(job.get("collected_request_keys", []))
    print(f"Collected so far: {collected} / {job['request_count']}")


def cmd_collect(args):
    job = load_job(args.job_file)
    provider = job["provider"]
    already_collected = set(job.get("collected_request_keys", []))
    pending_keys = set(job["requests"]) - already_collected

    if not pending_keys:
        print("Nothing new to collect (every request in this job has already been collected).")
        return

    results = model_providers.collect_batch(provider, job["provider_batch_id"], pending_keys)
    if not results:
        print("No results available yet for the pending requests.")
        return

    trials_by_id = {t["trial_id"]: t for t in load_trials(job["trials_file"])}
    counts = {"valid": 0, "invalid": 0, "error": 0}

    for key, outcome in results.items():
        meta = job["requests"][key]
        trial = trials_by_id.get(meta["trial_id"])
        timestamp = datetime.now(timezone.utc).isoformat()

        base_result = {
            "trial_id": meta["trial_id"],
            "provider": provider,
            "requested_model": job["requested_model"],
            "model": job["requested_model"],
            "reasoning_profile": job["reasoning_profile"],
            "provider_reasoning_settings": job["provider_reasoning_settings"],
            "execution_mode": "batch",
            "replicate_id": meta["replicate_id"],
            "attempt_id": 1,
            "timestamp": timestamp,
        }
        trial_meta = trial_metadata(trial) if trial is not None and trial["type"] in CONTEXT_TRIAL_TYPES else None

        if "error" in outcome:
            result = {
                **base_result, "response_model": None, "stop_reason": None,
                "input_tokens": None, "output_tokens": None, "response_text": None,
                "parsed_response": None, "validation_error": f"API call failed: {outcome['error']}",
            }
            counts["error"] += 1
        else:
            response_text = outcome["response_text"]
            if trial is not None:
                parsed_response, validation_error = parse_and_validate(
                    response_text, trial["type"], trial.get("choice_mode"), trial.get("rubric", "full"), trial.get("response_format")
                )
            else:
                parsed_response, validation_error = None, "unknown_trial_id: trial no longer exists in the trials file"
            result = {
                **base_result, "response_model": outcome.get("response_model"),
                "stop_reason": outcome.get("stop_reason"), "input_tokens": outcome.get("input_tokens"),
                "output_tokens": outcome.get("output_tokens"), "response_text": response_text,
                "parsed_response": parsed_response, "validation_error": validation_error,
            }
            counts["invalid" if validation_error else "valid"] += 1

        if trial_meta is not None:
            result["trial_meta"] = trial_meta
        result["sampling_regime"] = job["sampling_regime"]

        save_result(result, job["results_file"])
        already_collected.add(key)

    job["collected_request_keys"] = sorted(already_collected)
    save_job(args.job_file, job)

    print(f"Collected {len(results)} result(s): valid={counts['valid']} invalid={counts['invalid']} error={counts['error']}")
    print(f"Still pending: {len(job['requests']) - len(already_collected)}")


def main():
    parser = argparse.ArgumentParser(description="Native provider batch execution: submit / status / collect.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Select eligible trials, build requests, and submit a batch job")
    add_common_selection_arguments(submit_parser)
    submit_parser.set_defaults(func=cmd_submit)

    status_parser = subparsers.add_parser("status", help="Check a submitted batch job's status")
    status_parser.add_argument("--job-file", required=True)
    status_parser.set_defaults(func=cmd_status)

    collect_parser = subparsers.add_parser("collect", help="Idempotently collect normalized results into the results file")
    collect_parser.add_argument("--job-file", required=True)
    collect_parser.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
