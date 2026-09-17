"""v2 execution CLI: run the treatment or baseline manifest through one
evaluator, with randomised execution order, bounded retries, and full
evaluator-identity/attempt bookkeeping (see controllability_v2_execution.py).

Usage:
    # dry run -- prints the planned execution order and counts, makes no calls
    python3 run_controllability_v2.py treatment --provider anthropic --model claude-sonnet-5 \
        --reasoning-profile low --replicates 3 --seed 20260917 --dry-run

    python3 run_controllability_v2.py baseline --provider anthropic --model claude-sonnet-5 \
        --reasoning-profile low --replicates 3 --seed 20260917 --dry-run

Real execution (not --dry-run) refuses to run unless the study design is
frozen and matches its lock file (see freeze_controllability_v2.verify_frozen)
-- pass --allow-unfrozen to explicitly bypass this for exploratory/test runs
that are never meant to feed the frozen study's primary analysis.

Baseline and treatment are always written to SEPARATE results files (item 5:
"Baseline results must be collected independently from treatment results").
Every row records the full evaluator identity, the execution-order/replicate/
superblock bookkeeping from the plan, and every attempt made for that
planned observation (item 8) -- never just the final one.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import model_providers
from controllability_v2_execution import (
    build_evaluator_identity,
    limit_to_first_n_units,
    plan_baseline_execution_order,
    plan_treatment_execution_order,
    run_one_observation,
)
from controllability_v2_study_config import load_study_config
from controllability_v2_trials import (
    BASELINE_TRIALS_FILE,
    TRIALS_FILE,
)
from freeze_controllability_v2 import verify_frozen
from run_trial import load_trials

RESULTS_DIR = "results/controllability_v2"
TREATMENT_RESULTS_FILE = os.path.join(RESULTS_DIR, "treatment_raw.jsonl")
BASELINE_RESULTS_FILE = os.path.join(RESULTS_DIR, "baseline_raw.jsonl")


def save_result(row, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def make_call_fn(provider, model, reasoning_profile, max_output_tokens, sampling_params):
    def call_fn(trial):
        return model_providers.call_model(provider, model, trial["prompt"], max_output_tokens, reasoning_profile, sampling_params)
    return call_fn


def run_plan(plan, evaluator_identity, call_fn, retry_limit, results_file, unit_id_field):
    for entry in plan:
        observation = run_one_observation(entry["trial"], call_fn, retry_limit)
        row = {
            "trial_id": entry["trial_id"],
            unit_id_field: entry[unit_id_field],
            "replicate_number": entry["replicate_number"],
            "execution_order_index": entry["execution_order_index"],
            "random_seed": entry["random_seed"],
            "evaluator": evaluator_identity,
            "trial_meta": {k: v for k, v in entry["trial"].items() if k != "prompt"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: observation[k] for k in (
                "attempts", "total_attempts", "first_attempt_status", "first_valid_response",
                "parsing_status", "refusal_status", "api_error_status",
            )},
        }
        save_result(row, results_file)


def cmd_treatment(args):
    _run(args, TRIALS_FILE, TREATMENT_RESULTS_FILE, "superblock_id", plan_treatment_execution_order)


def cmd_baseline(args):
    _run(args, BASELINE_TRIALS_FILE, BASELINE_RESULTS_FILE, "block_id", plan_baseline_execution_order)


def _run(args, trials_file, results_file, unit_id_field, plan_fn):
    provider = args.provider
    model = args.model
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, args.reasoning_profile)
    max_output_tokens = model_providers.default_max_output_tokens(provider)
    sampling_params = {}  # provider-default sampling throughout v2 -- see STUDY_PROTOCOL_V2.md

    trials = load_trials(trials_file)
    trials = limit_to_first_n_units(trials, unit_id_field, args.limit)
    plan = plan_fn(trials, args.replicates, args.seed)

    print(f"Provider: {provider}  Model: {model}  Reasoning profile: {args.reasoning_profile}")
    print(f"Replicates: {args.replicates}  Seed: {args.seed}  Retry limit: {args.retry_limit}")
    print(f"Planned observations: {len(plan)}")

    if args.dry_run:
        print("Dry run: no network calls made.")
        print(f"First 5 execution-order entries: {[ (p['execution_order_index'], p['trial_id']) for p in plan[:5] ]}")
        return

    if not args.allow_unfrozen:
        ok, reason = verify_frozen()
        if not ok:
            raise SystemExit(
                f"Refusing to run: study design is not frozen or no longer matches its lock ({reason}). "
                f"Run freeze_controllability_v2.py first, or pass --allow-unfrozen for an exploratory run "
                f"that will never feed the frozen study's primary analysis."
            )

    evaluator_identity = build_evaluator_identity(
        provider=provider, requested_model=model, reasoning_profile=args.reasoning_profile,
        provider_reasoning_settings=reasoning_settings, sampling_settings=sampling_params, run_id=args.run_id,
    )
    call_fn = make_call_fn(provider, model, args.reasoning_profile, max_output_tokens, sampling_params)
    run_plan(plan, evaluator_identity, call_fn, args.retry_limit, results_file, unit_id_field)
    print(f"Done. Results appended to {results_file}")


def add_common_arguments(parser, default_replicates, default_retry_limit):
    parser.add_argument("--provider", required=True, choices=list(model_providers.PROVIDERS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-profile", choices=list(model_providers.REASONING_PROFILES_LOGICAL),
                         default=model_providers.DEFAULT_REASONING_PROFILE)
    parser.add_argument("--replicates", type=int, default=default_replicates)
    parser.add_argument("--seed", type=int, default=None, help="Default: random_seed from the study config")
    parser.add_argument("--retry-limit", type=int, default=default_retry_limit)
    parser.add_argument("--run-id", default=None, help="Wave/run identifier recorded on every result row")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only select the first N units (whole 8-cell superblocks for treatment, "
                              "whole 2-cell baseline units for baseline) -- never a partial unit")
    parser.add_argument("--allow-unfrozen", action="store_true",
                         help="Bypass the frozen-design check for a real (non-dry-run) execution -- "
                              "only for exploratory runs that will never feed the frozen study's primary analysis")
    parser.add_argument("--dry-run", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="v2 context-controllability execution CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    try:
        config = load_study_config()
        default_replicates_treatment = config["treatment_replicate_count"]
        default_replicates_baseline = config["baseline_replicate_count"]
        default_seed = config["random_seed"]
        default_retry_limit = config["retry_limit"]
    except FileNotFoundError:
        default_replicates_treatment = default_replicates_baseline = 1
        default_seed = 0
        default_retry_limit = 3

    treatment_parser = subparsers.add_parser("treatment", help="Run the primary treatment manifest")
    add_common_arguments(treatment_parser, default_replicates_treatment, default_retry_limit)
    treatment_parser.set_defaults(func=cmd_treatment, _default_seed=default_seed)

    baseline_parser = subparsers.add_parser("baseline", help="Run the blind baseline manifest")
    add_common_arguments(baseline_parser, default_replicates_baseline, default_retry_limit)
    baseline_parser.set_defaults(func=cmd_baseline, _default_seed=default_seed)

    args = parser.parse_args()
    if args.seed is None:
        args.seed = args._default_seed
    args.func(args)


if __name__ == "__main__":
    main()
