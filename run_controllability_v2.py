"""v2 execution CLI: run the treatment or baseline manifest through one
evaluator, with randomised execution order, bounded retries, bounded
concurrency, resumable/idempotent execution, and a production preflight.

Usage:
    # dry run -- prints the planned execution order and counts, makes no calls
    python3 run_controllability_v2.py treatment --dry-run
    python3 run_controllability_v2.py baseline --dry-run

    # preflight only -- validates the frozen design and prints a summary,
    # makes no calls regardless of --production
    python3 run_controllability_v2.py preflight

    # production (frozen, real API calls) -- provider/model/reasoning-profile/
    # replicates/seed/retry-limit are all read from the frozen study config;
    # passing one on the CLI anyway is only accepted if it matches exactly
    python3 run_controllability_v2.py treatment --production --concurrency 32
    python3 run_controllability_v2.py baseline --production --concurrency 32

    # exploratory (real API calls, never frozen-checked, never entering the
    # production results the paper's analysis reads by default)
    python3 run_controllability_v2.py treatment --allow-unfrozen \
        --provider anthropic --model claude-sonnet-5 --reasoning-profile low --replicates 1

A real (non---dry-run) execution must pass exactly one of --production or
--allow-unfrozen; --dry-run remains the default-safe mode and needs neither.

Production and exploratory results are written to entirely separate
directories (item 5) -- results/controllability_v2/production/ and
results/controllability_v2/exploratory/ -- so an exploratory/test run can
never contaminate the production results analyze_controllability_v2.py reads
by default.

Resumable execution (item 4): every planned observation has a stable
observation_id (see controllability_v2_execution.make_observation_id).
Before making a request, the current results file is inspected; an
observation_id already present (success OR a terminal, retries-exhausted
failure -- both are terminal, see run_one_observation) is skipped, never
re-run and never duplicated. Re-running the exact same command after an
interruption therefore executes only what's still missing.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import model_providers
from controllability_v2_deepseek_pricing import DeepSeekPricingGate, format_utc_z, is_deepseek_off_peak, next_off_peak_boundary
from controllability_v2_execution import (
    attach_observation_ids,
    build_evaluator_identity,
    filter_unresumed_plan,
    find_duplicate_observation_ids,
    limit_to_first_n_units,
    load_completed_observation_ids,
    plan_baseline_execution_order,
    plan_treatment_execution_order,
    resolve_production_settings,
    run_one_observation,
)
from controllability_v2_study_config import load_study_config
from controllability_v2_trials import (
    BASELINE_TRIALS_FILE,
    EXPERIMENT_ID,
    TRIALS_FILE,
    assert_baseline_trials_are_well_formed,
    assert_superblocks_are_well_formed,
)
from freeze_controllability_v2 import verify_frozen
from run_trial import load_trials

RESULTS_DIR = "results/controllability_v2"
PRODUCTION_DIR = os.path.join(RESULTS_DIR, "production")
EXPLORATORY_DIR = os.path.join(RESULTS_DIR, "exploratory")

# Analysis (analyze_controllability_v2.py) defaults to these -- the paper's
# analysis must never silently include exploratory results.
TREATMENT_RESULTS_FILE = os.path.join(PRODUCTION_DIR, "treatment_raw.jsonl")
BASELINE_RESULTS_FILE = os.path.join(PRODUCTION_DIR, "baseline_raw.jsonl")
EXPLORATORY_TREATMENT_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "treatment_raw.jsonl")
EXPLORATORY_BASELINE_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "baseline_raw.jsonl")

DEFAULT_CONCURRENCY = 32


# ---------------------------------------------------------------------------
# Thread-safe JSONL writer
# ---------------------------------------------------------------------------

class SafeJsonlWriter:
    """Append-only JSONL writer safe for concurrent use from a thread pool:
    one lock guards the single open file handle, and every write is
    followed by a flush so a killed process leaves whatever was written so
    far durably on disk (required for resumability)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8")

    def write(self, row):
        line = json.dumps(row)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


# ---------------------------------------------------------------------------
# Progress reporting (item 12) -- counters only, never prompt/story text,
# and never touches scheduling/ordering.
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total, already_completed=0, log_every=None):
        self._lock = threading.Lock()
        self.total = total
        self.completed = 0
        self.already_completed = already_completed
        self.successful = 0
        self.terminal_failures = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._start = time.time()
        self.log_every = log_every or max(1, total // 20 or 1)

    def record(self, observation):
        with self._lock:
            self.completed += 1
            if observation["parsing_status"] == "resolved":
                self.successful += 1
            else:
                self.terminal_failures += 1
            for attempt in observation["attempts"]:
                self.total_input_tokens += attempt.get("input_tokens") or 0
                self.total_output_tokens += attempt.get("output_tokens") or 0
            due = self.completed % self.log_every == 0 or self.completed == self.total
        if due:
            self._print()

    def _print(self):
        elapsed = time.time() - self._start
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        queued = self.total - self.completed
        print(
            f"[progress] {self.completed}/{self.total} this run (+{self.already_completed} already done)  "
            f"successful={self.successful}  terminal_failures={self.terminal_failures}  queued={queued}  "
            f"rate={rate:.2f}/s  elapsed={elapsed:.0f}s  "
            f"tokens~ in={self.total_input_tokens} out={self.total_output_tokens}"
        )


# ---------------------------------------------------------------------------
# Result row construction / execution
# ---------------------------------------------------------------------------

def build_result_row(entry, evaluator_identity, observation, unit_id_field):
    return {
        "observation_id": entry["observation_id"],
        "experiment_id": EXPERIMENT_ID,
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


def make_call_fn(provider, model, reasoning_profile, max_output_tokens, sampling_params, pricing_gate=None):
    """pricing_gate (see controllability_v2_deepseek_pricing.DeepSeekPricingGate)
    is consulted immediately before every dispatch, but ONLY for provider
    "deepseek" -- every other provider is completely unaffected, even when a
    gate object is passed in. The check happens on every attempt (a
    retried observation calls call_fn again, and each call re-checks), and
    happens before dispatch, never during an in-flight request -- a request
    already past this point always runs to completion even if a peak
    window begins mid-call."""
    def call_fn(trial):
        if provider == "deepseek" and pricing_gate is not None:
            pricing_gate.wait_until_dispatch_allowed()
        return model_providers.call_model(provider, model, trial["prompt"], max_output_tokens, reasoning_profile, sampling_params)
    return call_fn


def run_plan(plan, evaluator_identity, call_fn, retry_limit, writer, unit_id_field, concurrency, progress):
    """Execute every (already resumption-filtered) plan entry through
    call_fn, writing one result row per completed observation. Concurrency
    only changes WHEN each call executes -- futures are submitted in the
    plan's own (already randomised) order, never regrouped or resorted, so
    the pre-generated execution_order_index remains the sole record of
    experimental ordering; completion order (which is what concurrency
    actually reorders) never enters the saved data."""

    def task(entry):
        observation = run_one_observation(entry["trial"], call_fn, retry_limit)
        row = build_result_row(entry, evaluator_identity, observation, unit_id_field)
        writer.write(row)
        progress.record(observation)

    if concurrency <= 1:
        for entry in plan:
            task(entry)
        return

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, entry) for entry in plan]
        for future in as_completed(futures):
            future.result()  # re-raise anything task() itself didn't catch (run_one_observation catches API errors itself)


# ---------------------------------------------------------------------------
# Production preflight (item 11)
# ---------------------------------------------------------------------------

def run_preflight(study_config, cli_overrides_treatment, cli_overrides_baseline, concurrency, allow_peak_pricing=False):
    """Runs every check item 11 requires and returns (ok, checks, context).
    `checks` is a list of (name, ok, detail) tuples; `context` carries the
    loaded manifests and resolved settings for the caller (and for
    print_preflight_summary) so nothing is loaded twice.

    The DeepSeek pricing-window state (peak/off-peak, and the override flag)
    is informational only -- it is never added to `checks` and never
    affects `ok`: preflight must pass regardless of the current time."""
    checks = []
    context = {}

    now = datetime.now(timezone.utc)
    context["pricing_window_off_peak"] = is_deepseek_off_peak(now)
    context["pricing_window_next_off_peak"] = None if context["pricing_window_off_peak"] else next_off_peak_boundary(now)
    context["allow_peak_pricing"] = allow_peak_pricing

    is_frozen, reason = verify_frozen()
    checks.append(("study_lock_exists_and_hashes_validate", is_frozen, reason or "ok"))

    treatment_trials = load_trials(study_config["treatment_manifest_file"])
    baseline_trials = load_trials(study_config["baseline_manifest_file"])
    context["treatment_trials"] = treatment_trials
    context["baseline_trials"] = baseline_trials

    treatment_ids = {t["trial_id"] for t in treatment_trials}
    checks.append(("treatment_manifest_has_2640_unique_cells", len(treatment_ids) == 2640, f"found {len(treatment_ids)}"))

    baseline_ids = {t["trial_id"] for t in baseline_trials}
    checks.append(("baseline_manifest_has_132_unique_cells", len(baseline_ids) == 132, f"found {len(baseline_ids)}"))

    try:
        assert_superblocks_are_well_formed(treatment_trials)
        checks.append(("every_treatment_superblock_structurally_complete", True, "ok"))
    except (ValueError, KeyError) as e:
        checks.append(("every_treatment_superblock_structurally_complete", False, f"{type(e).__name__}: {e}"))

    try:
        assert_baseline_trials_are_well_formed(baseline_trials)
        checks.append(("every_baseline_unit_structurally_complete", True, "ok"))
    except (ValueError, KeyError) as e:
        checks.append(("every_baseline_unit_structurally_complete", False, str(e)))

    try:
        treatment_settings = resolve_production_settings(cli_overrides_treatment, study_config, "treatment")
        baseline_settings = resolve_production_settings(cli_overrides_baseline, study_config, "baseline")
        checks.append(("cli_settings_match_frozen_evaluator_and_replicate_counts", True, "ok"))
        context["treatment_settings"] = treatment_settings
        context["baseline_settings"] = baseline_settings
    except ValueError as e:
        checks.append(("cli_settings_match_frozen_evaluator_and_replicate_counts", False, str(e)))
        context["treatment_settings"] = context["baseline_settings"] = None

    dup_treatment = find_duplicate_observation_ids(TREATMENT_RESULTS_FILE)
    dup_baseline = find_duplicate_observation_ids(BASELINE_RESULTS_FILE)
    n_dup = len(dup_treatment) + len(dup_baseline)
    checks.append(("production_output_has_no_duplicate_observation_ids", n_dup == 0, "ok" if n_dup == 0 else f"{n_dup} duplicate id(s)"))

    provider = study_config["primary_evaluator"]["provider"]
    try:
        model_providers.get_api_key(provider)
        checks.append(("api_credential_present", True, "ok"))
    except RuntimeError as e:
        checks.append(("api_credential_present", False, str(e)))

    ok = all(c[1] for c in checks)
    context["concurrency"] = concurrency
    return ok, checks, context


def print_preflight_summary(study_config, context):
    primary = study_config["primary_evaluator"]
    treatment_trials, baseline_trials = context["treatment_trials"], context["baseline_trials"]
    treatment_replicates = study_config["treatment_replicate_count"]
    baseline_replicates = study_config["baseline_replicate_count"]
    planned_treatment = len(treatment_trials) * treatment_replicates
    planned_baseline = len(baseline_trials) * baseline_replicates

    experiment_id = study_config["experiment_id"]
    evaluator_id = primary["evaluator_id"]
    completed_treatment = len(load_completed_observation_ids(TREATMENT_RESULTS_FILE))
    completed_baseline = len(load_completed_observation_ids(BASELINE_RESULTS_FILE))

    print("=== Preflight summary ===")
    print(f"Model: {primary['requested_model']} (provider={primary['provider']})")
    print(f"Reasoning profile: {primary['reasoning_profile']}")
    print(f"Baseline replicate count: {baseline_replicates}")
    print(f"Treatment replicate count: {treatment_replicates}")
    print(f"Planned baseline observations: {planned_baseline}")
    print(f"Planned treatment observations: {planned_treatment}")
    print(f"Already completed (production): baseline={completed_baseline}  treatment={completed_treatment}")
    print(f"Remaining: baseline={planned_baseline - completed_baseline}  treatment={planned_treatment - completed_treatment}")
    print(f"Concurrency: {context['concurrency']}")
    print(f"Output directory: {PRODUCTION_DIR}")

    if context["pricing_window_off_peak"]:
        print("DeepSeek pricing window: OFF-PEAK")
    else:
        print("DeepSeek pricing window: PEAK")
        print(f"Next off-peak: {format_utc_z(context['pricing_window_next_off_peak'])}")
    print(f"Peak-pricing override: {'enabled' if context['allow_peak_pricing'] else 'disabled'}")


def cmd_preflight(args):
    study_config = load_study_config()
    ok, checks, context = run_preflight(study_config, {}, {}, args.concurrency, args.allow_peak_pricing)
    for name, check_ok, detail in checks:
        print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
    print_preflight_summary(study_config, context)
    if not ok:
        raise SystemExit("Preflight FAILED -- see the FAIL line(s) above. No API calls were made.")
    print("Preflight PASSED. No API calls were made.")


# ---------------------------------------------------------------------------
# treatment / baseline
# ---------------------------------------------------------------------------

def cmd_treatment(args):
    _run(args, TRIALS_FILE, "superblock_id", plan_treatment_execution_order, "treatment")


def cmd_baseline(args):
    _run(args, BASELINE_TRIALS_FILE, "block_id", plan_baseline_execution_order, "baseline")


def _cli_overrides(args):
    return {"provider": args.provider, "model": args.model, "reasoning_profile": args.reasoning_profile,
            "replicates": args.replicates, "seed": args.seed, "retry_limit": args.retry_limit}


def _run(args, trials_file, unit_id_field, plan_fn, unit_kind):
    if args.production and args.allow_unfrozen:
        raise SystemExit("--production and --allow-unfrozen are mutually exclusive.")
    if not args.dry_run and not args.production and not args.allow_unfrozen:
        raise SystemExit(
            "A real (non---dry-run) execution must pass exactly one of --production "
            "(frozen production run) or --allow-unfrozen (exploratory run)."
        )

    study_config = None
    try:
        study_config = load_study_config()
    except FileNotFoundError:
        pass

    cli_overrides = _cli_overrides(args)
    use_frozen = args.production or (args.dry_run and not args.allow_unfrozen and study_config is not None and not args.provider)

    if use_frozen:
        if study_config is None:
            raise SystemExit("No study config found at data/controllability_v2_study_config.json -- run controllability_v2_study_config.py first.")
        if args.production:
            ok, checks, context = run_preflight(
                study_config,
                cli_overrides if unit_kind == "treatment" else {},
                cli_overrides if unit_kind == "baseline" else {},
                args.concurrency,
                args.allow_peak_pricing,
            )
            for name, check_ok, detail in checks:
                print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
            print_preflight_summary(study_config, context)
            if not ok:
                raise SystemExit("Preflight FAILED -- refusing to start a production run. No API calls were made.")
            settings = context["treatment_settings"] if unit_kind == "treatment" else context["baseline_settings"]
        else:
            settings = resolve_production_settings(cli_overrides, study_config, unit_kind)
        results_file = TREATMENT_RESULTS_FILE if unit_kind == "treatment" else BASELINE_RESULTS_FILE
        output_label = "production"
    else:
        if not args.provider or not args.model:
            raise SystemExit("--provider and --model are required for an exploratory run (or a dry-run with no study config yet).")
        settings = {
            "provider": args.provider, "model": args.model,
            "reasoning_profile": args.reasoning_profile or model_providers.DEFAULT_REASONING_PROFILE,
            "replicates": args.replicates or 1,
            "seed": args.seed if args.seed is not None else 0,
            "retry_limit": args.retry_limit if args.retry_limit is not None else 3,
            "max_output_tokens": model_providers.default_max_output_tokens(args.provider),
            "evaluator_id": f"{args.provider}__{args.model}__{args.reasoning_profile or model_providers.DEFAULT_REASONING_PROFILE}",
        }
        results_file = EXPLORATORY_TREATMENT_RESULTS_FILE if unit_kind == "treatment" else EXPLORATORY_BASELINE_RESULTS_FILE
        output_label = "exploratory"

    provider, model, reasoning_profile = settings["provider"], settings["model"], settings["reasoning_profile"]
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, reasoning_profile)
    sampling_params = {}  # provider-default sampling throughout v2 -- see STUDY_PROTOCOL_V2.md

    trials = load_trials(trials_file)
    trials = limit_to_first_n_units(trials, unit_id_field, args.limit)
    plan = plan_fn(trials, settings["replicates"], settings["seed"])
    attach_observation_ids(plan, EXPERIMENT_ID, settings["evaluator_id"])

    already_completed = load_completed_observation_ids(results_file)
    remaining_plan = filter_unresumed_plan(plan, already_completed)

    print(f"Mode: {output_label}  Output: {results_file}")
    print(f"Provider: {provider}  Model: {model}  Reasoning profile: {reasoning_profile}")
    print(f"Replicates: {settings['replicates']}  Seed: {settings['seed']}  Retry limit: {settings['retry_limit']}")
    print(f"Planned observations: {len(plan)}  Already completed: {len(plan) - len(remaining_plan)}  Remaining: {len(remaining_plan)}")

    if args.dry_run:
        print("Dry run: no network calls made.")
        print(f"First 5 execution-order entries: {[(p['execution_order_index'], p['trial_id']) for p in remaining_plan[:5]]}")
        return

    if not remaining_plan:
        print("Nothing to do -- every planned observation already has a terminal result.")
        return

    evaluator_identity = build_evaluator_identity(
        provider=provider, requested_model=model, reasoning_profile=reasoning_profile,
        provider_reasoning_settings=reasoning_settings, sampling_settings=sampling_params, run_id=args.run_id,
        max_output_tokens=settings["max_output_tokens"], retry_limit=settings["retry_limit"],
    )
    # The gate is only ever consulted for provider == "deepseek" (see
    # make_call_fn); constructing it unconditionally here is harmless and
    # keeps the wiring simple for every other provider.
    pricing_gate = DeepSeekPricingGate(allow_peak=args.allow_peak_pricing)
    call_fn = make_call_fn(provider, model, reasoning_profile, settings["max_output_tokens"], sampling_params, pricing_gate)
    progress = ProgressTracker(total=len(remaining_plan), already_completed=len(plan) - len(remaining_plan))

    with SafeJsonlWriter(results_file) as writer:
        run_plan(remaining_plan, evaluator_identity, call_fn, settings["retry_limit"], writer, unit_id_field, args.concurrency, progress)

    print(f"Done. Results appended to {results_file}")


def add_common_arguments(parser):
    parser.add_argument("--provider", choices=list(model_providers.PROVIDERS), default=None,
                         help="Required for --allow-unfrozen; ignored (read from the frozen config) for --production unless "
                              "given, in which case it must match exactly")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-profile", choices=list(model_providers.REASONING_PROFILES_LOGICAL), default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--retry-limit", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Wave/run identifier recorded on every result row")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only select the first N units (whole 8-cell superblocks for treatment, "
                              "whole 2-cell baseline units for baseline) -- never a partial unit")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                         help=f"Bounded concurrent request execution (default: {DEFAULT_CONCURRENCY}). Changes transport "
                              "only -- never the planned/randomised observation set or order.")
    parser.add_argument("--production", action="store_true",
                         help="Frozen production run: settings are read from (and validated against) the frozen study "
                              "config; runs the item-11 preflight first; writes to results/controllability_v2/production/.")
    parser.add_argument("--allow-unfrozen", action="store_true",
                         help="Exploratory real run: no frozen-design check; writes to "
                              "results/controllability_v2/exploratory/, never into the production results.")
    parser.add_argument("--allow-peak-pricing", action="store_true",
                         help="Bypass the DeepSeek off-peak pricing guard (default: false). Applies only to the "
                              "DeepSeek provider; every other provider is unaffected either way.")
    parser.add_argument("--dry-run", action="store_true", help="Select and plan, print counts, make no network calls (default-safe mode)")


def main():
    parser = argparse.ArgumentParser(description="v2 context-controllability execution CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    treatment_parser = subparsers.add_parser("treatment", help="Run the primary treatment manifest")
    add_common_arguments(treatment_parser)
    treatment_parser.set_defaults(func=cmd_treatment)

    baseline_parser = subparsers.add_parser("baseline", help="Run the blind baseline manifest")
    add_common_arguments(baseline_parser)
    baseline_parser.set_defaults(func=cmd_baseline)

    preflight_parser = subparsers.add_parser("preflight", help="Run every production preflight check and print a summary; makes no API calls")
    preflight_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    preflight_parser.add_argument("--allow-peak-pricing", action="store_true",
                                   help="Reflects what a subsequent production run's override would be; preflight itself never dispatches.")
    preflight_parser.set_defaults(func=cmd_preflight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
