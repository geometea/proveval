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
import controllability_v2_recovery as recovery
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
    BASELINE_TEXT_ONLY_TRIALS_FILE,
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
# Wave 2 recovery: entirely separate from PRODUCTION_DIR -- recovery/
# validation/merged outputs are never written into, and never overwrite,
# the Wave 1 production directory above.
RECOVERY_DIR = os.path.join(RESULTS_DIR, "wave2_recovery")

# Analysis (analyze_controllability_v2.py) defaults to these -- the paper's
# analysis must never silently include exploratory results.
TREATMENT_RESULTS_FILE = os.path.join(PRODUCTION_DIR, "treatment_raw.jsonl")
BASELINE_RESULTS_FILE = os.path.join(PRODUCTION_DIR, "baseline_raw.jsonl")
EXPLORATORY_TREATMENT_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "treatment_raw.jsonl")
EXPLORATORY_BASELINE_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "baseline_raw.jsonl")
EXPLORATORY_BASELINE_TEXT_ONLY_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "baseline_text_only_raw.jsonl")

# Wave 2 recovery outputs (see controllability_v2_recovery.py). The Wave 1
# raw files above are only ever READ by the recovery/merge commands below --
# nothing in this module ever opens them for writing. baseline_text_only is
# a brand-new Wave 2 condition (never part of Wave 1), so its production
# raw file lives under RECOVERY_DIR too -- NOT under PRODUCTION_DIR -- so
# that the GitHub Actions recovery job's own cache restore/save (which only
# covers RECOVERY_DIR) actually preserves its progress across an
# interrupted/resumed recovery run instead of silently losing it every time.
BASELINE_TEXT_ONLY_RESULTS_FILE = os.path.join(RECOVERY_DIR, "baseline_text_only_raw.jsonl")
RECOVERY_RESULTS_FILE = os.path.join(RECOVERY_DIR, "recovery_raw.jsonl")
VALIDATION_RESULTS_FILE = os.path.join(RECOVERY_DIR, "validation_raw.jsonl")
MERGED_TREATMENT_RESULTS_FILE = os.path.join(RECOVERY_DIR, "merged_treatment_raw.jsonl")
MERGED_BASELINE_RESULTS_FILE = os.path.join(RECOVERY_DIR, "merged_baseline_raw.jsonl")
MERGE_DIAGNOSTICS_FILE = os.path.join(RECOVERY_DIR, "merge_diagnostics.json")
VALIDATION_DIAGNOSTICS_FILE = os.path.join(RECOVERY_DIR, "validation_diagnostics.json")
VALIDATION_AGREEMENT_BY_BIN_FILE = os.path.join(RECOVERY_DIR, "validation_agreement_by_bin.csv")
VALIDATION_CONTEXT_EFFECT_FILE = os.path.join(RECOVERY_DIR, "validation_context_effect_comparison.csv")

DEFAULT_CONCURRENCY = 32

# The real frozen v2 design's Wave 1 planned-observation total (2640
# treatment trials * 10 + 132 baseline trials * 10) -- cmd_merge's hard
# completeness gate checks the merged dataset against this literal, known
# constant of the CURRENT frozen design (the same style of hardcoded
# structural invariant run_preflight already uses for the 2640/132 unique-
# cell counts above), not merely "whatever this run's own arithmetic
# produced" -- so a --source-results pointed at the wrong design, or a
# study config that has silently drifted, is still caught.
EXPECTED_WAVE1_PLANNED_TOTAL = 27720
# The new no-context/text_only condition's known structural size: 132
# unique cells (66 story pairs x 2 positions) x 10 replicates each.
EXPECTED_BASELINE_TEXT_ONLY_TOTAL = 1320
EXPECTED_BASELINE_TEXT_ONLY_UNIQUE_CELLS = 132


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
# Wave 2 recovery: preflight, recovery run, new baseline_text_only condition,
# validation-duplicate run, and the merge step. See controllability_v2_recovery.py
# for the pure logic; every function below is orchestration/IO around it.
# Nothing here ever opens a Wave 1 production raw file for writing.
# ---------------------------------------------------------------------------

def _load_wave1_planned_index(study_config):
    treatment_trials = load_trials(study_config["treatment_manifest_file"])
    baseline_trials = load_trials(study_config["baseline_manifest_file"])
    evaluator_id = study_config["primary_evaluator"]["evaluator_id"]
    planned_index = recovery.build_wave1_planned_index(
        treatment_trials, baseline_trials, evaluator_id,
        study_config["treatment_replicate_count"], study_config["baseline_replicate_count"],
    )
    trials_by_id = {t["trial_id"]: t for t in treatment_trials + baseline_trials}
    return treatment_trials, baseline_trials, planned_index, trials_by_id


def run_recovery_preflight(study_config, source_results_dir, allow_peak_pricing=False):
    """Every check makes no API calls. Returns (ok, checks, context)."""
    checks = []
    context = {"allow_peak_pricing": allow_peak_pricing, "primary": study_config["primary_evaluator"]}

    try:
        treatment_trials, baseline_trials, planned_index, trials_by_id = _load_wave1_planned_index(study_config)
        checks.append(("wave1_planned_index_built", True, f"{len(planned_index)} planned observations"))
    except (FileNotFoundError, ValueError) as e:
        checks.append(("wave1_planned_index_built", False, str(e)))
        return False, checks, context

    checks.append(("wave1_planned_count_matches_frozen_design", len(planned_index) == 27720, f"found {len(planned_index)}"))

    try:
        treatment_rows, baseline_rows = recovery.load_wave1_raw_rows(source_results_dir)
        checks.append(("source_results_present", True, "ok"))
    except FileNotFoundError as e:
        checks.append(("source_results_present", False, str(e)))
        return False, checks, context

    classification = recovery.classify_wave1_rows(treatment_rows + baseline_rows, planned_index)
    try:
        recovery.validate_wave1_consistency(classification)
        checks.append(("wave1_internally_consistent", True, "ok"))
    except ValueError as e:
        checks.append(("wave1_internally_consistent", False, str(e)))

    recovery_plan = recovery.build_recovery_plan(classification["unresolved_ids"], planned_index, trials_by_id)
    try:
        recovery.assert_recovery_plan_matches_planned_index(recovery_plan, planned_index)
        checks.append(("recovery_creates_no_new_replicates", True, "ok"))
    except ValueError as e:
        checks.append(("recovery_creates_no_new_replicates", False, str(e)))

    baseline_text_only_trials = load_trials(BASELINE_TEXT_ONLY_TRIALS_FILE)
    n_new_baseline = len(baseline_text_only_trials) * study_config["baseline_replicate_count"]
    checks.append(("new_baseline_text_only_count_is_1320", n_new_baseline == 1320, f"found {n_new_baseline}"))

    validation_sample = recovery.select_validation_sample(
        list(classification["valid_by_id"].values()), planned_index, seed=study_config["random_seed"],
    )

    ok = all(c[1] for c in checks)
    context.update({
        "source_run_id": recovery.SOURCE_RUN_ID,
        "wave1_max_output_tokens": study_config.get("max_output_tokens"),
        "n_planned": len(planned_index),
        "n_valid": len(classification["valid_by_id"]),
        "n_unresolved": len(classification["unresolved_ids"]),
        "n_new_baseline_text_only": n_new_baseline,
        "n_validation": len(validation_sample),
        "planned_index": planned_index,
        "classification": classification,
    })
    return ok, checks, context


def print_recovery_preflight_summary(context):
    primary = context["primary"]
    total = context["n_unresolved"] + context["n_new_baseline_text_only"] + context["n_validation"]
    print(f"Source run ID: {context['source_run_id']}")
    print()
    print(f"Wave 1 planned observations: {context['n_planned']}")
    print(f"Wave 1 valid retained: {context['n_valid']}")
    print(f"Wave 1 unresolved: {context['n_unresolved']}")
    print(f"Recovery observations planned: {context['n_unresolved']}")
    print()
    print(f"New baseline_text_only observations: {context['n_new_baseline_text_only']}")
    print(f"Validation duplicates planned: {context['n_validation']}")
    print(f"Total Wave 2 model calls planned: {total}")
    print()
    print(f"Model: {primary['requested_model']}")
    print(f"Reasoning effort: {primary['reasoning_profile']}")
    print(f"Wave 1 max output tokens: {context['wave1_max_output_tokens']}")
    print(f"Wave 2 max output tokens: {recovery.RECOVERY_MAX_OUTPUT_TOKENS}")
    print(f"Peak-pricing override: {'enabled' if context['allow_peak_pricing'] else 'disabled'}")


def cmd_recovery_preflight(args):
    study_config = load_study_config()
    ok, checks, context = run_recovery_preflight(study_config, args.source_results, args.allow_peak_pricing)
    for name, check_ok, detail in checks:
        print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise SystemExit("Recovery preflight FAILED -- see the FAIL line(s) above. No API calls were made.")
    print_recovery_preflight_summary(context)
    print("Recovery preflight PASSED. No API calls were made.")


def _wave2_call_fn(study_config, pricing_gate):
    primary = study_config["primary_evaluator"]
    provider, model, reasoning_profile = primary["provider"], primary["requested_model"], primary["reasoning_profile"]
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, reasoning_profile)
    call_fn = make_call_fn(provider, model, reasoning_profile, recovery.RECOVERY_MAX_OUTPUT_TOKENS, {}, pricing_gate)
    evaluator_identity = build_evaluator_identity(
        provider=provider, requested_model=model, reasoning_profile=reasoning_profile,
        provider_reasoning_settings=reasoning_settings, sampling_settings={},
        max_output_tokens=recovery.RECOVERY_MAX_OUTPUT_TOKENS, retry_limit=study_config["retry_limit"],
    )
    return call_fn, evaluator_identity


def cmd_recovery_run(args):
    study_config = load_study_config()
    is_frozen, reason = verify_frozen()
    if not is_frozen:
        raise SystemExit(f"Refusing to run recovery against an unfrozen/mismatched study design: {reason}")

    _treatment_trials, _baseline_trials, planned_index, trials_by_id = _load_wave1_planned_index(study_config)
    treatment_rows, baseline_rows = recovery.load_wave1_raw_rows(args.source_results)
    classification = recovery.classify_wave1_rows(treatment_rows + baseline_rows, planned_index)
    recovery.validate_wave1_consistency(classification)

    plan = recovery.build_recovery_plan(classification["unresolved_ids"], planned_index, trials_by_id)
    recovery.assert_recovery_plan_matches_planned_index(plan, planned_index)

    # Wave 2 resume semantics (deliberately NOT load_completed_observation_ids):
    # only a genuinely VALID prior recovery answer counts as "done" -- a
    # 402/429/network failure, an exhausted-retries non-answer, or a
    # malformed response must remain eligible for this same command to
    # retry on a later invocation (e.g. a resumed GitHub Actions job).
    already_done = recovery.load_valid_completed_observation_ids(RECOVERY_RESULTS_FILE)
    remaining = [entry for entry in plan if entry["observation_id"] not in already_done]
    print(f"Recovery observations planned: {len(plan)}  Already completed (valid answer): {len(plan) - len(remaining)}  Remaining: {len(remaining)}")

    if args.dry_run:
        print("Dry run: no network calls made.")
        return
    if not remaining:
        print("Nothing to do -- every planned recovery observation already has a terminal result.")
        return

    call_fn, evaluator_identity = _wave2_call_fn(study_config, DeepSeekPricingGate(allow_peak=args.allow_peak_pricing))
    progress = ProgressTracker(total=len(remaining), already_completed=len(plan) - len(remaining))

    def task(entry):
        observation = run_one_observation(entry["trial"], call_fn, study_config["retry_limit"])
        row = recovery.build_recovery_result_row(entry, evaluator_identity, observation)
        writer.write(row)
        progress.record(observation)

    with SafeJsonlWriter(RECOVERY_RESULTS_FILE) as writer:
        if args.concurrency <= 1:
            for entry in remaining:
                task(entry)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(task, entry) for entry in remaining]
                for future in as_completed(futures):
                    future.result()

    print(f"Done. Recovery results appended to {RECOVERY_RESULTS_FILE}")


def cmd_validation_run(args):
    study_config = load_study_config()
    is_frozen, reason = verify_frozen()
    if not is_frozen:
        raise SystemExit(f"Refusing to run validation against an unfrozen/mismatched study design: {reason}")

    _treatment_trials, _baseline_trials, planned_index, trials_by_id = _load_wave1_planned_index(study_config)
    treatment_rows, baseline_rows = recovery.load_wave1_raw_rows(args.source_results)
    classification = recovery.classify_wave1_rows(treatment_rows + baseline_rows, planned_index)

    sample = recovery.select_validation_sample(
        list(classification["valid_by_id"].values()), planned_index,
        sample_size=recovery.VALIDATION_SAMPLE_SIZE, seed=study_config["random_seed"],
    )
    plan = recovery.build_validation_plan(sample, trials_by_id)
    for entry in plan:
        entry["observation_id"] = entry["validation_observation_id"]  # resumability key only -- never a real planned id

    # Same Wave 2 resume semantics as recovery-run: only a genuinely valid
    # rerun answer counts as "done" -- a failed validation attempt remains
    # eligible for a later invocation of this same command.
    already_done = recovery.load_valid_completed_observation_ids(VALIDATION_RESULTS_FILE)
    remaining = [entry for entry in plan if entry["observation_id"] not in already_done]
    print(f"Validation observations planned: {len(plan)}  Already completed (valid answer): {len(plan) - len(remaining)}  Remaining: {len(remaining)}")

    if args.dry_run:
        print("Dry run: no network calls made.")
        return
    if not remaining:
        print("Nothing to do -- every planned validation observation already has a terminal result.")
        return

    call_fn, evaluator_identity = _wave2_call_fn(study_config, DeepSeekPricingGate(allow_peak=args.allow_peak_pricing))
    progress = ProgressTracker(total=len(remaining), already_completed=len(plan) - len(remaining))

    def task(entry):
        observation = run_one_observation(entry["trial"], call_fn, study_config["retry_limit"])
        row = recovery.build_validation_result_row(entry, evaluator_identity, observation)
        row["observation_id"] = row["validation_observation_id"]
        writer.write(row)
        progress.record(observation)

    with SafeJsonlWriter(VALIDATION_RESULTS_FILE) as writer:
        if args.concurrency <= 1:
            for entry in remaining:
                task(entry)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(task, entry) for entry in remaining]
                for future in as_completed(futures):
                    future.result()

    print(f"Done. Validation results appended to {VALIDATION_RESULTS_FILE}")


def cmd_validation_report(args):
    """Pure post-processing over an already-collected validation_raw.jsonl
    -- makes no API calls. Reuses controllability_v2_recovery.compute_validation_agreement/
    compute_validation_context_effect_comparison rather than duplicating any
    logic. These validation duplicates are diagnostic only -- this command
    never writes into, and never affects, the primary merged dataset or the
    treatment/baseline/baseline_text_only analysis."""
    from analyze import write_csv

    rows = recovery.load_jsonl(args.validation_results) if os.path.exists(args.validation_results) else []
    n_total = len(rows)
    n_resolved = sum(1 for r in rows if r.get("parsing_status") == "resolved")
    n_failed = n_total - n_resolved

    agreement = recovery.compute_validation_agreement(rows)
    context_effects = recovery.compute_validation_context_effect_comparison(rows)

    os.makedirs(RECOVERY_DIR, exist_ok=True)
    summary = {
        "n_validation_attempted": n_total,
        "n_validation_resolved": n_resolved,
        "n_validation_failed": n_failed,
        **agreement,
    }
    with open(VALIDATION_DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bin_rows = [
        {"reasoning_token_bin": b, "n": agreement["n_by_reasoning_token_bin"].get(b, 0), "agreement_rate": rate}
        for b, rate in agreement["agreement_by_reasoning_token_bin"].items()
    ]
    if bin_rows:
        write_csv(bin_rows, ["reasoning_token_bin", "n", "agreement_rate"], VALIDATION_AGREEMENT_BY_BIN_FILE)
    if context_effects:
        write_csv(context_effects, list(context_effects[0].keys()), VALIDATION_CONTEXT_EFFECT_FILE)

    print(f"Validation attempted: {n_total}  resolved: {n_resolved}  failed: {n_failed}")
    print(f"Overall A/B agreement rate: {agreement['agreement_rate']}")
    print(f"Aggregate preference shift: {agreement['aggregate_preference_shift']}")
    print(f"Context-effect comparisons calculable: {len(context_effects)}")
    print(f"Wrote {VALIDATION_DIAGNOSTICS_FILE}")
    if bin_rows:
        print(f"Wrote {VALIDATION_AGREEMENT_BY_BIN_FILE}")
    if context_effects:
        print(f"Wrote {VALIDATION_CONTEXT_EFFECT_FILE}")


def cmd_merge(args):
    """Pure post-processing, no API calls: merges Wave 1 valid answers with
    Wave 2 recovery answers into MERGED_TREATMENT_RESULTS_FILE/
    MERGED_BASELINE_RESULTS_FILE, writes the required diagnostics, and then
    HARD-FAILS (after writing everything -- nothing is deleted) unless the
    merged dataset AND the new baseline_text_only dataset are both fully
    complete. This is the gate between recovery and analysis: an incomplete
    dataset must never silently reach analyze_controllability_v2.py.
    Never opens a Wave 1 raw file for writing."""
    study_config = load_study_config()
    _treatment_trials, _baseline_trials, planned_index, _trials_by_id = _load_wave1_planned_index(study_config)
    treatment_rows, baseline_rows = recovery.load_wave1_raw_rows(args.source_results)
    classification = recovery.classify_wave1_rows(treatment_rows + baseline_rows, planned_index)

    recovery_rows = recovery.load_jsonl(args.recovery_results) if os.path.exists(args.recovery_results) else []
    merged_by_id, diagnostics = recovery.merge_wave1_and_recovery(planned_index, classification["valid_by_id"], recovery_rows)
    split = recovery.split_merged_rows_by_unit_kind(merged_by_id, planned_index)

    os.makedirs(RECOVERY_DIR, exist_ok=True)
    with open(MERGED_TREATMENT_RESULTS_FILE, "w", encoding="utf-8") as f:
        for row in split["treatment"]:
            f.write(json.dumps(row) + "\n")
    with open(MERGED_BASELINE_RESULTS_FILE, "w", encoding="utf-8") as f:
        for row in split["baseline"]:
            f.write(json.dumps(row) + "\n")
    with open(MERGE_DIAGNOSTICS_FILE, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    for key, value in diagnostics.items():
        print(f"{key}: {value}")
    print(f"Wrote {len(split['treatment'])} merged treatment row(s) to {MERGED_TREATMENT_RESULTS_FILE}")
    print(f"Wrote {len(split['baseline'])} merged baseline row(s) to {MERGED_BASELINE_RESULTS_FILE}")
    print(f"Wrote diagnostics to {MERGE_DIAGNOSTICS_FILE}")

    baseline_text_only_trials = load_trials(BASELINE_TEXT_ONLY_TRIALS_FILE)
    bto_planned_index = recovery.enumerate_planned_ids(
        baseline_text_only_trials, study_config["baseline_replicate_count"],
        study_config["primary_evaluator"]["evaluator_id"], "baseline_text_only",
    )
    bto_rows = recovery.load_jsonl(BASELINE_TEXT_ONLY_RESULTS_FILE) if os.path.exists(BASELINE_TEXT_ONLY_RESULTS_FILE) else []
    bto_classification = recovery.classify_baseline_text_only_rows(bto_rows, bto_planned_index)
    print(f"baseline_text_only_valid: {len(bto_classification['valid_by_id'])} / {len(bto_planned_index)}")

    try:
        recovery.assert_recovery_complete(diagnostics, expected_planned_total=EXPECTED_WAVE1_PLANNED_TOTAL)
        recovery.assert_baseline_text_only_complete(
            bto_classification, bto_planned_index,
            expected_total=EXPECTED_BASELINE_TEXT_ONLY_TOTAL, expected_unique_cells=EXPECTED_BASELINE_TEXT_ONLY_UNIQUE_CELLS,
        )
    except ValueError as e:
        raise SystemExit(f"MERGE GATE FAILED -- analysis will NOT run: {e}")

    print("Wave 2 recovery and the new baseline_text_only condition are both fully complete. Analysis may proceed.")


# ---------------------------------------------------------------------------
# treatment / baseline
# ---------------------------------------------------------------------------

def cmd_treatment(args):
    _run(args, TRIALS_FILE, "superblock_id", plan_treatment_execution_order, "treatment")


def cmd_baseline(args):
    _run(args, BASELINE_TRIALS_FILE, "block_id", plan_baseline_execution_order, "baseline")


def cmd_baseline_text_only(args):
    """The Wave 2 recovery's new no-context + text_only condition (item:
    "add a new experimental condition"). Structurally an ordinary baseline
    run -- same evaluator/replicate count/off-peak guard as the original
    baseline -- writing to its OWN results file so it is never confused
    with (or pooled by default into) the original matched_control
    baseline's baseline_raw.jsonl. Like every other Wave 2 API call (never
    the historical Wave 1 ones), it uses the raised RECOVERY_MAX_OUTPUT_TOKENS
    ceiling instead of the frozen study config's original 512."""
    _run(args, BASELINE_TEXT_ONLY_TRIALS_FILE, "block_id", plan_baseline_execution_order, "baseline",
         production_results_file=BASELINE_TEXT_ONLY_RESULTS_FILE,
         exploratory_results_file=EXPLORATORY_BASELINE_TEXT_ONLY_RESULTS_FILE,
         max_output_tokens_override=recovery.RECOVERY_MAX_OUTPUT_TOKENS,
         use_valid_only_resume=True)


def _cli_overrides(args):
    return {"provider": args.provider, "model": args.model, "reasoning_profile": args.reasoning_profile,
            "replicates": args.replicates, "seed": args.seed, "retry_limit": args.retry_limit}


def _run(args, trials_file, unit_id_field, plan_fn, unit_kind, production_results_file=None,
         exploratory_results_file=None, max_output_tokens_override=None, use_valid_only_resume=False):
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
        results_file = production_results_file or (TREATMENT_RESULTS_FILE if unit_kind == "treatment" else BASELINE_RESULTS_FILE)
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
        results_file = exploratory_results_file or (EXPLORATORY_TREATMENT_RESULTS_FILE if unit_kind == "treatment" else EXPLORATORY_BASELINE_RESULTS_FILE)
        output_label = "exploratory"

    if max_output_tokens_override is not None:
        # Never affects the frozen-authoritative CLI-mismatch checks above
        # (max_output_tokens is not one of the checked fields) -- it only
        # changes the literal request setting for THIS command's own calls,
        # which correctly gives it its own run_config_id (see
        # build_evaluator_identity/make_run_config_id below).
        settings["max_output_tokens"] = max_output_tokens_override

    provider, model, reasoning_profile = settings["provider"], settings["model"], settings["reasoning_profile"]
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, reasoning_profile)
    sampling_params = {}  # provider-default sampling throughout v2 -- see STUDY_PROTOCOL_V2.md

    trials = load_trials(trials_file)
    trials = limit_to_first_n_units(trials, unit_id_field, args.limit)
    plan = plan_fn(trials, settings["replicates"], settings["seed"])
    attach_observation_ids(plan, EXPERIMENT_ID, settings["evaluator_id"])

    # Wave 1's own treatment/baseline commands keep the original "any
    # terminal row (success or retries-exhausted failure) is done" resume
    # semantics (use_valid_only_resume=False, the default). The Wave 2
    # baseline_text_only condition uses valid-answer-only resume semantics
    # instead -- a failed Wave 2 attempt must remain eligible for a later
    # rerun of this same command (see controllability_v2_recovery.
    # load_valid_completed_observation_ids).
    if use_valid_only_resume:
        already_completed = recovery.load_valid_completed_observation_ids(results_file)
    else:
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

    baseline_text_only_parser = subparsers.add_parser(
        "baseline-text-only", help="Run the Wave 2 recovery's new no-context + text_only condition"
    )
    add_common_arguments(baseline_text_only_parser)
    baseline_text_only_parser.set_defaults(func=cmd_baseline_text_only)

    recovery_preflight_parser = subparsers.add_parser(
        "recovery-preflight", help="Validate the Wave 1 source results and print the Wave 2 recovery plan; makes no API calls"
    )
    recovery_preflight_parser.add_argument("--source-results", required=True,
                                            help="Directory containing the extracted Wave 1 production artifact "
                                                 "(treatment_raw.jsonl + baseline_raw.jsonl)")
    recovery_preflight_parser.add_argument("--allow-peak-pricing", action="store_true")
    recovery_preflight_parser.set_defaults(func=cmd_recovery_preflight)

    recovery_run_parser = subparsers.add_parser(
        "recovery-run", help="Rerun only the Wave 1 planned observations that never resolved to a valid A/B answer"
    )
    recovery_run_parser.add_argument("--source-results", required=True)
    recovery_run_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    recovery_run_parser.add_argument("--allow-peak-pricing", action="store_true")
    recovery_run_parser.add_argument("--dry-run", action="store_true")
    recovery_run_parser.set_defaults(func=cmd_recovery_run)

    validation_run_parser = subparsers.add_parser(
        "validation-run", help="Rerun a stratified sample of already-resolved Wave 1 observations as a non-primary validation check"
    )
    validation_run_parser.add_argument("--source-results", required=True)
    validation_run_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    validation_run_parser.add_argument("--allow-peak-pricing", action="store_true")
    validation_run_parser.add_argument("--dry-run", action="store_true")
    validation_run_parser.set_defaults(func=cmd_validation_run)

    validation_report_parser = subparsers.add_parser(
        "validation-report", help="Compute and write A/B agreement / preference-shift / context-effect diagnostics "
                                   "from validation_raw.jsonl; makes no API calls"
    )
    validation_report_parser.add_argument("--validation-results", default=VALIDATION_RESULTS_FILE)
    validation_report_parser.set_defaults(func=cmd_validation_report)

    merge_parser = subparsers.add_parser(
        "merge", help="Merge Wave 1 valid answers with Wave 2 recovery answers into the analysis-ready merged dataset, "
                      "verify full completeness of both the merged dataset and the new baseline_text_only condition, "
                      "and refuse to proceed if either is incomplete; makes no API calls"
    )
    merge_parser.add_argument("--source-results", required=True)
    merge_parser.add_argument("--recovery-results", default=RECOVERY_RESULTS_FILE)
    merge_parser.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
