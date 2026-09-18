"""v3 selective-suppression execution CLI. Entirely separate from
run_controllability_v2.py: its own manifests, results directory
(results/controllability_v3/), study config, lock, and commands. It never
opens a v2 path for writing and its ids can never collide with v2's.

    python3 run_controllability_v3.py preflight          # zero API calls
    python3 run_controllability_v3.py pilot --production --concurrency 32
    python3 run_controllability_v3.py pilot-report       # zero API calls
    python3 run_controllability_v3.py production --production --concurrency 32
    python3 run_controllability_v3.py holdout --production --concurrency 32
    python3 run_controllability_v3.py analysis           # zero API calls

    # any executing command: --dry-run plans and prints, makes no calls
    python3 run_controllability_v3.py production --dry-run

Production/holdout/pilot with --production require the frozen v3 lock to
verify (and the v2 immutability guard to pass) and read every setting from
the frozen config. --allow-unfrozen exploratory runs write only to
results/controllability_v3/exploratory/ and never enter analysis.

Results layout (never committed; see .gitignore's results/):
    results/controllability_v3/pilot/pilot_raw.jsonl         pilot family only
    results/controllability_v3/production/primary_raw.jsonl  primary_context + primary_nocontext, interleaved
    results/controllability_v3/production/holdout_raw.jsonl  holdout family
    results/controllability_v3/analysis/                     analysis outputs

Resume: valid-answer-only (see controllability_v3_execution). Re-running the
same command after an interruption executes only what is still missing.
"""

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import model_providers
from controllability_v2_deepseek_pricing import DeepSeekPricingGate, format_utc_z, is_deepseek_off_peak, next_off_peak_boundary
from controllability_v3_corpus import CORPUS_FILE, assert_matches_v2_corpus, build_corpus_metadata
from controllability_v3_design import (
    CUE_ORDER,
    INTERVENTION_IDS,
    INTERVENTION_SENTENCES,
    all_instruction_sentences,
)
from controllability_v3_execution import (
    attach_observation_ids,
    build_evaluator_identity,
    build_result_row,
    count_rows,
    filter_unresumed_plan,
    find_duplicate_valid_ids,
    find_foreign_ids,
    limit_to_first_n_units,
    load_valid_completed_ids,
    plan_execution_order,
    resolve_production_settings,
    run_one_observation,
)
from controllability_v3_study_config import STUDY_CONFIG_FILE, load_study_config
from controllability_v3_trials import (
    CONTEXT_UNIQUE_CELLS,
    FAMILY_HOLDOUT,
    FAMILY_PILOT,
    FAMILY_PRIMARY_CONTEXT,
    FAMILY_PRIMARY_NOCONTEXT,
    HOLDOUT_UNIQUE_CELLS,
    NOCONTEXT_UNIQUE_CELLS,
    N_CUES,
    N_INTERVENTIONS,
    N_PAIRS,
    N_STORIES,
    assert_contrasts_are_well_formed,
    assert_holdout_manifest_well_formed,
    assert_no_id_overlap,
    assert_nocontext_manifest_well_formed,
    assert_pilot_manifest_well_formed,
    assert_primary_context_manifest_well_formed,
    build_prompt_for_trial,
    load_contrasts,
    load_manifest,
    load_story_texts,
    sha256_hex,
    verify_prompt_hashes,
)
from controllability_v3_v2_guard import V2_RESULT_DIRS, verify_v2_unchanged
from freeze_controllability_v3 import sha256_of_file, verify_frozen

RESULTS_DIR = "results/controllability_v3"
PRODUCTION_DIR = os.path.join(RESULTS_DIR, "production")
PILOT_DIR = os.path.join(RESULTS_DIR, "pilot")
EXPLORATORY_DIR = os.path.join(RESULTS_DIR, "exploratory")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

PRIMARY_RESULTS_FILE = os.path.join(PRODUCTION_DIR, "primary_raw.jsonl")
HOLDOUT_RESULTS_FILE = os.path.join(PRODUCTION_DIR, "holdout_raw.jsonl")
PILOT_RESULTS_FILE = os.path.join(PILOT_DIR, "pilot_raw.jsonl")
PILOT_REPORT_FILE = os.path.join(PILOT_DIR, "pilot_report.json")
EXPLORATORY_PRIMARY_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "primary_raw.jsonl")
EXPLORATORY_HOLDOUT_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "holdout_raw.jsonl")
EXPLORATORY_PILOT_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "pilot_raw.jsonl")

DEFAULT_CONCURRENCY = 32

# Literal design constants, checked against the independent arithmetic in
# design_arithmetic() and against the manifests themselves at preflight.
EXPECTED_CONTEXT_UNIQUE_CELLS = 10560
EXPECTED_NOCONTEXT_UNIQUE_CELLS = 1056
EXPECTED_HOLDOUT_UNIQUE_CELLS = 1320
EXPECTED_REPLICATES = 10
EXPECTED_CONTEXT_JUDGMENTS = 105600
EXPECTED_NOCONTEXT_JUDGMENTS = 10560
EXPECTED_PRIMARY_TOTAL = 116160
EXPECTED_HOLDOUT_JUDGMENTS = 13200

RUN_KIND_FILES = {
    "primary": {"families": (FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT), "production": PRIMARY_RESULTS_FILE, "exploratory": EXPLORATORY_PRIMARY_RESULTS_FILE},
    "holdout": {"families": (FAMILY_HOLDOUT,), "production": HOLDOUT_RESULTS_FILE, "exploratory": EXPLORATORY_HOLDOUT_RESULTS_FILE},
    "pilot": {"families": (FAMILY_PILOT,), "production": PILOT_RESULTS_FILE, "exploratory": EXPLORATORY_PILOT_RESULTS_FILE},
}


# ---------------------------------------------------------------------------
# Independent design arithmetic
# ---------------------------------------------------------------------------

def design_arithmetic(n_stories=N_STORIES, n_cues=None, n_interventions=None, replicates=EXPECTED_REPLICATES,
                      holdout_replicates=EXPECTED_REPLICATES):
    """Recompute every expected count from first principles (math.comb and
    the length of the frozen cue/intervention tuples) -- independent of the
    constants in controllability_v3_trials.py, which preflight then
    compares against these."""
    n_cues = n_cues if n_cues is not None else len(CUE_ORDER)
    n_interventions = n_interventions if n_interventions is not None else len(INTERVENTION_SENTENCES)
    n_pairs = math.comb(n_stories, 2)
    n_assignments, n_positions = 2, 2
    context_cells = n_pairs * n_cues * n_interventions * n_assignments * n_positions
    nocontext_cells = n_pairs * n_interventions * n_positions
    holdout_cells = n_pairs * n_cues * n_assignments * n_positions  # one enumerating intervention (I2)
    return {
        "stories": n_stories, "story_pairs": n_pairs, "cues": n_cues, "interventions": n_interventions,
        "context_unique_cells": context_cells, "context_judgments": context_cells * replicates,
        "nocontext_unique_cells": nocontext_cells, "nocontext_judgments": nocontext_cells * replicates,
        "primary_total_judgments": (context_cells + nocontext_cells) * replicates,
        "holdout_unique_cells": holdout_cells, "holdout_judgments": holdout_cells * holdout_replicates,
        "replicates": replicates, "holdout_replicates": holdout_replicates,
    }


# ---------------------------------------------------------------------------
# Thread-safe append-only JSONL writer + progress
# ---------------------------------------------------------------------------

class SafeJsonlWriter:
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

    def __exit__(self, *exc):
        self.close()


class ProgressTracker:
    def __init__(self, total, already_completed=0, log_every=None):
        self._lock = threading.Lock()
        self.total, self.already_completed = total, already_completed
        self.completed = self.successful = self.unresolved = 0
        self.input_tokens = self.output_tokens = self.reasoning_tokens = 0
        self._start = time.time()
        self.log_every = log_every or max(1, total // 20 or 1)

    def record(self, observation):
        with self._lock:
            self.completed += 1
            if observation["parsing_status"] == "resolved":
                self.successful += 1
            else:
                self.unresolved += 1
            for attempt in observation["attempts"]:
                self.input_tokens += attempt.get("input_tokens") or 0
                self.output_tokens += attempt.get("output_tokens") or 0
                self.reasoning_tokens += attempt.get("reasoning_tokens") or 0
            due = self.completed % self.log_every == 0 or self.completed == self.total
        if due:
            elapsed = time.time() - self._start
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            print(f"[progress] {self.completed}/{self.total} this run (+{self.already_completed} already valid)  "
                  f"resolved={self.successful}  unresolved={self.unresolved}  rate={rate:.2f}/s  elapsed={elapsed:.0f}s  "
                  f"tokens~ in={self.input_tokens} out={self.output_tokens} reasoning={self.reasoning_tokens}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def make_call_fn(provider, model, reasoning_profile, max_output_tokens, pricing_gate=None):
    """The ONLY place a v3 run reaches the network. The DeepSeek off-peak
    pricing gate is consulted before every dispatch for provider deepseek
    (and only then), exactly as in v2."""
    def call_fn(prompt):
        if provider == "deepseek" and pricing_gate is not None:
            pricing_gate.wait_until_dispatch_allowed()
        return model_providers.call_model(provider, model, prompt, max_output_tokens, reasoning_profile, None)
    return call_fn


def run_plan(plan, task, concurrency, deadline=None):
    """Submit entries in plan order with at most `concurrency` in flight.
    Stops submitting once `deadline` (time.time() seconds) has passed --
    in-flight calls always complete and are written. Returns the number of
    entries NOT submitted (0 when the plan ran to completion)."""
    if concurrency <= 1:
        for index, entry in enumerate(plan):
            if deadline is not None and time.time() > deadline:
                return len(plan) - index
            task(entry)
        return 0
    not_submitted = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        in_flight = set()
        for index, entry in enumerate(plan):
            if deadline is not None and time.time() > deadline:
                not_submitted = len(plan) - index
                break
            while len(in_flight) >= concurrency:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
            in_flight.add(pool.submit(task, entry))
        for future in in_flight:
            future.result()
    return not_submitted


# ---------------------------------------------------------------------------
# Preflight -- every check, zero API calls
# ---------------------------------------------------------------------------

def _check(checks, name, fn):
    try:
        detail = fn()
        checks.append((name, True, detail if isinstance(detail, str) and detail else "ok"))
        return True
    except Exception as e:  # any failure is a FAIL line, never a crash
        checks.append((name, False, f"{type(e).__name__}: {e}"))
        return False


def run_preflight(study_config, cli_overrides=None, concurrency=DEFAULT_CONCURRENCY, allow_peak_pricing=False,
                  skip_credential_check=False, results_files=None):
    checks, context = [], {}
    now = datetime.now(timezone.utc)
    context["pricing_window_off_peak"] = is_deepseek_off_peak(now)
    context["pricing_window_next_off_peak"] = None if context["pricing_window_off_peak"] else next_off_peak_boundary(now)
    context["allow_peak_pricing"] = allow_peak_pricing
    context["concurrency"] = concurrency

    ok, reason = verify_frozen()
    checks.append(("v3_study_lock_exists_and_hashes_validate", ok, reason or "ok"))

    ok, reason = verify_v2_unchanged()
    checks.append(("v2_files_unchanged_and_v2_lock_validates", ok, reason or "ok"))

    def contrasts_identical():
        if sha256_of_file(study_config["contrast_file"]) != sha256_of_file("data/controllability_v2_contrasts.jsonl"):
            raise ValueError("v3 contrasts differ from v2's")
        assert_contrasts_are_well_formed(load_contrasts(study_config["contrast_file"]))
    _check(checks, "cues_byte_identical_to_v2_and_well_formed", contrasts_identical)

    def corpus_ok():
        metadata = build_corpus_metadata()
        assert_matches_v2_corpus(metadata)
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        if on_disk["corpus_hash"] != metadata["corpus_hash"]:
            raise ValueError("data/controllability_v3_corpus.json does not match the story files on disk")
        return f"{len(metadata['item_ids'])} stories, corpus_hash {metadata['corpus_hash'][:12]}"
    _check(checks, "corpus_matches_v2_and_story_files_unchanged", corpus_ok)

    arithmetic = design_arithmetic(replicates=study_config["replicate_count"], holdout_replicates=study_config["holdout_replicate_count"])
    context["arithmetic"] = arithmetic

    def arithmetic_ok():
        expected = {
            "story_pairs": (N_PAIRS, 66), "cues": (N_CUES, 5), "interventions": (N_INTERVENTIONS, 8),
            "context_unique_cells": (CONTEXT_UNIQUE_CELLS, EXPECTED_CONTEXT_UNIQUE_CELLS),
            "nocontext_unique_cells": (NOCONTEXT_UNIQUE_CELLS, EXPECTED_NOCONTEXT_UNIQUE_CELLS),
            "holdout_unique_cells": (HOLDOUT_UNIQUE_CELLS, EXPECTED_HOLDOUT_UNIQUE_CELLS),
        }
        for key, (module_value, literal) in expected.items():
            if not (arithmetic[key] == module_value == literal):
                raise ValueError(f"{key}: independent arithmetic {arithmetic[key]}, module constant {module_value}, literal {literal}")
        if arithmetic["context_judgments"] != EXPECTED_CONTEXT_JUDGMENTS or arithmetic["nocontext_judgments"] != EXPECTED_NOCONTEXT_JUDGMENTS:
            raise ValueError("planned judgment totals do not match the design")
        if arithmetic["primary_total_judgments"] != EXPECTED_PRIMARY_TOTAL or arithmetic["holdout_judgments"] != EXPECTED_HOLDOUT_JUDGMENTS:
            raise ValueError("primary/holdout totals do not match the design")
    _check(checks, "design_arithmetic_independently_verified", arithmetic_ok)

    def replicates_ok():
        if study_config["replicate_count"] != EXPECTED_REPLICATES or study_config["holdout_replicate_count"] != EXPECTED_REPLICATES:
            raise ValueError(f"replicate counts {study_config['replicate_count']}/{study_config['holdout_replicate_count']} are not the designed {EXPECTED_REPLICATES}")
        if study_config["pilot_replicate_count"] != 1:
            raise ValueError("pilot_replicate_count must be 1 (a pilot never accumulates replicates)")
    _check(checks, "replicate_counts_as_designed", replicates_ok)

    manifests = {}
    for key, loader_name in (("context", "context_manifest_file"), ("nocontext", "nocontext_manifest_file"),
                             ("holdout", "holdout_manifest_file"), ("pilot", "pilot_manifest_file")):
        path = study_config[loader_name]
        try:
            manifests[key] = load_manifest(path)
        except FileNotFoundError as e:
            manifests[key] = None
            checks.append((f"{key}_manifest_present", False, str(e)))
    context["manifests"] = manifests

    if manifests["context"] is not None:
        ids = {t["trial_id"] for t in manifests["context"]}
        checks.append(("context_manifest_has_10560_unique_cells", len(ids) == EXPECTED_CONTEXT_UNIQUE_CELLS == len(manifests["context"]), f"found {len(ids)}"))
        _check(checks, "every_context_block_counterbalanced_and_all_interventions_present", lambda: assert_primary_context_manifest_well_formed(manifests["context"]))
    if manifests["nocontext"] is not None:
        ids = {t["trial_id"] for t in manifests["nocontext"]}
        checks.append(("nocontext_manifest_has_1056_unique_cells", len(ids) == EXPECTED_NOCONTEXT_UNIQUE_CELLS == len(manifests["nocontext"]), f"found {len(ids)}"))
        _check(checks, "every_nocontext_block_has_both_positions_and_no_context", lambda: assert_nocontext_manifest_well_formed(manifests["nocontext"]))
    if manifests["holdout"] is not None:
        ids = {t["trial_id"] for t in manifests["holdout"]}
        checks.append(("holdout_manifest_has_1320_unique_cells", len(ids) == EXPECTED_HOLDOUT_UNIQUE_CELLS == len(manifests["holdout"]), f"found {len(ids)}"))
        _check(checks, "holdout_variants_omit_exactly_the_tested_cue", lambda: assert_holdout_manifest_well_formed(manifests["holdout"]))

    frozen = [m for k, m in manifests.items() if k != "pilot" and m is not None]
    if len(frozen) == 3:
        _check(checks, "no_id_overlap_between_manifests_and_no_v2_ids", lambda: assert_no_id_overlap(*frozen))
        frozen_ids = {t["trial_id"] for m in frozen for t in m}
        if manifests["pilot"] is not None:
            _check(checks, "pilot_manifest_well_formed_and_disjoint_from_primary", lambda: assert_pilot_manifest_well_formed(manifests["pilot"], frozen_ids))

        def instruction_mapping():
            sentences = all_instruction_sentences()
            if study_config["instruction_strings"] != dict(INTERVENTION_SENTENCES):
                raise ValueError("study config instruction_strings differ from the frozen design wording")
            for iid, sentence in study_config["holdout_instruction_strings"].items():
                if sentences.get(iid) != sentence:
                    raise ValueError(f"holdout wording for {iid} differs between config and design")
            seen = set()
            for m in frozen:
                for t in m:
                    if t["instruction_id"] not in sentences:
                        raise ValueError(f"{t['trial_id']} has unknown instruction {t['instruction_id']!r}")
                    seen.add(t["instruction_id"])
            if set(INTERVENTION_IDS) - seen:
                raise ValueError(f"interventions never mapped to any cell: {sorted(set(INTERVENTION_IDS) - seen)}")
            return f"{len(seen)} instruction ids mapped"
        _check(checks, "instruction_mapping_matches_frozen_wording", instruction_mapping)

        def prompts_rebuild():
            texts = load_story_texts()
            for m in frozen:
                verify_prompt_hashes(m, texts)
            if manifests["pilot"] is not None:
                verify_prompt_hashes(manifests["pilot"], texts)
            unknown = {t["story_1_id"] for m in frozen for t in m} | {t["story_2_id"] for m in frozen for t in m}
            unknown -= set(texts)
            if unknown:
                raise ValueError(f"manifest references unknown stories: {sorted(unknown)}")
            return f"{sum(len(m) for m in frozen)} prompts rebuilt and hash-verified"
        _check(checks, "prompt_hashes_rebuild_from_story_files_and_all_stories_known", prompts_rebuild)

    def outputs_clean():
        files = results_files or {"primary": PRIMARY_RESULTS_FILE, "holdout": HOLDOUT_RESULTS_FILE, "pilot": PILOT_RESULTS_FILE}
        problems = []
        for kind, path in files.items():
            dup = find_duplicate_valid_ids(path)
            foreign = find_foreign_ids(path, RUN_KIND_FILES[kind]["families"])
            if dup:
                problems.append(f"{path}: {len(dup)} duplicate valid id(s)")
            if foreign:
                problems.append(f"{path}: {len(foreign)} foreign/non-{kind} row(s)")
        for d in V2_RESULT_DIRS:
            if os.path.abspath(PRODUCTION_DIR).startswith(os.path.abspath(d)):
                problems.append("v3 production dir lies inside a v2 results dir")
        if problems:
            raise ValueError("; ".join(problems))
    _check(checks, "production_and_pilot_outputs_have_no_duplicate_valid_or_foreign_ids", outputs_clean)

    def settings_ok():
        context["settings"] = {kind: resolve_production_settings(cli_overrides or {}, study_config, kind) for kind in ("primary", "holdout", "pilot")}
        if context["settings"]["primary"]["max_output_tokens"] < 4096:
            raise ValueError("max_output_tokens below the v2 recovery-era ceiling of 4096 (the 512 truncation failure mode)")
    _check(checks, "cli_settings_match_frozen_evaluator_replicates_and_token_ceiling", settings_ok)

    provider = study_config["primary_evaluator"]["provider"]
    if skip_credential_check:
        checks.append(("api_credential_present", True, "skipped (--skip-credential-check)"))
    else:
        try:
            model_providers.get_api_key(provider)
            checks.append(("api_credential_present", True, "ok"))
        except RuntimeError as e:
            checks.append(("api_credential_present", False, str(e)))

    return all(c[1] for c in checks), checks, context


def print_preflight_summary(study_config, context):
    a = context["arithmetic"]
    primary = study_config["primary_evaluator"]
    print("=== v3 preflight summary ===")
    print(f"Stories: {a['stories']}")
    print(f"Story pairs: {a['story_pairs']}")
    print(f"Cues: {a['cues']}")
    print(f"Interventions: {a['interventions']}")
    print(f"Context-present unique cells: {a['context_unique_cells']:,}")
    print(f"Context-present judgments @ {a['replicates']} reps: {a['context_judgments']:,}")
    print(f"No-context unique cells: {a['nocontext_unique_cells']:,}")
    print(f"No-context judgments @ {a['replicates']} reps: {a['nocontext_judgments']:,}")
    print(f"Primary total judgments: {a['primary_total_judgments']:,}")
    print(f"Held-out-cue unique cells (secondary, additional): {a['holdout_unique_cells']:,}")
    print(f"Held-out-cue judgments @ {a['holdout_replicates']} reps (secondary, additional): {a['holdout_judgments']:,}")
    pilot = context["manifests"].get("pilot")
    print(f"Pilot cells (never primary): {len(pilot) if pilot else 'n/a'} @ {study_config['pilot_replicate_count']} rep")
    print(f"Model: {primary['requested_model']} (provider={primary['provider']})  Reasoning profile: {primary['reasoning_profile']}")
    print(f"max_output_tokens: {study_config['max_output_tokens']}  Retry limit: {study_config['retry_limit']}  Seed: {study_config['random_seed']}")
    completed_primary = len(load_valid_completed_ids(PRIMARY_RESULTS_FILE))
    completed_holdout = len(load_valid_completed_ids(HOLDOUT_RESULTS_FILE))
    print(f"Already valid (production): primary={completed_primary:,}/{a['primary_total_judgments']:,}  holdout={completed_holdout:,}/{a['holdout_judgments']:,}")
    print(f"Concurrency: {context['concurrency']}  Output directory: {PRODUCTION_DIR}")
    if context["pricing_window_off_peak"]:
        print("DeepSeek pricing window: OFF-PEAK")
    else:
        print(f"DeepSeek pricing window: PEAK  (next off-peak: {format_utc_z(context['pricing_window_next_off_peak'])})")
    print(f"Peak-pricing override: {'enabled' if context['allow_peak_pricing'] else 'disabled'}")


def cmd_preflight(args):
    study_config = load_study_config()
    ok, checks, context = run_preflight(study_config, {}, args.concurrency, args.allow_peak_pricing, args.skip_credential_check)
    for name, check_ok, detail in checks:
        print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
    if "arithmetic" in context and "manifests" in context:
        print_preflight_summary(study_config, context)
    if not ok:
        raise SystemExit("v3 preflight FAILED -- see the FAIL line(s) above. No API calls were made.")
    print("v3 preflight PASSED. No API calls were made.")


# ---------------------------------------------------------------------------
# Executing commands
# ---------------------------------------------------------------------------

def _cli_overrides(args):
    return {"provider": args.provider, "model": args.model, "reasoning_profile": args.reasoning_profile,
            "replicates": args.replicates, "seed": args.seed, "retry_limit": args.retry_limit}


def _load_trials_for(run_kind, study_config):
    if run_kind == "primary":
        return load_manifest(study_config["context_manifest_file"]) + load_manifest(study_config["nocontext_manifest_file"])
    if run_kind == "holdout":
        return load_manifest(study_config["holdout_manifest_file"])
    if run_kind == "pilot":
        return load_manifest(study_config["pilot_manifest_file"])
    raise ValueError(run_kind)


def _run(args, run_kind):
    if args.production and args.allow_unfrozen:
        raise SystemExit("--production and --allow-unfrozen are mutually exclusive.")
    if not args.dry_run and not args.production and not args.allow_unfrozen:
        raise SystemExit("A real (non---dry-run) execution must pass exactly one of --production or --allow-unfrozen.")

    study_config = load_study_config()
    cli_overrides = _cli_overrides(args)
    files = RUN_KIND_FILES[run_kind]

    if args.allow_unfrozen:
        if not args.provider or not args.model:
            raise SystemExit("--provider and --model are required for an exploratory (--allow-unfrozen) run.")
        settings = {
            "provider": args.provider, "model": args.model,
            "reasoning_profile": args.reasoning_profile or model_providers.DEFAULT_REASONING_PROFILE,
            "replicates": args.replicates or 1, "seed": args.seed if args.seed is not None else 0,
            "retry_limit": args.retry_limit if args.retry_limit is not None else 3,
            "max_output_tokens": study_config["max_output_tokens"],
            "evaluator_id": f"{args.provider}__{args.model}__{args.reasoning_profile or model_providers.DEFAULT_REASONING_PROFILE}",
        }
        results_file, collection = files["exploratory"], "exploratory"
    else:
        if args.production:
            ok, checks, context = run_preflight(study_config, cli_overrides, args.concurrency, args.allow_peak_pricing)
            for name, check_ok, detail in checks:
                print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
            if "arithmetic" in context and "manifests" in context:
                print_preflight_summary(study_config, context)
            if not ok:
                raise SystemExit(f"v3 preflight FAILED -- refusing to start a {run_kind} run. No API calls were made.")
            settings = context["settings"][run_kind]
        else:
            settings = resolve_production_settings(cli_overrides, study_config, run_kind)
        results_file, collection = files["production"], ("pilot" if run_kind == "pilot" else "production")

    provider, model, reasoning_profile = settings["provider"], settings["model"], settings["reasoning_profile"]
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, reasoning_profile)

    trials = _load_trials_for(run_kind, study_config)
    trials = limit_to_first_n_units(trials, args.limit)
    texts = load_story_texts()
    verify_prompt_hashes(trials, texts)
    plan = attach_observation_ids(plan_execution_order(trials, settings["replicates"], settings["seed"]))
    already_valid = load_valid_completed_ids(results_file)
    remaining = filter_unresumed_plan(plan, already_valid)

    manifest_paths = {"primary": (study_config["context_manifest_file"], study_config["nocontext_manifest_file"]),
                      "holdout": (study_config["holdout_manifest_file"],), "pilot": (study_config["pilot_manifest_file"],)}[run_kind]
    manifest_sha = "+".join(sha256_of_file(p) for p in manifest_paths)

    print(f"Run kind: {run_kind}  Collection: {collection}  Output: {results_file}")
    print(f"Provider: {provider}  Model: {model}  Reasoning profile: {reasoning_profile}  max_output_tokens: {settings['max_output_tokens']}")
    print(f"Replicates: {settings['replicates']}  Seed: {settings['seed']}  Retry limit: {settings['retry_limit']}")
    print(f"Planned observations: {len(plan):,}  Already valid: {len(plan) - len(remaining):,}  Remaining: {len(remaining):,}")

    if args.dry_run:
        print("Dry run: no network calls made.")
        print(f"First 5 execution-order entries: {[(p['execution_order_index'], p['trial_id']) for p in remaining[:5]]}")
        return
    if not remaining:
        print("Nothing to do -- every planned observation already has a valid answer.")
        return

    evaluator_identity = build_evaluator_identity(
        provider=provider, requested_model=model, reasoning_profile=reasoning_profile,
        provider_reasoning_settings=reasoning_settings, sampling_settings={}, run_id=args.run_id,
        max_output_tokens=settings["max_output_tokens"], retry_limit=settings["retry_limit"],
    )
    pricing_gate = DeepSeekPricingGate(allow_peak=args.allow_peak_pricing)
    call_fn = make_call_fn(provider, model, reasoning_profile, settings["max_output_tokens"], pricing_gate)
    progress = ProgressTracker(total=len(remaining), already_completed=len(plan) - len(remaining))
    deadline = time.time() + args.time_budget_minutes * 60 if args.time_budget_minutes else None

    with SafeJsonlWriter(results_file) as writer:
        def task(entry):
            prompt = build_prompt_for_trial(entry["trial"], texts)
            observation = run_one_observation(entry["trial"], prompt, call_fn, settings["retry_limit"])
            row = build_result_row(entry, evaluator_identity, observation, sha256_hex(prompt), args.run_id, collection, manifest_sha)
            writer.write(row)
            progress.record(observation)
        not_submitted = run_plan(remaining, task, args.concurrency, deadline)

    still_missing = len(plan) - len(load_valid_completed_ids(results_file))
    print(f"Done. Results appended to {results_file}.  Not submitted (time budget): {not_submitted:,}  "
          f"Planned observations still without a valid answer: {still_missing:,}")
    if still_missing:
        print("Re-run this same command to resume; only the missing observations will be executed.")


def cmd_pilot(args):
    _run(args, "pilot")


def cmd_production(args):
    _run(args, "primary")


def cmd_holdout(args):
    _run(args, "holdout")


# ---------------------------------------------------------------------------
# Pilot report -- compliance/tokens/latency/position sanity only. Deliberately
# reports NO context-effect or suppression estimates: wording is never tuned
# on pilot effect sizes.
# ---------------------------------------------------------------------------

def build_pilot_report(rows, study_config):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["intervention_id"], bool(row["context_present"]))].append(row)

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    per_condition = []
    flags = []
    for (intervention_id, context_present), group in sorted(groups.items()):
        n = len(group)
        resolved = [r for r in group if r["parsing_status"] == "resolved"]
        first_valid = sum(1 for r in group if r["first_attempt_status"] == "valid")
        p_a = mean([1.0 if r["parsed_choice"] == "A" else 0.0 for r in resolved]) if resolved else None
        entry = {
            "intervention_id": intervention_id, "context_present": context_present, "n": n,
            "first_attempt_valid_rate": first_valid / n if n else None,
            "resolved_rate": len(resolved) / n if n else None,
            "mean_output_tokens": mean([r.get("output_tokens") for r in resolved]),
            "mean_reasoning_tokens": mean([r.get("reasoning_tokens") for r in resolved]),
            "max_output_tokens_seen": max([r.get("output_tokens") or 0 for r in resolved], default=None),
            "mean_latency_seconds": mean([r.get("latency_seconds") for r in resolved]),
            "p_choose_a": p_a,
            "first_attempt_status_counts": dict(sorted(
                {s: sum(1 for r in group if r["first_attempt_status"] == s) for s in {r["first_attempt_status"] for r in group}}.items())),
        }
        per_condition.append(entry)
        if entry["resolved_rate"] is not None and entry["resolved_rate"] < 0.95:
            flags.append(f"{intervention_id} ctx={context_present}: resolved rate {entry['resolved_rate']:.2f} < 0.95")
        if p_a is not None and n >= 20 and (p_a > 0.9 or p_a < 0.1):
            flags.append(f"{intervention_id} ctx={context_present}: P(A)={p_a:.2f} -- gross position/format pathology?")
        ceiling = study_config["max_output_tokens"]
        if entry["max_output_tokens_seen"] is not None and entry["max_output_tokens_seen"] >= ceiling:
            flags.append(f"{intervention_id} ctx={context_present}: an output hit the {ceiling}-token ceiling")

    families = {r["family"] for r in rows}
    if families - {FAMILY_PILOT}:
        flags.append(f"non-pilot rows present in the pilot file: {sorted(families - {FAMILY_PILOT})}")
    return {
        "experiment_id": study_config["experiment_id"], "n_rows": len(rows),
        "n_resolved": sum(1 for r in rows if r["parsing_status"] == "resolved"),
        "per_condition": per_condition, "flags": flags,
        "note": "Pilot diagnostics only (compliance, tokens, latency, P(A)). No context-effect or suppression estimates are computed from pilot data, and pilot rows never enter the primary analysis.",
    }


def cmd_pilot_report(args):
    from controllability_v3_execution import iter_rows
    study_config = load_study_config()
    rows = list(iter_rows(args.pilot_results))
    if not rows:
        raise SystemExit(f"No pilot rows found at {args.pilot_results}")
    report = build_pilot_report(rows, study_config)
    os.makedirs(os.path.dirname(args.report_file) or ".", exist_ok=True)
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Pilot rows: {report['n_rows']}  resolved: {report['n_resolved']}")
    for entry in report["per_condition"]:
        print(f"  {entry['intervention_id']:>3} ctx={str(entry['context_present']):5}  n={entry['n']:3}  "
              f"valid1st={entry['first_attempt_valid_rate']:.2f}  resolved={entry['resolved_rate']:.2f}  "
              f"out_tok={entry['mean_output_tokens'] or 0:.0f}  reasoning_tok={entry['mean_reasoning_tokens'] or 0:.0f}  "
              f"latency={entry['mean_latency_seconds'] or 0:.1f}s  P(A)={entry['p_choose_a'] if entry['p_choose_a'] is None else round(entry['p_choose_a'], 2)}")
    for flag in report["flags"]:
        print(f"  FLAG: {flag}")
    print(f"Wrote {args.report_file}")


def cmd_analysis(args):
    import analyze_controllability_v3
    analyze_controllability_v3.run_analysis(
        primary_results_file=args.primary_results, holdout_results_file=args.holdout_results,
        bootstrap_draws=args.bootstrap_draws, analysis_dir=args.analysis_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_execution_arguments(parser):
    parser.add_argument("--provider", choices=list(model_providers.PROVIDERS), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-profile", choices=list(model_providers.REASONING_PROFILES_LOGICAL), default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--retry-limit", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Wave/run identifier recorded on every result row")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N whole blocks (never a partial block)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--time-budget-minutes", type=float, default=None,
                         help="Stop submitting new calls after this long (in-flight calls complete); re-run to resume")
    parser.add_argument("--production", action="store_true", help="Frozen run: settings from the frozen v3 config; preflight first")
    parser.add_argument("--allow-unfrozen", action="store_true", help="Exploratory run into results/controllability_v3/exploratory/ only")
    parser.add_argument("--allow-peak-pricing", action="store_true", help="Bypass the DeepSeek off-peak guard (DeepSeek only)")
    parser.add_argument("--dry-run", action="store_true", help="Plan and print counts; make no network calls")


def build_parser():
    parser = argparse.ArgumentParser(description="v3 selective-suppression execution CLI (separate from v2).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="Every structural/frozen-design check; makes no API calls")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--allow-peak-pricing", action="store_true")
    p.add_argument("--skip-credential-check", action="store_true", help="For local structural checks without a provider key")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("pilot", help="Run the deterministic pilot manifest (pilot family only; never primary data)")
    add_execution_arguments(p)
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("pilot-report", help="Compliance/token/latency diagnostics from pilot_raw.jsonl; no API calls, no effect sizes")
    p.add_argument("--pilot-results", default=PILOT_RESULTS_FILE)
    p.add_argument("--report-file", default=PILOT_REPORT_FILE)
    p.set_defaults(func=cmd_pilot_report)

    p = sub.add_parser("production", help="Run the primary context + no-context manifests (interleaved, resumable)")
    add_execution_arguments(p)
    p.set_defaults(func=cmd_production)

    p = sub.add_parser("holdout", help="Run the held-out-cue generalization manifest (secondary, resumable)")
    add_execution_arguments(p)
    p.set_defaults(func=cmd_holdout)

    p = sub.add_parser("analysis", help="Run the v3 analysis on production results; no API calls")
    p.add_argument("--primary-results", default=PRIMARY_RESULTS_FILE)
    p.add_argument("--holdout-results", default=HOLDOUT_RESULTS_FILE)
    p.add_argument("--analysis-dir", default=ANALYSIS_DIR)
    p.add_argument("--bootstrap-draws", type=int, default=None)
    p.set_defaults(func=cmd_analysis)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
