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
from controllability_v3_evaluator_profiles import (
    PRIMARY_PROFILE_ID,
    get_profile,
    profile_ids,
    profile_matches_study_config,
    results_path_for_profile,
    validate_profile,
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
    resolve_profile_settings,
    run_one_observation,
)
import controllability_v3_stress_design as sd
import controllability_v3_stress_adversarial as adv
import controllability_v3_stress_dose as dose_module
from controllability_v3_stress_config import load_stress_config
from freeze_controllability_v3_stress import verify_frozen as verify_stress_frozen
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

# Stress families: their own directories, never mixed into the primary raw files.
STRESS_ADV_DIR = os.path.join(RESULTS_DIR, "stress_adversarial")
STRESS_DOSE_DIR = os.path.join(RESULTS_DIR, "stress_dose_response")
STRESS_ANALYSIS_DIR = os.path.join(ANALYSIS_DIR, "stress")
ADV_DEV_RESULTS_FILE = os.path.join(STRESS_ADV_DIR, "dev_raw.jsonl")
ADV_EVAL_RESULTS_FILE = os.path.join(STRESS_ADV_DIR, "eval_raw.jsonl")
DOSE_RESULTS_FILE = os.path.join(STRESS_DOSE_DIR, "dose_raw.jsonl")
EXPLORATORY_ADV_DEV_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "stress_adversarial_dev_raw.jsonl")
EXPLORATORY_ADV_EVAL_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "stress_adversarial_eval_raw.jsonl")
EXPLORATORY_DOSE_RESULTS_FILE = os.path.join(EXPLORATORY_DIR, "stress_dose_raw.jsonl")

# Run-kind registry. "families"/"production"/"exploratory" are the keys the
# original primary/holdout/pilot kinds always had; stress kinds add an id
# prefix, a replicate-count source, and the lock stage they require.
RUN_KIND_FILES = {
    "primary": {"families": (FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT), "production": PRIMARY_RESULTS_FILE, "exploratory": EXPLORATORY_PRIMARY_RESULTS_FILE},
    "holdout": {"families": (FAMILY_HOLDOUT,), "production": HOLDOUT_RESULTS_FILE, "exploratory": EXPLORATORY_HOLDOUT_RESULTS_FILE},
    "pilot": {"families": (FAMILY_PILOT,), "production": PILOT_RESULTS_FILE, "exploratory": EXPLORATORY_PILOT_RESULTS_FILE},
    "stress_adv_dev": {"families": (sd.FAMILY_ADV_DEV,), "production": ADV_DEV_RESULTS_FILE, "exploratory": EXPLORATORY_ADV_DEV_RESULTS_FILE,
                       "id_prefix": "v3s::", "stress_stage": "design", "replicates_key": ("adversarial", "dev_replicates")},
    "stress_adv_eval": {"families": (sd.FAMILY_ADV_EVAL,), "production": ADV_EVAL_RESULTS_FILE, "exploratory": EXPLORATORY_ADV_EVAL_RESULTS_FILE,
                        "id_prefix": "v3s::", "stress_stage": "attacks", "replicates_key": ("adversarial", "eval_replicates")},
    "stress_dose": {"families": (sd.FAMILY_DOSE,), "production": DOSE_RESULTS_FILE, "exploratory": EXPLORATORY_DOSE_RESULTS_FILE,
                    "id_prefix": "v3s::", "stress_stage": "design", "replicates_key": ("dose_response", "replicates")},
}
STRESS_RUN_KINDS = ("stress_adv_dev", "stress_adv_eval", "stress_dose")

# Assumed judge throughput for planning ONLY (no v2/v3 throughput was ever
# recorded in this repository); override with --assumed-throughput.
DEFAULT_ASSUMED_JUDGMENTS_PER_SECOND = 8.0


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
            foreign = find_foreign_ids(path, RUN_KIND_FILES[kind]["families"], (RUN_KIND_FILES[kind].get("id_prefix", "v3::"),))
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


def _load_trials_for(run_kind, study_config, stress_config=None):
    if run_kind == "primary":
        return load_manifest(study_config["context_manifest_file"]) + load_manifest(study_config["nocontext_manifest_file"])
    if run_kind == "holdout":
        return load_manifest(study_config["holdout_manifest_file"])
    if run_kind == "pilot":
        return load_manifest(study_config["pilot_manifest_file"])
    stress_config = stress_config or load_stress_config()
    if run_kind == "stress_adv_dev":
        return sd.load_manifest(stress_config["adversarial"]["dev_manifest_file"])
    if run_kind == "stress_adv_eval":
        return sd.load_manifest(stress_config["adversarial"]["eval_manifest_file"])
    if run_kind == "stress_dose":
        return sd.load_manifest(stress_config["dose_response"]["manifest_file"])
    raise ValueError(run_kind)


def _manifest_paths_for(run_kind, study_config, stress_config=None):
    if run_kind == "primary":
        return (study_config["context_manifest_file"], study_config["nocontext_manifest_file"])
    if run_kind == "holdout":
        return (study_config["holdout_manifest_file"],)
    if run_kind == "pilot":
        return (study_config["pilot_manifest_file"],)
    stress_config = stress_config or load_stress_config()
    return ({"stress_adv_dev": (stress_config["adversarial"]["dev_manifest_file"],),
             "stress_adv_eval": (stress_config["adversarial"]["eval_manifest_file"],),
             "stress_dose": (stress_config["dose_response"]["manifest_file"],)}[run_kind])


def _stress_replicates(run_kind, stress_config):
    section, key = RUN_KIND_FILES[run_kind]["replicates_key"]
    return stress_config[section][key]


def _id_maker_for(run_kind):
    from controllability_v3_trials import make_planned_observation_id
    return sd.make_stress_observation_id if run_kind in STRESS_RUN_KINDS else make_planned_observation_id


def _run(args, run_kind):
    if args.production and args.allow_unfrozen:
        raise SystemExit("--production and --allow-unfrozen are mutually exclusive.")
    if not args.dry_run and not args.production and not args.allow_unfrozen:
        raise SystemExit("A real (non---dry-run) execution must pass exactly one of --production or --allow-unfrozen.")

    study_config = load_study_config()
    stress_config = load_stress_config() if run_kind in STRESS_RUN_KINDS else None
    cli_overrides = _cli_overrides(args)
    files = RUN_KIND_FILES[run_kind]
    profile_id = getattr(args, "evaluator_profile", None) or PRIMARY_PROFILE_ID
    ok_profile, profile_reason = validate_profile(profile_id)
    if not ok_profile:
        raise SystemExit(f"Evaluator profile check failed: {profile_reason}")
    if profile_id.startswith("attacker_"):
        raise SystemExit(f"{profile_id!r} is an attacker profile and may never act as the judge.")
    profile = get_profile(profile_id)

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
        results_file, collection = results_path_for_profile(files["exploratory"], profile_id), "exploratory"
    else:
        if args.production:
            if run_kind in STRESS_RUN_KINDS:
                ok, checks, context = run_stress_preflight(stress_config, study_config, profile_id, cli_overrides, args.concurrency, args.allow_peak_pricing)
            else:
                ok, checks, context = run_preflight(study_config, cli_overrides if profile_id == PRIMARY_PROFILE_ID else {}, args.concurrency, args.allow_peak_pricing)
            for name, check_ok, detail in checks:
                print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
            if "arithmetic" in context and "manifests" in context:
                print_preflight_summary(study_config, context)
            if not ok:
                raise SystemExit(f"v3 preflight FAILED -- refusing to start a {run_kind} run. No API calls were made.")
            if run_kind in STRESS_RUN_KINDS and not context.get("stage_ready", False):
                raise SystemExit(f"Stress stage for {run_kind} is not ready (see FAIL lines). No API calls were made.")
        if run_kind in STRESS_RUN_KINDS:
            settings = resolve_profile_settings(profile, _stress_replicates(run_kind, stress_config), stress_config["random_seed"],
                                                stress_config["retry_limit"], cli_overrides)
        elif profile_id == PRIMARY_PROFILE_ID:
            settings = resolve_production_settings(cli_overrides, study_config, run_kind)
            if not profile_matches_study_config(profile_id, study_config):
                raise SystemExit("The primary evaluator profile no longer matches the frozen study config.")
        else:
            settings = resolve_profile_settings(profile, resolve_production_settings({}, study_config, run_kind)["replicates"],
                                                study_config["random_seed"], study_config["retry_limit"], cli_overrides)
        results_file = results_path_for_profile(files["production"], profile_id)
        collection = "pilot" if run_kind == "pilot" else "production"

    provider, model, reasoning_profile = settings["provider"], settings["model"], settings["reasoning_profile"]
    reasoning_settings = model_providers.resolve_reasoning_settings(provider, reasoning_profile)

    trials = _load_trials_for(run_kind, study_config, stress_config)
    trials = limit_to_first_n_units(trials, args.limit)
    texts = load_story_texts()
    verify_prompt_hashes(trials, texts)
    plan = attach_observation_ids(plan_execution_order(trials, settings["replicates"], settings["seed"]), profile_id, _id_maker_for(run_kind))
    already_valid = load_valid_completed_ids(results_file, profile_id)
    remaining = filter_unresumed_plan(plan, already_valid, key="evaluation_observation_id")

    manifest_sha = "+".join(sha256_of_file(p) for p in _manifest_paths_for(run_kind, study_config, stress_config))

    print(f"Run kind: {run_kind}  Collection: {collection}  Evaluator profile: {profile_id}  Output: {results_file}")
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

    still_missing = len(plan) - len(load_valid_completed_ids(results_file, profile_id))
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
    extra = [{"primary": results_path_for_profile(PRIMARY_RESULTS_FILE, p), "holdout": results_path_for_profile(HOLDOUT_RESULTS_FILE, p)}
             for p in (getattr(args, "extra_profiles", None) or [])]
    analyze_controllability_v3.run_analysis(
        primary_results_file=args.primary_results, holdout_results_file=args.holdout_results,
        bootstrap_draws=args.bootstrap_draws, analysis_dir=args.analysis_dir, extra_profile_files=extra,
    )


# ---------------------------------------------------------------------------
# Stress families: preflight, attack generation, selection, dose generation,
# cost planning. Every function here makes zero API calls EXCEPT
# cmd_stress_adversarial_generate (the attacker model) and the _run-based
# search/eval/dose commands (the judge).
# ---------------------------------------------------------------------------

def _stress_results_files(profile_id):
    return {kind: results_path_for_profile(RUN_KIND_FILES[kind]["production"], profile_id) for kind in STRESS_RUN_KINDS}


def stress_cost_plan(stress_config, candidates=None, selected=None, assumed_throughput=DEFAULT_ASSUMED_JUDGMENTS_PER_SECOND):
    a = stress_config["adversarial"]
    n_valid = len(adv.valid_candidates(candidates)) if candidates else None
    n_sel = len(selected["selected"]) if selected else None
    adv_plan = adv.design_arithmetic(n_valid, n_sel, a["dev_replicates"], a["eval_replicates"])
    dose_plan = dose_module.design_arithmetic(stress_config["dose_response"]["replicates"])
    stages = {
        "adversarial_attack_generation_calls": adv_plan["attacker_calls"],
        "adversarial_development_search_judgments": adv_plan["dev_judgments"],
        "adversarial_heldout_evaluation_judgments": adv_plan["eval_judgments"],
        "dose_response_judgments": dose_plan["judgments"],
    }
    hours = {k: v / assumed_throughput / 3600 for k, v in stages.items() if k.endswith("judgments")}
    return {"stages": stages, "assumed_judgments_per_second": assumed_throughput, "estimated_hours": hours,
            "candidates_basis": "actual valid candidates" if n_valid is not None else "planned (8 per cue x intervention, all assumed valid)",
            "selected_basis": "actual selected attacks" if n_sel is not None else "planned (K=3 per cue x intervention)",
            "adversarial": adv_plan, "dose": dose_plan}


def print_stress_cost_plan(plan):
    print("=== stress cost / run plan ===")
    for k, v in plan["stages"].items():
        line = f"{k}: {v:,}"
        if k in plan["estimated_hours"]:
            line += f"  (~{plan['estimated_hours'][k]:.1f} h at {plan['assumed_judgments_per_second']:.0f} judgments/s -- ASSUMED throughput; none was ever recorded for v2/v3)"
        print(line)
    print(f"candidates basis: {plan['candidates_basis']};  selected basis: {plan['selected_basis']}")
    d = plan["dose"]
    print(f"dose design: {d['story_pairs']} pairs x {d['doses']} doses x {d['interventions']} interventions x 4 cells = {d['unique_cells']:,} cells; "
          f"{d['replicates']} reps -> {d['judgments']:,} judgments (10 reps would be {d['judgments_at_10_replicates']:,})")


def run_profile_preflight(profile_id, study_config):
    checks = []
    ok, reason = validate_profile(profile_id)
    checks.append(("evaluator_profile_exists_and_provider_supports_it", ok, reason or f"{profile_id}: {get_profile(profile_id)['provider']}/{get_profile(profile_id)['model']}"))
    if ok:
        p = get_profile(profile_id)
        try:
            settings = model_providers.resolve_reasoning_settings(p["provider"], p["reasoning_profile"])
            checks.append(("reasoning_setting_valid", True, f"{p['reasoning_profile']} -> {settings}"))
        except ValueError as e:
            checks.append(("reasoning_setting_valid", False, str(e)))
        checks.append(("max_output_tokens_safe", 4096 <= p["max_output_tokens"] <= 32768, str(p["max_output_tokens"])))
        if profile_id.startswith("attacker_"):
            checks.append(("profile_role", True, "attacker (generation stage only; never a judge)"))
        else:
            checks.append(("profile_role", True, "judge"))
        if profile_id == PRIMARY_PROFILE_ID:
            checks.append(("primary_profile_matches_frozen_study_config", profile_matches_study_config(profile_id, study_config), "ok"))
    return all(c[1] for c in checks), checks


def run_stress_preflight(stress_config, study_config, profile_id=PRIMARY_PROFILE_ID, cli_overrides=None, concurrency=DEFAULT_CONCURRENCY,
                         allow_peak_pricing=False, skip_credential_check=False, results_files=None, families=("adversarial", "dose")):
    """Every stress check, zero API calls. Returns (ok, checks, context);
    context["stage_ready"] says whether the requested run kinds' inputs are
    frozen and consistent (design stage for dev search / dose; attacks
    stage for the held-out evaluation)."""
    checks, context = [], {"concurrency": concurrency, "allow_peak_pricing": allow_peak_pricing, "stage_ready": False}

    ok, reason = verify_frozen()
    checks.append(("primary_v3_lock_verifies_no_primary_manifest_mutation", ok, reason or "ok"))
    ok, reason = verify_v2_unchanged()
    checks.append(("v2_files_unchanged_and_v2_lock_validates", ok, reason or "ok"))
    ok, reason = verify_stress_frozen("design")
    checks.append(("stress_design_stage_frozen", ok, reason or "ok"))
    design_ok = ok

    p_ok, p_checks = run_profile_preflight(profile_id, study_config)
    checks.extend(p_checks)
    if profile_id.startswith("attacker_"):
        checks.append(("judge_profile_is_not_an_attacker", False, f"{profile_id} is an attacker profile and may never judge"))
        p_ok = False

    texts = load_story_texts()
    a, d = stress_config["adversarial"], stress_config["dose_response"]

    if "adversarial" in families:
        def split_ok():
            split = sd.load_split(a["split_file"])
            sd.assert_split_well_formed(split)
            context["split"] = split
            return f"{len(split['attack_development_pairs'])} development / {len(split['attack_evaluation_pairs'])} evaluation pairs, zero overlap"
        _check(checks, "attack_split_frozen_partitions_pairs_with_zero_leakage", split_ok)

        candidates = adv.load_candidates(a["candidates_file"])
        context["candidates"] = candidates
        if candidates:
            def cands_ok():
                adv.assert_candidates_well_formed(candidates, texts)
                valid = adv.valid_candidates(candidates)
                keys = {(c["cue_id"], c["intervention_id"]) for c in valid}
                return f"{len(valid)} valid of {sum(1 for c in candidates if c.get('generation_status') == 'ok')} generated; {len(keys)}/{len(CUE_ORDER) * len(INTERVENTION_IDS)} (cue, intervention) keys covered"
            _check(checks, "attack_candidates_satisfy_constraints_with_unique_ids", cands_ok)
            if os.path.exists(a["dev_manifest_file"]):
                def dev_ok():
                    trials = sd.load_manifest(a["dev_manifest_file"])
                    by_id = {c["attack_id"]: c for c in adv.valid_candidates(candidates)}
                    adv.assert_attack_manifest_well_formed(trials, sd.FAMILY_ADV_DEV, context["split"], by_id)
                    verify_prompt_hashes(trials, texts)
                    return f"{len(trials)} dev cells, development pairs only, prompts hash-verified"
                _check(checks, "attack_dev_manifest_uses_development_pairs_only_and_hashes", dev_ok)
        else:
            checks.append(("attack_candidates_present", True, "none generated yet (run stress-adversarial-generate)"))

        if os.path.exists(a["selected_attacks_file"]):
            def sel_ok():
                selected = adv.load_selected(a["selected_attacks_file"])
                adv.assert_selection_well_formed(selected, candidates, context["split"], a["top_k"])
                context["selected"] = selected
                return f"{len(selected['selected'])} selected attacks (K={a['top_k']})"
            _check(checks, "selected_top_k_attacks_exist_and_are_valid_candidates", sel_ok)
            if os.path.exists(a["eval_manifest_file"]) and "selected" in context:
                def eval_ok():
                    trials = sd.load_manifest(a["eval_manifest_file"])
                    by_id = {r["attack_id"]: r for r in context["selected"]["selected"]}
                    adv.assert_attack_manifest_well_formed(trials, sd.FAMILY_ADV_EVAL, context["split"], by_id)
                    verify_prompt_hashes(trials, texts)
                    return f"{len(trials)} held-out cells, evaluation pairs only, frozen attack ids, prompts hash-verified"
                _check(checks, "attack_eval_manifest_uses_heldout_pairs_and_frozen_attack_ids", eval_ok)
            ok, reason = verify_stress_frozen("attacks")
            checks.append(("stress_attacks_stage_frozen", ok, reason or "ok"))
            context["attacks_ok"] = ok
        else:
            context["attacks_ok"] = False
            checks.append(("selected_attacks_present", True, "not selected yet (run stress-adversarial-select after the search)"))

    if "dose" in families:
        def dose_ok():
            trials = sd.load_manifest(d["manifest_file"])
            dose_module.assert_dose_manifest_well_formed(trials)
            verify_prompt_hashes(trials, texts)
            arith = dose_module.design_arithmetic(d["replicates"])
            if len(trials) != arith["unique_cells"] or tuple(d["grid"]) != sd.DOSE_GRID:
                raise ValueError("dose manifest / grid disagree with the frozen design")
            return f"{len(trials)} cells, grid {tuple(d['grid'])}, direction/position/intervention balanced, prompts hash-verified"
        _check(checks, "dose_manifest_frozen_grid_balanced_expected_counts_and_hashes", dose_ok)

    def outputs_clean():
        files = results_files or _stress_results_files(profile_id)
        problems = []
        for kind, path in files.items():
            dup = find_duplicate_valid_ids(path)
            foreign = find_foreign_ids(path, RUN_KIND_FILES[kind]["families"], ("v3s::",))
            if dup:
                problems.append(f"{path}: {len(dup)} duplicate valid id(s)")
            if foreign:
                problems.append(f"{path}: {len(foreign)} foreign row(s)")
        if problems:
            raise ValueError("; ".join(problems))
    _check(checks, "stress_outputs_have_no_duplicate_valid_or_foreign_ids", outputs_clean)

    if p_ok:
        profile = get_profile(profile_id)
        try:
            for kind in STRESS_RUN_KINDS:
                resolve_profile_settings(profile, _stress_replicates(kind, stress_config), stress_config["random_seed"], stress_config["retry_limit"], cli_overrides)
            checks.append(("cli_settings_match_evaluator_profile", True, "ok"))
        except ValueError as e:
            checks.append(("cli_settings_match_evaluator_profile", False, str(e)))
        if skip_credential_check:
            checks.append(("api_credential_present", True, "skipped (--skip-credential-check)"))
        else:
            try:
                model_providers.get_api_key(profile["provider"])
                checks.append(("api_credential_present", True, "ok"))
            except RuntimeError as e:
                checks.append(("api_credential_present", False, str(e)))

    context["stage_ready"] = all(c[1] for c in checks) and design_ok
    context["plan"] = stress_cost_plan(stress_config, context.get("candidates"), context.get("selected"))
    return all(c[1] for c in checks), checks, context


def cmd_stress_preflight(args):
    study_config, stress_config = load_study_config(), load_stress_config()
    families = {"adversarial": ("adversarial",), "dose": ("dose",), "all": ("adversarial", "dose")}[args.family]
    ok, checks, context = run_stress_preflight(stress_config, study_config, args.evaluator_profile, {}, args.concurrency, args.allow_peak_pricing,
                                               args.skip_credential_check, families=families)
    for name, check_ok, detail in checks:
        print(f"[{'OK' if check_ok else 'FAIL'}] {name}: {detail}")
    plan = stress_cost_plan(stress_config, context.get("candidates"), context.get("selected"), args.assumed_throughput)
    print_stress_cost_plan(plan)
    if not ok:
        raise SystemExit("v3 stress preflight FAILED -- see the FAIL line(s) above. No API calls were made.")
    print("v3 stress preflight PASSED. No API calls were made.")


def cmd_profile_preflight(args):
    study_config = load_study_config()
    all_ok = True
    for pid in (args.profiles or profile_ids()):
        ok, checks = run_profile_preflight(pid, study_config)
        all_ok &= ok
        for name, check_ok, detail in checks:
            print(f"[{'OK' if check_ok else 'FAIL'}] {pid}: {name}: {detail}")
    if not all_ok:
        raise SystemExit("Evaluator profile preflight FAILED. No API calls were made.")
    print("Evaluator profile preflight PASSED. No API calls were made.")


def cmd_stress_dose_generate(args):
    trials = dose_module.build_dose_trials()
    dose_module.assert_dose_manifest_well_formed(trials)
    sd.write_manifest(trials, sd.DOSE_TRIALS_FILE)
    a = dose_module.design_arithmetic()
    print(f"Wrote {len(trials)} dose cells to {sd.DOSE_TRIALS_FILE}; grid {sd.DOSE_GRID}; {a['replicates']} reps -> {a['judgments']:,} judgments")


def cmd_stress_dose_preflight(args):
    args.family = "dose"
    cmd_stress_preflight(args)


def cmd_stress_adversarial_generate(args):
    """The ONLY command that calls the ATTACKER model. Resumable: (cue,
    intervention) keys with a successful generation are skipped. Writes
    candidates (with full provenance) then rebuilds the development
    manifest. --dry-run prints the plan and one prompt, making no calls."""
    stress_config = load_stress_config()
    a = stress_config["adversarial"]
    ok, reason = verify_stress_frozen("design")
    if not ok and not args.allow_unfrozen:
        raise SystemExit(f"Stress design stage is not frozen ({reason}); freeze it or pass --allow-unfrozen for an exploratory generation.")
    attacker_id = args.attacker_profile or a["default_attacker_profile_id"]
    p_ok, p_reason = validate_profile(attacker_id)
    if not p_ok:
        raise SystemExit(f"Attacker profile invalid: {p_reason}")
    attacker = get_profile(attacker_id)
    contrasts = load_contrasts(load_study_config()["contrast_file"])
    texts = load_story_texts()
    existing = adv.load_candidates(a["candidates_file"])
    todo = [(c, i) for c in CUE_ORDER for i in INTERVENTION_IDS if (c, i) not in adv.generated_keys(existing)]
    print(f"Attacker profile: {attacker_id} ({attacker['provider']}/{attacker['model']}/{attacker['reasoning_profile']}, max_output_tokens={attacker['max_output_tokens']})")
    print(f"(cue, intervention) keys to generate: {len(todo)} of {len(CUE_ORDER) * len(INTERVENTION_IDS)}  ({a['candidates_per_cue_intervention']} candidates each)")
    if args.dry_run:
        print("Dry run: no attacker calls made. Example prompt for the first key:")
        if todo:
            by_id = {c["contrast_id"]: c for c in contrasts}
            print(adv.build_generation_prompt(by_id[todo[0][0]], todo[0][1], a["candidates_per_cue_intervention"]))
        return
    pricing_gate = DeepSeekPricingGate(allow_peak=args.allow_peak_pricing)
    call_fn = make_call_fn(attacker["provider"], attacker["model"], attacker["reasoning_profile"], attacker["max_output_tokens"], pricing_gate)
    with SafeJsonlWriter(a["candidates_file"]) as writer:
        n = adv.generate_candidates(call_fn, attacker, contrasts, texts, existing, writer.write, a["candidates_per_cue_intervention"], a["generation_attempts"])
    print(f"Generated {n} (cue, intervention) key(s); candidates appended to {a['candidates_file']}")
    _build_dev_manifest(stress_config, texts)


def _build_dev_manifest(stress_config, texts=None):
    a = stress_config["adversarial"]
    texts = texts or load_story_texts()
    candidates = adv.load_candidates(a["candidates_file"])
    adv.assert_candidates_well_formed(candidates, texts)
    split = sd.load_split(a["split_file"])
    trials = adv.build_dev_trials(candidates, split, texts)
    adv.assert_attack_manifest_well_formed(trials, sd.FAMILY_ADV_DEV, split, {c["attack_id"]: c for c in adv.valid_candidates(candidates)})
    sd.write_manifest(trials, a["dev_manifest_file"])
    valid = adv.valid_candidates(candidates)
    print(f"Development manifest: {len(valid)} valid attacks x {sd.N_DEV_PAIRS} development pairs x 4 cells = {len(trials)} cells -> {a['dev_manifest_file']}")
    return trials


def cmd_stress_adversarial_build_dev(args):
    _build_dev_manifest(load_stress_config())


def cmd_stress_adversarial_search(args):
    _run(args, "stress_adv_dev")


def cmd_stress_adversarial_select(args):
    """Rank candidates by the pre-specified development score and freeze
    the top K per (cue, intervention) into SELECTED_ATTACKS_FILE + the
    held-out evaluation manifest. Uses development results only; refuses
    to run once a held-out evaluation file exists (never re-select after
    looking at evaluation data)."""
    import analyze_controllability_v3_stress as ast
    stress_config, study_config = load_stress_config(), load_study_config()
    a = stress_config["adversarial"]
    profile_id = args.evaluator_profile
    eval_file = results_path_for_profile(ADV_EVAL_RESULTS_FILE, profile_id)
    if os.path.exists(eval_file) and not args.force:
        raise SystemExit(f"Held-out evaluation results already exist at {eval_file}; selection must not be redone after evaluation (use --force only to rebuild an identical selection).")
    candidates = adv.load_candidates(a["candidates_file"])
    split = sd.load_split(a["split_file"])
    texts = load_story_texts()
    dev_rows = ast.load_rows(args.dev_results or results_path_for_profile(ADV_DEV_RESULTS_FILE, profile_id))
    dev_by_id, _ = ast.an.include_rows(dev_rows, (sd.FAMILY_ADV_DEV,), study_config, profile_id, id_prefix="v3s::")
    valid = [e["valid"] for e in dev_by_id.values() if e["valid"] is not None]
    kept, _, _ = ast.an.complete_units(valid, 4, dev_by_id)
    dev_effects = {k[2]: v for k, v in ast.variant_pair_effects(ast.stress_cell_rates(kept)).items()}
    leaked = {p for pairs in dev_effects.values() for p in pairs if f"{p[0]}_vs_{p[1]}" not in set(split["attack_development_pairs"])}
    if leaked:
        raise SystemExit(f"Development results contain evaluation pairs {sorted(leaked)[:3]} -- refusing to select.")
    ordinary = {}
    primary_file = args.primary_results or results_path_for_profile(PRIMARY_RESULTS_FILE, profile_id)
    primary_rows = ast.load_rows(primary_file)
    if primary_rows:
        by_id, _ = ast.an.include_rows(primary_rows, (FAMILY_PRIMARY_CONTEXT,), study_config, profile_id)
        pv = [e["valid"] for e in by_id.values() if e["valid"] is not None]
        kept_p, _, _ = ast.an.complete_units(pv, 4, by_id)
        ordinary = ast.ordinary_pair_effects(ast.an.pair_context_effects(ast.an.context_cell_rates(kept_p)), set(split["attack_development_pairs"]))
    selected, ranked = adv.rank_candidates(dev_effects, ordinary, candidates, a["top_k"], a["min_dev_pairs_for_ranking"])
    if not selected:
        raise SystemExit("No candidate reached the minimum number of complete development pairs; nothing selected.")
    out = {"k": a["top_k"], "score_definition": a["score_definition"], "evaluator_profile_id": profile_id, "dev_results_file": args.dev_results or results_path_for_profile(ADV_DEV_RESULTS_FILE, profile_id),
           "ordinary_reference_available": bool(ordinary), "split_sha256": adv.sha256_hex(json.dumps(split, sort_keys=True)),
           "candidates_sha256": sha256_of_file(a["candidates_file"]), "selected": selected, "ranked": ranked,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(a["selected_attacks_file"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    trials = adv.build_eval_trials(out, split, texts)
    adv.assert_attack_manifest_well_formed(trials, sd.FAMILY_ADV_EVAL, split, {r["attack_id"]: r for r in selected})
    sd.write_manifest(trials, a["eval_manifest_file"])
    keys = {(r["cue_id"], r["intervention_id"]) for r in selected}
    print(f"Selected {len(selected)} attacks over {len(keys)} (cue, intervention) keys (K={a['top_k']}; basis: {'attack minus ordinary' if ordinary else 'attack only'}) -> {a['selected_attacks_file']}")
    print(f"Held-out evaluation manifest: {len(selected)} attacks x {sd.N_EVAL_PAIRS} pairs x 4 cells = {len(trials)} cells -> {a['eval_manifest_file']}")
    print("Next: python3 freeze_controllability_v3_stress.py attacks   (then stress-adversarial-run --production)")


def cmd_stress_adversarial_run(args):
    _run(args, "stress_adv_eval")


def cmd_stress_dose_run(args):
    _run(args, "stress_dose")


def cmd_stress_analysis(args):
    import analyze_controllability_v3_stress as ast
    pid = args.evaluator_profile
    extra = [{"primary": results_path_for_profile(PRIMARY_RESULTS_FILE, p), "dev": results_path_for_profile(ADV_DEV_RESULTS_FILE, p),
              "eval": results_path_for_profile(ADV_EVAL_RESULTS_FILE, p), "dose": results_path_for_profile(DOSE_RESULTS_FILE, p)}
             for p in (args.extra_profiles or [])]
    ast.run_stress_analysis(results_path_for_profile(PRIMARY_RESULTS_FILE, pid), results_path_for_profile(ADV_DEV_RESULTS_FILE, pid),
                            results_path_for_profile(ADV_EVAL_RESULTS_FILE, pid), results_path_for_profile(DOSE_RESULTS_FILE, pid),
                            args.analysis_dir, args.bootstrap_draws, extra_profile_files=extra)


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
    parser.add_argument("--evaluator-profile", default=PRIMARY_PROFILE_ID, choices=[p for p in profile_ids() if not p.startswith("attacker_")],
                         help=f"Evaluator profile (model-side identity); the scientific manifest never changes. Default: {PRIMARY_PROFILE_ID}")
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
    p.add_argument("--extra-profiles", nargs="*", default=None, help="Other evaluator profiles whose result files should be analysed (separately) too")
    p.set_defaults(func=cmd_analysis)

    # ---- stress families ----
    judge_profiles = [x for x in profile_ids() if not x.startswith("attacker_")]

    p = sub.add_parser("profile-preflight", help="Validate evaluator profiles against model_providers; no API calls")
    p.add_argument("--profiles", nargs="*", default=None)
    p.set_defaults(func=cmd_profile_preflight)

    for name, fn, family in (("stress-preflight", cmd_stress_preflight, "all"), ("stress-dose-preflight", cmd_stress_dose_preflight, "dose")):
        p = sub.add_parser(name, help=f"Stress preflight ({family}); zero API calls")
        p.add_argument("--family", choices=["adversarial", "dose", "all"], default=family)
        p.add_argument("--evaluator-profile", default=PRIMARY_PROFILE_ID, choices=judge_profiles)
        p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
        p.add_argument("--allow-peak-pricing", action="store_true")
        p.add_argument("--skip-credential-check", action="store_true")
        p.add_argument("--assumed-throughput", type=float, default=DEFAULT_ASSUMED_JUDGMENTS_PER_SECOND, help="judgments/s used ONLY for the runtime estimate")
        p.set_defaults(func=fn)

    p = sub.add_parser("stress-dose-generate", help="(Re)generate the deterministic dose-response manifest; no API calls")
    p.set_defaults(func=cmd_stress_dose_generate)

    p = sub.add_parser("stress-adversarial-generate", help="ATTACKER model generates candidate re-framings (paid); resumable")
    p.add_argument("--attacker-profile", default=None, choices=[x for x in profile_ids()])
    p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("--allow-peak-pricing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_stress_adversarial_generate)

    p = sub.add_parser("stress-adversarial-build-dev", help="Rebuild the development manifest from the candidates file; no API calls")
    p.set_defaults(func=cmd_stress_adversarial_build_dev)

    p = sub.add_parser("stress-adversarial-search", help="Judge evaluates candidate attacks on DEVELOPMENT pairs (paid; resumable)")
    add_execution_arguments(p)
    p.set_defaults(func=cmd_stress_adversarial_search)

    p = sub.add_parser("stress-adversarial-select", help="Rank by the pre-specified dev score, freeze top-K, build the held-out manifest; no API calls")
    p.add_argument("--evaluator-profile", default=PRIMARY_PROFILE_ID, choices=judge_profiles)
    p.add_argument("--dev-results", default=None)
    p.add_argument("--primary-results", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_stress_adversarial_select)

    p = sub.add_parser("stress-adversarial-run", help="Judge evaluates the frozen selected attacks on HELD-OUT pairs (paid; resumable)")
    add_execution_arguments(p)
    p.set_defaults(func=cmd_stress_adversarial_run)

    p = sub.add_parser("stress-dose-run", help="Judge evaluates the dose-response manifest (paid; resumable)")
    add_execution_arguments(p)
    p.set_defaults(func=cmd_stress_dose_run)

    p = sub.add_parser("stress-analysis", help="Stress analysis for one profile (+ optional extra profiles, analysed separately); no API calls")
    p.add_argument("--evaluator-profile", default=PRIMARY_PROFILE_ID, choices=judge_profiles)
    p.add_argument("--extra-profiles", nargs="*", default=None, choices=judge_profiles)
    p.add_argument("--analysis-dir", default=STRESS_ANALYSIS_DIR)
    p.add_argument("--bootstrap-draws", type=int, default=None)
    p.set_defaults(func=cmd_stress_analysis)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
