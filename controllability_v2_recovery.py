"""Wave 2 recovery for the v2 context-controllability experiment: retain
every Wave 1 (run 35355521243) planned observation that already has a valid
parsed A/B answer, rerun only the ones that don't, add a genuinely new
no-context + text_only baseline condition, merge Wave 1 + recovery into one
analysis-ready dataset, and support a small non-primary validation sample
that reruns already-resolved Wave 1 observations to check answer stability.

This module is pure logic: it reads/writes plain Python data structures
(and, via the small file-loading helpers, JSONL files) and never makes a
model API call itself -- run_controllability_v2.py's recovery/validation/
baseline-text-only subcommands are the only things that ever dispatch a real
request, exactly the same call path (model_providers.call_model via
controllability_v2_execution.run_one_observation) the original Wave 1 run
used, just with max_output_tokens raised to RECOVERY_MAX_OUTPUT_TOKENS.

Key invariant, enforced structurally rather than merely by convention: a
recovery/validation observation's identity is always derived from something
that already existed in the frozen Wave 1 design (a planned observation_id
for recovery; an already-resolved Wave 1 observation_id for validation).
Neither path can manufacture a new scientific replicate -- see
build_recovery_plan and assert_recovery_plan_matches_planned_index.
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

from controllability_v2_execution import make_observation_id
from controllability_v2_trials import EXPERIMENT_ID
from run_trial import parse_plain_ab_response

# The specific Wave 1 GitHub Actions run this recovery wave is recovering.
SOURCE_RUN_ID = "35355521243"
SOURCE_WAVE = "wave1"

COLLECTION_WAVE_RECOVERY = "wave2_recovery"
COLLECTION_WAVE_BASELINE_TEXT_ONLY = "wave2_baseline_text_only"
COLLECTION_WAVE_VALIDATION = "wave2_validation"

# The ONLY setting changed for any Wave 2 API call relative to Wave 1's
# frozen max_output_tokens=512 (see controllability_v2_study_config.py).
# Historical Wave 1 records are never touched or retroactively relabeled.
RECOVERY_MAX_OUTPUT_TOKENS = 4096

VALIDATION_SAMPLE_SIZE = 500

WAVE1_TREATMENT_RAW_FILENAME = "treatment_raw.jsonl"
WAVE1_BASELINE_RAW_FILENAME = "baseline_raw.jsonl"

_RESOLVED_OBSERVATION_FIELDS = (
    "attempts", "total_attempts", "first_attempt_status", "first_valid_response",
    "parsing_status", "refusal_status", "api_error_status",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_valid_completed_observation_ids(results_file, id_field="observation_id"):
    """Wave 2 resume semantics -- deliberately DIFFERENT from
    controllability_v2_execution.load_completed_observation_ids, which is
    correct for Wave 1's OWN execution (any terminal row, success or a
    retries-exhausted failure, is "done, never repeat" -- and is left
    completely untouched by this module).

    For Wave 2 (recovery-run/baseline-text-only/validation-run), an
    observation counts as completed for RESUME purposes only if a row
    exists with a genuinely valid, re-parseable A/B answer
    (parsing_status == "resolved"). A Wave 2 API failure -- a 402, a 429,
    a network error, an exhausted-retries non-answer, a malformed/no-answer
    response, or hitting the 4096-token cap without ever producing A/B --
    must remain eligible for a later rerun of the SAME Wave 2 command; that
    is the entire point of a recovery wave that is itself resumable across
    interrupted/rerun GitHub Actions jobs. Multiple non-valid rows may
    accumulate for the same id across reruns (harmless, never confused
    with a valid answer); once one valid row exists for an id, every
    later invocation skips it -- see merge_wave1_and_recovery, which
    already only ever keeps a SINGLE valid answer per id and flags a
    second one as a duplicate (structurally shouldn't happen once this
    function is used to gate dispatch, but is still checked defensively).

    Returns an empty set if the file doesn't exist yet."""
    if not os.path.exists(results_file):
        return set()
    ids = set()
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("parsing_status") == "resolved":
                oid = row.get(id_field)
                if oid is not None:
                    ids.add(oid)
    return ids


def load_wave1_raw_rows(source_dir):
    """Reads treatment_raw.jsonl + baseline_raw.jsonl from `source_dir` (the
    extracted Wave 1 production artifact for run SOURCE_RUN_ID). Raises
    FileNotFoundError if EITHER file is missing -- recovery must never
    silently treat "no Wave 1 data found" as "everything is unresolved".
    Returns (treatment_rows, baseline_rows); the original files are only
    ever read here, never opened for writing by anything in this module."""
    treatment_path = os.path.join(source_dir, WAVE1_TREATMENT_RAW_FILENAME)
    baseline_path = os.path.join(source_dir, WAVE1_BASELINE_RAW_FILENAME)
    if not os.path.exists(treatment_path):
        raise FileNotFoundError(f"Wave 1 treatment results not found: {treatment_path}")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Wave 1 baseline results not found: {baseline_path}")
    return load_jsonl(treatment_path), load_jsonl(baseline_path)


# ---------------------------------------------------------------------------
# The frozen Wave 1 planned-observation index (item: "derive the exact count
# from the actual raw data" -- this is the independent, manifest-derived
# side of that comparison; classify_wave1_rows is the raw-data side)
# ---------------------------------------------------------------------------

def enumerate_planned_ids(trials, replicate_count, evaluator_id, unit_kind, experiment_id=EXPERIMENT_ID):
    """{observation_id: {"trial_id", "replicate_number", "unit_kind"}} for
    every planned (trial, replicate) pair, using the exact same
    make_observation_id construction the real execution path uses -- so
    these ids match real Wave 1 rows without depending on execution order
    (which is randomised and therefore not reproducible from the manifest
    alone)."""
    planned = {}
    for trial in trials:
        for replicate_number in range(1, replicate_count + 1):
            oid = make_observation_id(experiment_id, evaluator_id, trial["trial_id"], replicate_number)
            planned[oid] = {"trial_id": trial["trial_id"], "replicate_number": replicate_number, "unit_kind": unit_kind}
    return planned


def build_wave1_planned_index(treatment_trials, baseline_trials, evaluator_id,
                               treatment_replicate_count, baseline_replicate_count):
    """The full planned Wave 1 index: 2640*10 treatment + 132*10 baseline =
    27720 entries when run against the real frozen manifests/replicate
    counts. Raises ValueError if the treatment/baseline observation_id
    spaces somehow collide (they never should, since trial_ids differ)."""
    treatment_index = enumerate_planned_ids(treatment_trials, treatment_replicate_count, evaluator_id, "treatment")
    baseline_index = enumerate_planned_ids(baseline_trials, baseline_replicate_count, evaluator_id, "baseline")
    overlap = set(treatment_index) & set(baseline_index)
    if overlap:
        raise ValueError(f"Baseline and treatment observation_id spaces collide: {sorted(overlap)[:5]}")
    index = dict(treatment_index)
    index.update(baseline_index)
    return index


# ---------------------------------------------------------------------------
# Classifying the actual Wave 1 raw rows against the planned index
# ---------------------------------------------------------------------------

def _resolving_attempt(row):
    attempts = row.get("attempts") or []
    return attempts[-1] if attempts else None


def classify_wave1_rows(all_rows, planned_index):
    """Partitions every raw Wave 1 result row against `planned_index`.

    Returns a dict:
      valid_by_id: {observation_id: row} -- exactly one clean, re-parseable,
        non-duplicated valid answer per id.
      unresolved_ids: sorted list of ids in planned_index with no valid_by_id
        entry (covers both "row missing entirely" and "row present but
        parsing_status != resolved").
      duplicate_valid_ids: {observation_id: [rows]} -- more than one row
        independently claims parsing_status == "resolved" for the same
        planned id (file corruption/concatenation) -- a preflight failure.
      unknown_ids: [observation_id, ...] -- rows whose observation_id is not
        in planned_index at all (trial identity drift from the frozen
        design) -- a preflight failure.
      malformed_valid_ids: {observation_id: [rows]} -- rows claiming
        parsing_status == "resolved" whose own recorded response_text does
        not actually re-parse to the recorded first_valid_response -- a
        preflight failure; these are treated as neither valid nor as a
        normal duplicate.
    """
    by_id = defaultdict(list)
    unknown_ids = []
    for row in all_rows:
        oid = row.get("observation_id")
        if oid is None or oid not in planned_index:
            unknown_ids.append(oid)
        else:
            by_id[oid].append(row)

    valid_by_id = {}
    duplicate_valid_ids = {}
    malformed_valid_ids = {}
    for oid, rows in by_id.items():
        resolved_rows = [r for r in rows if r.get("parsing_status") == "resolved"]
        clean_rows = []
        for row in resolved_rows:
            attempt = _resolving_attempt(row)
            reparsed, _err = parse_plain_ab_response(attempt.get("response_text")) if attempt else (None, "no attempts recorded")
            if reparsed is None or reparsed != row.get("first_valid_response"):
                malformed_valid_ids.setdefault(oid, []).append(row)
            else:
                clean_rows.append(row)

        if len(clean_rows) > 1:
            duplicate_valid_ids[oid] = clean_rows
        elif len(clean_rows) == 1:
            valid_by_id[oid] = clean_rows[0]

    unresolved_ids = sorted(set(planned_index) - set(valid_by_id))
    return {
        "valid_by_id": valid_by_id,
        "unresolved_ids": unresolved_ids,
        "duplicate_valid_ids": duplicate_valid_ids,
        "unknown_ids": unknown_ids,
        "malformed_valid_ids": malformed_valid_ids,
    }


def validate_wave1_consistency(classification):
    """Raises ValueError on any of the internal-inconsistency conditions
    recovery-preflight must fail on. Does not check for missing source
    files (load_wave1_raw_rows already raises FileNotFoundError for that)
    or for recovery accidentally creating new replicates (structurally
    impossible -- see build_recovery_plan/assert_recovery_plan_matches_planned_index)."""
    if classification["unknown_ids"]:
        sample = classification["unknown_ids"][:5]
        raise ValueError(
            f"{len(classification['unknown_ids'])} Wave 1 result row(s) reference an observation_id "
            f"that does not belong to the frozen v2 design (trial identities do not match): {sample}"
        )
    if classification["duplicate_valid_ids"]:
        sample = sorted(classification["duplicate_valid_ids"])[:5]
        raise ValueError(
            f"Duplicate valid Wave 1 answers found for {len(classification['duplicate_valid_ids'])} "
            f"planned observation(s): {sample}"
        )
    if classification["malformed_valid_ids"]:
        sample = sorted(classification["malformed_valid_ids"])[:5]
        raise ValueError(
            f"{len(classification['malformed_valid_ids'])} Wave 1 row(s) marked resolved but do not "
            f"actually re-parse to their recorded answer: {sample}"
        )


# ---------------------------------------------------------------------------
# Recovery plan: one entry per unresolved planned observation, reusing its
# EXACT trial_id/replicate_number/observation_id -- never a new replicate.
# ---------------------------------------------------------------------------

def build_recovery_plan(unresolved_ids, planned_index, trials_by_id):
    """Deterministic (sorted-by-id) plan: one entry per unresolved planned
    observation, carrying the SAME observation_id/trial_id/replicate_number/
    unit_kind the original Wave 1 plan used. A recovery attempt is not a new
    scientific replicate -- this function cannot introduce one, since every
    entry is keyed off an id already present in `planned_index`."""
    plan = []
    for index, oid in enumerate(sorted(unresolved_ids), start=1):
        descriptor = planned_index[oid]
        trial = trials_by_id[descriptor["trial_id"]]
        plan.append({
            "observation_id": oid,
            "trial_id": descriptor["trial_id"],
            "replicate_number": descriptor["replicate_number"],
            "unit_kind": descriptor["unit_kind"],
            "trial": trial,
            "execution_order_index": index,
        })
    return plan


def assert_recovery_plan_matches_planned_index(recovery_plan, planned_index):
    """Defense-in-depth structural check: every recovery plan entry must
    reuse an observation_id/trial_id/replicate_number triple that already
    exists in the frozen planned index, and no observation_id may appear
    twice in the plan. Raises ValueError otherwise (recovery preflight's
    "recovery would accidentally create new replicates" failure mode)."""
    seen = set()
    for entry in recovery_plan:
        oid = entry["observation_id"]
        if oid in seen:
            raise ValueError(f"observation_id {oid!r} appears more than once in the recovery plan")
        seen.add(oid)
        if oid not in planned_index:
            raise ValueError(f"Recovery plan entry {oid!r} is not a planned Wave 1 observation")
        descriptor = planned_index[oid]
        if entry["trial_id"] != descriptor["trial_id"] or entry["replicate_number"] != descriptor["replicate_number"]:
            raise ValueError(f"Recovery plan entry {oid!r} does not match its planned trial_id/replicate_number")


def _unit_id_field(unit_kind):
    return "superblock_id" if unit_kind == "treatment" else "block_id"


def build_recovery_result_row(plan_entry, evaluator_identity, observation,
                               collection_wave=COLLECTION_WAVE_RECOVERY,
                               source_wave=SOURCE_WAVE, source_run_id=SOURCE_RUN_ID,
                               max_output_tokens=RECOVERY_MAX_OUTPUT_TOKENS):
    """The Wave 2 recovery result row schema: everything a normal v2 result
    row carries (trial_id/replicate_number/unit-id field/evaluator/
    trial_meta/attempts/parsing_status/...) PLUS the recovery provenance
    fields the task requires: collection_wave, max_output_tokens (the
    RECOVERY value, never the historical 512), source_wave, source_run_id."""
    trial = plan_entry["trial"]
    unit_id_field = _unit_id_field(plan_entry["unit_kind"])
    return {
        "observation_id": plan_entry["observation_id"],
        "experiment_id": EXPERIMENT_ID,
        "trial_id": plan_entry["trial_id"],
        unit_id_field: trial[unit_id_field],
        "replicate_number": plan_entry["replicate_number"],
        "execution_order_index": plan_entry["execution_order_index"],
        "evaluator": evaluator_identity,
        "trial_meta": {k: v for k, v in trial.items() if k != "prompt"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_wave": collection_wave,
        "max_output_tokens": max_output_tokens,
        "source_wave": source_wave,
        "source_run_id": source_run_id,
        **{k: observation[k] for k in _RESOLVED_OBSERVATION_FIELDS},
    }


# ---------------------------------------------------------------------------
# New no-context + text_only baseline condition -- ordinary v2 execution
# plan/result rows (reuses the treatment/baseline row shape), just tagged
# with a distinct collection_wave/condition_id so it is never confused with
# the original matched_control baseline.
# ---------------------------------------------------------------------------

def build_baseline_text_only_result_row(plan_entry, evaluator_identity, observation,
                                         collection_wave=COLLECTION_WAVE_BASELINE_TEXT_ONLY):
    trial = plan_entry["trial"]
    return {
        "observation_id": plan_entry["observation_id"],
        "experiment_id": EXPERIMENT_ID,
        "trial_id": plan_entry["trial_id"],
        "block_id": trial["block_id"],
        "replicate_number": plan_entry["replicate_number"],
        "execution_order_index": plan_entry["execution_order_index"],
        "random_seed": plan_entry.get("random_seed"),
        "evaluator": evaluator_identity,
        "trial_meta": {k: v for k, v in trial.items() if k != "prompt"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_wave": collection_wave,
        "condition_id": "baseline_text_only",
        **{k: observation[k] for k in _RESOLVED_OBSERVATION_FIELDS},
    }


# ---------------------------------------------------------------------------
# Merge: Wave 1 valid answers take precedence; recovery fills the rest.
# Never double-counts, never invents a replicate.
# ---------------------------------------------------------------------------

def merge_wave1_and_recovery(planned_index, wave1_valid_by_id, recovery_rows):
    """Deterministic merge: for every PLANNED Wave 1 observation, (1) use
    the Wave 1 answer if it is valid, (2) otherwise use the first valid
    Wave 2 recovery answer for that same id, (3) otherwise it remains
    unresolved and is simply absent from the merged dataset (exactly like
    an unresolved observation was already handled upstream of merging).
    Every kept row is tagged with which source it came from and, for
    recovery rows, keeps their collection_wave/source_wave/source_run_id
    provenance fields untouched (they were already stamped on write).

    Returns (merged_by_id, diagnostics). `merged_by_id` never contains more
    than one row per observation_id (dict-keyed) and never introduces an id
    outside `planned_index`."""
    recovery_valid_by_id = {}
    recovery_duplicate_ids = defaultdict(list)
    for row in recovery_rows:
        oid = row.get("observation_id")
        if oid not in planned_index or oid in wave1_valid_by_id:
            continue  # not a planned id, or Wave 1 already has the authoritative answer
        if row.get("parsing_status") != "resolved":
            continue
        if oid in recovery_valid_by_id:
            recovery_duplicate_ids[oid].append(row)
        else:
            recovery_valid_by_id[oid] = row

    merged_by_id = {}
    for oid in planned_index:
        if oid in wave1_valid_by_id:
            row = dict(wave1_valid_by_id[oid])
            row["result_source"] = "wave1_original"
        elif oid in recovery_valid_by_id:
            row = dict(recovery_valid_by_id[oid])
            row["result_source"] = "wave2_recovery"
        else:
            continue
        merged_by_id[oid] = row

    n_planned = len(planned_index)
    n_wave1_valid = len(wave1_valid_by_id)
    n_recovery_attempted = len(recovery_rows)
    n_recovery_resolved = len(recovery_valid_by_id)
    diagnostics = {
        "wave1_planned_total": n_planned,
        "wave1_valid_retained": n_wave1_valid,
        "wave1_unresolved": n_planned - n_wave1_valid,
        "recovery_attempted": n_recovery_attempted,
        "recovery_successfully_resolved": n_recovery_resolved,
        "still_unresolved_after_recovery": n_planned - len(merged_by_id),
        "duplicate_planned_observation_ids": sorted(recovery_duplicate_ids),
        "final_completeness": (len(merged_by_id) / n_planned) if n_planned else None,
    }
    return merged_by_id, diagnostics


def split_merged_rows_by_unit_kind(merged_by_id, planned_index):
    """{"treatment": [...], "baseline": [...]} -- what the merge step
    actually writes to disk (separate files, mirroring the original Wave 1
    treatment_raw.jsonl/baseline_raw.jsonl split, so analyze_controllability_v2.py's
    existing --treatment-results-file/--baseline-results-file arguments work
    unchanged against the merged dataset)."""
    out = {"treatment": [], "baseline": []}
    for oid, row in merged_by_id.items():
        out[planned_index[oid]["unit_kind"]].append(row)
    return out


# ---------------------------------------------------------------------------
# Recovery validation sample: 500 already-resolved Wave 1 observations,
# stratified across baseline/treatment, matched_control/text_only, cue, and
# original reasoning-token length. These are NEVER part of the merged
# primary dataset and never count as additional replicates.
# ---------------------------------------------------------------------------

def reasoning_token_bin(tokens):
    if tokens is None:
        return "unknown"
    if tokens == 0:
        return "0"
    if tokens < 100:
        return "1-99"
    if tokens < 500:
        return "100-499"
    return "500+"


def _stratum_key(row, planned_index):
    meta = row.get("trial_meta") or {}
    unit_kind = planned_index[row["observation_id"]]["unit_kind"]
    instruction_condition = meta.get("instruction_condition")
    cue = meta.get("contrast_id")
    attempt = _resolving_attempt(row)
    tokens = attempt.get("reasoning_tokens") if attempt else None
    return (unit_kind, instruction_condition, cue, reasoning_token_bin(tokens))


def select_validation_sample(wave1_valid_rows, planned_index, sample_size=VALIDATION_SAMPLE_SIZE, seed=0):
    """Proportional-allocation stratified sample (without replacement) of
    already-valid Wave 1 rows, deterministic from `seed`. Strata are
    (baseline/treatment, instruction_condition, cue, reasoning_token_bin).
    Returns exactly min(sample_size, len(wave1_valid_rows)) rows."""
    if not wave1_valid_rows:
        return []
    if sample_size >= len(wave1_valid_rows):
        return list(wave1_valid_rows)

    strata = defaultdict(list)
    for row in wave1_valid_rows:
        strata[_stratum_key(row, planned_index)].append(row)

    total = len(wave1_valid_rows)
    raw_alloc = {key: len(rows) * sample_size / total for key, rows in strata.items()}
    allocation = {key: int(raw_alloc[key]) for key in strata}
    remaining = sample_size - sum(allocation.values())
    # Largest-remainder method: give the +1 rounding slots to the strata
    # with the biggest fractional remainder first, so allocations always
    # sum to exactly sample_size.
    by_remainder = sorted(strata, key=lambda k: raw_alloc[k] - allocation[k], reverse=True)
    for key in by_remainder[:remaining]:
        allocation[key] += 1

    rng = random.Random(seed)
    sample = []
    for key, rows in strata.items():
        k = min(allocation[key], len(rows))
        sample.extend(rng.sample(rows, k))

    if len(sample) < sample_size:
        chosen_ids = {row["observation_id"] for row in sample}
        remaining_pool = [row for row in wave1_valid_rows if row["observation_id"] not in chosen_ids]
        rng.shuffle(remaining_pool)
        sample.extend(remaining_pool[: sample_size - len(sample)])
    elif len(sample) > sample_size:
        rng.shuffle(sample)
        sample = sample[:sample_size]
    return sample


def build_validation_plan(sample_rows, trials_by_id):
    """One plan entry per validation-sample row. `validation_observation_id`
    can never collide with a real planned observation_id (every planned id
    is `experiment_id::evaluator_id::trial_id::rN`, which never contains the
    "::validation_dup" suffix), so a validation row is structurally
    impossible to merge into the primary dataset or mistake for a real
    replicate."""
    plan = []
    for index, row in enumerate(sample_rows, start=1):
        trial_id = row["trial_id"]
        plan.append({
            "original_observation_id": row["observation_id"],
            "validation_observation_id": f"{row['observation_id']}::validation_dup",
            "trial_id": trial_id,
            "trial": trials_by_id[trial_id],
            "original_row": row,
            "execution_order_index": index,
        })
    return plan


def build_validation_result_row(plan_entry, evaluator_identity, observation,
                                 max_output_tokens=RECOVERY_MAX_OUTPUT_TOKENS):
    trial = plan_entry["trial"]
    original_row = plan_entry["original_row"]
    original_response = original_row.get("first_valid_response")
    new_response = observation.get("first_valid_response")
    agrees = None
    if original_response is not None and new_response is not None:
        agrees = original_response.get("overall_quality") == new_response.get("overall_quality")
    original_attempt = _resolving_attempt(original_row)
    return {
        "validation_observation_id": plan_entry["validation_observation_id"],
        "original_observation_id": plan_entry["original_observation_id"],
        "experiment_id": EXPERIMENT_ID,
        "trial_id": plan_entry["trial_id"],
        "trial_meta": {k: v for k, v in trial.items() if k != "prompt"},
        "evaluator": evaluator_identity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_wave": COLLECTION_WAVE_VALIDATION,
        "max_output_tokens": max_output_tokens,
        "original_first_valid_response": original_response,
        "original_reasoning_tokens": original_attempt.get("reasoning_tokens") if original_attempt else None,
        "agrees_with_original": agrees,
        **{k: observation[k] for k in _RESOLVED_OBSERVATION_FIELDS},
    }


def assert_recovery_complete(diagnostics, expected_planned_total=27720):
    """The hard gate between merge and analysis: the merged Wave 1 +
    recovery dataset must be FULLY complete before analysis is allowed to
    run on it -- a partially-recovered dataset silently analyzed would
    understate/misrepresent the true sample. Raises ValueError (never
    silently continues) listing every failing condition; never deletes or
    alters anything already written to disk, so a failing check still
    leaves a rerunnable state (rerun recovery-run, then merge again)."""
    problems = []
    if diagnostics["wave1_planned_total"] != expected_planned_total:
        problems.append(f"wave1_planned_total={diagnostics['wave1_planned_total']} (expected {expected_planned_total})")
    if diagnostics["still_unresolved_after_recovery"] != 0:
        problems.append(f"still_unresolved_after_recovery={diagnostics['still_unresolved_after_recovery']} (must be 0)")
    if diagnostics["final_completeness"] != 1.0:
        problems.append(f"final_completeness={diagnostics['final_completeness']} (must be 1.0)")
    if diagnostics["duplicate_planned_observation_ids"]:
        problems.append(f"duplicate_planned_observation_ids={diagnostics['duplicate_planned_observation_ids']} (must be empty)")
    if problems:
        raise ValueError(
            "Wave 2 recovery is INCOMPLETE -- refusing to proceed to analysis: " + "; ".join(problems) +
            ". Raw Wave 1, recovery, and merged output files have all been preserved on disk; "
            "rerun recovery-run (it will only attempt what's still missing) and then merge again."
        )


def classify_baseline_text_only_rows(rows, planned_index):
    """The new no-context/text_only condition uses the exact same identity/
    validity rules as Wave 1 (observation_id membership, single-clean-valid-
    answer-per-id, re-parseable response) -- reuses classify_wave1_rows
    directly rather than duplicating its logic under a new name."""
    return classify_wave1_rows(rows, planned_index)


def assert_baseline_text_only_complete(classification, planned_index,
                                        expected_total=1320, expected_unique_cells=132):
    """Verifies the new no-context/text_only dataset is not merely "1,320
    rows exist" but 1,320 genuinely VALID, re-parseable A/B judgments whose
    identities match the frozen baseline_text_only manifest exactly (132
    unique cells x 10 replicates each). Raises ValueError listing every
    failing condition; never mutates anything."""
    problems = []
    if len(planned_index) != expected_total:
        problems.append(f"planned baseline_text_only observations={len(planned_index)} (expected {expected_total})")
    unique_cells = {descriptor["trial_id"] for descriptor in planned_index.values()}
    if len(unique_cells) != expected_unique_cells:
        problems.append(f"unique baseline_text_only cells={len(unique_cells)} (expected {expected_unique_cells})")
    n_valid = len(classification["valid_by_id"])
    if n_valid != expected_total:
        problems.append(f"valid baseline_text_only judgments={n_valid} (expected {expected_total})")
    if classification["unresolved_ids"]:
        problems.append(f"{len(classification['unresolved_ids'])} baseline_text_only observation(s) still unresolved")
    if classification["unknown_ids"]:
        problems.append(f"{len(classification['unknown_ids'])} baseline_text_only row(s) with an id outside the frozen manifest")
    if classification["duplicate_valid_ids"]:
        problems.append(f"duplicate valid baseline_text_only answers for {len(classification['duplicate_valid_ids'])} id(s)")
    if classification["malformed_valid_ids"]:
        problems.append(f"{len(classification['malformed_valid_ids'])} baseline_text_only row(s) marked resolved but unparseable")
    if problems:
        raise ValueError(
            "New no-context/text_only baseline dataset is INCOMPLETE or invalid -- refusing to proceed to "
            "analysis: " + "; ".join(problems) + ". Rerun baseline-text-only (it only attempts what's missing)."
        )


def compute_validation_agreement(validation_rows):
    """A/B agreement rate overall and by original-reasoning-token bin, plus
    the aggregate preference shift (how much the overall A-rate moved
    between the original and the validation-duplicate answers). Never reads
    or writes anything outside `validation_rows` -- this never touches the
    primary analysis dataset."""
    resolved = [r for r in validation_rows if r.get("parsing_status") == "resolved" and r.get("agrees_with_original") is not None]
    n = len(resolved)
    agree_n = sum(1 for r in resolved if r["agrees_with_original"])

    by_bin = defaultdict(lambda: [0, 0])
    original_a = new_a = 0
    for r in resolved:
        b = reasoning_token_bin(r.get("original_reasoning_tokens"))
        by_bin[b][0] += 1 if r["agrees_with_original"] else 0
        by_bin[b][1] += 1
        if r["original_first_valid_response"].get("overall_quality") == "A":
            original_a += 1
        if r["first_valid_response"].get("overall_quality") == "A":
            new_a += 1

    return {
        "n_validation_pairs": n,
        "agreement_rate": (agree_n / n) if n else None,
        "agreement_by_reasoning_token_bin": {b: (agree / total if total else None) for b, (agree, total) in by_bin.items()},
        "n_by_reasoning_token_bin": {b: total for b, (_agree, total) in by_bin.items()},
        "original_a_rate": (original_a / n) if n else None,
        "validation_a_rate": (new_a / n) if n else None,
        "aggregate_preference_shift": ((new_a / n) - (original_a / n)) if n else None,
    }


def compute_validation_context_effect_comparison(validation_rows):
    """Best-effort ("where calculable") comparison of the treatment D_pair
    context effect computed from the ORIGINAL Wave 1 answers vs. from the
    validation-duplicate answers, restricted to (contrast, instruction_condition,
    story pair) groups where all 4 required counterbalanced cells happen to
    be present in this small 500-row sample. Most groups will have 0 or 1
    of their 4 cells sampled, so this list is typically short or empty --
    that is expected, not an error."""
    from analyze_controllability_v2 import story1_chosen as _story1_chosen
    import controllability_v2_stats as st

    def _cell_rate_map(response_field):
        groups = defaultdict(lambda: defaultdict(list))
        for row in validation_rows:
            meta = row.get("trial_meta") or {}
            if "contrast_id" not in meta or meta.get("contrast_id") is None:
                continue
            response = row.get(response_field)
            if response is None:
                continue
            chosen = meta["story_a_id"] if response.get("overall_quality") == "A" else meta["story_b_id"]
            chosen = chosen == meta["story_1_id"]
            key = (meta["contrast_id"], meta["instruction_condition"], meta["story_1_id"], meta["story_2_id"])
            groups[key][(meta["assignment"], meta["position"])].append(chosen)
        return {
            key: {cell: sum(vals) / len(vals) for cell, vals in cells.items()}
            for key, cells in groups.items()
        }

    original_rates = _cell_rate_map("original_first_valid_response")
    validation_rates = _cell_rate_map("first_valid_response")

    comparisons = []
    for key in sorted(set(original_rates) & set(validation_rates)):
        original_result = st.d_pair_from_cell_rates(original_rates[key])
        validation_result = st.d_pair_from_cell_rates(validation_rates[key])
        if original_result is None or validation_result is None:
            continue
        contrast_id, instruction_condition, s1, s2 = key
        comparisons.append({
            "contrast_id": contrast_id, "instruction_condition": instruction_condition,
            "story_1_id": s1, "story_2_id": s2,
            "d_pair_original": original_result["d_pair"], "d_pair_validation": validation_result["d_pair"],
            "d_pair_difference": validation_result["d_pair"] - original_result["d_pair"],
        })
    return comparisons
