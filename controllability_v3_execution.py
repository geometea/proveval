"""v3 execution planning and bookkeeping (pure logic, no API calls):
randomised execution order, per-attempt bookkeeping with latency/error
capture, valid-only resume semantics, the raw result-row schema, and
frozen-config-authoritative production settings.

Randomisation: units are the manifest's blocks (4-cell context blocks,
2-cell no-context blocks) x replicate. All units of a run -- context AND
no-context families together for the primary run -- are globally shuffled
with the frozen seed, then each unit's cells are shuffled with a seed
derived from (seed, block_id, replicate). Deterministic from the seed
alone. Because every family is interleaved in ONE order, temporal drift in
the provider cannot line up with intervention or context condition.

Resume semantics (v2 recovery era, adopted wholesale):
    valid parsed A/B response  -> completed, never re-run
    failed/malformed/unresolved -> eligible for retry on the next invocation
Raw files are append-only; a completed valid row is never overwritten.
Several non-valid rows may accumulate for one id across invocations; at
most one valid row ever exists per id (dispatch is gated on the valid set).

The evaluator identity / run_config_id logic is imported from v2's
execution module (pure functions) so the recorded identity is directly
comparable with v2's result rows. Nothing here writes to any v2 path.
"""

import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from controllability_v2_execution import build_evaluator_identity, classify_attempt  # pure helpers, re-used verbatim
from controllability_v3_design import EXPERIMENT, EXPERIMENT_ID, V2_ID_PREFIX
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID, make_evaluation_observation_id, profile_id_from_row
from controllability_v3_trials import make_planned_observation_id
from run_trial import parse_plain_ab_response

__all__ = [
    "build_evaluator_identity", "classify_attempt", "plan_execution_order", "attach_observation_ids",
    "run_one_observation", "load_valid_completed_ids", "find_duplicate_valid_ids", "filter_unresumed_plan",
    "build_result_row", "resolve_production_settings", "resolve_profile_settings", "limit_to_first_n_units", "count_rows",
    "evaluation_id_of",
]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def build_units(trials):
    units = defaultdict(list)
    for trial in trials:
        units[trial["block_id"]].append(trial)
    return units


def _derive_unit_seed(seed, unit_id, replicate_number):
    basis = f"{seed}:{unit_id}:{replicate_number}".encode("utf-8")
    return int(hashlib.sha256(basis).hexdigest()[:16], 16)


def plan_execution_order(trials, replicate_count, seed):
    """Flat, globally-randomised list of plan entries (one per planned
    observation) -- see the module docstring. Entries carry trial_id,
    block_id, family, replicate_number, random_seed, execution_order_index
    and the manifest row under "trial"."""
    units = build_units(trials)
    scheduled = [
        (block_id, replicate_number, list(cells))
        for block_id, cells in units.items()
        for replicate_number in range(1, replicate_count + 1)
    ]
    random.Random(seed).shuffle(scheduled)
    plan = []
    for block_id, replicate_number, cells in scheduled:
        shuffled = list(cells)
        random.Random(_derive_unit_seed(seed, block_id, replicate_number)).shuffle(shuffled)
        for cell in shuffled:
            plan.append({
                "trial_id": cell["trial_id"], "block_id": block_id, "family": cell["family"],
                "replicate_number": replicate_number, "random_seed": seed, "trial": cell,
            })
    for index, entry in enumerate(plan, start=1):
        entry["execution_order_index"] = index
    return plan


def attach_observation_ids(plan, profile_id=PRIMARY_PROFILE_ID, id_maker=make_planned_observation_id):
    """Stamp the scientific planned_observation_id (model-independent) and
    the evaluation_observation_id (planned id + evaluator profile) onto
    every entry. `id_maker` lets stress families use their own id grammar."""
    for entry in plan:
        entry["planned_observation_id"] = id_maker(entry["trial_id"], entry["replicate_number"])
        if entry["planned_observation_id"].startswith(V2_ID_PREFIX):
            raise ValueError("a v3 plan entry produced a v2-prefixed id")
        entry["evaluator_profile_id"] = profile_id
        entry["evaluation_observation_id"] = make_evaluation_observation_id(entry["planned_observation_id"], profile_id)
    return plan


def limit_to_first_n_units(trials, limit):
    """Keep only the first `limit` distinct blocks (whole blocks, never a
    partial one). None -> unchanged."""
    if limit is None:
        return trials
    keep, order = set(), []
    for trial in trials:
        if trial["block_id"] not in keep:
            keep.add(trial["block_id"])
            order.append(trial["block_id"])
    allowed = set(order[:limit])
    return [t for t in trials if t["block_id"] in allowed]


# ---------------------------------------------------------------------------
# One observation: bounded retries, latency + error capture per attempt
# ---------------------------------------------------------------------------

def run_one_observation(trial, prompt, call_fn, retry_limit, clock=time.perf_counter):
    """Up to 1 + retry_limit attempts; stops at the first valid A/B answer.
    call_fn(prompt) returns a normalize_response()-shaped dict or raises.
    Every attempt records its latency and, on failure, the error text."""
    attempts = []
    parsed_choice = None
    for attempt_number in range(1, retry_limit + 2):
        started = clock()
        error = None
        try:
            api_result = call_fn(prompt)
            response_text = api_result.get("response_text")
            parsed, validation_error = parse_plain_ab_response(response_text)
            status = classify_attempt(parsed, validation_error)
            if status != "valid":
                error = validation_error
        except Exception as e:  # provider/network failure -- recorded, never raised
            api_result = {}
            response_text = None
            parsed = None
            status = "api_error"
            error = f"API call failed: {type(e).__name__}: {e}"
        latency = clock() - started
        attempts.append({
            "attempt_number": attempt_number,
            "status": status,
            "raw_response": response_text,
            "parsed_choice": parsed["overall_quality"] if parsed else None,
            "error": error,
            "latency_seconds": latency,
            "response_model": api_result.get("response_model"),
            "request_id": api_result.get("request_id"),
            "input_tokens": api_result.get("input_tokens"),
            "cached_input_tokens": api_result.get("prompt_cache_hit_tokens"),
            "output_tokens": api_result.get("output_tokens"),
            "reasoning_tokens": api_result.get("reasoning_tokens"),
            "stop_reason": api_result.get("stop_reason"),
            "system_fingerprint": api_result.get("system_fingerprint"),
            "prompt_cache_hit_tokens": api_result.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": api_result.get("prompt_cache_miss_tokens"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if status == "valid":
            parsed_choice = parsed["overall_quality"]
            break
    return {
        "attempts": attempts,
        "total_attempts": len(attempts),
        "first_attempt_status": attempts[0]["status"],
        "final_attempt_status": attempts[-1]["status"],
        "parsed_choice": parsed_choice,
        "parsing_status": "resolved" if parsed_choice is not None else "unresolved",
        "error": None if parsed_choice is not None else attempts[-1]["error"],
    }


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

def build_result_row(entry, evaluator_identity, observation, prompt_sha256, run_id, collection, manifest_sha256=None):
    trial = entry["trial"]
    resolving = observation["attempts"][-1]
    return {
        "planned_observation_id": entry["planned_observation_id"],
        "evaluator_profile_id": entry.get("evaluator_profile_id", PRIMARY_PROFILE_ID),
        "evaluation_observation_id": entry.get("evaluation_observation_id") or make_evaluation_observation_id(entry["planned_observation_id"], PRIMARY_PROFILE_ID),
        "experiment": EXPERIMENT,
        "experiment_id": trial.get("experiment_id", EXPERIMENT_ID),
        "collection": collection,                # "pilot" | "production" | "exploratory"
        "family": entry["family"],
        "trial_id": entry["trial_id"],
        "block_id": entry["block_id"],
        "story_pair": trial["pair_id"],
        "story_1_id": trial["story_1_id"],
        "story_2_id": trial["story_2_id"],
        "story_a": trial["story_a_id"],
        "story_b": trial["story_b_id"],
        "cue": trial["cue_id"],
        "cue_family": trial["cue_family"],
        "context_assignment": trial["assignment"],
        "display_position": trial["position"],
        "intervention_id": trial["intervention_id"],
        "instruction_id": trial["instruction_id"],
        "base_intervention_id": trial.get("base_intervention_id"),
        "held_out_cue_id": trial.get("held_out_cue_id"),
        "source_trial_id": trial.get("source_trial_id"),
        "pilot_stratum": trial.get("pilot_stratum"),
        "variant": trial.get("variant"),
        "attack_id": trial.get("attack_id"),
        "attack_family": trial.get("attack_family"),
        "dose": trial.get("dose"),
        "dose_logit": trial.get("dose_logit"),
        "context_present": trial["context_present"],
        "replicate": entry["replicate_number"],
        "execution_order_index": entry["execution_order_index"],
        "random_seed": entry["random_seed"],
        "provider": evaluator_identity["provider"],
        "model": evaluator_identity["requested_model"],
        "response_model": resolving.get("response_model"),
        "reasoning_effort": evaluator_identity["reasoning_profile"],
        "provider_reasoning_settings": evaluator_identity["provider_reasoning_settings"],
        "max_output_tokens": evaluator_identity["max_output_tokens"],
        "run_config_id": evaluator_identity["run_config_id"],
        "evaluator": evaluator_identity,
        "raw_response": resolving.get("raw_response"),
        "parsed_choice": observation["parsed_choice"],
        "input_tokens": resolving.get("input_tokens"),
        "cached_input_tokens": resolving.get("cached_input_tokens"),
        "output_tokens": resolving.get("output_tokens"),
        "reasoning_tokens": resolving.get("reasoning_tokens"),
        "latency_seconds": resolving.get("latency_seconds"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "prompt_sha256": prompt_sha256,
        "manifest_sha256": manifest_sha256,
        "attempts": observation["attempts"],
        "total_attempts": observation["total_attempts"],
        "first_attempt_status": observation["first_attempt_status"],
        "final_attempt_status": observation["final_attempt_status"],
        "parsing_status": observation["parsing_status"],
        "error": observation["error"],
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def iter_rows(results_file):
    if not os.path.exists(results_file):
        return
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_rows(results_file):
    return sum(1 for _ in iter_rows(results_file))


def evaluation_id_of(row):
    """A row's evaluation_observation_id; rows written before evaluator
    profiles existed are attributed to the profile their evaluator fields
    identify (the frozen primary for historical primary rows)."""
    if row.get("evaluation_observation_id"):
        return row["evaluation_observation_id"]
    return make_evaluation_observation_id(row["planned_observation_id"], profile_id_from_row(row))


def load_valid_completed_ids(results_file, profile_id=None):
    """Observations with at least one row whose recorded raw_response
    genuinely re-parses to the recorded parsed_choice. A row claiming
    parsing_status=resolved that does not re-parse is NOT counted.

    With profile_id=None returns planned_observation_ids (legacy callers);
    with a profile returns evaluation_observation_ids for THAT profile
    only, so a file holding several profiles never lets one profile's
    answer satisfy another profile's plan."""
    ids = set()
    for row in iter_rows(results_file):
        if row.get("parsing_status") != "resolved":
            continue
        parsed, _ = parse_plain_ab_response(row.get("raw_response") or "")
        if not (parsed and parsed["overall_quality"] == row.get("parsed_choice")):
            continue
        if profile_id is None:
            ids.add(row["planned_observation_id"])
        elif profile_id_from_row(row) == profile_id:
            ids.add(evaluation_id_of(row))
    return ids


def find_duplicate_valid_ids(results_file):
    counts = Counter(evaluation_id_of(row) for row in iter_rows(results_file) if row.get("parsing_status") == "resolved")
    return sorted(oid for oid, n in counts.items() if n > 1)


def find_foreign_ids(results_file, allowed_families, id_prefixes=("v3::",)):
    """Ids in `results_file` whose family isn't one of `allowed_families`
    (e.g. a pilot row in the production file) or whose id prefix is not
    one of `id_prefixes` (primary "v3::", stress "v3s::")."""
    foreign = []
    for row in iter_rows(results_file):
        oid = row.get("planned_observation_id", "")
        if not oid.startswith(tuple(id_prefixes)) or row.get("family") not in allowed_families:
            foreign.append(oid)
    return foreign


def filter_unresumed_plan(plan, completed_ids, key="planned_observation_id"):
    return [entry for entry in plan if entry[key] not in completed_ids]


# ---------------------------------------------------------------------------
# Frozen-config-authoritative settings
# ---------------------------------------------------------------------------

_REPLICATE_KEY = {"primary": "replicate_count", "holdout": "holdout_replicate_count", "pilot": "pilot_replicate_count"}


def resolve_production_settings(cli_overrides, study_config, run_kind):
    """Read provider/model/reasoning/replicates/seed/retry/max tokens from
    the frozen config. A CLI value that is present and disagrees is a hard
    error -- a frozen study never silently runs under different settings."""
    if run_kind not in _REPLICATE_KEY:
        raise ValueError(f"run_kind must be one of {sorted(_REPLICATE_KEY)}, got {run_kind!r}")
    primary = study_config["primary_evaluator"]
    frozen = {
        "provider": primary["provider"],
        "model": primary["requested_model"],
        "reasoning_profile": primary["reasoning_profile"],
        "replicates": study_config[_REPLICATE_KEY[run_kind]],
        "seed": study_config["random_seed"],
        "retry_limit": study_config["retry_limit"],
        "max_output_tokens": study_config["max_output_tokens"],
    }
    mismatches = []
    for key, value in frozen.items():
        cli_value = (cli_overrides or {}).get(key)
        if cli_value is not None and cli_value != value:
            mismatches.append(f"{key}: got {cli_value!r}, frozen config requires {value!r}")
    if mismatches:
        raise ValueError("Settings conflict with the frozen v3 study config -- " + "; ".join(mismatches))
    resolved = dict(frozen)
    resolved["evaluator_id"] = primary["evaluator_id"]
    return resolved


def resolve_profile_settings(profile, replicates, seed, retry_limit, cli_overrides=None):
    """Settings for a run under an evaluator PROFILE (multi-model
    execution): provider/model/reasoning/max_output_tokens come from the
    profile, replicates/seed/retry_limit from the (frozen) design config.
    A CLI value that is present and disagrees with the profile is a hard
    error, exactly as for the frozen primary evaluator."""
    frozen = {"provider": profile["provider"], "model": profile["model"], "reasoning_profile": profile["reasoning_profile"],
              "replicates": replicates, "seed": seed, "retry_limit": retry_limit, "max_output_tokens": profile["max_output_tokens"]}
    mismatches = [f"{k}: got {cli_overrides[k]!r}, profile {profile['profile_id']!r} requires {v!r}"
                  for k, v in frozen.items() if (cli_overrides or {}).get(k) is not None and cli_overrides[k] != v]
    if mismatches:
        raise ValueError("Settings conflict with the evaluator profile -- " + "; ".join(mismatches))
    return {**frozen, "evaluator_id": f"{profile['provider']}__{profile['model']}__{profile['reasoning_profile']}", "evaluator_profile_id": profile["profile_id"]}
