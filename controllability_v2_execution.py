"""v2 execution planning: randomised order construction, bounded-retry
attempt bookkeeping, and evaluator identity -- shared by run_controllability_v2.py
(the real/dry-run CLI) and its tests, so the CLI is never the only thing
that exercises this logic.

Randomisation (item 7): treatment execution groups cells into
replicate-level 8-cell superblocks (superblock_id x replicate_number),
globally shuffles those units with a configurable seed, then independently
shuffles the 8 cells within each unit. Baseline execution does the same
with 2-cell (block_id x replicate_number) units. Everything is deterministic
from the seed alone (see _derive_unit_seed) -- the SAME seed always produces
the SAME order, and a re-run never executes "all replicates of one cell
followed by all replicates of another" because the global shuffle mixes
superblocks/blocks and replicates together before cells are ever flattened
into one list.

Retry bookkeeping (item 8): run_one_observation() records every attempt (up
to a fixed retry_limit), but the number of PLANNED replicate observations
is decided once by plan_treatment_execution_order/plan_baseline_execution_order
and never grows because of a retry -- a retried cell is still exactly one
planned observation with more than one attempt nested inside it.

Evaluator identity (item 9): build_evaluator_identity() is the one place an
"evaluator" is defined for v2 -- provider, requested_model, reasoning_profile,
provider_reasoning_settings, sampling_settings, plus (when available)
response_model/model_version and a run_id/wave. See
analyze_controllability_v2.py for how this identity is used to refuse to
silently pool results from different providers/models/reasoning profiles,
or different response_model versions of what was requested as one evaluator.
"""

import hashlib
import json
import os
import random
from collections import Counter

from run_trial import parse_plain_ab_response

# ---------------------------------------------------------------------------
# Grouping raw trial rows into randomisation units
# ---------------------------------------------------------------------------

def build_treatment_superblocks(trials):
    """{superblock_id: [8 cell dicts]} -- see controllability_v2_trials.py's
    assert_superblocks_are_well_formed for the same grouping used at
    generation time."""
    groups = {}
    for trial in trials:
        groups.setdefault(trial["superblock_id"], []).append(trial)
    return groups


def build_baseline_units(baseline_trials):
    """{block_id: [2 cell dicts]}."""
    groups = {}
    for trial in baseline_trials:
        groups.setdefault(trial["block_id"], []).append(trial)
    return groups


def _derive_unit_seed(seed, unit_id, replicate_number):
    """A distinct, deterministic sub-seed per (unit, replicate) derived from
    the one top-level seed -- so the within-unit cell shuffle is
    reproducible from `seed` alone, without every unit sharing the exact
    same shuffle pattern (which would silently preserve a fixed relative
    cell order across units)."""
    basis = f"{seed}:{unit_id}:{replicate_number}".encode("utf-8")
    return int(hashlib.sha256(basis).hexdigest()[:16], 16)


def _plan_execution_order(units_by_id, replicate_count, seed, unit_id_field):
    """Shared by plan_treatment_execution_order/plan_baseline_execution_order:
    build (unit_id, replicate_number, cells) triples for every unit x
    replicate, globally shuffle them with `seed`, then independently shuffle
    each unit's own cells with a seed derived from (seed, unit_id,
    replicate_number). Returns a flat list of plan entries, each carrying
    execution_order_index, random_seed, replicate_number, the unit id (under
    `unit_id_field`), and the trial dict itself under "trial"."""
    units = [
        (unit_id, replicate_number, list(cells))
        for unit_id, cells in units_by_id.items()
        for replicate_number in range(1, replicate_count + 1)
    ]
    random.Random(seed).shuffle(units)

    plan = []
    for unit_id, replicate_number, cells in units:
        shuffled_cells = list(cells)
        random.Random(_derive_unit_seed(seed, unit_id, replicate_number)).shuffle(shuffled_cells)
        for cell in shuffled_cells:
            plan.append(
                {
                    "trial_id": cell["trial_id"],
                    unit_id_field: unit_id,
                    "replicate_number": replicate_number,
                    "random_seed": seed,
                    "trial": cell,
                }
            )

    for index, entry in enumerate(plan, start=1):
        entry["execution_order_index"] = index
    return plan


def limit_to_first_n_units(trials, unit_id_field, limit):
    """Restrict `trials` to only the first `limit` distinct units (by
    first-seen order of `unit_id_field`), keeping every cell of each kept
    unit -- so a --limit on execution always selects whole superblocks/
    baseline units, never leaves one partially selected. `limit=None`
    returns `trials` unchanged."""
    if limit is None:
        return trials
    seen_order = []
    seen_ids = set()
    for trial in trials:
        unit_id = trial[unit_id_field]
        if unit_id not in seen_ids:
            seen_ids.add(unit_id)
            seen_order.append(unit_id)
    keep = set(seen_order[:limit])
    return [trial for trial in trials if trial[unit_id_field] in keep]


def plan_treatment_execution_order(treatment_trials, replicate_count, seed):
    superblocks = build_treatment_superblocks(treatment_trials)
    return _plan_execution_order(superblocks, replicate_count, seed, "superblock_id")


def plan_baseline_execution_order(baseline_trials, replicate_count, seed):
    units = build_baseline_units(baseline_trials)
    return _plan_execution_order(units, replicate_count, seed, "block_id")


# ---------------------------------------------------------------------------
# Bounded retry handling
# ---------------------------------------------------------------------------

def classify_attempt(parsed_response, validation_error):
    """One of "valid" / "refusal" / "invalid" / "api_error". "refusal" is a
    parse failure specifically flagged as an ambiguous/hedged response (see
    run_trial._PLAIN_AB_AMBIGUITY_MARKERS) -- distinct from an
    unparseable/malformed response ("invalid") and from the request itself
    failing ("api_error")."""
    if validation_error is not None and validation_error.startswith("API call failed"):
        return "api_error"
    if parsed_response is not None and validation_error is None:
        return "valid"
    if validation_error and "ambiguous/hedged" in validation_error:
        return "refusal"
    return "invalid"


def run_one_observation(trial, call_fn, retry_limit):
    """Run one planned observation (one trial, already assigned its
    replicate_number by the execution plan) with up to `retry_limit`
    retries beyond the first attempt. `call_fn(trial)` must return a dict
    with at least "response_text" (and may include provider usage metadata,
    which is preserved on each attempt record) or raise on an API failure.

    Stops retrying as soon as an attempt is "valid" (a retry exists to
    recover from failure, never to accumulate a second opinion). Every
    attempt actually made (whether or not it stopped the loop) is recorded
    in the returned "attempts" list -- retries never increase the number of
    PLANNED observations, only the number of attempts within this one.
    """
    attempts = []
    first_valid_response = None

    for attempt_number in range(1, retry_limit + 2):  # 1 initial attempt + retry_limit retries
        try:
            api_result = call_fn(trial)
            response_text = api_result.get("response_text")
            parsed_response, validation_error = parse_plain_ab_response(response_text)
            status = classify_attempt(parsed_response, validation_error)
        except Exception as e:
            api_result = {}
            response_text = None
            parsed_response = None
            validation_error = f"API call failed: {e}"
            status = "api_error"

        attempt_record = {
            "attempt_number": attempt_number,
            "status": status,
            "response_text": response_text,
            "parsed_response": parsed_response,
            "validation_error": validation_error,
            "response_model": api_result.get("response_model"),
            "request_id": api_result.get("request_id"),
            "input_tokens": api_result.get("input_tokens"),
            "output_tokens": api_result.get("output_tokens"),
            "reasoning_tokens": api_result.get("reasoning_tokens"),
            # DeepSeek-specific usage metadata (item 7) -- None for every
            # other provider, never fabricated.
            "system_fingerprint": api_result.get("system_fingerprint"),
            "prompt_cache_hit_tokens": api_result.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": api_result.get("prompt_cache_miss_tokens"),
        }
        attempts.append(attempt_record)

        if status == "valid":
            first_valid_response = parsed_response
            break

    first_attempt = attempts[0]
    return {
        "trial_id": trial["trial_id"],
        "attempts": attempts,
        "total_attempts": len(attempts),
        "first_attempt_status": first_attempt["status"],
        "first_valid_response": first_valid_response,
        "parsing_status": "resolved" if first_valid_response is not None else "unresolved",
        "refusal_status": first_attempt["status"] == "refusal",
        "api_error_status": first_attempt["status"] == "api_error",
    }


# ---------------------------------------------------------------------------
# Observation identity (resumable/idempotent execution) and run-config
# identity (item 9: an immutable record of the literal request settings)
# ---------------------------------------------------------------------------

def make_observation_id(experiment_id, evaluator_id, trial_id, replicate_number):
    """A stable id for one PLANNED model judgment, from exactly the fields
    item 4 of the production-readiness spec names: experiment_id,
    evaluator_id, trial_id, replicate_number ("replicate_id"). Deterministic
    and human-readable (never hashed) so a results file can be grepped/
    resumed-from by inspection. The same four inputs always produce the
    same id, so re-planning the exact same execution (same trials file,
    same evaluator, same replicate count) always reproduces the same set of
    observation_ids -- which is what makes skip-if-already-present resumable
    execution correct."""
    return f"{experiment_id}::{evaluator_id}::{trial_id}::r{replicate_number}"


def make_run_config_id(provider, requested_model, reasoning_profile, provider_reasoning_settings,
                        sampling_settings, max_output_tokens, retry_limit):
    """A short deterministic hash over every literal request setting this
    experiment holds constant across baseline and treatment and across
    every cell of one evaluator (item 9): provider, requested model,
    reasoning profile, the literal provider reasoning parameter, sampling
    settings, max output tokens, and the retry policy. Two observations
    sharing a run_config_id are guaranteed to have been requested under
    byte-identical settings; two different values mean something about the
    request configuration changed."""
    canonical = json.dumps(
        {
            "provider": provider,
            "requested_model": requested_model,
            "reasoning_profile": reasoning_profile,
            "provider_reasoning_settings": provider_reasoning_settings or {},
            "sampling_settings": sampling_settings or {},
            "max_output_tokens": max_output_tokens,
            "retry_limit": retry_limit,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Resumable, idempotent execution
# ---------------------------------------------------------------------------

def attach_observation_ids(plan, experiment_id, evaluator_id):
    """Stamp an "observation_id" onto every plan entry (see
    make_observation_id) -- done once execution knows which evaluator it's
    running as, since a bare plan (from plan_treatment_execution_order/
    plan_baseline_execution_order) doesn't carry evaluator identity."""
    for entry in plan:
        entry["observation_id"] = make_observation_id(experiment_id, evaluator_id, entry["trial_id"], entry["replicate_number"])
    return plan


def load_completed_observation_ids(results_file):
    """Every observation_id already present in `results_file` -- regardless
    of whether that observation ultimately succeeded or exhausted its retry
    budget. Both are TERMINAL: run_one_observation only returns (and a row
    is only ever written) after either a valid response or the configured
    retry_limit is exhausted, so a present row always means "done, do not
    repeat" -- a missing row always means "never reached a terminal result
    yet", which is exactly what makes an interrupted run resumable by
    re-running the same command. Returns an empty set if the file doesn't
    exist yet."""
    if not os.path.exists(results_file):
        return set()
    ids = set()
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            observation_id = json.loads(line).get("observation_id")
            if observation_id is not None:
                ids.add(observation_id)
    return ids


def filter_unresumed_plan(plan, completed_observation_ids):
    """Only the plan entries NOT already terminal in the results file --
    the set of observations an interrupted-and-resumed run still needs to
    execute. Never duplicates a completed observation_id."""
    return [entry for entry in plan if entry["observation_id"] not in completed_observation_ids]


def find_duplicate_observation_ids(results_file):
    """observation_ids that appear more than once in `results_file` -- used
    by the production preflight to catch corruption (e.g. a manually edited
    or concatenated results file) before a run starts. [] if the file is
    absent or clean."""
    if not os.path.exists(results_file):
        return []
    ids = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            observation_id = json.loads(line).get("observation_id")
            if observation_id is not None:
                ids.append(observation_id)
    return sorted(oid for oid, count in Counter(ids).items() if count > 1)


# ---------------------------------------------------------------------------
# Frozen-config-authoritative production settings (item 3)
# ---------------------------------------------------------------------------

#: cli_key -> (study_config lookup, human label). The baseline/treatment
#: replicate count key differs by unit_kind, so it's resolved separately.
_FROZEN_SCALAR_FIELDS = ("provider", "model", "reasoning_profile", "seed", "retry_limit")


def resolve_production_settings(cli_overrides, study_config, unit_kind):
    """Resolve provider/model/reasoning_profile/replicates/seed/retry_limit
    for a production (frozen) run. `cli_overrides` is a dict with any of
    those keys present only if the user actually passed the corresponding
    CLI flag (a key entirely absent, or None, means "not overridden" and
    is silently filled from the frozen config -- the preferred, no-flags-
    required path). A key that IS present and disagrees with the frozen
    value is a hard error: a study frozen at 10 replicates for deepseek-flash
    must never silently run with 3 replicates or a different model.

    `unit_kind` is "treatment" or "baseline" -- selects which of the two
    frozen replicate counts applies.
    """
    if unit_kind not in ("treatment", "baseline"):
        raise ValueError(f"unit_kind must be 'treatment' or 'baseline', got {unit_kind!r}")

    primary = study_config["primary_evaluator"]
    frozen = {
        "provider": primary["provider"],
        "model": primary["requested_model"],
        "reasoning_profile": primary["reasoning_profile"],
        "replicates": study_config["treatment_replicate_count" if unit_kind == "treatment" else "baseline_replicate_count"],
        "seed": study_config["random_seed"],
        "retry_limit": study_config["retry_limit"],
    }

    resolved = {}
    mismatches = []
    for key, frozen_value in frozen.items():
        cli_value = cli_overrides.get(key)
        if cli_value is not None and cli_value != frozen_value:
            mismatches.append(f"{key}: got {cli_value!r}, frozen config requires {frozen_value!r}")
        resolved[key] = frozen_value
    if mismatches:
        raise ValueError("Production run settings conflict with the frozen study config -- " + "; ".join(mismatches))

    resolved["max_output_tokens"] = study_config.get("max_output_tokens")
    resolved["evaluator_id"] = primary["evaluator_id"]
    return resolved


# ---------------------------------------------------------------------------
# Evaluator identity (item 9)
# ---------------------------------------------------------------------------

def build_evaluator_identity(provider, requested_model, reasoning_profile, provider_reasoning_settings=None,
                              sampling_settings=None, response_model=None, model_version=None, run_id=None,
                              max_output_tokens=None, retry_limit=None):
    """The one evaluator-identity record for v2: held constant within one
    evaluator across every experimental condition (both instruction
    conditions, every contrast, every story pair -- see
    STUDY_PROTOCOL_V2.md's "Fixed design rules"). response_model/
    model_version are filled in per attempt from what the provider actually
    returned, so they can be compared against this nominal identity by
    analyze_controllability_v2.match_evaluator_config -- never assumed equal
    to requested_model.

    max_output_tokens/retry_limit are recorded here too (alongside
    provider_reasoning_settings/sampling_settings) so `run_config_id` -- an
    immutable fingerprint of every literal request setting -- can be
    computed and compared across every result row for this evaluator; a
    changed run_config_id mid-study means the request configuration drifted
    and must never be silently pooled with the earlier one.
    """
    return {
        "provider": provider,
        "requested_model": requested_model,
        "reasoning_profile": reasoning_profile,
        "provider_reasoning_settings": provider_reasoning_settings or {},
        "sampling_settings": sampling_settings or {},
        "response_model": response_model,
        "model_version": model_version,
        "run_id": run_id,
        "max_output_tokens": max_output_tokens,
        "retry_limit": retry_limit,
        "run_config_id": make_run_config_id(
            provider, requested_model, reasoning_profile, provider_reasoning_settings,
            sampling_settings, max_output_tokens, retry_limit,
        ),
    }
