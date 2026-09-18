"""v3 selective-suppression experiment: deterministic manifest generation.

Three FROZEN manifest families (each written as one JSONL file, one row per
unique cell, rows in a fixed deterministic order, keys sorted -- so
regenerating produces a byte-identical file):

  primary_context   data/controllability_v3_context_trials.jsonl
      66 story pairs x 5 cues x 8 interventions x 4 counterbalance cells
      (context assignment forward/flipped x display position) = 10,560 cells

  primary_nocontext data/controllability_v3_nocontext_trials.jsonl
      66 story pairs x 8 interventions x 2 display positions = 1,056 cells
      -- the SAME instruction sentence as the context-present cell, with
      every contextual clause removed. I0's no-context cells are the blind
      baseline (byte-identical prompts to v2's blind baseline).

  holdout           data/controllability_v3_holdout_trials.jsonl
      The held-out-cue generalization experiment: for the one intervention
      that enumerates cue families (I2), five variants each omit exactly one
      family; each variant is tested ONLY on the cue it omits.
      66 pairs x 5 held-out cues x 4 cells = 1,320 cells

plus one non-frozen, regenerable pilot manifest
(data/controllability_v3_pilot_trials.jsonl) -- a deterministic subset of
the three families for a handful of story pairs, whose ids live in their
own "pilot" family so no pilot observation can ever be mistaken for (or
resumed as) a primary one.

Unlike v2's manifests, v3 rows do NOT embed the prompt text (the full
design would be ~250 MB of duplicated story text). Each row carries
prompt_sha256 instead; the prompt is rebuilt deterministically from the
frozen wording + the story files (build_prompt_for_trial) and verified
against that hash at load time (verify_prompt_hashes), so any change to a
story file, the wording, or the template is caught before a single call.

Ids. trial_id (one per unique cell) has the form

    v3::{family}::{pair_id}::{cue_id|none}::{assignment|none}::{position}::{instruction_id}::{ctx|noctx}

and a planned observation id appends "::r{replicate}". Every component the
task requires (story pair, cue, context assignment, display position,
intervention, context-present flag, replicate, experiment family) is
therefore readable straight off the id. v2 ids start with
"context_controllability_v2::"; v3 ids start with "v3::" -- disjoint.

No model API calls happen anywhere in this module.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict

from context_comparisons import load_story
from context_trials import ITEMS_FILE, load_items, story_pairs
from controllability_v3_design import (
    CUE_ENUMERATING_INTERVENTIONS,
    CUE_FAMILY,
    CUE_ORDER,
    EXPERIMENT_ID,
    ID_PREFIX,
    INTERVENTION_IDS,
    NO_CONTEXT_INTRO,
    V2_ID_PREFIX,
    all_instruction_sentences,
    build_prompt,
    context_intro,
    holdout_instruction_id,
    instruction_sentence,
)

CONTRASTS_FILE = "data/controllability_v3_contrasts.jsonl"
CONTEXT_TRIALS_FILE = "data/controllability_v3_context_trials.jsonl"
NOCONTEXT_TRIALS_FILE = "data/controllability_v3_nocontext_trials.jsonl"
HOLDOUT_TRIALS_FILE = "data/controllability_v3_holdout_trials.jsonl"
PILOT_TRIALS_FILE = "data/controllability_v3_pilot_trials.jsonl"

FAMILY_PRIMARY_CONTEXT = "primary_context"
FAMILY_PRIMARY_NOCONTEXT = "primary_nocontext"
FAMILY_HOLDOUT = "holdout"
FAMILY_PILOT = "pilot"
FROZEN_FAMILIES = (FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT, FAMILY_HOLDOUT)
PRIMARY_FAMILIES = (FAMILY_PRIMARY_CONTEXT, FAMILY_PRIMARY_NOCONTEXT)

RESPONSE_FORMAT = "plain_ab"

ASSIGNMENTS = ("forward", "flipped")
POSITIONS = ("story1_as_a", "story2_as_a")
REQUIRED_CONTEXT_CELLS = {(a, p) for a in ASSIGNMENTS for p in POSITIONS}
REQUIRED_NOCONTEXT_CELLS = set(POSITIONS)

# ---------------------------------------------------------------------------
# Design arithmetic. Computed from first principles here AND re-derived
# independently by the preflight (see run_controllability_v3.py) -- the two
# must agree with each other and with the literal constants below.
# ---------------------------------------------------------------------------

N_STORIES = 12
N_PAIRS = N_STORIES * (N_STORIES - 1) // 2               # 66
N_CUES = len(CUE_ORDER)                                    # 5
N_INTERVENTIONS = len(INTERVENTION_IDS)                    # 8
N_CONTEXT_CELLS_PER_BLOCK = len(REQUIRED_CONTEXT_CELLS)   # 4
N_POSITIONS = len(POSITIONS)                               # 2

CONTEXT_UNIQUE_CELLS = N_PAIRS * N_CUES * N_INTERVENTIONS * N_CONTEXT_CELLS_PER_BLOCK   # 10,560
NOCONTEXT_UNIQUE_CELLS = N_PAIRS * N_INTERVENTIONS * N_POSITIONS                        # 1,056
HOLDOUT_UNIQUE_CELLS = N_PAIRS * N_CUES * N_CONTEXT_CELLS_PER_BLOCK * len(CUE_ENUMERATING_INTERVENTIONS)  # 1,320

assert CONTEXT_UNIQUE_CELLS == 10560
assert NOCONTEXT_UNIQUE_CELLS == 1056
assert HOLDOUT_UNIQUE_CELLS == 1320


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

REQUIRED_CONTRAST_FIELDS = {"contrast_id", "dimension", "a", "b", "a_clause", "b_clause", "expected_direction"}


def load_contrasts(path=CONTRASTS_FILE):
    contrasts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                contrasts.append(json.loads(line))
    return contrasts


def assert_contrasts_are_well_formed(contrasts):
    seen = []
    for contrast in contrasts:
        missing = REQUIRED_CONTRAST_FIELDS - set(contrast)
        if missing:
            raise ValueError(f"{contrast.get('contrast_id', '?')!r} is missing required field(s): {sorted(missing)}")
        if contrast["expected_direction"] not in ("a", "b"):
            raise ValueError(f"{contrast['contrast_id']!r}: expected_direction must be 'a' or 'b'")
        if contrast["a"] == contrast["b"]:
            raise ValueError(f"{contrast['contrast_id']!r}: 'a' and 'b' must be distinct")
        if CUE_FAMILY.get(contrast["contrast_id"]) != contrast["dimension"]:
            raise ValueError(f"{contrast['contrast_id']!r}: dimension {contrast['dimension']!r} does not match the frozen cue table")
        seen.append(contrast["contrast_id"])
    if seen != list(CUE_ORDER):
        raise ValueError(f"Expected exactly the 5 cues in canonical order {list(CUE_ORDER)}, found {seen}")


def load_story_texts(items=None):
    """{story_id: text} for the 12 corpus stories."""
    items = items if items is not None else load_items(ITEMS_FILE)
    return {item["id"]: load_story(item["path"]) for item in items}


def assert_items_are_the_frozen_corpus(items):
    ids = sorted(item["id"] for item in items)
    if len(ids) != N_STORIES or len(set(ids)) != N_STORIES:
        raise ValueError(f"Expected exactly {N_STORIES} distinct stories, found {len(ids)}")


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

def pair_id_for(story_1_id, story_2_id):
    return f"{story_1_id}_vs_{story_2_id}"


def make_trial_id(family, pair_id, cue_id, assignment, position, instruction_id, context_present):
    return "::".join([
        ID_PREFIX, family, pair_id, cue_id or "none", assignment or "none", position, instruction_id,
        "ctx" if context_present else "noctx",
    ])


def make_planned_observation_id(trial_id, replicate_number):
    return f"{trial_id}::r{replicate_number}"


def parse_planned_observation_id(observation_id):
    """Inverse of make_trial_id + make_planned_observation_id. Returns a
    dict; raises ValueError for anything that isn't a v3 id."""
    parts = observation_id.split("::")
    if len(parts) != 9 or parts[0] != ID_PREFIX or not parts[8].startswith("r"):
        raise ValueError(f"Not a v3 planned_observation_id: {observation_id!r}")
    return {
        "family": parts[1], "pair_id": parts[2], "cue_id": None if parts[3] == "none" else parts[3],
        "assignment": None if parts[4] == "none" else parts[4], "position": parts[5],
        "instruction_id": parts[6], "context_present": parts[7] == "ctx", "replicate_number": int(parts[8][1:]),
        "trial_id": "::".join(parts[:8]),
    }


# ---------------------------------------------------------------------------
# Prompt (re)construction
# ---------------------------------------------------------------------------

def _display(story_1, story_2, position):
    if position == "story1_as_a":
        return story_1, story_2
    if position == "story2_as_a":
        return story_2, story_1
    raise ValueError(f"Unknown position {position!r}")


def build_prompt_for_trial(trial, texts):
    """Rebuild the exact prompt for a manifest row from the frozen wording
    and the story texts. The manifest never stores the prompt itself."""
    text_a, text_b = texts[trial["story_a_id"]], texts[trial["story_b_id"]]
    instruction = instruction_sentence(trial["instruction_id"])
    return build_prompt(trial["intro"], instruction, text_a, text_b)


def verify_prompt_hashes(trials, texts):
    """Raise ValueError naming the first row whose rebuilt prompt does not
    hash to its recorded prompt_sha256."""
    for trial in trials:
        if sha256_hex(build_prompt_for_trial(trial, texts)) != trial["prompt_sha256"]:
            raise ValueError(f"{trial['trial_id']}: rebuilt prompt does not match prompt_sha256 (story text, wording, or template changed)")


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

def _base_cell(family, story_1, story_2, position, instruction_id, intervention_id, context_present, intro, texts):
    story_a, story_b = _display(story_1, story_2, position)
    prompt = build_prompt(intro, instruction_sentence(instruction_id), texts[story_a["id"]], texts[story_b["id"]])
    return {
        "experiment_id": EXPERIMENT_ID,
        "family": family,
        "type": "context_pairwise",
        "choice_mode": "forced",
        "response_format": RESPONSE_FORMAT,
        "pair_id": pair_id_for(story_1["id"], story_2["id"]),
        "story_1_id": story_1["id"],
        "story_2_id": story_2["id"],
        "story_a_id": story_a["id"],
        "story_b_id": story_b["id"],
        "position": position,
        "intervention_id": intervention_id,
        "instruction_id": instruction_id,
        "context_present": context_present,
        "intro": intro,
        "prompt_sha256": sha256_hex(prompt),
    }


def build_context_block(family, contrast, story_1, story_2, instruction_id, intervention_id, texts, extra=None):
    """The 4-cell counterbalanced block for one (story pair, cue,
    instruction): assignment (which cue value each STORY gets) x position
    (which story is displayed as Passage A). Same structure as v2."""
    if story_1["id"] == story_2["id"]:
        raise ValueError("a block needs two different stories")
    pair_id = pair_id_for(story_1["id"], story_2["id"])
    block_id = "::".join([ID_PREFIX, family, pair_id, contrast["contrast_id"], instruction_id])
    a_clause, b_clause = contrast["a_clause"], contrast["b_clause"]
    assignments = {
        "forward": ((contrast["a"], a_clause), (contrast["b"], b_clause)),
        "flipped": ((contrast["b"], b_clause), (contrast["a"], a_clause)),
    }
    cells = []
    for assignment in ASSIGNMENTS:
        context_1, context_2 = assignments[assignment]
        for position in POSITIONS:
            story_a, story_b = _display(story_1, story_2, position)
            context_by_id = {story_1["id"]: context_1, story_2["id"]: context_2}
            value_a, clause_a = context_by_id[story_a["id"]]
            value_b, clause_b = context_by_id[story_b["id"]]
            intro = context_intro(clause_a, clause_b)
            cell = _base_cell(family, story_1, story_2, position, instruction_id, intervention_id, True, intro, texts)
            cell.update({
                "trial_id": make_trial_id(family, pair_id, contrast["contrast_id"], assignment, position, instruction_id, True),
                "block_id": block_id,
                "cue_id": contrast["contrast_id"],
                "cue_family": contrast["dimension"],
                "expected_direction": contrast["expected_direction"],
                "assignment": assignment,
                "context_a": {"value": value_a, "clause": clause_a},
                "context_b": {"value": value_b, "clause": clause_b},
            })
            if extra:
                cell.update(extra)
            cells.append(cell)
    return cells


def build_nocontext_block(family, story_1, story_2, intervention_id, texts):
    """The 2-cell (display position) block for one (story pair, intervention)
    with NO contextual framing: identical instruction sentence, NO_CONTEXT_INTRO."""
    pair_id = pair_id_for(story_1["id"], story_2["id"])
    block_id = "::".join([ID_PREFIX, family, pair_id, intervention_id])
    cells = []
    for position in POSITIONS:
        cell = _base_cell(family, story_1, story_2, position, intervention_id, intervention_id, False, NO_CONTEXT_INTRO, texts)
        cell.update({
            "trial_id": make_trial_id(family, pair_id, None, None, position, intervention_id, False),
            "block_id": block_id,
            "cue_id": None,
            "cue_family": None,
            "expected_direction": None,
            "assignment": None,
            "context_a": None,
            "context_b": None,
        })
        cells.append(cell)
    return cells


def build_context_trials(contrasts, items, texts, family=FAMILY_PRIMARY_CONTEXT):
    trials = []
    for story_1, story_2 in story_pairs(items):
        for contrast in contrasts:
            for intervention_id in INTERVENTION_IDS:
                trials.extend(build_context_block(family, contrast, story_1, story_2, intervention_id, intervention_id, texts))
    return trials


def build_nocontext_trials(items, texts, family=FAMILY_PRIMARY_NOCONTEXT):
    trials = []
    for story_1, story_2 in story_pairs(items):
        for intervention_id in INTERVENTION_IDS:
            trials.extend(build_nocontext_block(family, story_1, story_2, intervention_id, texts))
    return trials


def build_holdout_trials(contrasts, items, texts, family=FAMILY_HOLDOUT):
    """For each enumerating intervention and each cue k: the variant that
    omits k, tested on cue k only."""
    by_id = {c["contrast_id"]: c for c in contrasts}
    trials = []
    for story_1, story_2 in story_pairs(items):
        for base in CUE_ENUMERATING_INTERVENTIONS:
            for cue_id in CUE_ORDER:
                instruction_id = holdout_instruction_id(base, cue_id)
                extra = {"base_intervention_id": base, "held_out_cue_id": cue_id}
                trials.extend(build_context_block(family, by_id[cue_id], story_1, story_2, instruction_id, instruction_id, texts, extra))
    return trials


# ---------------------------------------------------------------------------
# Pilot: a deterministic small subset spanning every intervention, every
# cue, context/no-context, both positions, and (when a baseline-strength
# table is supplied) strong and ambiguous story pairs.
# ---------------------------------------------------------------------------

PILOT_DEFAULT_N_STRONG = 2
PILOT_DEFAULT_N_AMBIGUOUS = 2


def select_pilot_pairs(items, n_strong, n_ambiguous, seed, baseline_strength_rows=None):
    """Returns [(story_1_id, story_2_id, stratum)]. With a baseline-strength
    table (rows with story_1_id, story_2_id, baseline_strength -- e.g. an
    independent blind-baseline analysis), takes the n_strong strongest and
    n_ambiguous weakest pairs. Without one, takes a seeded random sample
    labelled "seeded" -- ambiguity cannot be inferred without blind data
    and is never guessed."""
    pairs = [(s1["id"], s2["id"]) for s1, s2 in story_pairs(items)]
    if baseline_strength_rows:
        strength = {}
        for row in baseline_strength_rows:
            key = (row["story_1_id"], row["story_2_id"])
            if key not in set(pairs):
                raise ValueError(f"baseline-strength row names an unknown pair {key}")
            strength[key] = float(row["baseline_strength"])
        missing = [p for p in pairs if p not in strength]
        if missing:
            raise ValueError(f"baseline-strength table is missing {len(missing)} pair(s), e.g. {missing[:3]}")
        ranked = sorted(pairs, key=lambda p: (-strength[p], p))
        strong = ranked[:n_strong]
        ambiguous = [p for p in reversed(ranked) if p not in strong][:n_ambiguous]
        return [(*p, "strong") for p in strong] + [(*p, "ambiguous") for p in ambiguous]
    rng = random.Random(seed)
    chosen = rng.sample(pairs, n_strong + n_ambiguous)
    return [(*p, "seeded") for p in sorted(chosen)]


def build_pilot_trials(context_trials, nocontext_trials, holdout_trials, pilot_pairs):
    """Copies of every frozen-family cell for the pilot pairs, re-identified
    into the "pilot" family. Each copy records its source_trial_id and
    pilot_stratum; its trial_id / block_id start with "v3::pilot::" so a
    pilot observation can never collide with, or be resumed as, a primary
    or holdout one."""
    strata = {(s1, s2): stratum for s1, s2, stratum in pilot_pairs}
    trials = []
    for source in context_trials + nocontext_trials + holdout_trials:
        key = (source["story_1_id"], source["story_2_id"])
        if key not in strata:
            continue
        cell = dict(source)
        cell["source_family"] = source["family"]
        cell["source_trial_id"] = source["trial_id"]
        cell["family"] = FAMILY_PILOT
        cell["trial_id"] = _reid_into_pilot(source["trial_id"])
        cell["block_id"] = _reid_into_pilot(source["block_id"])
        cell["pilot_stratum"] = strata[key]
        trials.append(cell)
    return trials


def _reid_into_pilot(identifier):
    parts = identifier.split("::")
    if parts[0] != ID_PREFIX or parts[1] not in FROZEN_FAMILIES:
        raise ValueError(f"Cannot re-identify {identifier!r} into the pilot family")
    return "::".join([ID_PREFIX, FAMILY_PILOT, parts[1]] + parts[2:])


# ---------------------------------------------------------------------------
# Validation -- fail loudly on any structural drift
# ---------------------------------------------------------------------------

def _assert_unique_ids(trials, label):
    ids = [t["trial_id"] for t in trials]
    duplicates = sorted(tid for tid, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"{label}: duplicate trial_id(s): {duplicates[:5]}")
    for tid in ids:
        if tid.startswith(V2_ID_PREFIX) or not tid.startswith(ID_PREFIX + "::"):
            raise ValueError(f"{label}: {tid!r} is not a v3 id")


def _assert_common_cell_fields(trial, family, label):
    if trial["family"] != family:
        raise ValueError(f"{label}: {trial['trial_id']} has family {trial['family']!r}, expected {family!r}")
    if trial["experiment_id"] != EXPERIMENT_ID:
        raise ValueError(f"{label}: {trial['trial_id']} has experiment_id {trial['experiment_id']!r}")
    if trial["response_format"] != RESPONSE_FORMAT or trial["choice_mode"] != "forced":
        raise ValueError(f"{label}: {trial['trial_id']} is not a forced plain_ab cell")
    if trial["story_1_id"] >= trial["story_2_id"]:
        raise ValueError(f"{label}: {trial['trial_id']} story_1/story_2 are not in canonical order")
    expected_a, expected_b = _display({"id": trial["story_1_id"]}, {"id": trial["story_2_id"]}, trial["position"])
    if (trial["story_a_id"], trial["story_b_id"]) != (expected_a["id"], expected_b["id"]):
        raise ValueError(f"{label}: {trial['trial_id']} story_a/story_b do not match its position")
    parsed = parse_planned_observation_id(make_planned_observation_id(trial["trial_id"], 1))
    for key in ("family", "pair_id", "cue_id", "assignment", "position", "instruction_id", "context_present"):
        if parsed[key] != trial[key]:
            raise ValueError(f"{label}: {trial['trial_id']} id component {key} disagrees with the row ({parsed[key]!r} vs {trial[key]!r})")


def assert_context_family_well_formed(trials, family, expected_instruction_ids, expected_blocks, label=None):
    """Shared by primary_context and holdout: unique v3 ids, every block
    exactly the 4 required (assignment, position) cells, instruction ids
    from the expected set, instruction_id consistent with intervention_id,
    and exactly `expected_blocks` blocks. Returns {block_id: cells}."""
    label = label or family
    _assert_unique_ids(trials, label)
    by_block = defaultdict(list)
    sentences = all_instruction_sentences()
    for trial in trials:
        _assert_common_cell_fields(trial, family, label)
        if not trial["context_present"]:
            raise ValueError(f"{label}: {trial['trial_id']} must be context_present")
        if trial["cue_id"] not in CUE_ORDER or trial["cue_family"] != CUE_FAMILY[trial["cue_id"]]:
            raise ValueError(f"{label}: {trial['trial_id']} has an unknown cue")
        if trial["instruction_id"] not in expected_instruction_ids or trial["instruction_id"] != trial["intervention_id"]:
            raise ValueError(f"{label}: {trial['trial_id']} has instruction {trial['instruction_id']!r} (intervention {trial['intervention_id']!r})")
        if trial["instruction_id"] not in sentences:
            raise ValueError(f"{label}: {trial['trial_id']} has no frozen wording for {trial['instruction_id']!r}")
        if trial["intro"] != context_intro(trial["context_a"]["clause"], trial["context_b"]["clause"]):
            raise ValueError(f"{label}: {trial['trial_id']} intro does not match its context clauses")
        by_block[trial["block_id"]].append(trial)
    for block_id, cells in by_block.items():
        if {(c["assignment"], c["position"]) for c in cells} != REQUIRED_CONTEXT_CELLS or len(cells) != 4:
            raise ValueError(f"{label}: block {block_id!r} does not have exactly the 4 required counterbalance cells")
        if len({(c["cue_id"], c["instruction_id"], c["pair_id"]) for c in cells}) != 1:
            raise ValueError(f"{label}: block {block_id!r} mixes cues/instructions/pairs")
    if len(by_block) != expected_blocks:
        raise ValueError(f"{label}: expected {expected_blocks} blocks, found {len(by_block)}")
    return by_block


def assert_primary_context_manifest_well_formed(trials):
    by_block = assert_context_family_well_formed(
        trials, FAMILY_PRIMARY_CONTEXT, set(INTERVENTION_IDS), N_PAIRS * N_CUES * N_INTERVENTIONS,
    )
    if len(trials) != CONTEXT_UNIQUE_CELLS:
        raise ValueError(f"primary_context: expected {CONTEXT_UNIQUE_CELLS} cells, found {len(trials)}")
    per_pair_cue = defaultdict(set)
    for trial in trials:
        per_pair_cue[(trial["pair_id"], trial["cue_id"])].add(trial["intervention_id"])
    bad = [k for k, v in per_pair_cue.items() if v != set(INTERVENTION_IDS)]
    if bad or len(per_pair_cue) != N_PAIRS * N_CUES:
        raise ValueError(f"primary_context: not every (pair, cue) has all {N_INTERVENTIONS} interventions (e.g. {bad[:3]})")
    return by_block


def assert_nocontext_manifest_well_formed(trials, family=FAMILY_PRIMARY_NOCONTEXT):
    _assert_unique_ids(trials, family)
    by_block = defaultdict(list)
    for trial in trials:
        _assert_common_cell_fields(trial, family, family)
        if trial["context_present"] or trial["cue_id"] is not None or trial["assignment"] is not None:
            raise ValueError(f"{family}: {trial['trial_id']} must carry no context/cue/assignment")
        if trial["context_a"] is not None or trial["context_b"] is not None or trial["intro"] != NO_CONTEXT_INTRO:
            raise ValueError(f"{family}: {trial['trial_id']} carries contextual framing")
        if trial["instruction_id"] not in INTERVENTION_IDS or trial["instruction_id"] != trial["intervention_id"]:
            raise ValueError(f"{family}: {trial['trial_id']} has instruction {trial['instruction_id']!r}")
        by_block[trial["block_id"]].append(trial)
    for block_id, cells in by_block.items():
        if len(cells) != 2 or {c["position"] for c in cells} != REQUIRED_NOCONTEXT_CELLS:
            raise ValueError(f"{family}: block {block_id!r} does not have exactly both display positions")
    if len(by_block) != N_PAIRS * N_INTERVENTIONS or len(trials) != NOCONTEXT_UNIQUE_CELLS:
        raise ValueError(f"{family}: expected {NOCONTEXT_UNIQUE_CELLS} cells in {N_PAIRS * N_INTERVENTIONS} blocks, found {len(trials)} in {len(by_block)}")
    return by_block


def assert_holdout_manifest_well_formed(trials):
    expected_ids = {holdout_instruction_id(b, c) for b in CUE_ENUMERATING_INTERVENTIONS for c in CUE_ORDER}
    by_block = assert_context_family_well_formed(
        trials, FAMILY_HOLDOUT, expected_ids, N_PAIRS * N_CUES * len(CUE_ENUMERATING_INTERVENTIONS),
    )
    for trial in trials:
        if trial["held_out_cue_id"] != trial["cue_id"]:
            raise ValueError(f"holdout: {trial['trial_id']} is tested on {trial['cue_id']!r} but omits {trial['held_out_cue_id']!r}")
        if trial["instruction_id"] != holdout_instruction_id(trial["base_intervention_id"], trial["held_out_cue_id"]):
            raise ValueError(f"holdout: {trial['trial_id']} instruction/held-out cue mismatch")
    if len(trials) != HOLDOUT_UNIQUE_CELLS:
        raise ValueError(f"holdout: expected {HOLDOUT_UNIQUE_CELLS} cells, found {len(trials)}")
    return by_block


def assert_pilot_manifest_well_formed(trials, frozen_trial_ids):
    _assert_unique_ids(trials, FAMILY_PILOT)
    for trial in trials:
        if trial["family"] != FAMILY_PILOT or not trial["trial_id"].startswith(f"{ID_PREFIX}::{FAMILY_PILOT}::"):
            raise ValueError(f"pilot: {trial['trial_id']} is not in the pilot family")
        if trial["trial_id"] in frozen_trial_ids:
            raise ValueError(f"pilot: {trial['trial_id']} collides with a frozen-family id")
        if trial["source_trial_id"] not in frozen_trial_ids:
            raise ValueError(f"pilot: {trial['trial_id']} has an unknown source_trial_id")
        if trial["pilot_stratum"] not in ("strong", "ambiguous", "seeded"):
            raise ValueError(f"pilot: {trial['trial_id']} has stratum {trial['pilot_stratum']!r}")
    families = {t["source_family"] for t in trials}
    if families != set(FROZEN_FAMILIES):
        raise ValueError(f"pilot: must span every frozen family, found {sorted(families)}")
    context = [t for t in trials if t["source_family"] == FAMILY_PRIMARY_CONTEXT]
    if {t["intervention_id"] for t in context} != set(INTERVENTION_IDS):
        raise ValueError("pilot: context cells must span all 8 interventions")
    if {t["cue_id"] for t in context} != set(CUE_ORDER):
        raise ValueError("pilot: context cells must span all 5 cues")
    if {t["position"] for t in trials} != set(POSITIONS):
        raise ValueError("pilot: must span both display positions")
    nocontext = [t for t in trials if t["source_family"] == FAMILY_PRIMARY_NOCONTEXT]
    if {t["intervention_id"] for t in nocontext} != set(INTERVENTION_IDS):
        raise ValueError("pilot: no-context cells must span all 8 interventions")


def assert_no_id_overlap(*trial_lists):
    seen = {}
    for trials in trial_lists:
        for trial in trials:
            if trial["trial_id"] in seen:
                raise ValueError(f"trial_id {trial['trial_id']!r} appears in more than one manifest")
            seen[trial["trial_id"]] = True


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def write_manifest(trials, path):
    """Byte-stable: fixed row order (generation order), sorted keys, one
    row per line, trailing newline."""
    with open(path, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial, sort_keys=True) + "\n")


def load_manifest(path):
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def generate_frozen_manifests(items=None, contrasts=None, texts=None):
    """Build + validate the three frozen families in memory. Returns
    (context_trials, nocontext_trials, holdout_trials)."""
    items = items if items is not None else load_items(ITEMS_FILE)
    contrasts = contrasts if contrasts is not None else load_contrasts()
    texts = texts if texts is not None else load_story_texts(items)
    assert_items_are_the_frozen_corpus(items)
    assert_contrasts_are_well_formed(contrasts)

    context_trials = build_context_trials(contrasts, items, texts)
    nocontext_trials = build_nocontext_trials(items, texts)
    holdout_trials = build_holdout_trials(contrasts, items, texts)

    assert_primary_context_manifest_well_formed(context_trials)
    assert_nocontext_manifest_well_formed(nocontext_trials)
    assert_holdout_manifest_well_formed(holdout_trials)
    assert_no_id_overlap(context_trials, nocontext_trials, holdout_trials)
    return context_trials, nocontext_trials, holdout_trials


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Generate the v3 manifests (no API calls).")
    parser.add_argument("--pilot-seed", type=int, default=20260918)
    parser.add_argument("--pilot-n-strong", type=int, default=PILOT_DEFAULT_N_STRONG)
    parser.add_argument("--pilot-n-ambiguous", type=int, default=PILOT_DEFAULT_N_AMBIGUOUS)
    parser.add_argument("--baseline-strength-file", default=None,
                         help="Optional CSV with story_1_id, story_2_id, baseline_strength (e.g. an independent "
                              "blind-baseline analysis) used to pick strong/ambiguous pilot pairs")
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()

    context_trials, nocontext_trials, holdout_trials = generate_frozen_manifests()
    write_manifest(context_trials, CONTEXT_TRIALS_FILE)
    write_manifest(nocontext_trials, NOCONTEXT_TRIALS_FILE)
    write_manifest(holdout_trials, HOLDOUT_TRIALS_FILE)

    print(f"v3 selective-suppression experiment: {EXPERIMENT_ID}")
    print(f"Stories: {N_STORIES}   Story pairs: {N_PAIRS}   Cues: {N_CUES}   Interventions: {N_INTERVENTIONS}")
    print(f"primary_context cells:   {len(context_trials)} -> {CONTEXT_TRIALS_FILE}")
    print(f"primary_nocontext cells: {len(nocontext_trials)} -> {NOCONTEXT_TRIALS_FILE}")
    print(f"holdout cells:           {len(holdout_trials)} -> {HOLDOUT_TRIALS_FILE}")

    if not args.skip_pilot:
        strength_rows = None
        if args.baseline_strength_file:
            with open(args.baseline_strength_file, newline="", encoding="utf-8") as f:
                strength_rows = list(csv.DictReader(f))
        items = load_items(ITEMS_FILE)
        pilot_pairs = select_pilot_pairs(items, args.pilot_n_strong, args.pilot_n_ambiguous, args.pilot_seed, strength_rows)
        pilot_trials = build_pilot_trials(context_trials, nocontext_trials, holdout_trials, pilot_pairs)
        frozen_ids = {t["trial_id"] for t in context_trials + nocontext_trials + holdout_trials}
        assert_pilot_manifest_well_formed(pilot_trials, frozen_ids)
        write_manifest(pilot_trials, PILOT_TRIALS_FILE)
        print(f"pilot cells:             {len(pilot_trials)} -> {PILOT_TRIALS_FILE}  "
              f"(pairs: {[(s1, s2, st) for s1, s2, st in pilot_pairs]}; seed={args.pilot_seed}; "
              f"strength file={'yes' if strength_rows else 'no (seeded selection)'})")


if __name__ == "__main__":
    main()
