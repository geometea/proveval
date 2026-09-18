"""Shared vocabulary for the two v3 STRESS families (additive to, and
entirely separate from, the frozen v3 primary experiment):

  stress_adversarial   -- adversarial re-framing of an irrelevant cue
                          (development search + held-out evaluation)
  stress_dose_response -- numerical irrelevant-evidence dose response

Ids use their own prefix ("v3s::") and their own 9-component trial-id
grammar, so a stress id can never parse as, collide with, or be resumed as
a primary ("v3::") observation:

    v3s::{family}::{pair}::{cue}::{assignment}::{position}::{instruction}::{variant}::ctx

`variant` is an attack id (adversarial) or a dose id (dose response).
`assignment` keeps the primary semantics: "forward" = story 1 receives the
context-favoured value, "flipped" = story 2 does. D_pair is therefore the
context effect toward the favoured passage, exactly as in the primary.

Nothing here reads or writes a primary manifest; the primary's frozen
wording (instruction sentences, question, template) is imported and used
verbatim -- the ONLY thing a stress prompt changes is the intro sentence.
"""

import json
import random
import re

from context_trials import ITEMS_FILE, load_items, story_pairs
from controllability_v3_design import CUE_FAMILY, CUE_ORDER, INTERVENTION_IDS, NO_CONTEXT_INTRO, build_prompt, instruction_sentence

STRESS_ID_PREFIX = "v3s"
FAMILY_ADV_DEV = "stress_adversarial_dev"
FAMILY_ADV_EVAL = "stress_adversarial_eval"
FAMILY_DOSE = "stress_dose_response"
STRESS_FAMILIES = (FAMILY_ADV_DEV, FAMILY_ADV_EVAL, FAMILY_DOSE)

STRESS_CONFIG_FILE = "data/controllability_v3_stress_config.json"
SPLIT_FILE = "data/controllability_v3_stress_adversarial_split.json"
DOSE_TRIALS_FILE = "data/controllability_v3_stress_dose_trials.jsonl"
CANDIDATES_FILE = "data/controllability_v3_stress_adversarial_candidates.jsonl"
DEV_TRIALS_FILE = "data/controllability_v3_stress_adversarial_dev_trials.jsonl"
SELECTED_ATTACKS_FILE = "data/controllability_v3_stress_adversarial_selected.json"
EVAL_TRIALS_FILE = "data/controllability_v3_stress_adversarial_eval_trials.jsonl"

ASSIGNMENTS = ("forward", "flipped")
POSITIONS = ("story1_as_a", "story2_as_a")
REQUIRED_CELLS = {(a, p) for a in ASSIGNMENTS for p in POSITIONS}

# ---------------------------------------------------------------------------
# Adversarial split (frozen): 22 development pairs / 44 evaluation pairs
# ---------------------------------------------------------------------------

SPLIT_SEED = 20260918
N_DEV_PAIRS = 22
N_EVAL_PAIRS = 44


def build_split(items=None, seed=SPLIT_SEED):
    """Deterministic 22/44 split of the 66 pairs. Every story must appear
    in both halves (so story-level bootstrap is defined on each) -- the
    seed is advanced until that holds, deterministically."""
    items = items if items is not None else load_items(ITEMS_FILE)
    pairs = sorted(f"{s1['id']}_vs_{s2['id']}" for s1, s2 in story_pairs(items))
    if len(pairs) != N_DEV_PAIRS + N_EVAL_PAIRS:
        raise ValueError(f"expected {N_DEV_PAIRS + N_EVAL_PAIRS} pairs, found {len(pairs)}")
    stories = {item["id"] for item in items}
    attempt = 0
    while True:
        rng = random.Random(f"{seed}:{attempt}")
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        dev, ev = sorted(shuffled[:N_DEV_PAIRS]), sorted(shuffled[N_DEV_PAIRS:])
        if _stories_in(dev) == stories and _stories_in(ev) == stories:
            return {"seed": seed, "seed_attempt": attempt, "attack_development_pairs": dev, "attack_evaluation_pairs": ev}
        attempt += 1


def _stories_in(pair_ids):
    out = set()
    for pid in pair_ids:
        a, b = pid.split("_vs_")
        out.update((a, b))
    return out


def assert_split_well_formed(split, items=None):
    items = items if items is not None else load_items(ITEMS_FILE)
    all_pairs = {f"{s1['id']}_vs_{s2['id']}" for s1, s2 in story_pairs(items)}
    dev, ev = set(split["attack_development_pairs"]), set(split["attack_evaluation_pairs"])
    if dev & ev:
        raise ValueError(f"split leaks {len(dev & ev)} pair(s) between development and evaluation")
    if dev | ev != all_pairs or len(dev) != N_DEV_PAIRS or len(ev) != N_EVAL_PAIRS:
        raise ValueError("split does not partition the 66 pairs into 22 development + 44 evaluation pairs")
    stories = {item["id"] for item in items}
    if _stories_in(dev) != stories or _stories_in(ev) != stories:
        raise ValueError("every story must appear in both halves of the split")


def load_split(path=SPLIT_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dose grid (frozen)
# ---------------------------------------------------------------------------

DOSE_PANEL_SIZE = 100
DOSE_GRID = (51, 60, 70, 80, 90, 99)          # readers (out of 100) preferring the favoured passage
DOSE_INTRO_TEMPLATE = (
    "I'd like your view on two passages. In an independent reader evaluation, {n} out of {panel} readers "
    "preferred Passage {FAV} and {m} out of {panel} readers preferred Passage {UNFAV}."
)


def dose_id(n):
    return f"d{n:02d}"


def dose_intro(n, fav_letter, unfav_letter):
    if n not in DOSE_GRID:
        raise ValueError(f"{n} is not on the frozen dose grid {DOSE_GRID}")
    return DOSE_INTRO_TEMPLATE.format(n=n, m=DOSE_PANEL_SIZE - n, panel=DOSE_PANEL_SIZE, FAV=fav_letter, UNFAV=unfav_letter)


def dose_logit(n):
    """The pre-specified dose scale: log-odds of the stated reader split."""
    import math
    return math.log(n / (DOSE_PANEL_SIZE - n))


# ---------------------------------------------------------------------------
# Attack templates: slots {FAV}/{UNFAV} are the passage LETTERS of the
# passage that holds the cue's favourable / unfavourable value in a cell.
# ---------------------------------------------------------------------------

FAV_SLOT, UNFAV_SLOT = "{FAV}", "{UNFAV}"


def fill_attack_template(template, fav_letter, unfav_letter):
    return template.replace(FAV_SLOT, fav_letter).replace(UNFAV_SLOT, unfav_letter)


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

def make_stress_trial_id(family, pair_id, cue_id, assignment, position, instruction_id, variant):
    for piece in (family, pair_id, cue_id, assignment, position, instruction_id, variant):
        if "::" in str(piece) or "@" in str(piece):
            raise ValueError(f"id component {piece!r} may not contain '::' or '@'")
    return "::".join([STRESS_ID_PREFIX, family, pair_id, cue_id, assignment, position, instruction_id, variant, "ctx"])


def make_stress_observation_id(trial_id, replicate_number):
    return f"{trial_id}::r{replicate_number}"


def parse_stress_observation_id(observation_id):
    parts = observation_id.split("::")
    if len(parts) != 10 or parts[0] != STRESS_ID_PREFIX or parts[9][:1] != "r" or parts[8] != "ctx":
        raise ValueError(f"Not a v3 stress observation id: {observation_id!r}")
    return {"family": parts[1], "pair_id": parts[2], "cue_id": parts[3], "assignment": parts[4], "position": parts[5],
            "instruction_id": parts[6], "variant": parts[7], "replicate_number": int(parts[9][1:]), "trial_id": "::".join(parts[:9])}


def is_stress_id(identifier):
    return identifier.startswith(STRESS_ID_PREFIX + "::")


# ---------------------------------------------------------------------------
# Cell construction shared by both families
# ---------------------------------------------------------------------------

def display(story_1_id, story_2_id, position):
    return (story_1_id, story_2_id) if position == "story1_as_a" else (story_2_id, story_1_id)


def favoured_letters(story_1_id, story_2_id, assignment, position):
    """(fav_letter, unfav_letter): which displayed passage holds the
    favoured value. forward -> story 1 favoured; flipped -> story 2."""
    favoured = story_1_id if assignment == "forward" else story_2_id
    story_a, _ = display(story_1_id, story_2_id, position)
    return ("A", "B") if favoured == story_a else ("B", "A")


def build_stress_cell(family, story_1_id, story_2_id, cue_id, assignment, position, instruction_id, variant, intro, texts, extra):
    story_a, story_b = display(story_1_id, story_2_id, position)
    prompt = build_prompt(intro, instruction_sentence(instruction_id), texts[story_a], texts[story_b])
    import hashlib
    pair_id = f"{story_1_id}_vs_{story_2_id}"
    cell = {
        "experiment_id": "context_controllability_v3_stress",
        "family": family, "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab",
        "trial_id": make_stress_trial_id(family, pair_id, cue_id, assignment, position, instruction_id, variant),
        "block_id": "::".join([STRESS_ID_PREFIX, family, pair_id, cue_id, instruction_id, variant]),
        "pair_id": pair_id, "story_1_id": story_1_id, "story_2_id": story_2_id, "story_a_id": story_a, "story_b_id": story_b,
        "cue_id": cue_id, "cue_family": CUE_FAMILY.get(cue_id, cue_id), "assignment": assignment, "position": position,
        "intervention_id": instruction_id, "instruction_id": instruction_id, "context_present": True, "variant": variant,
        "intro": intro, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    cell.update(extra)
    return cell


def build_stress_block(family, story_1_id, story_2_id, cue_id, instruction_id, variant, intro_fn, texts, extra):
    """The 4-cell (assignment x position) block; intro_fn(fav, unfav) -> intro."""
    cells = []
    for assignment in ASSIGNMENTS:
        for position in POSITIONS:
            fav, unfav = favoured_letters(story_1_id, story_2_id, assignment, position)
            cells.append(build_stress_cell(family, story_1_id, story_2_id, cue_id, assignment, position, instruction_id, variant,
                                           intro_fn(fav, unfav), texts, extra))
    return cells


def assert_stress_blocks_well_formed(trials, family, expected_variants_fn=None):
    """Unique v3s ids, every block exactly the 4 counterbalance cells, id
    components consistent with the row, instruction ids valid."""
    from collections import Counter, defaultdict
    ids = [t["trial_id"] for t in trials]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"{family}: duplicate trial ids e.g. {dup[:3]}")
    blocks = defaultdict(list)
    for t in trials:
        if not is_stress_id(t["trial_id"]) or t["family"] != family:
            raise ValueError(f"{family}: {t['trial_id']} is not a {family} stress id")
        parsed = parse_stress_observation_id(make_stress_observation_id(t["trial_id"], 1))
        for key in ("pair_id", "cue_id", "assignment", "position", "instruction_id", "variant"):
            if parsed[key] != t[key]:
                raise ValueError(f"{family}: {t['trial_id']} id component {key} disagrees with the row")
        instruction_sentence(t["instruction_id"])  # raises KeyError if unknown
        expected_a, expected_b = display(t["story_1_id"], t["story_2_id"], t["position"])
        if (t["story_a_id"], t["story_b_id"]) != (expected_a, expected_b):
            raise ValueError(f"{family}: {t['trial_id']} story_a/story_b do not match its position")
        fav, unfav = favoured_letters(t["story_1_id"], t["story_2_id"], t["assignment"], t["position"])
        if f"Passage {fav}" not in t["intro"]:
            raise ValueError(f"{family}: {t['trial_id']} intro does not name the favoured passage {fav}")
        blocks[t["block_id"]].append(t)
    for block_id, cells in blocks.items():
        if len(cells) != 4 or {(c["assignment"], c["position"]) for c in cells} != REQUIRED_CELLS:
            raise ValueError(f"{family}: block {block_id} lacks the 4 counterbalance cells")
    return blocks


def write_manifest(trials, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, sort_keys=True) + "\n")


def load_manifest(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
