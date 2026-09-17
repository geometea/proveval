"""v2 context-controllability experiment: generate the primary treatment
manifest, the optional minimal-instruction manifest, and the independent
blind baseline manifest.

This is a from-scratch v2 design living ALONGSIDE the v1 implementation
(controllability_trials.py / analyze_controllability.py / data/
controllability_*.jsonl), which is left completely untouched. See
STUDY_PROTOCOL_V2.md for the full design rationale. Key differences from v1:

  - One standardised writing-quality question (QUESTION), used verbatim in
    every v2 prompt -- no ratings, no JSON, no five-category rubric, no
    ties.
  - Two primary instruction conditions replace v1's naturalistic/text_only_
    invariance regimes: MATCHED_CONTROL_INSTRUCTION and TEXT_ONLY_INSTRUCTION
    (plus an optional, never-primary MINIMAL_INSTRUCTION_CONDITION with no
    added sentence at all). For a given story-pair/contrast/assignment/
    position cell, the matched_control and text_only prompts differ in
    EXACTLY that one instruction sentence -- everything else (intro,
    passages, question) is byte-identical between them.
  - The four-cell counterbalance (context assignment x display position) is
    generated once per instruction_condition, and every
    (story pair, contrast) additionally gets a `superblock_id` spanning its
    matched_control block (4 cells) + text_only block (4 cells) = 8 cells.
    Primary analysis only ever uses complete 8-cell superblocks (see
    analyze_controllability_v2.filter_complete_superblocks).
  - The blind baseline (66 pairs x 2 display positions = 132 prompts) has no
    contextual framing AND uses the matched_control instruction sentence
    (there is no naturalistic/suppression distinction to draw for a prompt
    that never had context in the first place). There is no second
    text-only baseline.

The 5 canonical context contrasts (data/controllability_v2_contrasts.jsonl)
keep the EXACT wording of v1's data/controllability_contrasts.jsonl --
a_clause/b_clause are read verbatim, never regenerated or paraphrased.

Run this file directly to write the treatment, minimal-instruction, and
baseline manifests and print a computed dataset-size summary. No model API
calls happen here.
"""

import hashlib
import json

from context_comparisons import load_story
from context_trials import load_items, story_pairs

CONTRASTS_FILE = "data/controllability_v2_contrasts.jsonl"
TRIALS_FILE = "data/controllability_v2_trials.jsonl"
MINIMAL_TRIALS_FILE = "data/controllability_v2_minimal_trials.jsonl"
BASELINE_TRIALS_FILE = "data/controllability_v2_baseline_trials.jsonl"

EXPERIMENT_ID = "context_controllability_v2"
BASELINE_EXPERIMENT_ID = "context_controllability_baseline_v2"
MINIMAL_EXPERIMENT_ID = "context_controllability_minimal_v2"

RESPONSE_FORMAT = "plain_ab"
PRIMARY_CATEGORY = "overall_quality"

# The two PRIMARY instruction conditions (item 2). "minimal_instruction" is
# OPTIONAL, generated only into MINIMAL_TRIALS_FILE, and never part of a
# primary superblock or primary analysis.
PRIMARY_INSTRUCTION_CONDITIONS = ("matched_control", "text_only")
MINIMAL_INSTRUCTION_CONDITION = "minimal_instruction"

REQUIRED_CONTRAST_IDS = {
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
}

CONTRAST_DISPLAY_NAMES = {
    "provenance_human_vs_llm": "human vs LLM",
    "source_journal_vs_random": "journal vs random",
    "reception_positive_vs_negative": "positive vs negative reception",
    "opinion_liked_vs_disliked": "liked vs disliked",
    "editing_edited_vs_first_draft": "edited vs first draft",
}

REQUIRED_CELLS = {
    ("forward", "story1_as_a"),
    ("forward", "story2_as_a"),
    ("flipped", "story1_as_a"),
    ("flipped", "story2_as_a"),
}

BASELINE_POSITIONS = {"story1_as_a", "story2_as_a"}

# ---------------------------------------------------------------------------
# Frozen prompt wording (item 1/2). Every string below is used VERBATIM --
# never paraphrased, reworded, or generated per-cell.
# ---------------------------------------------------------------------------

BASELINE_INTRO = "I'd like your view on two passages."

QUESTION = (
    "Which passage is better written overall? If they're close, choose the "
    "one you think is slightly better written. Please answer A or B."
)

MATCHED_CONTROL_INSTRUCTION = "For this comparison, please read both passages carefully before making your choice."
TEXT_ONLY_INSTRUCTION = "For this comparison, please read both passages carefully and judge only the writing itself."

INSTRUCTION_SENTENCES = {
    "matched_control": MATCHED_CONTROL_INSTRUCTION,
    "text_only": TEXT_ONLY_INSTRUCTION,
    "minimal_instruction": None,  # no added sentence at all
}

_BODY_WITH_INSTRUCTION = "{intro}\n\n{instruction}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{question}"
_BODY_WITHOUT_INSTRUCTION = "{intro}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{question}"


def intro_sentence(a_clause, b_clause):
    """Deterministic opening sentence naming Passage A's and Passage B's
    attribution -- never generated/paraphrased per cell."""
    return f"I'd like your view on two passages. Passage A {a_clause}, while Passage B {b_clause}."


def build_prompt(intro, text_a, text_b, instruction_condition):
    """Build one prompt. For corresponding matched_control/text_only cells
    (same intro/text_a/text_b), the returned prompt differs from the other
    condition's ONLY in the instruction sentence -- see
    test_matched_control_and_text_only_differ_only_in_the_instruction_sentence."""
    instruction = INSTRUCTION_SENTENCES[instruction_condition]
    if instruction is None:
        return _BODY_WITHOUT_INSTRUCTION.format(intro=intro, text_a=text_a, text_b=text_b, question=QUESTION)
    return _BODY_WITH_INSTRUCTION.format(intro=intro, instruction=instruction, text_a=text_a, text_b=text_b, question=QUESTION)


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_contrasts(path=CONTRASTS_FILE):
    contrasts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                contrasts.append(json.loads(line))
    return contrasts


# ---------------------------------------------------------------------------
# Validation: fail loudly on a malformed contrast/trial set
# ---------------------------------------------------------------------------

REQUIRED_CONTRAST_FIELDS = {"contrast_id", "dimension", "a", "b", "a_clause", "b_clause", "expected_direction"}


def assert_contrasts_are_well_formed(contrasts):
    seen_ids = set()
    for contrast in contrasts:
        missing = REQUIRED_CONTRAST_FIELDS - set(contrast)
        if missing:
            raise ValueError(f"{contrast.get('contrast_id', '?')!r} is missing required field(s): {sorted(missing)}")
        if contrast["contrast_id"] in seen_ids:
            raise ValueError(f"Duplicate contrast_id: {contrast['contrast_id']!r}")
        seen_ids.add(contrast["contrast_id"])
        if contrast["expected_direction"] not in ("a", "b"):
            raise ValueError(f"{contrast['contrast_id']!r}: expected_direction must be 'a' or 'b', got {contrast['expected_direction']!r}")
        if contrast["a"] == contrast["b"]:
            raise ValueError(f"{contrast['contrast_id']!r}: 'a' and 'b' must be distinct values")

    if seen_ids != REQUIRED_CONTRAST_IDS:
        raise ValueError(f"Expected exactly the 5 contrasts {sorted(REQUIRED_CONTRAST_IDS)}, found {sorted(seen_ids)}")


def assert_trials_are_well_formed(trials, expected_instruction_conditions):
    """Fail loudly if generation ever drifts: unique trial ids, every trial
    an ordinary forced-choice plain_ab cell, prompt_sha256 matching its own
    prompt, and every (contrast, story-pair, instruction_condition) block
    has exactly the 4 required counterbalanced cells. Returns
    {block_id: [cells]}."""
    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        duplicates = sorted({tid for tid in trial_ids if trial_ids.count(tid) > 1})
        raise ValueError(f"Duplicate trial_id(s): {duplicates}")

    by_block = {}
    for trial in trials:
        if trial["type"] != "context_pairwise":
            raise ValueError(f"{trial['trial_id']}: expected type 'context_pairwise', got {trial['type']!r}")
        if trial["choice_mode"] != "forced":
            raise ValueError(f"{trial['trial_id']}: expected choice_mode 'forced', got {trial['choice_mode']!r}")
        if trial["response_format"] != RESPONSE_FORMAT:
            raise ValueError(f"{trial['trial_id']}: expected response_format {RESPONSE_FORMAT!r}, got {trial['response_format']!r}")
        if trial["contrast_id"] not in REQUIRED_CONTRAST_IDS:
            raise ValueError(f"{trial['trial_id']}: unexpected contrast_id {trial['contrast_id']!r}")
        if trial["instruction_condition"] not in expected_instruction_conditions:
            raise ValueError(f"{trial['trial_id']}: unexpected instruction_condition {trial['instruction_condition']!r}")
        if trial["prompt_sha256"] != sha256_hex(trial["prompt"]):
            raise ValueError(f"{trial['trial_id']}: prompt_sha256 does not match its own prompt")
        by_block.setdefault(trial["block_id"], []).append(trial)

    for block_id, cells in by_block.items():
        if len(cells) != 4:
            raise ValueError(f"Block {block_id!r} has {len(cells)} cell(s), expected exactly 4")
        actual_cells = {(c["assignment"], c["position"]) for c in cells}
        if actual_cells != REQUIRED_CELLS:
            raise ValueError(f"Block {block_id!r} has cells {sorted(actual_cells)}, expected {sorted(REQUIRED_CELLS)}")

    return by_block


def assert_superblocks_are_well_formed(treatment_trials):
    """Every superblock_id (one per story-pair x contrast) must contain
    exactly 8 cells: 4 matched_control + 4 text_only. Returns
    {superblock_id: [cells]}."""
    by_superblock = {}
    for trial in treatment_trials:
        by_superblock.setdefault(trial["superblock_id"], []).append(trial)

    for superblock_id, cells in by_superblock.items():
        if len(cells) != 8:
            raise ValueError(f"Superblock {superblock_id!r} has {len(cells)} cell(s), expected exactly 8")
        by_condition = {}
        for c in cells:
            by_condition.setdefault(c["instruction_condition"], []).append(c)
        if set(by_condition) != set(PRIMARY_INSTRUCTION_CONDITIONS):
            raise ValueError(f"Superblock {superblock_id!r} has instruction_conditions {sorted(by_condition)}, expected {sorted(PRIMARY_INSTRUCTION_CONDITIONS)}")
        for condition, condition_cells in by_condition.items():
            if len(condition_cells) != 4:
                raise ValueError(f"Superblock {superblock_id!r}/{condition}: {len(condition_cells)} cell(s), expected 4")

    return by_superblock


def assert_baseline_trials_are_well_formed(trials):
    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        duplicates = sorted({tid for tid in trial_ids if trial_ids.count(tid) > 1})
        raise ValueError(f"Duplicate baseline trial_id(s): {duplicates}")

    by_block = {}
    for trial in trials:
        if trial["type"] != "context_pairwise":
            raise ValueError(f"{trial['trial_id']}: expected type 'context_pairwise', got {trial['type']!r}")
        if trial["choice_mode"] != "forced":
            raise ValueError(f"{trial['trial_id']}: expected choice_mode 'forced', got {trial['choice_mode']!r}")
        if trial["response_format"] != RESPONSE_FORMAT:
            raise ValueError(f"{trial['trial_id']}: expected response_format {RESPONSE_FORMAT!r}, got {trial['response_format']!r}")
        if "assignment" in trial or "contrast_id" in trial:
            raise ValueError(f"{trial['trial_id']}: baseline trial must not carry any contextual clause/assignment/contrast_id")
        if trial["instruction_condition"] != "matched_control":
            raise ValueError(f"{trial['trial_id']}: baseline must use matched_control, got {trial['instruction_condition']!r}")
        if trial["prompt_sha256"] != sha256_hex(trial["prompt"]):
            raise ValueError(f"{trial['trial_id']}: prompt_sha256 does not match its own prompt")
        by_block.setdefault(trial["block_id"], []).append(trial)

    for block_id, cells in by_block.items():
        if len(cells) != 2:
            raise ValueError(f"Baseline block {block_id!r} has {len(cells)} cell(s), expected exactly 2")
        actual_positions = {c["position"] for c in cells}
        if actual_positions != BASELINE_POSITIONS:
            raise ValueError(f"Baseline block {block_id!r} has positions {actual_positions}, expected {BASELINE_POSITIONS}")

    return by_block


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def build_treatment_block(contrast, story_1, story_2, instruction_condition, superblock_id):
    """Build the 4-cell counterbalanced block for one contrast/story-pair/
    instruction_condition: crosses "assignment" (which contrast value each
    text is attributed) with "position" (which text is DISPLAYED as
    Passage A). Structurally identical to v1's four-cell counterbalance --
    only the prompt wording (this experiment's own frozen QUESTION/
    instruction sentences) and instruction_condition/superblock_id
    bookkeeping differ."""
    if story_1["id"] == story_2["id"]:
        raise ValueError("build_treatment_block requires two different stories, not the same story twice")

    text_1 = load_story(story_1["path"])
    text_2 = load_story(story_2["path"])
    pair_id = f"{story_1['id']}_vs_{story_2['id']}"
    block_id = f"{superblock_id}__{instruction_condition}"
    a_clause, b_clause = contrast["a_clause"], contrast["b_clause"]

    assignments = {
        "forward": ((contrast["a"], a_clause), (contrast["b"], b_clause)),
        "flipped": ((contrast["b"], b_clause), (contrast["a"], a_clause)),
    }
    stories = {"story1_as_a": (story_1, story_2, text_1, text_2), "story2_as_a": (story_2, story_1, text_2, text_1)}

    cells = []
    for assignment, (context_1, context_2) in assignments.items():
        for position, (story_a, story_b, text_a, text_b) in stories.items():
            context_by_id = {story_1["id"]: context_1, story_2["id"]: context_2}
            value_a, clause_a = context_by_id[story_a["id"]]
            value_b, clause_b = context_by_id[story_b["id"]]
            intro = intro_sentence(clause_a, clause_b)
            prompt = build_prompt(intro, text_a, text_b, instruction_condition)
            cells.append(
                {
                    "trial_id": f"{block_id}__{assignment}__{position}",
                    "block_id": block_id,
                    "superblock_id": superblock_id,
                    "type": "context_pairwise",
                    "choice_mode": "forced",
                    "response_format": RESPONSE_FORMAT,
                    "instruction_condition": instruction_condition,
                    "contrast_id": contrast["contrast_id"],
                    "dimension": contrast["dimension"],
                    "expected_direction": contrast["expected_direction"],
                    "story_1_id": story_1["id"],
                    "story_2_id": story_2["id"],
                    "story_a_id": story_a["id"],
                    "story_b_id": story_b["id"],
                    "assignment": assignment,
                    "position": position,
                    "context_a": {"value": value_a, "clause": clause_a},
                    "context_b": {"value": value_b, "clause": clause_b},
                    "intro": intro,
                    "prompt": prompt,
                    "prompt_sha256": sha256_hex(prompt),
                }
            )
    return cells


def build_baseline_trials(items):
    """66 pairs x 2 display positions = 132 trials: no contextual framing at
    all, matched_control instruction (item 5) -- there is no
    naturalistic/text-only distinction to draw when there was never any
    context to suppress. All 132 trials carry contrast_id=None and no
    "assignment" field."""
    trials = []
    for story_1, story_2 in story_pairs(items):
        text_1 = load_story(story_1["path"])
        text_2 = load_story(story_2["path"])
        pair_id = f"{story_1['id']}_vs_{story_2['id']}"
        block_id = f"baseline_superblock__{pair_id}"
        stories = {"story1_as_a": (story_1, story_2, text_1, text_2), "story2_as_a": (story_2, story_1, text_2, text_1)}
        for position, (story_a, story_b, text_a, text_b) in stories.items():
            prompt = build_prompt(BASELINE_INTRO, text_a, text_b, "matched_control")
            trials.append(
                {
                    "trial_id": f"{block_id}__{position}",
                    "block_id": block_id,
                    "type": "context_pairwise",
                    "choice_mode": "forced",
                    "response_format": RESPONSE_FORMAT,
                    "instruction_condition": "matched_control",
                    "story_1_id": story_1["id"],
                    "story_2_id": story_2["id"],
                    "story_a_id": story_a["id"],
                    "story_b_id": story_b["id"],
                    "position": position,
                    "intro": BASELINE_INTRO,
                    "prompt": prompt,
                    "prompt_sha256": sha256_hex(prompt),
                    "experiment_id": BASELINE_EXPERIMENT_ID,
                }
            )
    return trials


def build_all_treatment_trials(contrasts, items, instruction_conditions):
    """One superblock_id per (story pair, contrast); one 4-cell block per
    instruction_condition within it."""
    all_trials = []
    for story_1, story_2 in story_pairs(items):
        for contrast in contrasts:
            pair_id = f"{story_1['id']}_vs_{story_2['id']}"
            superblock_id = f"superblock__{contrast['contrast_id']}__{pair_id}"
            for instruction_condition in instruction_conditions:
                for cell in build_treatment_block(contrast, story_1, story_2, instruction_condition, superblock_id):
                    all_trials.append(cell)
    return all_trials


def main():
    items = load_items()
    contrasts = load_contrasts()
    assert_contrasts_are_well_formed(contrasts)

    treatment_trials = build_all_treatment_trials(contrasts, items, PRIMARY_INSTRUCTION_CONDITIONS)
    for trial in treatment_trials:
        trial["experiment_id"] = EXPERIMENT_ID
    by_block = assert_trials_are_well_formed(treatment_trials, PRIMARY_INSTRUCTION_CONDITIONS)
    by_superblock = assert_superblocks_are_well_formed(treatment_trials)

    with open(TRIALS_FILE, "w", encoding="utf-8") as f:
        for trial in treatment_trials:
            f.write(json.dumps(trial) + "\n")

    minimal_trials = build_all_treatment_trials(contrasts, items, [MINIMAL_INSTRUCTION_CONDITION])
    for trial in minimal_trials:
        trial["experiment_id"] = MINIMAL_EXPERIMENT_ID
    assert_trials_are_well_formed(minimal_trials, [MINIMAL_INSTRUCTION_CONDITION])
    with open(MINIMAL_TRIALS_FILE, "w", encoding="utf-8") as f:
        for trial in minimal_trials:
            f.write(json.dumps(trial) + "\n")

    baseline_trials = build_baseline_trials(items)
    by_baseline_block = assert_baseline_trials_are_well_formed(baseline_trials)
    with open(BASELINE_TRIALS_FILE, "w", encoding="utf-8") as f:
        for trial in baseline_trials:
            f.write(json.dumps(trial) + "\n")

    n_stories = len(items)
    n_pairs = n_stories * (n_stories - 1) // 2
    n_contrasts = len(contrasts)
    n_blocks = n_pairs * n_contrasts * len(PRIMARY_INSTRUCTION_CONDITIONS)
    assert len(by_block) == n_blocks
    assert len(treatment_trials) == n_blocks * 4 == n_pairs * n_contrasts * 2 * 4
    assert len(treatment_trials) == 2640, f"expected exactly 2640 primary treatment prompts, got {len(treatment_trials)}"
    assert len(by_superblock) == n_pairs * n_contrasts
    assert len(minimal_trials) == n_pairs * n_contrasts * 4
    assert len(by_baseline_block) == n_pairs
    assert len(baseline_trials) == n_pairs * 2
    assert len(baseline_trials) == 132, f"expected exactly 132 baseline prompts, got {len(baseline_trials)}"

    print(f"Context controllability experiment v2: {EXPERIMENT_ID}")
    print(f"Stories: {n_stories}")
    print(f"Story pairs: {n_pairs}")
    print(f"Context contrasts: {n_contrasts}")
    print(f"Primary instruction conditions: {', '.join(PRIMARY_INSTRUCTION_CONDITIONS)}")
    print(f"Response format: {RESPONSE_FORMAT} (primary category: {PRIMARY_CATEGORY})")
    print(f"Superblocks: {len(by_superblock)} (8 cells each)")
    print(f"Blocks: {n_blocks} (4 cells each)")
    print(f"Primary treatment trials: {len(treatment_trials)}")
    print(f"Output: {TRIALS_FILE}")
    print()
    print(f"Optional minimal-instruction trials: {len(minimal_trials)} (excluded from all primary analyses)")
    print(f"Output: {MINIMAL_TRIALS_FILE}")
    print()
    print(f"Blind baseline: {BASELINE_EXPERIMENT_ID}")
    print(f"Baseline pairs: {n_pairs}")
    print(f"Baseline trials: {len(baseline_trials)} ({n_pairs} pairs x 2 positions, matched_control instruction)")
    print(f"Output: {BASELINE_TRIALS_FILE}")


if __name__ == "__main__":
    main()
