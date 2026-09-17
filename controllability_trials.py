"""Standalone context-controllability experiment: generate the focused
treatment manifest of ordinary context_pairwise trials, plus a separate
no-context blind baseline manifest.

For each of 5 isolated context contrasts (provenance human-vs-LLM, source
venue journal-vs-random, reception positive-vs-negative, user opinion
liked-vs-disliked, editing status edited-vs-first-draft -- see
data/controllability_contrasts.jsonl, whose a_clause/b_clause wordings are
frozen and read verbatim, never generated or paraphrased here), every text
pair is run under BOTH evaluation regimes:
  - "naturalistic": the ordinary framing, unchanged.
  - "text_only_invariance" (the suppression condition): the exact same
    contextual claim is shown, plus exactly one added sentence asking the
    evaluator to judge the writing itself. The context is never removed in
    this regime, and nothing else differs -- see SUPPRESSION_INSTRUCTION.
    suppression_magnitude = abs(naturalistic_effect) - abs(text_only_effect)
    (see analyze_controllability.py) measures how much of the naturalistic
    effect's MAGNITUDE the instruction removes.

Single outcome, plain A/B: this experiment asks one question --
"Which passage do you prefer overall?" -- normalized internally to
{"overall_quality": "A"|"B"} (response_format="plain_ab"; see
run_trial.parse_plain_ab_response) so the existing pairwise analysis
(context_analysis_pairwise.py, with categories=["overall_quality"]) is
reused unmodified rather than a second five-category rubric. No JSON is
requested from the model for these trials.

Every emitted treatment trial is an ordinary "type": "context_pairwise",
"choice_mode": "forced" trial built with the SAME 4-cell counterbalance
(context assignment x display position, one shared block_id) as the
general v0.2 benchmark's build_contrast_block() -- reimplemented here only
because the prompt TEXT itself is different (plain "Passage A/B" wording,
no categories block, no JSON schema), never a second counterbalancing
algorithm: the assignment/position crossing and block_id/story identity
bookkeeping below are structurally identical to
context_contrasts.build_contrast_block. Each trial also carries
"experiment_id": EXPERIMENT_ID and a deterministic "prompt_sha256" (see
run_batch.is_completed) so a saved result can never be mistaken for
satisfying a trial whose prompt wording has since changed.

The blind baseline (data/controllability_baseline_trials.jsonl) shows the
same 66 text pairs with NO contextual framing at all, once in each display
order (66 x 2 = 132 trials, naturalistic regime only -- there is no context
to instruct the evaluator to ignore). It measures whether the evaluator
was close to indifferent between the two texts before any context was
introduced (baseline_margin, see analyze_controllability.py).

Run this file directly to write both files and print a computed
dataset-size summary. No model API calls happen here.
"""

import hashlib
import json

from context_packets import load_dimensions
from context_contrasts import load_contrasts
from context_comparisons import load_story
from context_trials import EVALUATION_REGIMES, load_items, story_pairs

CONTRASTS_FILE = "data/controllability_contrasts.jsonl"
TRIALS_FILE = "data/controllability_trials.jsonl"
BASELINE_TRIALS_FILE = "data/controllability_baseline_trials.jsonl"
EXPERIMENT_ID = "context_controllability_v1"
BASELINE_EXPERIMENT_ID = "context_controllability_baseline_v1"

# This experiment's dedicated response format (see run_trial.parse_and_validate
# / run_trial.parse_plain_ab_response): a single plain "A" or "B" answer,
# normalized to {"overall_quality": "A"|"B"} -- not the shared 5-field JSON
# rubric every other context_pairwise trial uses.
RESPONSE_FORMAT = "plain_ab"
PRIMARY_CATEGORY = "overall_quality"

# The five isolated context contrasts this experiment tests -- checked
# against data/controllability_contrasts.jsonl by id, not assumed.
REQUIRED_CONTRAST_IDS = {
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
}

# Human-readable display name for each contrast -- used by
# analyze_controllability.py's primary suppression table. Spelled out once
# here (never derived by parsing contrast_id) since the exact phrasing
# isn't mechanically recoverable from the id.
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
# Prompt wording -- frozen, plain-language, "passage" (never "story") and
# plain A/B (never JSON). Only the intro sentence's contextual clause
# differs between cells within one contrast; the suppression instruction is
# the ONLY difference between the naturalistic and text_only_invariance
# versions of the same cell (see build_treatment_cell).
# ---------------------------------------------------------------------------

BASELINE_INTRO = "I'd like your view on two passages."

QUESTION = "Which passage do you prefer overall? If they're close, choose the one you slightly prefer. Please answer A or B."

# Deliberately generic: it says only "the writing itself" / "as written",
# never enumerating authorship/source/reception/editing/opinion, and never
# using the words "irrelevant", "context", "bias", "invariance", or
# "suppression".
SUPPRESSION_INSTRUCTION = "For this comparison, I'm only interested in the writing itself. Please base your judgment on the passages as written."

_NATURALISTIC_BODY = "{intro}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{question}"
_SUPPRESSION_BODY = "{intro}\n\n{instruction}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{question}"


def intro_sentence(a_clause, b_clause):
    """Build the deterministic opening sentence naming Passage A's and
    Passage B's attribution -- the passage-wording analogue of
    context_contrasts.intro_sentence, never generated/paraphrased."""
    return f"I'd like your view on two passages. Passage A {a_clause}, while Passage B {b_clause}."


def build_plain_ab_prompt(intro, text_a, text_b, evaluation_regime):
    if evaluation_regime == "naturalistic":
        return _NATURALISTIC_BODY.format(intro=intro, text_a=text_a, text_b=text_b, question=QUESTION)
    if evaluation_regime == "text_only_invariance":
        return _SUPPRESSION_BODY.format(
            intro=intro, instruction=SUPPRESSION_INSTRUCTION, text_a=text_a, text_b=text_b, question=QUESTION
        )
    raise ValueError(f"Unknown evaluation_regime: {evaluation_regime!r}")


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validation: fail loudly on a malformed contrast/trial set
# ---------------------------------------------------------------------------

def assert_contrasts_are_well_formed(dimensions, contrasts):
    """Contrast ids must be exactly REQUIRED_CONTRAST_IDS (no more, no
    fewer, no substitutions), each dimension used must be registered, each
    contrast must isolate exactly one dimension (no combined-context
    trials), and every "a"/"b" value must be a real registered value of
    that dimension."""
    seen_ids = set()
    seen_dimensions = set()
    for contrast in contrasts:
        if contrast["id"] in seen_ids:
            raise ValueError(f"Duplicate contrast id: {contrast['id']!r}")
        seen_ids.add(contrast["id"])

        dim_name = contrast["dimension"]
        if dim_name not in dimensions:
            raise ValueError(f"{contrast['id']!r} references unregistered dimension {dim_name!r}")
        if dim_name in seen_dimensions:
            raise ValueError(
                f"{contrast['id']!r} reuses dimension {dim_name!r} already used by another contrast in this "
                f"focused manifest -- each of the 5 contrasts must isolate a distinct context type"
            )
        seen_dimensions.add(dim_name)

        dim_values = dimensions[dim_name]["values"]
        for value_id in (contrast["a"], contrast["b"]):
            if value_id not in dim_values:
                raise ValueError(f"{contrast['id']!r} references unregistered value {value_id!r} of {dim_name!r}")

    if seen_ids != REQUIRED_CONTRAST_IDS:
        raise ValueError(
            f"Expected exactly the 5 contrasts {sorted(REQUIRED_CONTRAST_IDS)}, found {sorted(seen_ids)}"
        )


def assert_trials_are_well_formed(trials):
    """Fail loudly if generation ever drifts: every trial must be an
    ordinary forced-choice context_pairwise cell using response_format
    "plain_ab", trial ids must be globally unique, every trial must carry a
    prompt_sha256 matching its own prompt, and every block must contain
    exactly the four required counterbalanced cells. Returns
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
        if trial["prompt_sha256"] != sha256_hex(trial["prompt"]):
            raise ValueError(f"{trial['trial_id']}: prompt_sha256 does not match its own prompt")
        by_block.setdefault(trial["block_id"], []).append(trial)

    for block_id, cells in by_block.items():
        if len(cells) != 4:
            raise ValueError(f"Block {block_id!r} has {len(cells)} cell(s), expected exactly 4")
        actual_cells = {(c["assignment"], c["position"]) for c in cells}
        if actual_cells != REQUIRED_CELLS:
            raise ValueError(
                f"Block {block_id!r} has cells {sorted(actual_cells)}, expected {sorted(REQUIRED_CELLS)}"
            )

    return by_block


def assert_baseline_trials_are_well_formed(trials):
    """Fail loudly: unique trial ids, ordinary forced-choice context_pairwise
    cells with no context assignment, a matching prompt_sha256, and every
    block has exactly the 2 required display positions. Returns
    {block_id: [cells]}."""
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
        if "assignment" in trial:
            raise ValueError(f"{trial['trial_id']}: baseline trial must not carry a context 'assignment'")
        if trial["evaluation_regime"] != "naturalistic":
            raise ValueError(f"{trial['trial_id']}: baseline is naturalistic-only, got {trial['evaluation_regime']!r}")
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

def build_treatment_block(contrast, story_1, story_2, evaluation_regime):
    """Build the 4-cell counterbalanced block for one contrast/story-pair/
    regime: crosses "assignment" (which contrast value each text is
    attributed -- "forward": text_1 gets value "a", text_2 gets value "b";
    "flipped": the reverse) with "position" (which text is DISPLAYED as
    Passage A). Structurally identical to
    context_contrasts.build_contrast_block's counterbalance -- only the
    prompt TEXT (plain A/B, "Passage" wording, no categories/JSON) and
    response_format differ.
    """
    if story_1["id"] == story_2["id"]:
        raise ValueError("build_treatment_block requires two different stories, not the same story twice")

    text_1 = load_story(story_1["path"])
    text_2 = load_story(story_2["path"])
    pair_id = f"{story_1['id']}_vs_{story_2['id']}"
    block_id = f"block__{contrast['id']}__{pair_id}__{evaluation_regime}__forced__{RESPONSE_FORMAT}"
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
            prompt = build_plain_ab_prompt(intro, text_a, text_b, evaluation_regime)
            cells.append(
                {
                    "trial_id": f"{block_id}__{assignment}__{position}",
                    "block_id": block_id,
                    "type": "context_pairwise",
                    "choice_mode": "forced",
                    "response_format": RESPONSE_FORMAT,
                    "contrast_id": contrast["id"],
                    "dimension": contrast["dimension"],
                    "story_1_id": story_1["id"],
                    "story_2_id": story_2["id"],
                    "story_a_id": story_a["id"],
                    "story_b_id": story_b["id"],
                    "assignment": assignment,
                    "position": position,
                    "evaluation_regime": evaluation_regime,
                    "context_a": {"value": value_a, "clause": clause_a},
                    "context_b": {"value": value_b, "clause": clause_b},
                    "intro": intro,
                    "prompt": prompt,
                    "prompt_sha256": sha256_hex(prompt),
                }
            )
    return cells


def build_baseline_trials(items):
    """66 pairs x 2 display positions = 132 trials, no contextual framing at
    all (see BASELINE_INTRO), naturalistic regime only -- there is nothing
    for a "text_only_invariance" instruction to suppress. All 132 trials
    share contrast_id="no_context_baseline" and carry no "assignment" field
    (there is no context to assign)."""
    trials = []
    for story_1, story_2 in story_pairs(items):
        text_1 = load_story(story_1["path"])
        text_2 = load_story(story_2["path"])
        pair_id = f"{story_1['id']}_vs_{story_2['id']}"
        block_id = f"baseline_block__{pair_id}"
        stories = {"story1_as_a": (story_1, story_2, text_1, text_2), "story2_as_a": (story_2, story_1, text_2, text_1)}
        for position, (story_a, story_b, text_a, text_b) in stories.items():
            prompt = f"{BASELINE_INTRO}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{QUESTION}"
            trials.append(
                {
                    "trial_id": f"{block_id}__{position}",
                    "block_id": block_id,
                    "type": "context_pairwise",
                    "choice_mode": "forced",
                    "response_format": RESPONSE_FORMAT,
                    "evaluation_regime": "naturalistic",
                    "contrast_id": "no_context_baseline",
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


def main():
    dimensions = load_dimensions()
    items = load_items()
    contrasts = load_contrasts(CONTRASTS_FILE)
    assert_contrasts_are_well_formed(dimensions, contrasts)

    all_trials = []
    for regime in EVALUATION_REGIMES:
        for story_1, story_2 in story_pairs(items):
            for contrast in contrasts:
                for cell in build_treatment_block(contrast, story_1, story_2, regime):
                    cell["experiment_id"] = EXPERIMENT_ID
                    all_trials.append(cell)

    by_block = assert_trials_are_well_formed(all_trials)

    with open(TRIALS_FILE, "w", encoding="utf-8") as f:
        for trial in all_trials:
            f.write(json.dumps(trial) + "\n")

    baseline_trials = build_baseline_trials(items)
    by_baseline_block = assert_baseline_trials_are_well_formed(baseline_trials)

    with open(BASELINE_TRIALS_FILE, "w", encoding="utf-8") as f:
        for trial in baseline_trials:
            f.write(json.dumps(trial) + "\n")

    n_stories = len(items)
    n_pairs = n_stories * (n_stories - 1) // 2
    n_contrasts = len(contrasts)
    n_blocks = n_pairs * n_contrasts * len(EVALUATION_REGIMES)
    assert len(by_block) == n_blocks, f"computed {n_blocks} expected blocks but found {len(by_block)}"
    assert len(all_trials) == n_blocks * 4
    assert len(by_baseline_block) == n_pairs
    assert len(baseline_trials) == n_pairs * 2

    print(f"Context controllability experiment: {EXPERIMENT_ID}")
    print(f"Stories: {n_stories}")
    print(f"Story pairs: {n_pairs}")
    print(f"Context contrasts: {n_contrasts}")
    print(f"Evaluation regimes: {', '.join(EVALUATION_REGIMES)}")
    print(f"Response format: {RESPONSE_FORMAT} (primary category: {PRIMARY_CATEGORY})")
    print(f"Blocks: {n_blocks}")
    print(f"Trials: {len(all_trials)}")
    print("Cells per block: 4")
    print(f"Output: {TRIALS_FILE}")
    print()
    print(f"Blind baseline: {BASELINE_EXPERIMENT_ID}")
    print(f"Baseline pairs: {n_pairs}")
    print(f"Baseline trials: {len(baseline_trials)} ({n_pairs} pairs x 2 positions, naturalistic only)")
    print(f"Output: {BASELINE_TRIALS_FILE}")


if __name__ == "__main__":
    main()
