"""Generate the v0.2 context-benchmark trial manifest: data/context_trials.jsonl.

Three trial families, each carrying explicit structured metadata (no need to
parse trial_id to recover it -- see item 2/6 of the design):

- context_single:   one story, one story-scope context signal, v0.2's own
                     1.0-10.0 decimal rating task (data/context_tasks.jsonl,
                     separate from the v0.1 pilot's integer 1-5 task).
- context_pairwise: two different stories, one story-scope contrast, in the
                     full 4-cell counterbalance of context assignment x
                     display position (reuses
                     context_contrasts.build_contrast_block).
- context_prompt:   two different stories with neutral (no) story-level
                     context, one prompt-scope contrast -- an extraneous
                     sentence appended to the shared request, not attributed
                     to either story. This is deliberately NOT forced through
                     the story-A-vs-story-B contrast abstraction (there is
                     nothing to attribute to a specific story), so it has its
                     own small builder below instead.

This is a separate manifest from data/trials.jsonl on purpose: the v0.1 pilot
pipeline (prompts.py, comparisons.py, make_trials.py) is untouched, and
running this file never modifies data/trials.jsonl or any pilot result.
data/items.jsonl now registers all 12 corpus stories; make_trials.py has been
pinned to the original 4 so the v0.1 pilot set stays reproducible.

Run this file directly to write data/context_trials.jsonl and print a
dataset-size summary. No model API calls happen here.
"""

import json
from itertools import combinations

import prompts
import context_comparisons
from context_packets import load_dimensions, dimensions_with_scope, single_variable_conditions, neutral_condition
from context_contrasts import load_contrasts, build_contrast_block

ITEMS_FILE = "data/items.jsonl"
PROMPT_CONTRASTS_FILE = "data/context_prompt_contrasts.jsonl"
CONTEXT_TRIALS_FILE = "data/context_trials.jsonl"

# A separate, OPTIONAL file: same-context pairwise trials (see
# build_context_pairwise_same_trials). Deliberately not merged into
# CONTEXT_TRIALS_FILE, so nothing that runs against the required manifest
# picks these up by accident -- see item 6 of the context-benchmark spec and
# "Optional same-context pairwise family" in FINAL_DESIGN.md.
OPTIONAL_SAME_CONTEXT_TRIALS_FILE = "data/context_trials_optional_same_context.jsonl"

# Used for context_prompt trials: story-level context is neutral (nothing
# attributed to either story), so only the appended sentence varies.
NEUTRAL_PAIRWISE_INTRO = "Hi! Can you give me some feedback on these two stories? I'm trying to make up my mind."


def load_items(path=ITEMS_FILE):
    """Read a .jsonl file and return a list of dicts (one per line)."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_prompt_contrasts(path=PROMPT_CONTRASTS_FILE):
    """Read data/context_prompt_contrasts.jsonl into a list of contrast dicts."""
    contrasts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                contrasts.append(json.loads(line))
    return contrasts


def story_pairs(items):
    """Deterministic unordered pairs, alphabetical by id.

    This fixes which story is story_1 (always Story A) and which is story_2
    (always Story B) reproducibly across runs -- see
    context_contrasts.build_contrast_trials for why position is held fixed.
    """
    return list(combinations(sorted(items, key=lambda item: item["id"]), 2))


# ---------------------------------------------------------------------------
# context_single: one story, one story-scope signal, 1-5 ratings
# ---------------------------------------------------------------------------

CONTEXT_TASKS_FILE = "data/context_tasks.jsonl"


def build_context_single_trials(dimensions, items):
    # v0.2 uses its own 1.0-10.0 decimal rating task (data/context_tasks.jsonl),
    # not the v0.1 pilot's integer 1-5 "rating" task in data/tasks.jsonl --
    # see run_trial.validate_context_single_response.
    tasks = prompts.load_items(CONTEXT_TASKS_FILE)
    rating_task = next(t for t in tasks if t["id"] == "context_rating")

    # neutral_condition() first: one no-context baseline per story, so v0.2
    # has its own clean single-text baseline (dimension="neutral",
    # value="neutral") independent of the v0.1 pilot's "neutral" condition.
    conditions = [neutral_condition()] + single_variable_conditions(dimensions)

    trials = []
    for item in items:
        text = prompts.load_story(item["path"])
        for condition in conditions:
            prompt = prompts.build_prompt(text, condition["context_text"], rating_task["instruction"])
            trials.append(
                {
                    "trial_id": f"context_single__{item['id']}__{condition['condition_id']}",
                    "type": "context_single",
                    "story_id": item["id"],
                    "dimension": condition["dimension"],
                    "value": condition["value"],
                    "condition_id": condition["condition_id"],
                    "context_text": condition["context_text"],
                    "prompt": prompt,
                }
            )
    return trials


# ---------------------------------------------------------------------------
# context_pairwise: two stories, one story-scope contrast, full 4-cell
# counterbalance of context assignment x display position (see
# context_contrasts.build_contrast_block)
# ---------------------------------------------------------------------------

def build_context_pairwise_trials(dimensions, items):
    contrasts = load_contrasts()
    trials = []

    for story_1, story_2 in story_pairs(items):
        for contrast in contrasts:
            for cell in build_contrast_block(dimensions, contrast, story_1, story_2):
                trials.append({**cell, "type": "context_pairwise"})
    return trials


# ---------------------------------------------------------------------------
# context_prompt: two stories, neutral story-level context, one extraneous
# prompt-level sentence present or absent
# ---------------------------------------------------------------------------

def build_context_prompt_trials(dimensions, items):
    prompt_dims = dimensions_with_scope(dimensions, "prompt")
    contrasts = load_prompt_contrasts()
    trials = []

    for story_1, story_2 in story_pairs(items):
        text_1 = context_comparisons.load_story(story_1["path"])
        text_2 = context_comparisons.load_story(story_2["path"])

        for contrast in contrasts:
            dim = prompt_dims[contrast["dimension"]]
            for value_id in (contrast["a"], contrast["b"]):
                extra_sentence = dim["values"][value_id]
                intro = f"{NEUTRAL_PAIRWISE_INTRO} {extra_sentence}".strip() if extra_sentence else NEUTRAL_PAIRWISE_INTRO
                prompt = context_comparisons.build_prompt(intro, text_1, text_2)
                trials.append(
                    {
                        "trial_id": f"context_prompt__{contrast['id']}__{story_1['id']}_vs_{story_2['id']}__{value_id}",
                        "type": "context_prompt",
                        "story_a_id": story_1["id"],
                        "story_b_id": story_2["id"],
                        "contrast_id": contrast["id"],
                        "dimension": contrast["dimension"],
                        "value": value_id,
                        "prompt_context_text": extra_sentence,
                        "prompt": prompt,
                    }
                )
    return trials


# ---------------------------------------------------------------------------
# OPTIONAL, not part of the required manifest: same-context pairwise trials
# ---------------------------------------------------------------------------

def build_context_pairwise_same_trials(dimensions, items):
    """OPTIONAL trial family: Story A and Story B are attributed the SAME
    context value (e.g. "both of these were generated by Claude"), instead of
    the forward/flipped contrast used by build_context_pairwise_trials.

    Why this exists: the forward/flipped contrast trials are built to measure
    *causal context sensitivity* (does swapping which story gets which value
    change the A/B/tie choice?), not to give a ranking "under" one context
    condition -- each pairwise contrast observation only ever involves two
    *different* claimed values, never one shared value applied to both sides.
    A same-context family, if ever run, would instead let a full round-robin
    of the corpus be judged under one single, globally consistent context
    condition (e.g. "everything here was written by Claude"), which is what
    would be needed for a true per-condition *pairwise* ranking analogous to
    the per-(model, dimension, value) *single-text* rankings in
    analyze_context.py. See "Optional same-context pairwise family" in
    FINAL_DESIGN.md.

    Deliberately minimal to avoid bloating the design:
    - Reuses each dimension's existing rendered phrase (no new hand-written
      "same" clauses, no new data file) followed by a fixed "That's true of
      both of them." sentence.
    - Skips dependent dimensions (currently just writer_status): resolving
      their pronoun would need a provenance carrier threaded through, which
      isn't worth the complexity for a family that is never run by default.

    Writes to OPTIONAL_SAME_CONTEXT_TRIALS_FILE, a separate file from
    CONTEXT_TRIALS_FILE, so it is never picked up by a run against the
    required manifest. Not part of any required API run; do not run it
    without a deliberate, separate decision to do so.
    """
    story_dims = dimensions_with_scope(dimensions, "story")
    trials = []

    for story_1, story_2 in story_pairs(items):
        text_1 = context_comparisons.load_story(story_1["path"])
        text_2 = context_comparisons.load_story(story_2["path"])

        for dimension_id, dim in story_dims.items():
            if dim.get("depends_on"):
                continue
            for value_id, phrase in dim["values"].items():
                intro = (
                    f"Hi! Can you give me some feedback on these two stories? "
                    f"{phrase} That's true of both of them. I'm trying to make up my mind."
                )
                prompt = context_comparisons.build_prompt(intro, text_1, text_2)
                trials.append(
                    {
                        "trial_id": f"context_pairwise_same__{dimension_id}__{value_id}__{story_1['id']}_vs_{story_2['id']}",
                        "type": "context_pairwise_same",
                        "story_a_id": story_1["id"],
                        "story_b_id": story_2["id"],
                        "dimension": dimension_id,
                        "value": value_id,
                        "prompt": prompt,
                    }
                )
    return trials


def main():
    dimensions = load_dimensions()
    items = load_items()

    single_trials = build_context_single_trials(dimensions, items)
    pairwise_trials = build_context_pairwise_trials(dimensions, items)
    prompt_trials = build_context_prompt_trials(dimensions, items)
    all_trials = single_trials + pairwise_trials + prompt_trials

    with open(CONTEXT_TRIALS_FILE, "w") as f:
        for trial in all_trials:
            f.write(json.dumps(trial) + "\n")

    trial_ids = [t["trial_id"] for t in all_trials]
    print(f"Total context trials: {len(all_trials)}")
    print(f"  context_single:   {len(single_trials)} (includes 1 neutral baseline per story)")
    print(f"  context_pairwise: {len(pairwise_trials)}")
    print(f"  context_prompt:   {len(prompt_trials)}")
    print(f"All trial IDs unique: {len(trial_ids) == len(set(trial_ids))}")

    # Optional, separate file -- not part of the required manifest above and
    # not folded into all_trials/CONTEXT_TRIALS_FILE. Generation is free, so
    # it's written for inspection, but running it is a distinct future
    # decision (see build_context_pairwise_same_trials docstring).
    same_context_trials = build_context_pairwise_same_trials(dimensions, items)
    with open(OPTIONAL_SAME_CONTEXT_TRIALS_FILE, "w") as f:
        for trial in same_context_trials:
            f.write(json.dumps(trial) + "\n")
    print(
        f"\nOptional (NOT part of the required manifest, NOT run): "
        f"{len(same_context_trials)} context_pairwise_same trials -> {OPTIONAL_SAME_CONTEXT_TRIALS_FILE}"
    )


if __name__ == "__main__":
    main()
