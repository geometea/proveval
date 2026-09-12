"""Generate the v0.2 context-benchmark trial manifest: data/context_trials.jsonl.

Three trial families, each carrying explicit structured metadata (no need to
parse trial_id to recover it -- see item 2/6 of the design):

- context_single:   one story, one story-scope context signal, 1-5 ratings
                     (reuses prompts.py's existing "rating" task, unmodified).
- context_pairwise: two different stories, one story-scope contrast, in both
                     forward and flipped attribution (reuses
                     context_contrasts.build_contrast_trials).
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
from context_packets import load_dimensions, dimensions_with_scope, single_variable_conditions
from context_contrasts import load_contrasts, build_contrast_trials

ITEMS_FILE = "data/items.jsonl"
PROMPT_CONTRASTS_FILE = "data/context_prompt_contrasts.jsonl"
CONTEXT_TRIALS_FILE = "data/context_trials.jsonl"

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

def build_context_single_trials(dimensions, items):
    tasks = prompts.load_items(prompts.TASKS_FILE)
    rating_task = next(t for t in tasks if t["id"] == "rating")

    trials = []
    for item in items:
        text = prompts.load_story(item["path"])
        for condition in single_variable_conditions(dimensions):
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
# context_pairwise: two stories, one story-scope contrast, forward + flipped
# ---------------------------------------------------------------------------

def build_context_pairwise_trials(dimensions, items):
    contrasts = load_contrasts()
    trials = []

    for story_1, story_2 in story_pairs(items):
        for contrast in contrasts:
            forward, flipped = build_contrast_trials(dimensions, contrast, story_1, story_2)

            for trial, assignment, context_a, context_b in (
                (forward, "forward", {"value": contrast["a"], "clause": contrast["a_clause"]}, {"value": contrast["b"], "clause": contrast["b_clause"]}),
                (flipped, "flipped", {"value": contrast["b"], "clause": contrast["b_clause"]}, {"value": contrast["a"], "clause": contrast["a_clause"]}),
            ):
                trials.append(
                    {
                        "trial_id": trial["trial_id"],
                        "type": "context_pairwise",
                        "story_a_id": trial["story_a_id"],
                        "story_b_id": trial["story_b_id"],
                        "contrast_id": contrast["id"],
                        "dimension": contrast["dimension"],
                        "assignment": assignment,
                        "context_a": context_a,
                        "context_b": context_b,
                        "prompt": trial["prompt"],
                    }
                )
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
    print(f"  context_single:   {len(single_trials)}")
    print(f"  context_pairwise: {len(pairwise_trials)}")
    print(f"  context_prompt:   {len(prompt_trials)}")
    print(f"All trial IDs unique: {len(trial_ids) == len(set(trial_ids))}")


if __name__ == "__main__":
    main()
