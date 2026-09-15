"""Generate all v0.1 pilot trials (single-story and comparative) and save them.

Run this file directly to write data/trials.jsonl.
No model API calls happen here yet.

data/items.jsonl now registers all 12 corpus stories (for the newer
context-benchmark pipeline in context_trials.py), but the v0.1 pilot was run
against only the original 4. To keep this script reproducing the exact same
108-trial pilot set regardless of how many stories get registered later,
it filters down to PILOT_STORY_IDS explicitly rather than using every
registered story.
"""

import json
from itertools import permutations

import prompts
import comparisons

TRIALS_FILE = "data/trials.jsonl"

# The 4 stories the v0.1 pilot actually ran against (see PILOT_RESULTS.md).
# Fixed on purpose: data/items.jsonl now holds all 12 corpus stories for the
# newer context-benchmark pipeline, but this script must keep producing the
# same pilot trial set no matter how many stories get registered there.
PILOT_STORY_IDS = ["gilbert", "dunnest_smoke", "prophet", "santa"]


def pilot_items(items):
    """Filter a loaded items list down to the fixed v0.1 pilot subset.

    Raises if any PILOT_STORY_IDS entry has no match in `items` -- silently
    returning fewer than 4 stories would produce a smaller-than-108 trial
    set with no error, directly contradicting this module's stated purpose
    of reproducing the exact same pilot set regardless of how many stories
    data/items.jsonl registers later.
    """
    found = [item for item in items if item["id"] in PILOT_STORY_IDS]
    missing = set(PILOT_STORY_IDS) - {item["id"] for item in found}
    if missing:
        raise ValueError(
            f"pilot_items: {sorted(missing)} not found in the loaded items list -- "
            f"the v0.1 pilot set can no longer be reproduced. Check data/items.jsonl."
        )
    return found


def build_single_trials():
    """One trial per story x condition x task, using prompts.py's builder."""
    items = pilot_items(prompts.load_items(prompts.ITEMS_FILE))
    conditions = prompts.load_items(prompts.CONDITIONS_FILE)
    tasks = prompts.load_items(prompts.TASKS_FILE)

    trials = []
    for item in items:
        text = prompts.load_story(item["path"])
        for condition in conditions:
            for task in tasks:
                prompt = prompts.build_prompt(text, condition["context"], task["instruction"])
                trial_id = f"single__{item['id']}__{condition['id']}__{task['id']}"
                trials.append(
                    {
                        "trial_id": trial_id,
                        "type": "single",
                        "story_id": item["id"],
                        "condition_id": condition["id"],
                        "task_id": task["id"],
                        "prompt": prompt,
                    }
                )
    return trials


def build_comparison_trials():
    """One trial per ordered story pair x comparison condition, using comparisons.py's builder."""
    items = pilot_items(comparisons.load_items(comparisons.ITEMS_FILE))
    conditions = comparisons.load_items(comparisons.CONDITIONS_FILE)

    trials = []
    for item_a, item_b in permutations(items, 2):
        text_a = comparisons.load_story(item_a["path"])
        text_b = comparisons.load_story(item_b["path"])
        for condition in conditions:
            prompt = comparisons.build_comparison_prompt(
                text_a, text_b, condition["a_context"], condition["b_context"]
            )
            trial_id = f"comparison__{item_a['id']}__{item_b['id']}__{condition['id']}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "type": "comparison",
                    "story_a_id": item_a["id"],
                    "story_b_id": item_b["id"],
                    "condition_id": condition["id"],
                    "prompt": prompt,
                }
            )
    return trials


def build_comparison_control_trials():
    """One control trial per story x comparison condition, story compared against itself."""
    items = pilot_items(comparisons.load_items(comparisons.ITEMS_FILE))
    conditions = comparisons.load_items(comparisons.CONDITIONS_FILE)

    trials = []
    for item_a, item_b in comparisons.get_control_pairs(items):
        text = comparisons.load_story(item_a["path"])
        for condition in conditions:
            prompt = comparisons.build_comparison_prompt(
                text, text, condition["a_context"], condition["b_context"]
            )
            trial_id = f"comparison_control__{item_a['id']}__{condition['id']}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "type": "comparison_control",
                    "story_a_id": item_a["id"],
                    "story_b_id": item_b["id"],
                    "condition_id": condition["id"],
                    "prompt": prompt,
                }
            )
    return trials


def main():
    trials = build_single_trials() + build_comparison_trials() + build_comparison_control_trials()

    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        duplicates = sorted({tid for tid in trial_ids if trial_ids.count(tid) > 1})
        raise ValueError(
            f"Refusing to write {TRIALS_FILE}: duplicate trial_id(s) found: {duplicates}"
        )

    with open(TRIALS_FILE, "w") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")

    num_single = sum(1 for t in trials if t["type"] == "single")
    num_comparison = sum(1 for t in trials if t["type"] == "comparison")
    num_comparison_control = sum(1 for t in trials if t["type"] == "comparison_control")

    print(f"Total trials: {len(trials)}")
    print(f"Single-story trials: {num_single}")
    print(f"Comparative trials: {num_comparison}")
    print(f"Identical-text control trials: {num_comparison_control}")
    print(f"All trial IDs unique: {len(trial_ids) == len(set(trial_ids))}")


if __name__ == "__main__":
    main()
