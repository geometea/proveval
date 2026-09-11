"""Generate all trials (single-story and comparative) and save them.

Run this file directly to write data/trials.jsonl.
No model API calls happen here yet.
"""

import json
from itertools import permutations

import prompts
import comparisons

TRIALS_FILE = "data/trials.jsonl"


def build_single_trials():
    """One trial per story x condition x task, using prompts.py's builder."""
    items = prompts.load_items(prompts.ITEMS_FILE)
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
    items = comparisons.load_items(comparisons.ITEMS_FILE)
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


def main():
    trials = build_single_trials() + build_comparison_trials()

    with open(TRIALS_FILE, "w") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")

    trial_ids = [t["trial_id"] for t in trials]
    num_single = sum(1 for t in trials if t["type"] == "single")
    num_comparison = sum(1 for t in trials if t["type"] == "comparison")

    print(f"Total trials: {len(trials)}")
    print(f"Single-story trials: {num_single}")
    print(f"Comparative trials: {num_comparison}")
    print(f"All trial IDs unique: {len(trial_ids) == len(set(trial_ids))}")


if __name__ == "__main__":
    main()
