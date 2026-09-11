"""Drive prompts.py's existing single-story numeric rating with a context packet.

prompts.build_prompt(text, context, instruction) already accepts an arbitrary
context string, so a context_packets.render_packet() result plugs straight in --
no changes to prompts.py are needed. This file is a thin demonstration of that,
kept separate so context_packets.py and context_comparisons.py don't need to
import prompts.py themselves.

Run this file directly to print a small example set of single-story,
single-variable prompts using the existing 1-5 rating task. No model API calls
happen here, and nothing is written to disk.
"""

import prompts
from context_packets import load_dimensions, single_variable_conditions


def single_variable_rating_examples(dimensions, story):
    """One single-story rating prompt per single-variable condition, for one story.

    Each prompt is directly comparable to prompts.py's existing "neutral"
    single-story trials (context = ""), just with a context_packets-driven
    signal instead of one of the hand-written conditions in
    data/conditions.jsonl.
    """
    tasks = prompts.load_items(prompts.TASKS_FILE)
    rating_task = next(t for t in tasks if t["id"] == "rating")
    text = prompts.load_story(story["path"])

    examples = []
    for condition in single_variable_conditions(dimensions):
        prompt = prompts.build_prompt(text, condition["context_text"], rating_task["instruction"])
        examples.append(
            {
                "trial_id": f"context_single__{story['id']}__{condition['condition_id']}",
                "dimension": condition["dimension"],
                "value": condition["value"],
                "note": condition["note"],
                "prompt": prompt,
            }
        )
    return examples


def main():
    dimensions = load_dimensions()
    items = prompts.load_items(prompts.ITEMS_FILE)
    story = items[0]
    examples = single_variable_rating_examples(dimensions, story)

    print(
        f"{len(examples)} example single-story rating prompts "
        f"(story: {story['id']}, using prompts.py's existing 'rating' task unchanged)\n"
    )

    preview_count = 1
    for ex in examples[:preview_count]:
        print(f"=== {ex['trial_id']} ===")
        if ex["note"]:
            print(f"(note: {ex['note']})")
        print(ex["prompt"])
        print()

    print(f"...and {len(examples) - preview_count} more conditions:")
    for ex in examples[preview_count:]:
        print(f"  {ex['trial_id']}")


if __name__ == "__main__":
    main()
