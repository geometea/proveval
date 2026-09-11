"""Build the natural, non-experiment-flavored comparison prompt for context-packet trials.

This is a distinct prompt format from comparisons.py's numeric-rating format: it
frames the request as a casual ask for help choosing between two stories, and asks
for a plain A / B / tie choice per category instead of 1-5 ratings. It is
intentionally kept separate from prompts.py, comparisons.py, make_trials.py,
run_trial.py, run_batch.py, and analyze.py -- none of those are touched by this
file, and data/trials.jsonl is never written to, so the existing pipeline and its
results/raw.jsonl schema are unaffected.

Run this file directly to print a small example set of single-variable-isolated
prompts, built from context_packets.py's condition generator. No model API calls
happen here, and nothing is written to disk.
"""

import json

from context_packets import load_dimensions, single_variable_conditions

ITEMS_FILE = "data/items.jsonl"

INSTRUCTION_TEMPLATE = """Hi! Can you help me choose between these two stories?

Story A:
{context_a}

{story_a}

Story B:
{context_b}

{story_b}

For each category, choose which story is better.

Categories:
- Plot structure
- Prose style
- Characterisation
- Originality
- Overall quality

Return only JSON:

{{
  "plot_structure": "A",
  "prose_style": "B",
  "characterisation": "A",
  "originality": "tie",
  "overall_quality": "B"
}}"""


def load_items(path=ITEMS_FILE):
    """Read a .jsonl file and return a list of dicts (one per line)."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_story(path):
    """Read the story text from its .txt file."""
    with open(path, "r") as f:
        return f.read().strip()


def build_prompt(story_a, context_a_text, story_b, context_b_text):
    """Fill in the natural comparison template with two stories and their context text.

    context_a_text / context_b_text may be "" when a story is given no context at
    all (the baseline condition).
    """
    return INSTRUCTION_TEMPLATE.format(
        context_a=context_a_text,
        story_a=story_a,
        context_b=context_b_text,
        story_b=story_b,
    )


def single_variable_examples(dimensions, story):
    """Build one example comparison per single-variable condition.

    Story A always gets no context (the empty baseline); Story B is the *same*
    underlying text, given exactly one context signal. Using the same text on
    both sides means the context sentence is the only thing that can possibly
    account for any difference in the model's choices.
    """
    text = load_story(story["path"])
    examples = []
    for condition in single_variable_conditions(dimensions):
        prompt = build_prompt(text, "", text, condition["context_text"])
        examples.append(
            {
                "trial_id": f"context__{story['id']}__{condition['condition_id']}",
                "story_id": story["id"],
                "dimension": condition["dimension"],
                "value": condition["value"],
                "context_a": "",
                "context_b": condition["context_text"],
                "note": condition["note"],
                "prompt": prompt,
            }
        )
    return examples


def main():
    dimensions = load_dimensions()
    items = load_items()
    story = items[0]  # keep the example small: one story is enough to show the mechanism
    examples = single_variable_examples(dimensions, story)

    print(
        f"{len(examples)} example single-variable context-comparison prompts "
        f"(story: {story['id']}, Story A = no context, Story B = one signal)\n"
    )

    preview_count = 2
    for ex in examples[:preview_count]:
        print(f"=== {ex['trial_id']} ===")
        if ex["note"]:
            print(f"(note: {ex['note']})")
        print(ex["prompt"])
        print()

    print(f"...and {len(examples) - preview_count} more conditions:")
    for ex in examples[preview_count:]:
        print(f"  {ex['trial_id']}  (context B: {ex['context_b']!r})")


if __name__ == "__main__":
    main()
