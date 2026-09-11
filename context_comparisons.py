"""Build the natural, non-experiment-flavored comparison prompt for context-packet trials.

This is a distinct prompt format from comparisons.py's numeric-rating format: it
frames the request as a casual ask for help choosing between two stories, and asks
for a plain A / B / tie choice per category instead of 1-5 ratings. It is
intentionally kept separate from prompts.py, comparisons.py, make_trials.py,
run_trial.py, run_batch.py, and analyze.py -- none of those are touched by this
file, and data/trials.jsonl is never written to, so the existing pipeline and its
results/raw.jsonl schema are unaffected.

Story A and Story B are always two DIFFERENT underlying texts (models can tell
when they're being shown the same text twice, which would give the game away and
contaminate the comparison). The design here is a context *flip*: pick two
stories, pick one context signal, and run it twice -- once with story_1 carrying
the signal and story_2 getting no context, once with the assignment swapped. If
the model's preference is really about the underlying stories, whichever story
is "better" should win in both trials, just switching A/B position. If the
preference instead flips to whichever story is carrying the context signal, that
signal is doing the work.

Run this file directly to print a small example set of single-variable flip
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


def single_variable_flip_examples(dimensions, story_1, story_2):
    """Build a forward/swapped pair of comparisons per single-variable condition.

    story_1 and story_2 are two different underlying texts. For each condition,
    "forward" gives story_1 the context signal and story_2 nothing; "swapped"
    gives story_2 the signal and story_1 nothing. Whichever story is carrying
    the signal moves between A and B across the pair, so a genuine story-quality
    difference should keep picking the same *story* in both trials, while a
    genuine context effect should keep picking whichever *position/story* holds
    the signal.
    """
    text_1 = load_story(story_1["path"])
    text_2 = load_story(story_2["path"])
    pair_id = f"{story_1['id']}_vs_{story_2['id']}"

    examples = []
    for condition in single_variable_conditions(dimensions):
        treatment_text = condition["context_text"]
        base_fields = {
            "dimension": condition["dimension"],
            "value": condition["value"],
            "note": condition["note"],
        }

        examples.append(
            {
                **base_fields,
                "trial_id": f"context__{pair_id}__{condition['condition_id']}__forward",
                "story_with_treatment": story_1["id"],
                "story_baseline": story_2["id"],
                "treatment_position": "A",
                "prompt": build_prompt(text_1, treatment_text, text_2, ""),
            }
        )
        examples.append(
            {
                **base_fields,
                "trial_id": f"context__{pair_id}__{condition['condition_id']}__swapped",
                "story_with_treatment": story_2["id"],
                "story_baseline": story_1["id"],
                "treatment_position": "B",
                "prompt": build_prompt(text_1, "", text_2, treatment_text),
            }
        )
    return examples


def main():
    dimensions = load_dimensions()
    items = load_items()
    story_1, story_2 = items[0], items[1]  # a fixed pair keeps the example small
    examples = single_variable_flip_examples(dimensions, story_1, story_2)

    print(
        f"{len(examples)} example context-flip prompts "
        f"({story_1['id']} vs {story_2['id']}, forward + swapped per condition)\n"
    )

    preview_count = 2
    for ex in examples[:preview_count]:
        print(f"=== {ex['trial_id']} ===")
        if ex["note"]:
            print(f"(note: {ex['note']})")
        print(ex["prompt"])
        print()

    print(f"...and {len(examples) - preview_count} more:")
    for ex in examples[preview_count:]:
        print(
            f"  {ex['trial_id']}  "
            f"(treatment on {ex['story_with_treatment']}, position {ex['treatment_position']})"
        )


if __name__ == "__main__":
    main()
