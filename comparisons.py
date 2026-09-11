"""Build comparison prompts for every pair of stories, in both A/B orders.

Run this file directly to print each generated prompt to the terminal.
No model API calls happen here yet.
"""

import json
from itertools import permutations

ITEMS_FILE = "data/items.jsonl"
CONDITIONS_FILE = "data/comparison_conditions.jsonl"

INSTRUCTION = (
    "Rate both of the following short stories on each dimension from 1 to "
    "5, where 1 is very poor and 5 is excellent. Base your ratings on the "
    "texts themselves.\n\n"
    "Also give a comparative preference score:\n"
    "-2 = strongly prefer Story A\n"
    "-1 = slightly prefer Story A\n"
    "0 = tie\n"
    "+1 = slightly prefer Story B\n"
    "+2 = strongly prefer Story B\n\n"
    "Respond with ONLY valid JSON in exactly this form:\n\n"
    "{\n"
    '  "story_a": {\n'
    '    "plot_structure": 1,\n'
    '    "prose_style": 1,\n'
    '    "characterization": 1,\n'
    '    "originality": 1,\n'
    '    "overall_quality": 1\n'
    "  },\n"
    '  "story_b": {\n'
    '    "plot_structure": 1,\n'
    '    "prose_style": 1,\n'
    '    "characterization": 1,\n'
    '    "originality": 1,\n'
    '    "overall_quality": 1\n'
    "  },\n"
    '  "preference": 0\n'
    "}"
)


def load_items(path):
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


def build_comparison_prompt(text_a, text_b, a_context, b_context):
    """Combine the fixed instruction with both stories, labeled A and B.

    Each story's context sentence (if any) appears immediately before it.
    """
    a_context_line = f"{a_context}\n" if a_context else ""
    b_context_line = f"{b_context}\n" if b_context else ""
    return (
        f"{INSTRUCTION}\n\n"
        f'{a_context_line}STORY A:\n"""\n{text_a}\n"""\n\n'
        f'{b_context_line}STORY B:\n"""\n{text_b}\n"""'
    )


def get_control_pairs(items):
    """Pair each story with itself, for identical-text control comparisons."""
    return [(item, item) for item in items]


def main():
    items = load_items(ITEMS_FILE)
    conditions = load_items(CONDITIONS_FILE)
    # permutations gives both A/B orderings for every pair
    for item_a, item_b in permutations(items, 2):
        text_a = load_story(item_a["path"])
        text_b = load_story(item_b["path"])
        for condition in conditions:
            prompt = build_comparison_prompt(
                text_a, text_b, condition["a_context"], condition["b_context"]
            )
            print(
                f"--- Comparison: {item_a['id']} vs {item_b['id']} "
                f"| Condition: {condition['id']} ---"
            )
            print(prompt)
            print()

    # identical-text controls: each story compared against itself
    for item_a, item_b in get_control_pairs(items):
        text_a = load_story(item_a["path"])
        text_b = load_story(item_b["path"])
        for condition in conditions:
            prompt = build_comparison_prompt(
                text_a, text_b, condition["a_context"], condition["b_context"]
            )
            print(f"--- Comparison Control: {item_a['id']} | Condition: {condition['id']} ---")
            print(prompt)
            print()


if __name__ == "__main__":
    main()
