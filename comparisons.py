"""Build comparison prompts for every unique pair of stories.

Run this file directly to print each generated prompt to the terminal.
No model API calls happen here yet.
"""

import json
from itertools import combinations

ITEMS_FILE = "data/items.jsonl"

INSTRUCTION = (
    "Compare the literary quality of the following two short stories. "
    "Which is stronger overall? Discuss the strengths and weaknesses of "
    "each, explain your preference, and base your judgment only on the "
    "texts."
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


def build_comparison_prompt(text_a, text_b):
    """Combine the fixed instruction with both stories, labeled A and B."""
    return (
        f"{INSTRUCTION}\n\n"
        f'STORY A:\n"""\n{text_a}\n"""\n\n'
        f'STORY B:\n"""\n{text_b}\n"""'
    )


def main():
    items = load_items(ITEMS_FILE)
    for item_a, item_b in combinations(items, 2):
        text_a = load_story(item_a["path"])
        text_b = load_story(item_b["path"])
        prompt = build_comparison_prompt(text_a, text_b)
        print(f"--- Comparison: {item_a['id']} vs {item_b['id']} ---")
        print(prompt)
        print()


if __name__ == "__main__":
    main()
