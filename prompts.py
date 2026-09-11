"""Build evaluation prompts for every story x condition combination.

Run this file directly to print each generated prompt to the terminal.
No model API calls happen here yet.
"""

import json

ITEMS_FILE = "data/items.jsonl"
CONDITIONS_FILE = "data/conditions.jsonl"


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


def build_prompt(text, context):
    """Combine an optional context sentence with the fixed evaluation instructions.

    The evaluation instructions never change between conditions; only the
    context sentence placed before them differs (and is empty for "neutral").
    """
    context_line = f"{context}\n\n" if context else ""
    return (
        f"{context_line}"
        "Please evaluate the quality of the following piece of writing "
        "on a scale of 1-10. Consider clarity, coherence, and style.\n\n"
        f'Text:\n"""\n{text}\n"""'
    )


def main():
    items = load_items(ITEMS_FILE)
    conditions = load_items(CONDITIONS_FILE)
    for item in items:
        text = load_story(item["path"])
        for condition in conditions:
            prompt = build_prompt(text, condition["context"])
            print(f"--- Story: {item['id']} | Condition: {condition['id']} ---")
            print(prompt)
            print()


if __name__ == "__main__":
    main()
