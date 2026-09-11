"""Build evaluation prompts from the toy prose samples.

Run this file directly to print each generated prompt to the terminal.
No model API calls happen here yet.
"""

import json

ITEMS_FILE = "data/items.jsonl"


def load_items(path):
    """Read a .jsonl file and return a list of dicts (one per line)."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_prompt(text):
    """Wrap a piece of prose in a neutral evaluation prompt."""
    return (
        "Please evaluate the quality of the following piece of writing "
        "on a scale of 1-10. Consider clarity, coherence, and style.\n\n"
        f'Text:\n"""\n{text}\n"""'
    )


def main():
    items = load_items(ITEMS_FILE)
    for item in items:
        prompt = build_prompt(item["text"])
        print(f"--- Prompt for item {item['id']} ---")
        print(prompt)
        print()


if __name__ == "__main__":
    main()
