"""Build evaluation prompts for every story x condition x task combination.

Run this file directly to print each generated prompt to the terminal.
No model API calls happen here yet.
"""

import json

ITEMS_FILE = "data/items.jsonl"
CONDITIONS_FILE = "data/conditions.jsonl"
TASKS_FILE = "data/tasks.jsonl"


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


def build_prompt(text, context, instruction):
    """Combine an optional context sentence, the task instruction, and the story.

    Order is: context sentence (if any), blank line, task instruction, story.
    The context sentence is omitted entirely for "neutral".
    """
    context_line = f"{context}\n\n" if context else ""
    return (
        f"{context_line}"
        f"{instruction}\n\n"
        f'Text:\n"""\n{text}\n"""'
    )


def main():
    items = load_items(ITEMS_FILE)
    conditions = load_items(CONDITIONS_FILE)
    tasks = load_items(TASKS_FILE)
    for item in items:
        text = load_story(item["path"])
        for condition in conditions:
            for task in tasks:
                prompt = build_prompt(text, condition["context"], task["instruction"])
                print(
                    f"--- Story: {item['id']} | Condition: {condition['id']} "
                    f"| Task: {task['id']} ---"
                )
                print(prompt)
                print()


if __name__ == "__main__":
    main()
