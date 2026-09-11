"""The natural, non-experiment-flavored comparison prompt for context-packet trials.

This is a distinct prompt format from comparisons.py's numeric-rating format: it
frames the request as a casual ask for help choosing between two stories, and asks
for a plain A / B / tie choice per category instead of 1-5 ratings. It is
intentionally kept separate from prompts.py, comparisons.py, make_trials.py,
run_trial.py, run_batch.py, and analyze.py -- none of those are touched by this
file, and data/trials.jsonl is never written to, so the existing pipeline and its
results/raw.jsonl schema are unaffected.

This module just holds the prompt template and the shared load_items/load_story
helpers. The actual trial generator lives in context_contrasts.py, which reuses
build_prompt() from here to fill it in with two different stories and two
explicitly contrasted context values.
"""

import json

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
    """Fill in the natural comparison template with two stories and their context text."""
    return INSTRUCTION_TEMPLATE.format(
        context_a=context_a_text,
        story_a=story_a,
        context_b=context_b_text,
        story_b=story_b,
    )
