"""Generate forward/flipped A-B trials from explicit context contrasts.

A contrast names two specific values of one dimension to pit against each other
(e.g. provenance "ai_claude" vs "human_unknown"), defined in
data/context_contrasts.jsonl -- not hardcoded here. For a fixed pair of two
DIFFERENT stories, this builds:

- a forward trial: story_1 gets context value "a", story_2 gets value "b"
- a flipped trial: the assignment is swapped

Both stories always carry some context (never nothing) in both trials -- the
control comes from swapping which value each story gets, not from comparing
against an empty baseline. If the model's choice is really about the
underlying stories, the same story should keep winning in both trials, just
moving between A and B. If it instead flips to whichever story is currently
carrying value "a" (or "b"), the context is doing the work, not the prose.

Reuses context_comparisons.build_prompt() and its INSTRUCTION_TEMPLATE for the
actual prompt text (the existing natural "help me choose" / A-B-tie JSON
format) rather than duplicating it. Does not modify prompts.py, comparisons.py,
make_trials.py, run_trial.py, run_batch.py, analyze.py, or data/trials.jsonl,
and makes no API calls.

Run this file directly to print a few example forward/flipped prompt pairs.
"""

import json

from context_packets import load_dimensions, render_packet
from context_comparisons import build_prompt, load_items, load_story

CONTRASTS_FILE = "data/context_contrasts.jsonl"


def load_contrasts(path=CONTRASTS_FILE):
    """Read data/context_contrasts.jsonl into a list of contrast dicts."""
    contrasts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                contrasts.append(json.loads(line))
    return contrasts


def assert_no_shared_identity_contradiction(dimensions, contrast):
    """Refuse contrasts that would make one claimed identity contradict itself.

    A two-story prompt has a single "I" speaking for both stories at once. Any
    dimension that depends_on another dimension (currently just writer_status,
    which depends on provenance) describes a property of *the person*, not of
    the individual story -- so it cannot validly take two different values
    across Story A and Story B in the same prompt, no matter what carrier is
    used: "I am a published author" for Story A and "this is my first time
    writing" for Story B contradict each other, since they're both said by the
    same "I" in the same breath.

    Dimensions like this are single-story only (see context_single_prompts.py,
    where each API call only ever claims one identity for one story).
    """
    dim = dimensions[contrast["dimension"]]
    if dim.get("depends_on"):
        raise ValueError(
            f"Contrast {contrast['id']!r} uses dimension {contrast['dimension']!r}, which "
            f"depends on {dim['depends_on']!r}. Dependent dimensions describe a claimed "
            f"identity shared by both stories in a two-story prompt, so they can't be "
            f"contrasted across Story A/B without contradicting themselves. Use "
            f"context_single_prompts.py for this dimension instead."
        )


def contrast_context_texts(dimensions, contrast):
    """Render the two context strings named by a contrast's "a"/"b" values."""
    assert_no_shared_identity_contradiction(dimensions, contrast)
    carrier = contrast.get("carrier", {})
    packet_a = {**carrier, contrast["dimension"]: contrast["a"]}
    packet_b = {**carrier, contrast["dimension"]: contrast["b"]}
    return render_packet(dimensions, packet_a), render_packet(dimensions, packet_b)


def build_contrast_trials(dimensions, contrast, story_1, story_2):
    """Build the forward and flipped trial for one contrast, across two different stories."""
    if story_1["id"] == story_2["id"]:
        raise ValueError("build_contrast_trials requires two different stories, not the same story twice")

    context_a, context_b = contrast_context_texts(dimensions, contrast)
    text_1 = load_story(story_1["path"])
    text_2 = load_story(story_2["path"])
    pair_id = f"{story_1['id']}_vs_{story_2['id']}"

    forward = {
        "trial_id": f"contrast__{contrast['id']}__{pair_id}__forward",
        "contrast_id": contrast["id"],
        "dimension": contrast["dimension"],
        "story_a_id": story_1["id"],
        "story_b_id": story_2["id"],
        "context_a": context_a,
        "context_b": context_b,
        "prompt": build_prompt(text_1, context_a, text_2, context_b),
    }
    flipped = {
        "trial_id": f"contrast__{contrast['id']}__{pair_id}__flipped",
        "contrast_id": contrast["id"],
        "dimension": contrast["dimension"],
        "story_a_id": story_1["id"],
        "story_b_id": story_2["id"],
        "context_a": context_b,
        "context_b": context_a,
        "prompt": build_prompt(text_1, context_b, text_2, context_a),
    }
    return forward, flipped


def main():
    dimensions = load_dimensions()
    contrasts = load_contrasts()
    items = load_items()
    story_1, story_2 = items[0], items[1]  # a fixed, distinct pair keeps the example small

    all_trials = []
    for contrast in contrasts:
        forward, flipped = build_contrast_trials(dimensions, contrast, story_1, story_2)
        all_trials.append(forward)
        all_trials.append(flipped)

    print(
        f"{len(contrasts)} contrasts x 2 (forward/flipped) = {len(all_trials)} example trials "
        f"({story_1['id']} vs {story_2['id']})\n"
    )

    preview_count = 3
    for trial in all_trials[:preview_count]:
        print(f"=== {trial['trial_id']} ===")
        print(f"(A: {trial['context_a']!r} | B: {trial['context_b']!r})")
        print(trial["prompt"])
        print()

    print(f"...and {len(all_trials) - preview_count} more:")
    for trial in all_trials[preview_count:]:
        print(f"  {trial['trial_id']}  (A: {trial['context_a']!r} | B: {trial['context_b']!r})")


if __name__ == "__main__":
    main()
