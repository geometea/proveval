"""Load context-packet dimensions and render a packet into natural-sounding text.

A context packet is a small dict, e.g. {"provenance": "ai_claude", "editing_status": "edited"}.
Dimensions and their phrasing are defined in data/context_dimensions.jsonl, not
hardcoded here, so new signals or wording can be added without touching this file.

This module is standalone: it does not import or modify prompts.py, comparisons.py,
make_trials.py, run_trial.py, run_batch.py, or analyze.py, and data/trials.jsonl is
never touched.

Run this file directly to print a small example set of single-variable conditions.
No model API calls happen here.
"""

import json

DIMENSIONS_FILE = "data/context_dimensions.jsonl"

# The order pieces are joined into one natural paragraph when a packet has more
# than one dimension set. Deliberately excludes "prompt_context": that
# dimension has scope "prompt" (it's not attributed to any one story), and is
# handled separately by context_trials.py rather than through render_packet.
DIMENSION_ORDER = ["provenance", "writer_status", "source_venue", "editing_status", "reception", "user_opinion"]


def load_dimensions(path=DIMENSIONS_FILE):
    """Read data/context_dimensions.jsonl into {dimension_id: dimension_dict}."""
    dimensions = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                dim = json.loads(line)
                dimensions[dim["dimension"]] = dim
    return dimensions


def dimensions_with_scope(dimensions, scope):
    """Return {dimension_id: dimension_dict} filtered to a given "scope" value.

    "story" dimensions (provenance, writer_status, source_venue,
    editing_status, reception, user_opinion) are attributed to one story.
    "prompt" dimensions (currently just prompt_context) apply to the whole
    request and aren't attached to either story -- see context_trials.py.
    """
    return {dim_id: dim for dim_id, dim in dimensions.items() if dim.get("scope") == scope}


def value_text(dimensions, dimension_id, value_id, packet):
    """Look up the phrase for one (dimension, value), resolving subject-dependent wording.

    Some dimensions (currently just writer_status) phrase their values differently
    depending on another dimension already in the packet (e.g. "I" vs "they",
    depending on provenance). Those are stored as {parent_value: phrase} instead
    of a plain string.
    """
    dim = dimensions[dimension_id]
    text = dim["values"][value_id]
    if isinstance(text, dict):
        parent_value = packet[dim["depends_on"]]
        text = text[parent_value]
    return text


def validate_packet(dimensions, packet):
    """Check every dimension/value in the packet is real, and dependencies are met."""
    for dimension_id, value_id in packet.items():
        if dimension_id not in dimensions:
            raise ValueError(f"Unknown dimension: {dimension_id}")
        dim = dimensions[dimension_id]
        if value_id not in dim["values"]:
            raise ValueError(f"Unknown value {value_id!r} for dimension {dimension_id!r}")
        depends_on = dim.get("depends_on")
        if depends_on:
            if depends_on not in packet:
                raise ValueError(f"{dimension_id!r} requires {depends_on!r} to also be set")
            if packet[depends_on] not in dim["applies_when"]:
                raise ValueError(
                    f"{dimension_id!r} only applies when {depends_on!r} is one of {dim['applies_when']}"
                )


def render_packet(dimensions, packet):
    """Turn a context packet into one natural-sounding paragraph, or "" if empty."""
    validate_packet(dimensions, packet)
    sentences = [
        value_text(dimensions, dimension_id, packet[dimension_id], packet)
        for dimension_id in DIMENSION_ORDER
        if dimension_id in packet
    ]
    return " ".join(sentences)


def neutral_condition():
    """The empty-context baseline for context_single trials: no dimension set
    at all, so context_text is "". This is v0.2's own clean single-text
    baseline (separate from the v0.1 pilot's "neutral" condition in
    data/conditions.jsonl) -- every single-variable condition and every
    per-condition ranking in analyze_context.py is implicitly compared
    against this. See context_trials.build_context_single_trials, which adds
    one of these per story alongside every single_variable_conditions() entry.
    """
    return {
        "condition_id": "neutral",
        "dimension": "neutral",
        "value": "neutral",
        "packet": {},
        "context_text": "",
        "note": "",
        "hypothesis": "Baseline: no context signal presented at all.",
    }


def single_variable_conditions(dimensions):
    """Build one example condition per (dimension, value), isolating that one variable.

    Each condition is meant to be compared against an empty baseline packet (no
    context sentence at all) -- so only the one named signal differs between
    baseline and treatment.

    writer_status cannot be expressed without a provenance, so its conditions pin
    provenance to a fixed carrier value (the first entry in applies_when, "self").
    That carrier is held constant across every writer_status condition, so
    writer_status is still the only thing varying *among those conditions* -- but
    each of them differs from the true empty baseline in two fields, not one. This
    is called out in each condition's "note".
    """
    conditions = []
    for dimension_id in DIMENSION_ORDER:
        dim = dimensions[dimension_id]
        depends_on = dim.get("depends_on")
        for value_id in dim["values"]:
            if depends_on:
                carrier_value = dim["applies_when"][0]
                packet = {depends_on: carrier_value, dimension_id: value_id}
                note = f"carrier: {depends_on}={carrier_value} held constant (not a pure single-field isolation)"
            else:
                packet = {dimension_id: value_id}
                note = ""
            conditions.append(
                {
                    "condition_id": f"{dimension_id}__{value_id}",
                    "dimension": dimension_id,
                    "value": value_id,
                    "packet": packet,
                    "context_text": render_packet(dimensions, packet),
                    "note": note,
                    "hypothesis": dim["hypothesis"],
                }
            )
    return conditions


def main():
    dimensions = load_dimensions()
    conditions = single_variable_conditions(dimensions)
    print(f"{len(conditions)} single-variable example conditions (baseline = no context at all):\n")
    for c in conditions:
        print(f"--- {c['condition_id']} ---")
        print(f"  context: {c['context_text']!r}")
        if c["note"]:
            print(f"  note: {c['note']}")
        print(f"  hypothesis: {c['hypothesis']}")
        print()


if __name__ == "__main__":
    main()
