"""Offline helper for building a fixed human-preference ranking from pairwise judgments.

Reads data/human_pairwise.jsonl (winner/loser judgments collected by hand,
never invented here), checks the implied partial order for cycles or direct
contradictions, and reports whether it uniquely determines a full ranking of
every registered story. Only writes data/human_reference.json when the order
is both complete (covers every story in data/items.jsonl) and unique (no two
valid orderings disagree) -- otherwise it explains what's missing and writes
nothing.

This is meant to support collecting just enough adaptive pairwise judgments
to pin down a full order, rather than rating all C(12,2)=66 pairs up front.

Run this file directly to print a report. No model API calls happen here.
"""

import json

PAIRWISE_FILE = "data/human_pairwise.jsonl"
ITEMS_FILE = "data/items.jsonl"
REFERENCE_FILE = "data/human_reference.json"


def load_pairwise(path=PAIRWISE_FILE):
    """Read a winner/loser judgments file into a list of (winner, loser) tuples."""
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                pairs.append((row["winner"], row["loser"]))
    return pairs


def load_story_ids(path=ITEMS_FILE):
    """Read data/items.jsonl and return the full list of registered story ids."""
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    return ids


def find_unknown_story_ids(pairs, known_ids):
    """Return the sorted set of winner/loser ids in `pairs` that aren't in
    known_ids (data/items.jsonl) -- e.g. a typo in a hand-edited judgment.
    Checked before anything else touches `pairs`, since topological_order's
    adjacency/in_degree dicts are built only from known_ids and would raise
    a raw KeyError on an unrecognized id instead of a clear diagnostic.
    """
    known = set(known_ids)
    unknown = set()
    for winner, loser in pairs:
        if winner not in known:
            unknown.add(winner)
        if loser not in known:
            unknown.add(loser)
    return sorted(unknown)


def find_direct_contradictions(pairs):
    """Return [(winner, loser), ...] for any pair judged both ways."""
    beats = set(pairs)
    seen_unordered = set()
    contradictions = []
    for winner, loser in pairs:
        unordered = tuple(sorted((winner, loser)))
        if unordered in seen_unordered:
            continue
        seen_unordered.add(unordered)
        if (loser, winner) in beats:
            contradictions.append((winner, loser))
    return contradictions


def topological_order(nodes, pairs):
    """Attempt a best-first topological order (winners before losers), via Kahn's algorithm.

    Returns (order_or_None, is_unique, message):
      order_or_None: a full ordering if one exists at all (no cycle), else None
      is_unique: True only if every step had exactly one available next story
                 (i.e. the judgments actually form a total order, not just a
                 valid partial one)
      message:   a human-readable explanation of the result
    """
    in_degree = {n: 0 for n in nodes}
    adjacency = {n: [] for n in nodes}
    for winner, loser in pairs:
        adjacency[winner].append(loser)
        in_degree[loser] += 1

    remaining = set(nodes)
    order = []
    is_unique = True
    tie_steps = []

    while remaining:
        ready = sorted(n for n in remaining if in_degree[n] == 0)
        if not ready:
            return None, False, f"Cycle detected: no valid next story among {sorted(remaining)}"
        if len(ready) > 1:
            is_unique = False
            tie_steps.append(ready)
        # Pick deterministically (alphabetical) so we can still finish the
        # cycle check; is_unique already records that this step was ambiguous.
        node = ready[0]
        order.append(node)
        remaining.discard(node)
        for next_node in adjacency[node]:
            in_degree[next_node] -= 1

    if is_unique:
        return order, True, "The pairwise judgments uniquely determine a full ranking."
    return (
        order,
        False,
        f"The pairwise judgments do not uniquely determine a full ranking: "
        f"{len(tie_steps)} point(s) had more than one story that could come next "
        f"(e.g. {tie_steps[0]}). More pairwise judgments are needed.",
    )


def main():
    pairs = load_pairwise()
    story_ids = load_story_ids()

    print(f"Loaded {len(pairs)} human pairwise judgment(s) over {len(story_ids)} registered stories.\n")
    for winner, loser in pairs:
        print(f"  {winner} > {loser}")
    print()

    unknown_ids = find_unknown_story_ids(pairs, story_ids)
    if unknown_ids:
        print(f"INCONSISTENT: judgment(s) reference unrecognized story id(s): {unknown_ids}")
        print(f"(not in {ITEMS_FILE} -- check for a typo in {PAIRWISE_FILE})")
        print("\nNot writing a reference ranking.")
        return

    contradictions = find_direct_contradictions(pairs)
    if contradictions:
        print("INCONSISTENT: direct contradictions found (judged both ways):")
        for a, b in contradictions:
            print(f"  {a} vs {b}")
        print("\nNot writing a reference ranking.")
        return

    order, is_unique, message = topological_order(story_ids, pairs)

    if order is None:
        print(f"INCONSISTENT: {message}")
        print("\nNot writing a reference ranking.")
        return

    print(message)
    if not is_unique:
        print("\nNot writing a reference ranking -- collect more pairwise judgments to narrow it down.")
        return

    with open(REFERENCE_FILE, "w") as f:
        json.dump({"ranking": order, "source": PAIRWISE_FILE, "num_judgments": len(pairs)}, f, indent=2)
    print(f"\nWrote complete, unique ranking to {REFERENCE_FILE}:")
    for i, story_id in enumerate(order, start=1):
        print(f"  {i}. {story_id}")


if __name__ == "__main__":
    main()
