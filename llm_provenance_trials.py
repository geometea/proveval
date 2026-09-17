"""Standalone named-LLM provenance pairwise experiment: generate the
focused manifest of ordinary context_pairwise trials.

Given the same underlying pair of stories, does an evaluator's preference
change depending on whether each story is claimed to have been generated
by Claude, ChatGPT, Gemini, or DeepSeek? This is a provenance-LABEL
experiment: the prose is always one of the existing 12 registered corpus
stories, byte-identical across every cell -- only the claimed authorship
changes. It is NOT a comparison of actual outputs produced by those
models, and no non-Anthropic API is called anywhere in this repository
(see README.md's "Standalone named-LLM provenance experiment" section).

This file is orchestration only. It reuses the existing v0.2 machinery
exactly as-is:
  - context_contrasts.build_contrast_block() (via
    context_trials.build_context_pairwise_trials) for the 4-cell
    counterbalance of context assignment x display position -- no second
    counterbalancing implementation.
  - context_comparisons.py's existing forced-choice prompt format and JSON
    response schema -- no new prompt builder, no new validator.
  - run_trial.py / run_batch.py, unmodified, for execution, retries, and
    sampling regimes.
  - context_analysis_pairwise.py / context_analysis_io.py, unmodified, for
    analysis (see analyze_llm_provenance.py).

Every emitted trial is an ordinary "type": "context_pairwise",
"choice_mode": "forced" trial -- there is no new trial type. The only
thing narrowing this manifest to the focused experiment is which
contrasts get passed to build_context_pairwise_trials(): the six rows of
data/llm_provenance_contrasts.jsonl (all C(4,2) unordered pairs of
Claude/ChatGPT/Gemini/DeepSeek) instead of the full
data/context_contrasts.jsonl set. Each trial also carries an
"experiment_id" field so a shared results file (or a report built from
one) can still tell this manifest's rows apart from the general
benchmark's if they were ever pooled -- see EXPERIMENT_ID.

Run this file directly to write data/llm_provenance_trials.jsonl and print
a computed dataset-size summary. No model API calls happen here.
"""

import argparse
import json

from context_packets import load_dimensions
from context_contrasts import load_contrasts
from context_trials import EVALUATION_REGIMES, build_context_pairwise_trials, load_items

CONTRASTS_FILE = "data/llm_provenance_contrasts.jsonl"
TRIALS_FILE = "data/llm_provenance_trials.jsonl"
EXPERIMENT_ID = "llm_provenance_pairwise_v1"

# The four claimed-provenance labels this experiment manipulates. The six
# rows of CONTRASTS_FILE must be exactly C(4,2) unordered pairs of these --
# checked by assert_contrasts_are_well_formed, not assumed.
PROVENANCE_LABELS = ["ai_claude", "ai_chatgpt", "ai_gemini", "ai_deepseek"]

# Human-readable display name for each label -- irregular capitalization
# (ChatGPT, DeepSeek) means these can't be derived from the value id by a
# generic rule, so they're spelled out once here and reused by
# analyze_llm_provenance.py rather than re-derived from contrast_id text.
PROVENANCE_LABEL_NAMES = {
    "ai_claude": "Claude",
    "ai_chatgpt": "ChatGPT",
    "ai_gemini": "Gemini",
    "ai_deepseek": "DeepSeek",
}

DEFAULT_EVALUATION_REGIMES = ["naturalistic"]

REQUIRED_CELLS = {
    ("forward", "story1_as_a"),
    ("forward", "story2_as_a"),
    ("flipped", "story1_as_a"),
    ("flipped", "story2_as_a"),
}


def assert_contrasts_are_well_formed(dimensions, contrasts):
    """Fail loudly rather than silently generating a malformed focused
    manifest: every contrast must reference the 'provenance' dimension and
    two of exactly the four experimental labels, contrast ids must be
    unique, and the six rows must cover each unordered pairing exactly
    once (i.e. genuinely be all of C(4,2), not just six non-duplicate rows).
    """
    provenance_values = dimensions["provenance"]["values"]

    seen_ids = set()
    seen_unordered_pairs = set()
    for contrast in contrasts:
        if contrast["id"] in seen_ids:
            raise ValueError(f"Duplicate contrast id: {contrast['id']!r}")
        seen_ids.add(contrast["id"])

        if contrast["dimension"] != "provenance":
            raise ValueError(f"{contrast['id']!r}: expected dimension 'provenance', got {contrast['dimension']!r}")

        for value_id in (contrast["a"], contrast["b"]):
            if value_id not in provenance_values:
                raise ValueError(f"{contrast['id']!r} references unregistered provenance value {value_id!r}")
            if value_id not in PROVENANCE_LABELS:
                raise ValueError(
                    f"{contrast['id']!r} references {value_id!r}, which is not one of the four "
                    f"experimental labels {PROVENANCE_LABELS}"
                )

        pair = frozenset((contrast["a"], contrast["b"]))
        if pair in seen_unordered_pairs:
            raise ValueError(f"Duplicate unordered pairing: {contrast['a']} vs {contrast['b']}")
        seen_unordered_pairs.add(pair)

    expected_n = len(PROVENANCE_LABELS) * (len(PROVENANCE_LABELS) - 1) // 2
    if len(contrasts) != expected_n:
        raise ValueError(
            f"Expected exactly {expected_n} contrasts (C({len(PROVENANCE_LABELS)},2) unordered pairs "
            f"of {PROVENANCE_LABELS}), found {len(contrasts)}"
        )


def assert_trials_are_well_formed(trials):
    """Fail loudly if generation ever drifts from the shared v0.2
    architecture: every trial must be an ordinary forced-choice
    context_pairwise cell, trial ids must be globally unique, and every
    block must contain exactly the four required counterbalanced cells --
    never an orphan or a duplicate. Returns {block_id: [cells]}.
    """
    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        duplicates = sorted({tid for tid in trial_ids if trial_ids.count(tid) > 1})
        raise ValueError(f"Duplicate trial_id(s): {duplicates}")

    by_block = {}
    for trial in trials:
        if trial["type"] != "context_pairwise":
            raise ValueError(f"{trial['trial_id']}: expected type 'context_pairwise', got {trial['type']!r}")
        if trial["choice_mode"] != "forced":
            raise ValueError(f"{trial['trial_id']}: expected choice_mode 'forced', got {trial['choice_mode']!r}")
        by_block.setdefault(trial["block_id"], []).append(trial)

    for block_id, cells in by_block.items():
        if len(cells) != 4:
            raise ValueError(f"Block {block_id!r} has {len(cells)} cell(s), expected exactly 4")
        actual_cells = {(c["assignment"], c["position"]) for c in cells}
        if actual_cells != REQUIRED_CELLS:
            raise ValueError(
                f"Block {block_id!r} has cells {sorted(actual_cells)}, expected {sorted(REQUIRED_CELLS)}"
            )

    return by_block


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate the standalone named-LLM provenance pairwise experiment manifest."
    )
    parser.add_argument(
        "--evaluation-regime",
        action="append",
        dest="evaluation_regimes",
        choices=list(EVALUATION_REGIMES),
        help=f"Repeatable, matching run_batch.py's convention (default: {DEFAULT_EVALUATION_REGIMES[0]} only).",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    evaluation_regimes = args.evaluation_regimes or DEFAULT_EVALUATION_REGIMES

    dimensions = load_dimensions()
    items = load_items()
    contrasts = load_contrasts(CONTRASTS_FILE)
    assert_contrasts_are_well_formed(dimensions, contrasts)

    all_trials = []
    for regime in evaluation_regimes:
        trials = build_context_pairwise_trials(dimensions, items, regime, choice_mode="forced", contrasts=contrasts)
        for trial in trials:
            trial["experiment_id"] = EXPERIMENT_ID
        all_trials.extend(trials)

    by_block = assert_trials_are_well_formed(all_trials)

    with open(TRIALS_FILE, "w") as f:
        for trial in all_trials:
            f.write(json.dumps(trial) + "\n")

    n_stories = len(items)
    n_pairs = n_stories * (n_stories - 1) // 2
    n_contrasts = len(contrasts)
    n_blocks = n_pairs * n_contrasts * len(evaluation_regimes)
    assert len(by_block) == n_blocks, f"computed {n_blocks} expected blocks but found {len(by_block)}"
    assert len(all_trials) == n_blocks * 4

    print(f"LLM provenance experiment: {EXPERIMENT_ID}")
    print(f"Stories: {n_stories}")
    print(f"Story pairs: {n_pairs}")
    print(f"Named LLMs: {len(PROVENANCE_LABELS)}")
    print(f"Contrasts: {n_contrasts}")
    print(f"Evaluation regimes: {', '.join(evaluation_regimes)}")
    print(f"Blocks: {n_blocks}")
    print(f"Trials: {len(all_trials)}")
    print("Cells per block: 4")
    print(f"Output: {TRIALS_FILE}")


if __name__ == "__main__":
    main()
