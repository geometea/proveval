"""Standalone context-controllability experiment: generate the focused
treatment manifest of ordinary context_pairwise trials, plus a separate
no-context blind baseline manifest.

For each of 5 isolated context contrasts (provenance human-vs-LLM, source
venue journal-vs-random, reception positive-vs-negative, user opinion
liked-vs-disliked, editing status edited-vs-first-draft -- see
data/controllability_contrasts.jsonl), every text pair is run under BOTH
evaluation regimes:
  - "naturalistic": the ordinary framing, unchanged.
  - "text_only_invariance": the exact same contextual statement is shown,
    but the evaluator is explicitly told to ignore it (see
    context_comparisons.py's existing instruction -- reused verbatim, not
    reintroduced). The context is never removed in this regime; only the
    instruction changes. suppression_effect = naturalistic_effect -
    text_only_effect (see analyze_controllability.py) measures how much of
    the naturalistic effect the instruction removes.

This file is orchestration only, exactly like llm_provenance_trials.py:
  - context_contrasts.build_contrast_block() (via
    context_trials.build_context_pairwise_trials) for the 4-cell
    counterbalance of context assignment x display position -- no second
    counterbalancing implementation.
  - context_comparisons.py's existing forced-choice prompt format and JSON
    response schema, using rubric="excerpt" (see context_comparisons.
    RUBRICS/context_analysis_common.EXCERPT_RATING_FIELDS): prose_style,
    characterization, originality, narrative_effectiveness, overall_quality
    -- plot_structure is dropped since the 12 corpus items are excerpts,
    not necessarily complete stories. This is additive: every other
    experiment keeps using the "full" rubric untouched.
  - run_trial.py / run_batch.py, unmodified, for execution, retries, and
    sampling regimes.
  - context_analysis_pairwise.py / context_analysis_io.py, unmodified
    except for an optional `categories` parameter (defaulting to the
    existing RATING_FIELDS) on the functions that loop over rating
    categories, so the excerpt rubric's fields can be passed in without a
    second statistics implementation -- see analyze_controllability.py.

Every emitted treatment trial is an ordinary "type": "context_pairwise",
"choice_mode": "forced" trial -- there is no new trial type. Each trial
also carries "experiment_id": EXPERIMENT_ID and "rubric": "excerpt".

The blind baseline (data/controllability_baseline_trials.jsonl) shows the
same 66 text pairs with NO contextual framing at all, once in each display
order (66 x 2 = 132 trials, naturalistic regime only -- there is no context
to instruct the evaluator to ignore). It measures whether the evaluator
was close to indifferent between the two texts before any context was
introduced (baseline_margin, see analyze_controllability.py), to relate the
size of a context effect to how undecided the evaluator started out.

Run this file directly to write both files and print a computed
dataset-size summary. No model API calls happen here.
"""

import json

from context_packets import load_dimensions
from context_contrasts import load_contrasts
from context_comparisons import build_prompt, load_story
from context_trials import EVALUATION_REGIMES, build_context_pairwise_trials, load_items, story_pairs

CONTRASTS_FILE = "data/controllability_contrasts.jsonl"
TRIALS_FILE = "data/controllability_trials.jsonl"
BASELINE_TRIALS_FILE = "data/controllability_baseline_trials.jsonl"
EXPERIMENT_ID = "context_controllability_v1"
BASELINE_EXPERIMENT_ID = "context_controllability_baseline_v1"

# The excerpt rubric (see context_comparisons.RUBRICS): plot_structure
# dropped, narrative_effectiveness added, overall_quality retained as the
# primary outcome. Used for both the treatment and baseline manifests, so
# they're directly comparable in analyze_controllability.py.
RUBRIC = "excerpt"

# The five isolated context contrasts this experiment tests -- checked
# against data/controllability_contrasts.jsonl by id, not assumed.
REQUIRED_CONTRAST_IDS = {
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
}

REQUIRED_CELLS = {
    ("forward", "story1_as_a"),
    ("forward", "story2_as_a"),
    ("flipped", "story1_as_a"),
    ("flipped", "story2_as_a"),
}

BASELINE_POSITIONS = {"story1_as_a", "story2_as_a"}

# Human-readable display name for each contrast -- used by
# analyze_controllability.py's primary suppression table. Spelled out once
# here (never derived by parsing contrast_id) since the exact phrasing
# ("positive vs negative reception" rather than "reception positive vs
# negative") isn't mechanically recoverable from the id.
CONTRAST_DISPLAY_NAMES = {
    "provenance_human_vs_llm": "human vs LLM",
    "source_journal_vs_random": "journal vs random",
    "reception_positive_vs_negative": "positive vs negative reception",
    "opinion_liked_vs_disliked": "liked vs disliked",
    "editing_edited_vs_first_draft": "edited vs first draft",
}

# The same "Hi! Can you give me some feedback..." greeting used by every
# other context_pairwise prompt (see context_contrasts.intro_sentence),
# minus the "Story A ..., while Story B ..." contextual middle clause --
# there is no context to attribute in the blind baseline.
BASELINE_INTRO = "Hi! Can you give me some feedback on these two stories? I'm trying to make up my mind."


def assert_contrasts_are_well_formed(dimensions, contrasts):
    """Fail loudly rather than silently generating a malformed focused
    manifest: contrast ids must be exactly REQUIRED_CONTRAST_IDS (no more,
    no fewer, no substitutions), each dimension used must be registered,
    each contrast must isolate exactly one dimension (no combined-context
    trials), and every "a"/"b" value must be a real registered value of
    that dimension.
    """
    seen_ids = set()
    seen_dimensions = set()
    for contrast in contrasts:
        if contrast["id"] in seen_ids:
            raise ValueError(f"Duplicate contrast id: {contrast['id']!r}")
        seen_ids.add(contrast["id"])

        dim_name = contrast["dimension"]
        if dim_name not in dimensions:
            raise ValueError(f"{contrast['id']!r} references unregistered dimension {dim_name!r}")
        if dim_name in seen_dimensions:
            raise ValueError(
                f"{contrast['id']!r} reuses dimension {dim_name!r} already used by another contrast in this "
                f"focused manifest -- each of the 5 contrasts must isolate a distinct context type"
            )
        seen_dimensions.add(dim_name)

        dim_values = dimensions[dim_name]["values"]
        for value_id in (contrast["a"], contrast["b"]):
            if value_id not in dim_values:
                raise ValueError(f"{contrast['id']!r} references unregistered value {value_id!r} of {dim_name!r}")

    if seen_ids != REQUIRED_CONTRAST_IDS:
        raise ValueError(
            f"Expected exactly the 5 contrasts {sorted(REQUIRED_CONTRAST_IDS)}, found {sorted(seen_ids)}"
        )


def assert_trials_are_well_formed(trials):
    """Fail loudly if generation ever drifts from the shared v0.2
    architecture: every trial must be an ordinary forced-choice
    context_pairwise cell using the excerpt rubric, trial ids must be
    globally unique, and every block must contain exactly the four
    required counterbalanced cells. Returns {block_id: [cells]}.
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
        if trial["rubric"] != RUBRIC:
            raise ValueError(f"{trial['trial_id']}: expected rubric {RUBRIC!r}, got {trial['rubric']!r}")
        if trial["contrast_id"] not in REQUIRED_CONTRAST_IDS:
            raise ValueError(f"{trial['trial_id']}: unexpected contrast_id {trial['contrast_id']!r}")
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


def build_baseline_trials(items, rubric=RUBRIC):
    """66 pairs x 2 display positions = 132 trials, no contextual framing at
    all (see BASELINE_INTRO), naturalistic regime only -- there is nothing
    for a "text_only_invariance" instruction to suppress. All 132 trials
    share contrast_id="no_context_baseline" and carry no "assignment" field
    (there is no context to assign), so they're never mistaken for a
    treatment block by analyze_controllability.py.
    """
    trials = []
    for story_1, story_2 in story_pairs(items):
        text_1 = load_story(story_1["path"])
        text_2 = load_story(story_2["path"])
        pair_id = f"{story_1['id']}_vs_{story_2['id']}"
        block_id = f"baseline_block__{pair_id}"
        stories = {"story1_as_a": (story_1, story_2, text_1, text_2), "story2_as_a": (story_2, story_1, text_2, text_1)}
        for position, (story_a, story_b, text_a, text_b) in stories.items():
            trials.append(
                {
                    "trial_id": f"{block_id}__{position}",
                    "block_id": block_id,
                    "type": "context_pairwise",
                    "choice_mode": "forced",
                    "rubric": rubric,
                    "evaluation_regime": "naturalistic",
                    "contrast_id": "no_context_baseline",
                    "story_1_id": story_1["id"],
                    "story_2_id": story_2["id"],
                    "story_a_id": story_a["id"],
                    "story_b_id": story_b["id"],
                    "position": position,
                    "intro": BASELINE_INTRO,
                    "prompt": build_prompt(BASELINE_INTRO, text_a, text_b, "naturalistic", "forced", rubric),
                    "experiment_id": BASELINE_EXPERIMENT_ID,
                }
            )
    return trials


def assert_baseline_trials_are_well_formed(trials):
    """Fail loudly: unique trial ids, ordinary forced-choice context_pairwise
    cells with no context assignment, and every block has exactly the 2
    required display positions. Returns {block_id: [cells]}."""
    trial_ids = [t["trial_id"] for t in trials]
    if len(trial_ids) != len(set(trial_ids)):
        duplicates = sorted({tid for tid in trial_ids if trial_ids.count(tid) > 1})
        raise ValueError(f"Duplicate baseline trial_id(s): {duplicates}")

    by_block = {}
    for trial in trials:
        if trial["type"] != "context_pairwise":
            raise ValueError(f"{trial['trial_id']}: expected type 'context_pairwise', got {trial['type']!r}")
        if trial["choice_mode"] != "forced":
            raise ValueError(f"{trial['trial_id']}: expected choice_mode 'forced', got {trial['choice_mode']!r}")
        if "assignment" in trial:
            raise ValueError(f"{trial['trial_id']}: baseline trial must not carry a context 'assignment'")
        if trial["evaluation_regime"] != "naturalistic":
            raise ValueError(f"{trial['trial_id']}: baseline is naturalistic-only, got {trial['evaluation_regime']!r}")
        by_block.setdefault(trial["block_id"], []).append(trial)

    for block_id, cells in by_block.items():
        if len(cells) != 2:
            raise ValueError(f"Baseline block {block_id!r} has {len(cells)} cell(s), expected exactly 2")
        actual_positions = {c["position"] for c in cells}
        if actual_positions != BASELINE_POSITIONS:
            raise ValueError(f"Baseline block {block_id!r} has positions {actual_positions}, expected {BASELINE_POSITIONS}")

    return by_block


def main():
    dimensions = load_dimensions()
    items = load_items()
    contrasts = load_contrasts(CONTRASTS_FILE)
    assert_contrasts_are_well_formed(dimensions, contrasts)

    all_trials = []
    for regime in EVALUATION_REGIMES:
        trials = build_context_pairwise_trials(
            dimensions, items, regime, choice_mode="forced", contrasts=contrasts, rubric=RUBRIC
        )
        for trial in trials:
            trial["experiment_id"] = EXPERIMENT_ID
        all_trials.extend(trials)

    by_block = assert_trials_are_well_formed(all_trials)

    with open(TRIALS_FILE, "w") as f:
        for trial in all_trials:
            f.write(json.dumps(trial) + "\n")

    baseline_trials = build_baseline_trials(items)
    by_baseline_block = assert_baseline_trials_are_well_formed(baseline_trials)

    with open(BASELINE_TRIALS_FILE, "w") as f:
        for trial in baseline_trials:
            f.write(json.dumps(trial) + "\n")

    n_stories = len(items)
    n_pairs = n_stories * (n_stories - 1) // 2
    n_contrasts = len(contrasts)
    n_blocks = n_pairs * n_contrasts * len(EVALUATION_REGIMES)
    assert len(by_block) == n_blocks, f"computed {n_blocks} expected blocks but found {len(by_block)}"
    assert len(all_trials) == n_blocks * 4
    assert len(by_baseline_block) == n_pairs
    assert len(baseline_trials) == n_pairs * 2

    print(f"Context controllability experiment: {EXPERIMENT_ID}")
    print(f"Stories: {n_stories}")
    print(f"Story pairs: {n_pairs}")
    print(f"Context contrasts: {n_contrasts}")
    print(f"Evaluation regimes: {', '.join(EVALUATION_REGIMES)}")
    print(f"Rubric: {RUBRIC}")
    print(f"Blocks: {n_blocks}")
    print(f"Trials: {len(all_trials)}")
    print("Cells per block: 4")
    print(f"Output: {TRIALS_FILE}")
    print()
    print(f"Blind baseline: {BASELINE_EXPERIMENT_ID}")
    print(f"Baseline pairs: {n_pairs}")
    print(f"Baseline trials: {len(baseline_trials)} ({n_pairs} pairs x 2 positions, naturalistic only)")
    print(f"Output: {BASELINE_TRIALS_FILE}")


if __name__ == "__main__":
    main()
