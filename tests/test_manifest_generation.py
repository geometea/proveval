"""Integration-level tests against the real repository data files.

These are slower and depend on data/*.jsonl, unlike the other test modules
which use synthetic fixtures exclusively. They persist the "does the
manifest still regenerate to the expected shape" check that was previously
done by hand after every change, reading the printed summary and eyeballing
it -- now asserted automatically. No API calls; no files under data/ or
results/ are modified (context_trials.main() writes files, so this module
calls the builder functions directly instead of main()).
"""

import pytest

from context_contrasts import load_contrasts
from context_packets import DIMENSION_ORDER, load_dimensions
from context_trials import (
    build_context_pairwise_trials,
    build_context_prompt_trials,
    build_context_single_trials,
    build_tie_allowed_pairwise_trials,
    load_items,
    EVALUATION_REGIMES,
)
from make_trials import main as _unused  # noqa: F401  (import-time sanity: v0.1 module still importable)


@pytest.fixture(scope="module")
def dims():
    return load_dimensions()


@pytest.fixture(scope="module")
def items():
    return load_items()


def all_required_trials(dims, items):
    trials = []
    for regime in EVALUATION_REGIMES:
        trials += build_context_single_trials(dims, items, regime)
        trials += build_context_pairwise_trials(dims, items, regime)
        trials += build_context_prompt_trials(dims, items, regime)
    return trials


def expected_context_single_count(dims, items):
    """Number of single-variable conditions is a function of how many
    values each DIMENSION_ORDER dimension currently registers (see
    context_packets.single_variable_conditions) -- computed here rather
    than hardcoded, since adding a legitimate new dimension value (e.g. a
    new named-model provenance label) is expected to grow this count, and
    a hardcoded literal would make that an unrelated test failure instead
    of the expected, calculable consequence it actually is.
    """
    n_conditions = sum(len(dims[d]["values"]) for d in DIMENSION_ORDER) + 1  # +1 neutral baseline
    return len(items) * n_conditions * len(EVALUATION_REGIMES)


def expected_context_pairwise_count(items):
    n_pairs = len(items) * (len(items) - 1) // 2
    return n_pairs * len(load_contrasts()) * 4 * len(EVALUATION_REGIMES)


def test_required_manifest_context_single_count_matches_dimension_sizes(dims, items):
    trials = all_required_trials(dims, items)
    single = [t for t in trials if t["type"] == "context_single"]
    assert len(single) == expected_context_single_count(dims, items)


def test_required_manifest_totals(dims, items):
    """Total = context_single (a function of dimension sizes -- see above)
    + context_pairwise (66 pairs x len(contrasts) x 4 cells x 2 regimes,
    structurally unaffected by dimension-value counts) + context_prompt
    (unaffected). Adding a new provenance value legitimately grows the
    total via context_single alone."""
    trials = all_required_trials(dims, items)
    expected = expected_context_single_count(dims, items) + expected_context_pairwise_count(items) + 264
    assert len(trials) == expected


def test_required_manifest_trial_ids_are_globally_unique(dims, items):
    trials = all_required_trials(dims, items)
    trial_ids = [t["trial_id"] for t in trials]
    assert len(trial_ids) == len(set(trial_ids))


def test_required_manifest_splits_evenly_by_evaluation_regime(dims, items):
    trials = all_required_trials(dims, items)
    counts = {regime: sum(1 for t in trials if t["evaluation_regime"] == regime) for regime in EVALUATION_REGIMES}
    assert counts["naturalistic"] == counts["text_only_invariance"] == len(trials) // 2


def test_required_manifest_pairwise_trials_are_all_forced_choice(dims, items):
    """The pairwise family's own structure (66 pairs x 13 contrasts x 4
    cells x 2 regimes) is entirely unaffected by the provenance dimension
    growing new values -- contrasts are explicit rows in
    data/context_contrasts.jsonl, not derived from "all values of a
    dimension". The required manifest must also never silently include
    tie-allowed trials -- that family lives only in the separate, optional
    tie-allowed file (see test_tie_allowed_family_is_the_same_size_but_separate)."""
    trials = all_required_trials(dims, items)
    pairwise = [t for t in trials if t["type"] == "context_pairwise"]
    assert len(pairwise) == expected_context_pairwise_count(items) == 6864
    assert all(t["choice_mode"] == "forced" for t in pairwise)


def test_tie_allowed_family_is_the_same_size_but_separate(dims, items):
    forced = []
    tie_allowed = []
    for regime in EVALUATION_REGIMES:
        forced += build_context_pairwise_trials(dims, items, regime)
        tie_allowed += build_tie_allowed_pairwise_trials(dims, items, regime)
    assert len(forced) == len(tie_allowed) == 6864
    assert all(t["choice_mode"] == "tie_allowed" for t in tie_allowed)
    assert set(t["trial_id"] for t in forced).isdisjoint(t["trial_id"] for t in tie_allowed)


def test_corpus_has_twelve_stories(items):
    assert len(items) == 12
    assert len({i["id"] for i in items}) == 12


class TestV1PilotUnaffected:
    """The v0.1 pilot manifest must stay pinned to its original 4 stories
    and 108 trials regardless of any v0.2 work in this repo."""

    def test_make_trials_still_pinned_to_original_four_stories(self):
        import make_trials

        assert len(make_trials.PILOT_STORY_IDS) == 4

    def test_v01_and_v02_use_disjoint_trials_files(self):
        import make_trials
        import context_trials

        assert make_trials.TRIALS_FILE != context_trials.CONTEXT_TRIALS_FILE
