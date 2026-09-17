"""Tests for the 4-cell counterbalanced pairwise block builder."""

import pytest

from context_packets import load_dimensions
from context_contrasts import build_contrast_block, load_contrasts


@pytest.fixture(scope="module")
def dimensions():
    return load_dimensions()


@pytest.fixture(scope="module")
def a_contrast():
    return load_contrasts()[0]


STORY_1 = {"id": "story_one", "path": None}
STORY_2 = {"id": "story_two", "path": None}


def _build(dimensions, contrast, choice_mode="forced", evaluation_regime="naturalistic", monkeypatch=None):
    return build_contrast_block(dimensions, contrast, STORY_1, STORY_2, evaluation_regime, choice_mode)


@pytest.fixture(autouse=True)
def _stub_story_loading(monkeypatch):
    # build_contrast_block loads story text from disk via story["path"]; stub
    # it so this test module doesn't depend on which corpus files exist.
    import context_contrasts

    monkeypatch.setattr(context_contrasts, "load_story", lambda path: f"TEXT[{path or 'unknown'}]")
    yield


def test_produces_exactly_four_cells(dimensions, a_contrast):
    cells = _build(dimensions, a_contrast)
    assert len(cells) == 4
    combos = {(c["assignment"], c["position"]) for c in cells}
    assert combos == {
        ("forward", "story1_as_a"),
        ("forward", "story2_as_a"),
        ("flipped", "story1_as_a"),
        ("flipped", "story2_as_a"),
    }


def test_all_cells_share_one_block_id(dimensions, a_contrast):
    cells = _build(dimensions, a_contrast)
    assert len({c["block_id"] for c in cells}) == 1


def test_trial_ids_are_unique_within_block(dimensions, a_contrast):
    cells = _build(dimensions, a_contrast)
    assert len({c["trial_id"] for c in cells}) == 4


def test_choice_mode_is_recorded_on_every_cell_and_baked_into_block_id(dimensions, a_contrast):
    forced = _build(dimensions, a_contrast, choice_mode="forced")
    tie_allowed = _build(dimensions, a_contrast, choice_mode="tie_allowed")

    assert all(c["choice_mode"] == "forced" for c in forced)
    assert all(c["choice_mode"] == "tie_allowed" for c in tie_allowed)

    forced_block_ids = {c["block_id"] for c in forced}
    tie_block_ids = {c["block_id"] for c in tie_allowed}
    assert forced_block_ids.isdisjoint(tie_block_ids)


def test_evaluation_regime_is_baked_into_block_id(dimensions, a_contrast):
    naturalistic = _build(dimensions, a_contrast, evaluation_regime="naturalistic")
    invariance = _build(dimensions, a_contrast, evaluation_regime="text_only_invariance")
    assert {c["block_id"] for c in naturalistic}.isdisjoint({c["block_id"] for c in invariance})


def test_forced_cells_use_forced_prompt_wording(dimensions, a_contrast):
    forced = _build(dimensions, a_contrast, choice_mode="forced")
    tie_allowed = _build(dimensions, a_contrast, choice_mode="tie_allowed")
    assert all('"tie"' not in c["prompt"] for c in forced)
    assert all('"tie"' in c["prompt"] for c in tie_allowed)


def test_same_story_twice_is_rejected(dimensions, a_contrast):
    with pytest.raises(ValueError):
        build_contrast_block(dimensions, a_contrast, STORY_1, STORY_1, "naturalistic")


def test_position_pairs_share_assignment_and_differ_only_in_position(dimensions, a_contrast):
    cells = _build(dimensions, a_contrast)
    by_key = {(c["assignment"], c["position"]): c for c in cells}
    forward_a = by_key[("forward", "story1_as_a")]
    forward_b = by_key[("forward", "story2_as_a")]
    assert forward_a["context_a"]["value"] == forward_b["context_b"]["value"]
    assert forward_a["context_b"]["value"] == forward_b["context_a"]["value"]


# ---------------------------------------------------------------------------
# rubric threading -- "full" is the only rubric now (an "excerpt" rubric
# used to live here for the standalone context-controllability experiment;
# that experiment now uses its own dedicated plain-A/B prompt/response
# format and no longer calls build_contrast_block() at all -- see
# controllability_trials.py).
# ---------------------------------------------------------------------------

def test_default_rubric_is_full_and_block_id_is_unaffected(dimensions, a_contrast):
    default_cells = _build(dimensions, a_contrast)
    explicit_full_cells = build_contrast_block(dimensions, a_contrast, STORY_1, STORY_2, "naturalistic", "forced", "full")
    assert [c["block_id"] for c in default_cells] == [c["block_id"] for c in explicit_full_cells]
    assert all(c["rubric"] == "full" for c in default_cells)
