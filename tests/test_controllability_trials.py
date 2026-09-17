"""Tests for the standalone context-controllability experiment's manifest
generator (controllability_trials.py).

Integration-level, like test_manifest_generation.py / test_llm_provenance_trials.py:
reads the real data/context_dimensions.jsonl, data/items.jsonl, and
data/controllability_contrasts.jsonl. No file under data/ is written by
these tests -- main() is only ever exercised with the output paths
monkeypatched to a tmp_path.
"""

import copy
import json

import pytest

import controllability_trials as ct
import run_batch as rb
from context_analysis_common import EXCERPT_RATING_FIELDS, RATING_FIELDS
from context_packets import load_dimensions
from context_trials import EVALUATION_REGIMES, build_context_pairwise_trials, load_items
from run_trial import parse_and_validate


@pytest.fixture(scope="module")
def dims():
    return load_dimensions()


@pytest.fixture(scope="module")
def items():
    return load_items()


@pytest.fixture(scope="module")
def contrasts():
    return ct.load_contrasts(ct.CONTRASTS_FILE)


# ---------------------------------------------------------------------------
# 1. The dedicated contrast file itself
# ---------------------------------------------------------------------------

def test_contrasts_file_has_exactly_the_five_required_contrasts(contrasts):
    assert {c["id"] for c in contrasts} == ct.REQUIRED_CONTRAST_IDS
    assert len(contrasts) == 5


def test_each_contrast_isolates_a_distinct_dimension(contrasts):
    dimensions_used = [c["dimension"] for c in contrasts]
    assert len(dimensions_used) == len(set(dimensions_used))
    assert set(dimensions_used) == {"provenance", "source_venue", "reception", "user_opinion", "editing_status"}


def test_contrasts_use_registered_dimension_values(dims, contrasts):
    for c in contrasts:
        values = dims[c["dimension"]]["values"]
        assert c["a"] in values
        assert c["b"] in values


def test_every_contrast_has_a_display_name():
    for contrast_id in ct.REQUIRED_CONTRAST_IDS:
        assert contrast_id in ct.CONTRAST_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# 2. assert_contrasts_are_well_formed: fail loudly on malformed input
# ---------------------------------------------------------------------------

def test_assert_contrasts_are_well_formed_accepts_the_real_file(dims, contrasts):
    ct.assert_contrasts_are_well_formed(dims, contrasts)  # must not raise


def test_rejects_duplicate_contrast_id(dims, contrasts):
    broken = contrasts + [dict(contrasts[0])]
    with pytest.raises(ValueError, match="Duplicate contrast id"):
        ct.assert_contrasts_are_well_formed(dims, broken)


def test_rejects_two_contrasts_sharing_a_dimension(dims, contrasts):
    """Each trial must test only one context type -- no combined-context
    trials, so two contrasts may never isolate the same dimension."""
    duplicate_dimension = copy.deepcopy(contrasts[0])
    duplicate_dimension["id"] = "some_other_id"
    broken = contrasts + [duplicate_dimension]
    with pytest.raises(ValueError, match="already used by another contrast"):
        ct.assert_contrasts_are_well_formed(dims, broken)


def test_rejects_unregistered_dimension(dims, contrasts):
    broken = copy.deepcopy(contrasts)
    broken[0]["dimension"] = "not_a_real_dimension"
    with pytest.raises(ValueError, match="unregistered dimension"):
        ct.assert_contrasts_are_well_formed(dims, broken)


def test_rejects_unregistered_value(dims, contrasts):
    broken = copy.deepcopy(contrasts)
    broken[0]["b"] = "not_a_real_value"
    with pytest.raises(ValueError, match="unregistered value"):
        ct.assert_contrasts_are_well_formed(dims, broken)


def test_rejects_missing_or_substituted_contrast_ids(dims, contrasts):
    with pytest.raises(ValueError, match="Expected exactly the 5 contrasts"):
        ct.assert_contrasts_are_well_formed(dims, contrasts[:4])


# ---------------------------------------------------------------------------
# 3. assert_trials_are_well_formed: fail loudly on a malformed treatment manifest
# ---------------------------------------------------------------------------

def _one_real_block(dims, items, contrasts):
    trials = build_context_pairwise_trials(
        dims, items[:2], "naturalistic", choice_mode="forced", contrasts=contrasts[:1], rubric=ct.RUBRIC
    )
    for t in trials:
        t["experiment_id"] = ct.EXPERIMENT_ID
    assert len(trials) == 4
    return trials


def test_assert_trials_are_well_formed_accepts_one_real_block(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    by_block = ct.assert_trials_are_well_formed(trials)
    assert len(by_block) == 1
    assert len(list(by_block.values())[0]) == 4


def test_rejects_duplicate_trial_id(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    trials.append(dict(trials[0]))
    with pytest.raises(ValueError, match="Duplicate trial_id"):
        ct.assert_trials_are_well_formed(trials)


def test_rejects_wrong_type(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    trials[0] = {**trials[0], "type": "context_single"}
    with pytest.raises(ValueError, match="expected type 'context_pairwise'"):
        ct.assert_trials_are_well_formed(trials)


def test_rejects_wrong_choice_mode(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    trials[0] = {**trials[0], "choice_mode": "tie_allowed"}
    with pytest.raises(ValueError, match="expected choice_mode 'forced'"):
        ct.assert_trials_are_well_formed(trials)


def test_rejects_wrong_rubric(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    trials[0] = {**trials[0], "rubric": "full"}
    with pytest.raises(ValueError, match="expected rubric 'excerpt'"):
        ct.assert_trials_are_well_formed(trials)


def test_rejects_unexpected_contrast_id(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    trials[0] = {**trials[0], "contrast_id": "provenance_claude_vs_human"}  # general-benchmark contrast, not ours
    with pytest.raises(ValueError, match="unexpected contrast_id"):
        ct.assert_trials_are_well_formed(trials)


def test_rejects_block_with_fewer_than_four_cells(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)[:3]
    with pytest.raises(ValueError, match="expected exactly 4"):
        ct.assert_trials_are_well_formed(trials)


def test_rejects_block_missing_a_required_cell(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    trials[3] = {**trials[3], "assignment": trials[0]["assignment"], "position": trials[0]["position"],
                 "trial_id": trials[3]["trial_id"] + "_dup"}
    with pytest.raises(ValueError, match="expected \\["):
        ct.assert_trials_are_well_formed(trials)


# ---------------------------------------------------------------------------
# 4. Manifest sizes: calculated, not hardcoded -- pinned to the spec's own
#    documented expectation too.
# ---------------------------------------------------------------------------

def test_treatment_manifest_matches_spec_expected_counts(dims, items, contrasts):
    n_pairs = len(items) * (len(items) - 1) // 2
    all_trials = []
    for regime in EVALUATION_REGIMES:
        trials = build_context_pairwise_trials(
            dims, items, regime, choice_mode="forced", contrasts=contrasts, rubric=ct.RUBRIC
        )
        for t in trials:
            t["experiment_id"] = ct.EXPERIMENT_ID
        all_trials.extend(trials)

    n_blocks = n_pairs * len(contrasts) * len(EVALUATION_REGIMES)
    assert (n_pairs, n_blocks, len(all_trials)) == (66, 660, 2640)  # spec's documented expectation
    by_block = ct.assert_trials_are_well_formed(all_trials)
    assert len(by_block) == n_blocks


def test_baseline_manifest_matches_spec_expected_counts(items):
    baseline_trials = ct.build_baseline_trials(items)
    n_pairs = len(items) * (len(items) - 1) // 2
    assert (n_pairs, len(baseline_trials)) == (66, 132)  # spec's documented expectation
    by_block = ct.assert_baseline_trials_are_well_formed(baseline_trials)
    assert len(by_block) == n_pairs


# ---------------------------------------------------------------------------
# 5. main(): end-to-end, output redirected to tmp_path
# ---------------------------------------------------------------------------

def test_main_writes_2640_treatment_and_132_baseline_trials(tmp_path, monkeypatch):
    treatment_path = tmp_path / "controllability_trials.jsonl"
    baseline_path = tmp_path / "controllability_baseline_trials.jsonl"
    monkeypatch.setattr(ct, "TRIALS_FILE", str(treatment_path))
    monkeypatch.setattr(ct, "BASELINE_TRIALS_FILE", str(baseline_path))

    ct.main()

    treatment_lines = treatment_path.read_text().splitlines()
    baseline_lines = baseline_path.read_text().splitlines()
    assert len(treatment_lines) == 2640
    assert len(baseline_lines) == 132

    treatment_trials = [json.loads(line) for line in treatment_lines]
    assert all(t["experiment_id"] == ct.EXPERIMENT_ID for t in treatment_trials)
    assert all(t["rubric"] == "excerpt" for t in treatment_trials)
    assert all(t["type"] == "context_pairwise" and t["choice_mode"] == "forced" for t in treatment_trials)

    baseline_trials = [json.loads(line) for line in baseline_lines]
    assert all(t["experiment_id"] == ct.BASELINE_EXPERIMENT_ID for t in baseline_trials)
    assert all("assignment" not in t for t in baseline_trials)
    assert all(t["evaluation_regime"] == "naturalistic" for t in baseline_trials)


def test_broad_general_benchmark_is_unaffected_by_this_experiment(dims, items):
    """Omitting contrasts/rubric still generates the full general-benchmark
    contrast set with the "full" rubric -- this experiment's own contrasts
    file and rubric never leak into the shared default."""
    from context_contrasts import load_contrasts as load_general_contrasts

    general = build_context_pairwise_trials(dims, items[:2], "naturalistic")
    assert len(general) == len(load_general_contrasts()) * 4
    assert all(t["rubric"] == "full" for t in general)
    assert all("experiment_id" not in t for t in general)


# ---------------------------------------------------------------------------
# 6. Manual-inspection checklist, automated: fixed prose, correct context
#    and display-position swaps, one shared block_id, naturalistic vs
#    text_only regime parity.
# ---------------------------------------------------------------------------

def test_one_block_satisfies_the_full_manual_inspection_checklist(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    by_cell = {(t["assignment"], t["position"]): t for t in trials}
    assert set(by_cell) == ct.REQUIRED_CELLS
    assert len({t["block_id"] for t in trials}) == 1

    story_1_id = trials[0]["story_1_id"]
    story_2_id = trials[0]["story_2_id"]
    assert len({t["story_1_id"] for t in trials}) == 1
    assert len({t["story_2_id"] for t in trials}) == 1
    assert story_1_id != story_2_id

    for (assignment, position), t in by_cell.items():
        if position == "story1_as_a":
            assert t["story_a_id"] == story_1_id and t["story_b_id"] == story_2_id
        else:
            assert t["story_a_id"] == story_2_id and t["story_b_id"] == story_1_id

    contrast = contrasts[0]
    for (assignment, position), t in by_cell.items():
        story_1_value = contrast["a"] if assignment == "forward" else contrast["b"]
        story_2_value = contrast["b"] if assignment == "forward" else contrast["a"]
        expected_a_value = story_1_value if position == "story1_as_a" else story_2_value
        expected_b_value = story_2_value if position == "story1_as_a" else story_1_value
        assert t["context_a"]["value"] == expected_a_value
        assert t["context_b"]["value"] == expected_b_value

    def story_bodies(prompt):
        return prompt.split("I'm trying to make up my mind.", 1)[1]

    for position in ("story1_as_a", "story2_as_a"):
        bodies = {story_bodies(by_cell[(assignment, position)]["prompt"]) for assignment in ("forward", "flipped")}
        assert len(bodies) == 1, f"story prose at position={position} must be identical across forward/flipped"

    reference_bodies = {
        by_cell[("forward", "story1_as_a")]["prompt"].split("I'm trying to make up my mind.", 1)[1],
        by_cell[("forward", "story2_as_a")]["prompt"].split("I'm trying to make up my mind.", 1)[1],
    }
    for t in trials:
        assert story_bodies(t["prompt"]) in reference_bodies


def test_naturalistic_and_text_only_blocks_carry_identical_context_and_only_the_instruction_differs(dims, items, contrasts):
    naturalistic_trials = build_context_pairwise_trials(
        dims, items[:2], "naturalistic", choice_mode="forced", contrasts=contrasts[:1], rubric=ct.RUBRIC
    )
    invariance_trials = build_context_pairwise_trials(
        dims, items[:2], "text_only_invariance", choice_mode="forced", contrasts=contrasts[:1], rubric=ct.RUBRIC
    )
    by_key_nat = {(t["assignment"], t["position"]): t for t in naturalistic_trials}
    by_key_inv = {(t["assignment"], t["position"]): t for t in invariance_trials}

    for key in by_key_nat:
        nat, inv = by_key_nat[key], by_key_inv[key]
        assert nat["intro"] == inv["intro"]
        assert nat["context_a"] == inv["context_a"]
        assert nat["context_b"] == inv["context_b"]
        assert "Judge only the two pieces of prose" not in nat["prompt"]
        assert "Judge only the two pieces of prose" in inv["prompt"]
        # everything up to the invariance instruction sentence is identical
        nat_before = nat["prompt"].split("For each category")[0]
        inv_before = inv["prompt"].split("Judge only the two pieces of prose")[0]
        assert nat_before == inv_before


def test_baseline_block_has_no_context_and_only_two_positions(items):
    baseline_trials = ct.build_baseline_trials(items[:2])
    assert len(baseline_trials) == 2
    by_position = {t["position"]: t for t in baseline_trials}
    assert set(by_position) == ct.BASELINE_POSITIONS
    # story_pairs() sorts by id, so story_1/story_2 identity isn't
    # necessarily items[0]/items[1] in file order -- derive it the same way.
    story_1_id, story_2_id = sorted(item["id"] for item in items[:2])
    assert by_position["story1_as_a"]["story_a_id"] == story_1_id
    assert by_position["story1_as_a"]["story_b_id"] == story_2_id
    assert by_position["story2_as_a"]["story_a_id"] == story_2_id
    assert by_position["story2_as_a"]["story_b_id"] == story_1_id
    assert len({t["block_id"] for t in baseline_trials}) == 1
    for t in baseline_trials:
        assert "was generated by" not in t["prompt"]
        assert "was written by" not in t["prompt"]


# ---------------------------------------------------------------------------
# 7. Runner integration: reuse run_batch.py unmodified.
# ---------------------------------------------------------------------------

def test_run_batch_limit_1_selects_exactly_one_complete_treatment_block(dims, items, contrasts):
    trials = build_context_pairwise_trials(
        dims, items, "naturalistic", choice_mode="forced", contrasts=contrasts, rubric=ct.RUBRIC
    )
    selected = rb.select_trials(trials, None, None, None, None, None, limit=1)
    assert len(selected) == 4
    assert len({t["block_id"] for t in selected}) == 1


def test_run_batch_limit_2_selects_exactly_two_complete_treatment_blocks(dims, items, contrasts):
    trials = build_context_pairwise_trials(
        dims, items, "naturalistic", choice_mode="forced", contrasts=contrasts, rubric=ct.RUBRIC
    )
    selected = rb.select_trials(trials, None, None, None, None, None, limit=2)
    assert len(selected) == 8
    block_ids = {t["block_id"] for t in selected}
    assert len(block_ids) == 2
    for bid in block_ids:
        assert len([t for t in selected if t["block_id"] == bid]) == 4


def test_run_batch_treats_baseline_trials_as_singletons_not_blocks(items):
    """The baseline manifest has 2-cell blocks, not 4-cell blocks --
    run_batch.py's block grouping keys purely on block_id/type, so this
    just confirms --limit still selects whole baseline pairs together."""
    baseline_trials = ct.build_baseline_trials(items)
    selected = rb.select_trials(baseline_trials, None, None, None, None, None, limit=1)
    assert len(selected) == 2
    assert len({t["block_id"] for t in selected}) == 1


def test_treatment_trials_pass_existing_forced_choice_validation_with_no_new_validator(dims, items, contrasts):
    trials = _one_real_block(dims, items, contrasts)
    well_formed_response = json.dumps(
        {"prose_style": "A", "characterization": "B", "originality": "A",
         "narrative_effectiveness": "B", "overall_quality": "A"}
    )
    for t in trials:
        parsed, error = parse_and_validate(well_formed_response, t["type"], t.get("choice_mode"), t.get("rubric", "full"))
        assert error is None
        assert parsed["overall_quality"] == "A"


def test_baseline_trials_pass_existing_forced_choice_validation_with_no_new_validator(items):
    baseline_trials = ct.build_baseline_trials(items[:2])
    well_formed_response = json.dumps(
        {"prose_style": "A", "characterization": "B", "originality": "A",
         "narrative_effectiveness": "B", "overall_quality": "A"}
    )
    for t in baseline_trials:
        parsed, error = parse_and_validate(well_formed_response, t["type"], t.get("choice_mode"), t.get("rubric", "full"))
        assert error is None
