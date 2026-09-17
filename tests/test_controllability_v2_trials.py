"""Tests for controllability_v2_trials.py: prompt wording, contrast schema,
counterbalance/superblock structure, and manifest sizes.

These operate on freshly-generated in-memory trials (via the module's own
build_* functions against the real data/items.jsonl stories), never on the
possibly-stale files on disk -- so a stale data/controllability_v2_trials.jsonl
left over from a previous run can never make these tests pass or fail
incorrectly.
"""

import itertools

import pytest

import controllability_v2_trials as t
from context_trials import load_items, story_pairs


@pytest.fixture(scope="module")
def items():
    return load_items()


@pytest.fixture(scope="module")
def contrasts():
    return t.load_contrasts()


@pytest.fixture(scope="module")
def treatment_trials(contrasts, items):
    trials = t.build_all_treatment_trials(contrasts, items, t.PRIMARY_INSTRUCTION_CONDITIONS)
    for trial in trials:
        trial["experiment_id"] = t.EXPERIMENT_ID
    return trials


@pytest.fixture(scope="module")
def minimal_trials(contrasts, items):
    trials = t.build_all_treatment_trials(contrasts, items, [t.MINIMAL_INSTRUCTION_CONDITION])
    for trial in trials:
        trial["experiment_id"] = t.MINIMAL_EXPERIMENT_ID
    return trials


@pytest.fixture(scope="module")
def baseline_trials(items):
    return t.build_baseline_trials(items)


# ---------------------------------------------------------------------------
# Contrasts: exactly five, well-formed, canonical wording
# ---------------------------------------------------------------------------

class TestContrasts:
    def test_exactly_five_contrasts(self, contrasts):
        assert len(contrasts) == 5
        assert {c["contrast_id"] for c in contrasts} == t.REQUIRED_CONTRAST_IDS

    def test_every_contrast_has_the_required_fields(self, contrasts):
        for contrast in contrasts:
            assert t.REQUIRED_CONTRAST_FIELDS <= set(contrast)

    def test_a_is_always_the_declared_favourable_direction(self, contrasts):
        for contrast in contrasts:
            assert contrast["expected_direction"] == "a"

    def test_canonical_wording_matches_v1(self, contrasts):
        """v2 must retain v1's exact frozen a_clause/b_clause wording --
        never regenerated or paraphrased."""
        import controllability_trials as v1

        v1_contrasts = {c["id"]: c for c in v1.load_contrasts(v1.CONTRASTS_FILE)}
        for contrast in contrasts:
            v1_contrast = v1_contrasts[contrast["contrast_id"]]
            assert contrast["a_clause"] == v1_contrast["a_clause"]
            assert contrast["b_clause"] == v1_contrast["b_clause"]

    def test_assert_contrasts_are_well_formed_rejects_duplicate_id(self, contrasts):
        broken = contrasts + [dict(contrasts[0])]
        with pytest.raises(ValueError):
            t.assert_contrasts_are_well_formed(broken)

    def test_assert_contrasts_are_well_formed_rejects_missing_field(self, contrasts):
        broken = [dict(c) for c in contrasts]
        del broken[0]["a_clause"]
        with pytest.raises(ValueError):
            t.assert_contrasts_are_well_formed(broken)


# ---------------------------------------------------------------------------
# Frozen wording: question and instruction strings, identical everywhere
# ---------------------------------------------------------------------------

class TestFrozenWording:
    def test_exact_question_wording(self):
        assert t.QUESTION == (
            "Which passage is better written overall? If they're close, choose the "
            "one you think is slightly better written. Please answer A or B."
        )

    def test_exact_instruction_strings(self):
        assert t.MATCHED_CONTROL_INSTRUCTION == "For this comparison, please read both passages carefully before making your choice."
        assert t.TEXT_ONLY_INSTRUCTION == "For this comparison, please read both passages carefully and judge only the writing itself."

    def test_question_is_identical_across_every_treatment_prompt(self, treatment_trials):
        assert all(trial["prompt"].rstrip().endswith(t.QUESTION) for trial in treatment_trials)

    def test_question_is_identical_in_the_baseline_too(self, baseline_trials):
        assert all(trial["prompt"].rstrip().endswith(t.QUESTION) for trial in baseline_trials)

    def test_matched_control_and_text_only_differ_only_in_the_instruction_sentence(self, treatment_trials):
        by_cell = {}
        for trial in treatment_trials:
            key = (trial["superblock_id"], trial["assignment"], trial["position"])
            by_cell.setdefault(key, {})[trial["instruction_condition"]] = trial["prompt"]

        checked = 0
        for cells in by_cell.values():
            if set(cells) != {"matched_control", "text_only"}:
                continue
            checked += 1
            control_prompt = cells["matched_control"].replace(t.MATCHED_CONTROL_INSTRUCTION, "\x00")
            text_only_prompt = cells["text_only"].replace(t.TEXT_ONLY_INSTRUCTION, "\x00")
            assert control_prompt == text_only_prompt
        assert checked == 1320  # 330 superblocks x 4 (assignment, position) cells each


# ---------------------------------------------------------------------------
# Story pairs / counts
# ---------------------------------------------------------------------------

class TestCounts:
    def test_exactly_66_unordered_story_pairs(self, items):
        pairs = story_pairs(items)
        assert len(pairs) == 66
        assert len(items) == 12
        assert len(set(itertools.combinations(sorted(i["id"] for i in items), 2))) == 66

    def test_exactly_2640_primary_treatment_prompts(self, treatment_trials):
        assert len(treatment_trials) == 2640

    def test_exactly_132_baseline_prompts(self, baseline_trials):
        assert len(baseline_trials) == 132

    def test_exactly_8_cells_per_treatment_superblock(self, treatment_trials):
        by_superblock = t.assert_superblocks_are_well_formed(treatment_trials)
        assert len(by_superblock) == 330
        assert all(len(cells) == 8 for cells in by_superblock.values())

    def test_exactly_four_assignment_position_cells_per_instruction_condition(self, treatment_trials):
        by_block = t.assert_trials_are_well_formed(treatment_trials, t.PRIMARY_INSTRUCTION_CONDITIONS)
        assert len(by_block) == 660
        for cells in by_block.values():
            assert len(cells) == 4
            assert {(c["assignment"], c["position"]) for c in cells} == t.REQUIRED_CELLS
            assert len({c["instruction_condition"] for c in cells}) == 1


# ---------------------------------------------------------------------------
# Identity / hashing
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_unique_trial_ids_treatment(self, treatment_trials):
        ids = [tr["trial_id"] for tr in treatment_trials]
        assert len(ids) == len(set(ids))

    def test_unique_trial_ids_baseline(self, baseline_trials):
        ids = [tr["trial_id"] for tr in baseline_trials]
        assert len(ids) == len(set(ids))

    def test_correct_prompt_hashes(self, treatment_trials, baseline_trials):
        for trial in treatment_trials + baseline_trials:
            assert trial["prompt_sha256"] == t.sha256_hex(trial["prompt"])

    def test_assert_trials_are_well_formed_rejects_a_tampered_hash(self, treatment_trials):
        tampered = [dict(tr) for tr in treatment_trials[:4]]
        tampered[0]["prompt_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            t.assert_trials_are_well_formed(tampered, t.PRIMARY_INSTRUCTION_CONDITIONS)


# ---------------------------------------------------------------------------
# Optional minimal-instruction manifest
# ---------------------------------------------------------------------------

class TestMinimalInstructionManifest:
    def test_minimal_trials_are_absent_from_the_primary_manifest(self, treatment_trials, minimal_trials):
        primary_ids = {tr["trial_id"] for tr in treatment_trials}
        minimal_ids = {tr["trial_id"] for tr in minimal_trials}
        assert primary_ids.isdisjoint(minimal_ids)

    def test_minimal_trials_carry_a_distinct_experiment_id(self, treatment_trials, minimal_trials):
        assert {tr["experiment_id"] for tr in treatment_trials} == {t.EXPERIMENT_ID}
        assert {tr["experiment_id"] for tr in minimal_trials} == {t.MINIMAL_EXPERIMENT_ID}

    def test_minimal_instruction_has_no_added_sentence(self, minimal_trials):
        for trial in minimal_trials:
            assert t.MATCHED_CONTROL_INSTRUCTION not in trial["prompt"]
            assert t.TEXT_ONLY_INSTRUCTION not in trial["prompt"]

    def test_minimal_manifest_size(self, minimal_trials):
        assert len(minimal_trials) == 66 * 5 * 4  # 1 instruction condition x 4 cells


# ---------------------------------------------------------------------------
# Baseline: no contextual clauses, matched_control instruction only
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_baseline_contains_no_contextual_clauses(self, baseline_trials):
        for trial in baseline_trials:
            assert "assignment" not in trial
            assert "contrast_id" not in trial
            assert "context_a" not in trial and "context_b" not in trial

    def test_baseline_uses_matched_control_instruction(self, baseline_trials):
        for trial in baseline_trials:
            assert trial["instruction_condition"] == "matched_control"
            assert t.MATCHED_CONTROL_INSTRUCTION in trial["prompt"]
            assert t.TEXT_ONLY_INSTRUCTION not in trial["prompt"]

    def test_baseline_blocks_have_exactly_two_positions(self, baseline_trials):
        by_block = t.assert_baseline_trials_are_well_formed(baseline_trials)
        assert len(by_block) == 66
        assert all(len(cells) == 2 for cells in by_block.values())
