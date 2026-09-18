"""Tests for controllability_v2_recovery.py: Wave 1 planned-id enumeration,
classification of raw Wave 1 rows (valid/unresolved/duplicate/unknown/
malformed), the recovery plan (never a new replicate), the deterministic
merge (Wave 1 valid > recovery, never double-counted), stratified validation
sampling, and validation agreement/context-effect diagnostics.

No model API calls happen anywhere in this file or in the module under test.
"""

import pytest

import controllability_v2_recovery as rec
from controllability_v2_execution import make_observation_id
from controllability_v2_trials import EXPERIMENT_ID

EVALUATOR_ID = "deepseek__deepseek-flash__low"


def _treatment_trial(trial_id, block_id, superblock_id, contrast_id="c1", instruction_condition="matched_control",
                      assignment="forward", position="story1_as_a", s1="s1", s2="s2"):
    story_a_id, story_b_id = (s1, s2) if position == "story1_as_a" else (s2, s1)
    return {
        "trial_id": trial_id, "block_id": block_id, "superblock_id": superblock_id,
        "contrast_id": contrast_id, "instruction_condition": instruction_condition,
        "assignment": assignment, "position": position,
        "story_1_id": s1, "story_2_id": s2, "story_a_id": story_a_id, "story_b_id": story_b_id,
        "prompt": f"prompt for {trial_id}",
    }


def _baseline_trial(trial_id, block_id, instruction_condition="matched_control"):
    return {"trial_id": trial_id, "block_id": block_id, "instruction_condition": instruction_condition,
            "prompt": f"prompt for {trial_id}"}


@pytest.fixture
def small_design():
    treatment_trials = [
        _treatment_trial("t1", "b1", "sb1"),
        _treatment_trial("t2", "b1", "sb1", position="story2_as_a"),
    ]
    baseline_trials = [
        _baseline_trial("base1", "bb1"),
        _baseline_trial("base2", "bb2"),
    ]
    planned_index = rec.build_wave1_planned_index(treatment_trials, baseline_trials, EVALUATOR_ID,
                                                   treatment_replicate_count=2, baseline_replicate_count=1)
    trials_by_id = {t["trial_id"]: t for t in treatment_trials + baseline_trials}
    return {"treatment_trials": treatment_trials, "baseline_trials": baseline_trials,
            "planned_index": planned_index, "trials_by_id": trials_by_id}


def _oid(trial_id, replicate_number):
    return make_observation_id(EXPERIMENT_ID, EVALUATOR_ID, trial_id, replicate_number)


def _resolved_row(oid, trial_id, response="A"):
    return {
        "observation_id": oid, "trial_id": trial_id, "parsing_status": "resolved",
        "first_valid_response": {"overall_quality": response},
        "attempts": [{"response_text": response, "reasoning_tokens": 42}],
    }


def _unresolved_row(oid, trial_id):
    return {
        "observation_id": oid, "trial_id": trial_id, "parsing_status": "unresolved",
        "first_valid_response": None,
        "attempts": [{"response_text": "I cannot decide", "reasoning_tokens": 10}],
    }


# ---------------------------------------------------------------------------
# Planned-id enumeration
# ---------------------------------------------------------------------------

class TestPlannedIndex:
    def test_planned_index_has_treatment_times_replicates_plus_baseline_times_replicates(self, small_design):
        # 2 treatment trials * 2 replicates + 2 baseline trials * 1 replicate = 6
        assert len(small_design["planned_index"]) == 6

    def test_every_planned_id_round_trips_through_make_observation_id(self, small_design):
        assert _oid("t1", 1) in small_design["planned_index"]
        assert _oid("t1", 2) in small_design["planned_index"]
        assert _oid("base1", 1) in small_design["planned_index"]
        assert small_design["planned_index"][_oid("t1", 1)] == {"trial_id": "t1", "replicate_number": 1, "unit_kind": "treatment"}

    def test_historical_wave1_max_output_tokens_is_still_the_original_512(self):
        """Wave 1's own frozen setting is untouched by the Wave 2 recovery
        addition -- RECOVERY_MAX_OUTPUT_TOKENS (4096) is a NEW value used
        only for recovery/validation/new-condition calls, never retroactively
        applied to (or recorded on) the historical Wave 1 study config."""
        from controllability_v2_study_config import load_study_config
        study_config = load_study_config()
        assert study_config["max_output_tokens"] == 512
        assert rec.RECOVERY_MAX_OUTPUT_TOKENS == 4096
        assert study_config["max_output_tokens"] != rec.RECOVERY_MAX_OUTPUT_TOKENS

    def test_new_baseline_text_only_manifest_yields_exactly_1320_planned_observations_at_10_replicates(self):
        from run_trial import load_trials
        from controllability_v2_study_config import load_study_config
        from controllability_v2_trials import BASELINE_TEXT_ONLY_TRIALS_FILE
        study_config = load_study_config()
        baseline_text_only_trials = load_trials(BASELINE_TEXT_ONLY_TRIALS_FILE)
        assert len(baseline_text_only_trials) == 132
        assert study_config["baseline_replicate_count"] == 10
        planned = rec.enumerate_planned_ids(
            baseline_text_only_trials, study_config["baseline_replicate_count"],
            study_config["primary_evaluator"]["evaluator_id"], "baseline_text_only",
        )
        assert len(planned) == 1320

    def test_real_frozen_manifest_counts_produce_the_expected_27720(self):
        """Sanity-check against the ACTUAL frozen v2 manifests (never
        modified by this module) -- 2640 treatment trials * 10 + 132
        baseline trials * 10 = 27720, matching the task's stated Wave 1
        planned-observation count."""
        from run_trial import load_trials
        from controllability_v2_study_config import load_study_config
        study_config = load_study_config()
        treatment_trials = load_trials(study_config["treatment_manifest_file"])
        baseline_trials = load_trials(study_config["baseline_manifest_file"])
        index = rec.build_wave1_planned_index(
            treatment_trials, baseline_trials, study_config["primary_evaluator"]["evaluator_id"],
            study_config["treatment_replicate_count"], study_config["baseline_replicate_count"],
        )
        assert len(index) == 27720


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassifyWave1Rows:
    def test_valid_rows_are_retained_and_never_flagged_unresolved(self, small_design):
        rows = [_resolved_row(_oid("t1", 1), "t1")]
        result = rec.classify_wave1_rows(rows, small_design["planned_index"])
        assert _oid("t1", 1) in result["valid_by_id"]
        assert _oid("t1", 1) not in result["unresolved_ids"]

    def test_missing_and_present_but_unresolved_rows_both_count_as_unresolved(self, small_design):
        rows = [_unresolved_row(_oid("t1", 1), "t1")]  # t1/r2, base1/r1, base2/r1 never even appear
        result = rec.classify_wave1_rows(rows, small_design["planned_index"])
        assert set(result["unresolved_ids"]) == set(small_design["planned_index"]) - set()  # nothing valid at all
        assert len(result["unresolved_ids"]) == 6

    def test_unknown_observation_id_is_reported_separately_never_silently_dropped(self, small_design):
        rows = [{"observation_id": "not-a-real-id", "trial_id": "ghost", "parsing_status": "resolved",
                 "first_valid_response": {"overall_quality": "A"}, "attempts": [{"response_text": "A"}]}]
        result = rec.classify_wave1_rows(rows, small_design["planned_index"])
        assert result["unknown_ids"] == ["not-a-real-id"]
        assert result["valid_by_id"] == {}

    def test_two_independent_valid_rows_for_the_same_id_are_flagged_as_duplicates(self, small_design):
        oid = _oid("t1", 1)
        rows = [_resolved_row(oid, "t1", "A"), _resolved_row(oid, "t1", "B")]
        result = rec.classify_wave1_rows(rows, small_design["planned_index"])
        assert oid in result["duplicate_valid_ids"]
        assert len(result["duplicate_valid_ids"][oid]) == 2
        assert oid not in result["valid_by_id"]

    def test_a_resolved_row_whose_response_text_does_not_reparse_is_malformed_not_valid(self, small_design):
        oid = _oid("t1", 1)
        row = {"observation_id": oid, "trial_id": "t1", "parsing_status": "resolved",
               "first_valid_response": {"overall_quality": "A"},
               "attempts": [{"response_text": "I cannot decide, it's ambiguous", "reasoning_tokens": 5}]}
        result = rec.classify_wave1_rows([row], small_design["planned_index"])
        assert oid in result["malformed_valid_ids"]
        assert oid not in result["valid_by_id"]
        assert oid in result["unresolved_ids"]

    def test_exact_counts_derived_from_raw_data_match_a_realistic_mixed_scenario(self, small_design):
        rows = [
            _resolved_row(_oid("t1", 1), "t1"),
            _resolved_row(_oid("t1", 2), "t1"),
            _unresolved_row(_oid("t2", 1), "t2"),
            _resolved_row(_oid("base1", 1), "base1"),
        ]
        result = rec.classify_wave1_rows(rows, small_design["planned_index"])
        assert len(result["valid_by_id"]) == 3
        assert len(result["unresolved_ids"]) == 3  # t2/r1 (present, unresolved), t2/r2, base2/r1 (missing)


class TestValidateWave1Consistency:
    def test_clean_classification_passes(self, small_design):
        result = rec.classify_wave1_rows([_resolved_row(_oid("t1", 1), "t1")], small_design["planned_index"])
        rec.validate_wave1_consistency(result)  # must not raise

    def test_unknown_ids_fail_validation(self, small_design):
        rows = [{"observation_id": "bogus", "trial_id": "x", "parsing_status": "resolved",
                 "first_valid_response": {"overall_quality": "A"}, "attempts": [{"response_text": "A"}]}]
        result = rec.classify_wave1_rows(rows, small_design["planned_index"])
        with pytest.raises(ValueError, match="trial identities do not match"):
            rec.validate_wave1_consistency(result)

    def test_duplicate_valid_answers_fail_validation(self, small_design):
        oid = _oid("t1", 1)
        result = rec.classify_wave1_rows([_resolved_row(oid, "t1", "A"), _resolved_row(oid, "t1", "B")], small_design["planned_index"])
        with pytest.raises(ValueError, match="Duplicate valid Wave 1 answers"):
            rec.validate_wave1_consistency(result)

    def test_malformed_valid_rows_fail_validation(self, small_design):
        oid = _oid("t1", 1)
        row = {"observation_id": oid, "trial_id": "t1", "parsing_status": "resolved",
               "first_valid_response": {"overall_quality": "A"},
               "attempts": [{"response_text": "not really an answer at all and ambiguous", "reasoning_tokens": 1}]}
        result = rec.classify_wave1_rows([row], small_design["planned_index"])
        with pytest.raises(ValueError, match="do not actually re-parse"):
            rec.validate_wave1_consistency(result)


# ---------------------------------------------------------------------------
# Loading raw files
# ---------------------------------------------------------------------------

class TestLoadWave1RawRows:
    def test_missing_source_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rec.load_wave1_raw_rows(str(tmp_path / "does_not_exist"))

    def test_missing_one_of_the_two_files_raises(self, tmp_path):
        (tmp_path / "treatment_raw.jsonl").write_text('{"observation_id": "x"}\n', encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            rec.load_wave1_raw_rows(str(tmp_path))

    def test_loads_both_files_when_present(self, tmp_path):
        (tmp_path / "treatment_raw.jsonl").write_text('{"observation_id": "t"}\n', encoding="utf-8")
        (tmp_path / "baseline_raw.jsonl").write_text('{"observation_id": "b"}\n', encoding="utf-8")
        treatment_rows, baseline_rows = rec.load_wave1_raw_rows(str(tmp_path))
        assert treatment_rows == [{"observation_id": "t"}]
        assert baseline_rows == [{"observation_id": "b"}]


# ---------------------------------------------------------------------------
# Recovery plan: never a new replicate
# ---------------------------------------------------------------------------

class TestBuildRecoveryPlan:
    def test_recovery_plan_covers_exactly_the_unresolved_ids(self, small_design):
        unresolved = [_oid("t1", 2), _oid("base2", 1)]
        plan = rec.build_recovery_plan(unresolved, small_design["planned_index"], small_design["trials_by_id"])
        assert {e["observation_id"] for e in plan} == set(unresolved)

    def test_recovery_plan_preserves_trial_id_and_replicate_number(self, small_design):
        unresolved = [_oid("t1", 2)]
        plan = rec.build_recovery_plan(unresolved, small_design["planned_index"], small_design["trials_by_id"])
        assert plan[0]["trial_id"] == "t1"
        assert plan[0]["replicate_number"] == 2
        assert plan[0]["unit_kind"] == "treatment"

    def test_recovery_plan_never_appears_twice_for_the_same_id(self, small_design):
        unresolved = [_oid("t1", 2), _oid("t1", 2)]  # pathological duplicate input
        plan = rec.build_recovery_plan(list(set(unresolved)), small_design["planned_index"], small_design["trials_by_id"])
        ids = [e["observation_id"] for e in plan]
        assert len(ids) == len(set(ids))

    def test_assert_recovery_plan_matches_planned_index_passes_for_a_valid_plan(self, small_design):
        unresolved = [_oid("t1", 2), _oid("base2", 1)]
        plan = rec.build_recovery_plan(unresolved, small_design["planned_index"], small_design["trials_by_id"])
        rec.assert_recovery_plan_matches_planned_index(plan, small_design["planned_index"])  # must not raise

    def test_assert_recovery_plan_rejects_a_fabricated_id_not_in_the_planned_index(self, small_design):
        plan = [{"observation_id": "not-planned", "trial_id": "t1", "replicate_number": 99, "unit_kind": "treatment"}]
        with pytest.raises(ValueError):
            rec.assert_recovery_plan_matches_planned_index(plan, small_design["planned_index"])

    def test_assert_recovery_plan_rejects_a_duplicated_id_within_the_plan(self, small_design):
        oid = _oid("t1", 2)
        plan = [
            {"observation_id": oid, "trial_id": "t1", "replicate_number": 2, "unit_kind": "treatment"},
            {"observation_id": oid, "trial_id": "t1", "replicate_number": 2, "unit_kind": "treatment"},
        ]
        with pytest.raises(ValueError):
            rec.assert_recovery_plan_matches_planned_index(plan, small_design["planned_index"])


class TestBuildRecoveryResultRow:
    def test_recovery_result_row_carries_the_required_provenance_metadata(self, small_design):
        plan_entry = rec.build_recovery_plan([_oid("t1", 2)], small_design["planned_index"], small_design["trials_by_id"])[0]
        observation = {"attempts": [], "total_attempts": 1, "first_attempt_status": "valid",
                       "first_valid_response": {"overall_quality": "A"}, "parsing_status": "resolved",
                       "refusal_status": False, "api_error_status": False}
        row = rec.build_recovery_result_row(plan_entry, {"provider": "deepseek"}, observation)
        assert row["collection_wave"] == "wave2_recovery"
        assert row["max_output_tokens"] == 4096
        assert row["source_wave"] == "wave1"
        assert row["source_run_id"] == "35355521243"
        assert row["trial_id"] == "t1"
        assert row["replicate_number"] == 2
        assert row["superblock_id"] == "sb1"  # treatment trial's own unit-id field
        assert row["observation_id"] == _oid("t1", 2)

    def test_baseline_recovery_result_row_carries_block_id_not_superblock_id(self, small_design):
        plan_entry = rec.build_recovery_plan([_oid("base2", 1)], small_design["planned_index"], small_design["trials_by_id"])[0]
        observation = {"attempts": [], "total_attempts": 1, "first_attempt_status": "valid",
                       "first_valid_response": {"overall_quality": "B"}, "parsing_status": "resolved",
                       "refusal_status": False, "api_error_status": False}
        row = rec.build_recovery_result_row(plan_entry, {}, observation)
        assert row["block_id"] == "bb2"
        assert "superblock_id" not in row


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMergeWave1AndRecovery:
    def test_wave1_valid_answer_is_preferred_over_a_recovery_answer(self, small_design):
        oid = _oid("t1", 1)
        wave1_valid = {oid: _resolved_row(oid, "t1", "A")}
        recovery_rows = [dict(_resolved_row(oid, "t1", "B"), result_source_would_be="wave2_recovery")]
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], wave1_valid, recovery_rows)
        assert merged[oid]["first_valid_response"]["overall_quality"] == "A"
        assert merged[oid]["result_source"] == "wave1_original"

    def test_recovery_answer_is_used_only_when_wave1_has_none(self, small_design):
        oid = _oid("t1", 1)
        recovery_rows = [_resolved_row(oid, "t1", "B")]
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], {}, recovery_rows)
        assert merged[oid]["first_valid_response"]["overall_quality"] == "B"
        assert merged[oid]["result_source"] == "wave2_recovery"

    def test_merged_dataset_never_contains_more_than_one_row_per_planned_id(self, small_design):
        oid = _oid("t1", 1)
        wave1_valid = {oid: _resolved_row(oid, "t1", "A")}
        recovery_rows = [_resolved_row(oid, "t1", "B")]  # would-be duplicate if merge were naive
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], wave1_valid, recovery_rows)
        assert len(merged) == 1

    def test_still_unresolved_after_recovery_is_simply_absent_from_the_merged_dataset(self, small_design):
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], {}, [])
        assert merged == {}
        assert diag["still_unresolved_after_recovery"] == 6

    def test_diagnostics_report_every_required_field(self, small_design):
        oid1, oid2 = _oid("t1", 1), _oid("t1", 2)
        wave1_valid = {oid1: _resolved_row(oid1, "t1", "A")}
        recovery_rows = [_resolved_row(oid2, "t1", "B"), _unresolved_row(_oid("base1", 1), "base1")]
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], wave1_valid, recovery_rows)
        assert diag["wave1_planned_total"] == 6
        assert diag["wave1_valid_retained"] == 1
        assert diag["wave1_unresolved"] == 5
        assert diag["recovery_attempted"] == 2
        assert diag["recovery_successfully_resolved"] == 1
        assert diag["still_unresolved_after_recovery"] == 4
        assert diag["duplicate_planned_observation_ids"] == []
        assert diag["final_completeness"] == pytest.approx(2 / 6)

    def test_duplicate_recovery_answers_for_one_id_are_reported_not_silently_picked(self, small_design):
        oid = _oid("t1", 1)
        recovery_rows = [_resolved_row(oid, "t1", "A"), _resolved_row(oid, "t1", "B")]
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], {}, recovery_rows)
        assert diag["duplicate_planned_observation_ids"] == [oid]
        assert oid in merged  # first one seen is kept deterministically

    def test_split_merged_rows_by_unit_kind(self, small_design):
        oid_t, oid_b = _oid("t1", 1), _oid("base1", 1)
        wave1_valid = {oid_t: _resolved_row(oid_t, "t1", "A"), oid_b: _resolved_row(oid_b, "base1", "A")}
        merged, _diag = rec.merge_wave1_and_recovery(small_design["planned_index"], wave1_valid, [])
        split = rec.split_merged_rows_by_unit_kind(merged, small_design["planned_index"])
        assert [r["observation_id"] for r in split["treatment"]] == [oid_t]
        assert [r["observation_id"] for r in split["baseline"]] == [oid_b]


# ---------------------------------------------------------------------------
# Validation sample
# ---------------------------------------------------------------------------

class TestSelectValidationSample:
    def test_sample_size_capped_at_available_rows(self, small_design):
        rows = [_resolved_row(_oid("t1", 1), "t1"), _resolved_row(_oid("t1", 2), "t1")]
        sample = rec.select_validation_sample(rows, small_design["planned_index"], sample_size=500, seed=1)
        assert len(sample) == 2

    def test_sample_is_deterministic_for_a_fixed_seed(self, small_design):
        rows = [_resolved_row(_oid("t1", 1), "t1"), _resolved_row(_oid("t1", 2), "t1"),
                _resolved_row(_oid("base1", 1), "base1"), _resolved_row(_oid("base2", 1), "base2")]
        sample_a = rec.select_validation_sample(rows, small_design["planned_index"], sample_size=2, seed=42)
        sample_b = rec.select_validation_sample(rows, small_design["planned_index"], sample_size=2, seed=42)
        assert [r["observation_id"] for r in sample_a] == [r["observation_id"] for r in sample_b]

    def test_sample_never_exceeds_requested_size(self, small_design):
        rows = [_resolved_row(_oid("t1", 1), "t1"), _resolved_row(_oid("t1", 2), "t1"),
                _resolved_row(_oid("base1", 1), "base1"), _resolved_row(_oid("base2", 1), "base2")]
        sample = rec.select_validation_sample(rows, small_design["planned_index"], sample_size=3, seed=7)
        assert len(sample) == 3

    def test_empty_input_returns_empty_sample(self, small_design):
        assert rec.select_validation_sample([], small_design["planned_index"], sample_size=500, seed=0) == []

    def test_reasoning_token_bin_boundaries(self):
        assert rec.reasoning_token_bin(None) == "unknown"
        assert rec.reasoning_token_bin(0) == "0"
        assert rec.reasoning_token_bin(99) == "1-99"
        assert rec.reasoning_token_bin(499) == "100-499"
        assert rec.reasoning_token_bin(500) == "500+"


class TestValidationPlanAndAgreement:
    def test_validation_observation_id_never_collides_with_a_real_planned_id(self, small_design):
        sample = [_resolved_row(_oid("t1", 1), "t1")]
        plan = rec.build_validation_plan(sample, small_design["trials_by_id"])
        assert plan[0]["validation_observation_id"] not in small_design["planned_index"]
        assert plan[0]["validation_observation_id"].endswith("::validation_dup")

    def test_agreement_is_true_when_the_rerun_matches_the_original(self, small_design):
        original = _resolved_row(_oid("t1", 1), "t1", "A")
        plan_entry = rec.build_validation_plan([original], small_design["trials_by_id"])[0]
        observation = {"attempts": [{"response_text": "A", "reasoning_tokens": 1}], "total_attempts": 1,
                       "first_attempt_status": "valid", "first_valid_response": {"overall_quality": "A"},
                       "parsing_status": "resolved", "refusal_status": False, "api_error_status": False}
        row = rec.build_validation_result_row(plan_entry, {}, observation)
        assert row["agrees_with_original"] is True

    def test_agreement_is_false_when_the_rerun_disagrees(self, small_design):
        original = _resolved_row(_oid("t1", 1), "t1", "A")
        plan_entry = rec.build_validation_plan([original], small_design["trials_by_id"])[0]
        observation = {"attempts": [{"response_text": "B", "reasoning_tokens": 1}], "total_attempts": 1,
                       "first_attempt_status": "valid", "first_valid_response": {"overall_quality": "B"},
                       "parsing_status": "resolved", "refusal_status": False, "api_error_status": False}
        row = rec.build_validation_result_row(plan_entry, {}, observation)
        assert row["agrees_with_original"] is False

    def test_compute_validation_agreement_overall_rate_and_bins(self):
        rows = [
            {"parsing_status": "resolved", "agrees_with_original": True, "original_reasoning_tokens": 10,
             "original_first_valid_response": {"overall_quality": "A"}, "first_valid_response": {"overall_quality": "A"}},
            {"parsing_status": "resolved", "agrees_with_original": False, "original_reasoning_tokens": 600,
             "original_first_valid_response": {"overall_quality": "A"}, "first_valid_response": {"overall_quality": "B"}},
        ]
        result = rec.compute_validation_agreement(rows)
        assert result["n_validation_pairs"] == 2
        assert result["agreement_rate"] == 0.5
        assert result["agreement_by_reasoning_token_bin"]["1-99"] == 1.0
        assert result["agreement_by_reasoning_token_bin"]["500+"] == 0.0
        assert result["aggregate_preference_shift"] == pytest.approx(0.5 - 1.0)

    def test_compute_validation_agreement_ignores_unresolved_rows(self):
        rows = [{"parsing_status": "unresolved", "agrees_with_original": None}]
        result = rec.compute_validation_agreement(rows)
        assert result["n_validation_pairs"] == 0
        assert result["agreement_rate"] is None

    def test_validation_never_enters_a_normal_merge(self, small_design):
        """A validation row's id is never in planned_index, so even if
        someone accidentally fed validation rows into merge_wave1_and_recovery
        as if they were recovery rows, they would be silently ignored --
        never merged into the primary dataset."""
        original = _resolved_row(_oid("t1", 1), "t1", "A")
        plan_entry = rec.build_validation_plan([original], small_design["trials_by_id"])[0]
        observation = {"attempts": [{"response_text": "B", "reasoning_tokens": 1}], "total_attempts": 1,
                       "first_attempt_status": "valid", "first_valid_response": {"overall_quality": "B"},
                       "parsing_status": "resolved", "refusal_status": False, "api_error_status": False}
        validation_row = rec.build_validation_result_row(plan_entry, {}, observation)
        validation_row["observation_id"] = validation_row["validation_observation_id"]  # if misused as a recovery row
        merged, diag = rec.merge_wave1_and_recovery(small_design["planned_index"], {}, [validation_row])
        assert merged == {}
        assert diag["recovery_successfully_resolved"] == 0
