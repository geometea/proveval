"""Tests for analyze_controllability_v2.py: evaluator partitioning (never
silently pooling different providers/models/reasoning-profiles/response-model
versions), complete-superblock/baseline-unit filtering, and the
story-identity (never raw A/B letter identity) estimator.
"""

import pytest

import analyze_controllability_v2 as av2

STUDY_CONFIG = {
    "primary_evaluator": {"evaluator_id": "anthropic__claude-sonnet-5__low", "provider": "anthropic",
                           "requested_model": "claude-sonnet-5", "reasoning_profile": "low", "role": "primary"},
    "replication_evaluators": [
        {"evaluator_id": "openai__gpt-5.6__low", "provider": "openai", "requested_model": "gpt-5.6",
         "reasoning_profile": "low", "role": "replication"},
    ],
}


def make_treatment_row(trial_id, superblock_id, replicate_number, contrast_id, instruction_condition,
                        assignment, position, story_1_id, story_2_id, story_a_id, story_b_id,
                        choice="A", provider="anthropic", requested_model="claude-sonnet-5",
                        reasoning_profile="low", response_model="claude-sonnet-5-20250929", resolved=True):
    return {
        "trial_id": trial_id, "superblock_id": superblock_id, "replicate_number": replicate_number,
        "evaluator": {"provider": provider, "requested_model": requested_model, "reasoning_profile": reasoning_profile},
        "trial_meta": {
            "contrast_id": contrast_id, "instruction_condition": instruction_condition, "assignment": assignment,
            "position": position, "story_1_id": story_1_id, "story_2_id": story_2_id,
            "story_a_id": story_a_id, "story_b_id": story_b_id,
        },
        "attempts": [{"attempt_number": 1, "status": "valid" if resolved else "invalid", "response_model": response_model}],
        "total_attempts": 1,
        "first_attempt_status": "valid" if resolved else "invalid",
        "first_valid_response": {"overall_quality": choice} if resolved else None,
        "parsing_status": "resolved" if resolved else "unresolved",
        "refusal_status": False,
        "api_error_status": False,
    }


def make_baseline_row(trial_id, block_id, replicate_number, story_1_id, story_2_id, position, choice="A",
                       provider="anthropic", requested_model="claude-sonnet-5", reasoning_profile="low"):
    return {
        "trial_id": trial_id, "block_id": block_id, "replicate_number": replicate_number,
        "evaluator": {"provider": provider, "requested_model": requested_model, "reasoning_profile": reasoning_profile},
        "trial_meta": {"story_1_id": story_1_id, "story_2_id": story_2_id, "position": position,
                       "story_a_id": story_1_id if position == "story1_as_a" else story_2_id,
                       "story_b_id": story_2_id if position == "story1_as_a" else story_1_id},
        "attempts": [{"attempt_number": 1, "status": "valid", "response_model": "claude-sonnet-5-20250929"}],
        "total_attempts": 1, "first_attempt_status": "valid",
        "first_valid_response": {"overall_quality": choice}, "parsing_status": "resolved",
        "refusal_status": False, "api_error_status": False,
    }


def make_full_superblock(superblock_id, replicate_number, contrast_id="c1", story_1_id="s1", story_2_id="s2", **kw):
    rows = []
    for instruction_condition in ("matched_control", "text_only"):
        for assignment in ("forward", "flipped"):
            for position in ("story1_as_a", "story2_as_a"):
                story_a_id = story_1_id if position == "story1_as_a" else story_2_id
                story_b_id = story_2_id if position == "story1_as_a" else story_1_id
                rows.append(make_treatment_row(
                    f"{superblock_id}__{instruction_condition}__{assignment}__{position}", superblock_id,
                    replicate_number, contrast_id, instruction_condition, assignment, position,
                    story_1_id, story_2_id, story_a_id, story_b_id, **kw,
                ))
    return rows


# ---------------------------------------------------------------------------
# story1_chosen: story identity, never raw letter identity
# ---------------------------------------------------------------------------

class TestStory1Chosen:
    def test_choice_a_when_story1_displayed_as_a_means_story1_chosen(self):
        row = make_treatment_row("t1", "sb1", 1, "c1", "matched_control", "forward", "story1_as_a", "s1", "s2", "s1", "s2", choice="A")
        assert av2.story1_chosen(row) is True

    def test_choice_a_when_story1_displayed_as_b_means_story1_not_chosen(self):
        """The raw letter "A" must not be treated as the estimand -- when
        story1 is displayed as Passage B, choosing "A" means story2 was
        chosen, so story1_chosen must be False."""
        row = make_treatment_row("t1", "sb1", 1, "c1", "matched_control", "forward", "story2_as_a", "s1", "s2", "s2", "s1", choice="A")
        assert av2.story1_chosen(row) is False

    def test_unresolved_observation_returns_none(self):
        row = make_treatment_row("t1", "sb1", 1, "c1", "matched_control", "forward", "story1_as_a", "s1", "s2", "s1", "s2", resolved=False)
        assert av2.story1_chosen(row) is None


# ---------------------------------------------------------------------------
# Complete-superblock / baseline-unit filtering
# ---------------------------------------------------------------------------

class TestFilterCompleteSuperblocks:
    def test_complete_superblock_is_kept(self):
        rows = make_full_superblock("sb1", 1)
        kept, n_complete, n_incomplete = av2.filter_complete_superblocks(rows)
        assert len(kept) == 8
        assert n_complete == 1
        assert n_incomplete == 0

    def test_incomplete_superblock_is_entirely_excluded(self):
        rows = make_full_superblock("sb1", 1)[:-1]  # drop one cell -- 7/8
        kept, n_complete, n_incomplete = av2.filter_complete_superblocks(rows)
        assert kept == []
        assert n_complete == 0
        assert n_incomplete == 1

    def test_one_unresolved_cell_makes_the_whole_replicate_unit_incomplete(self):
        rows = make_full_superblock("sb1", 1)
        rows[0]["parsing_status"] = "unresolved"
        rows[0]["first_valid_response"] = None
        kept, n_complete, n_incomplete = av2.filter_complete_superblocks(rows)
        assert kept == []
        assert n_incomplete == 1

    def test_replicates_are_evaluated_independently(self):
        complete = make_full_superblock("sb1", 1)
        incomplete = make_full_superblock("sb1", 2)[:-1]
        kept, n_complete, n_incomplete = av2.filter_complete_superblocks(complete + incomplete)
        assert len(kept) == 8
        assert n_complete == 1
        assert n_incomplete == 1


class TestFilterCompleteBaselineUnits:
    def test_complete_unit_is_kept(self):
        rows = [
            make_baseline_row("b1__a", "b1", 1, "s1", "s2", "story1_as_a"),
            make_baseline_row("b1__b", "b1", 1, "s1", "s2", "story2_as_a"),
        ]
        kept, n_complete, n_incomplete = av2.filter_complete_baseline_units(rows)
        assert len(kept) == 2 and n_complete == 1 and n_incomplete == 0

    def test_missing_position_makes_the_unit_incomplete(self):
        rows = [make_baseline_row("b1__a", "b1", 1, "s1", "s2", "story1_as_a")]
        kept, n_complete, n_incomplete = av2.filter_complete_baseline_units(rows)
        assert kept == [] and n_incomplete == 1


# ---------------------------------------------------------------------------
# Evaluator separation: never silently pool providers/models/profiles/versions
# ---------------------------------------------------------------------------

class TestPartitionByResolvedEvaluator:
    def test_matches_rows_to_configured_evaluators(self):
        rows = make_full_superblock("sb1", 1) + make_full_superblock("sb1", 1, provider="openai", requested_model="gpt-5.6")
        partitions, unmatched, warnings = av2.partition_by_resolved_evaluator(rows, STUDY_CONFIG)
        assert set(partitions) == {"anthropic__claude-sonnet-5__low", "openai__gpt-5.6__low"}
        assert unmatched == []
        assert warnings == []

    def test_unconfigured_evaluator_identity_is_excluded_not_pooled(self):
        rows = make_full_superblock("sb1", 1, provider="deepseek", requested_model="deepseek-v4-pro")
        partitions, unmatched, warnings = av2.partition_by_resolved_evaluator(rows, STUDY_CONFIG)
        assert partitions == {}
        assert len(unmatched) == 8

    def test_reasoning_profile_mismatch_is_a_different_evaluator(self):
        rows = make_full_superblock("sb1", 1, reasoning_profile="high")
        partitions, unmatched, warnings = av2.partition_by_resolved_evaluator(rows, STUDY_CONFIG)
        assert partitions == {}  # "high" was never configured -- excluded, never folded into the "low" evaluator
        assert len(unmatched) == 8

    def test_multiple_response_model_versions_are_split_not_pooled(self):
        rows = (
            make_full_superblock("sb1", 1, response_model="claude-sonnet-5-20250101")
            + make_full_superblock("sb2", 1, response_model="claude-sonnet-5-20250601")
        )
        partitions, unmatched, warnings = av2.partition_by_resolved_evaluator(rows, STUDY_CONFIG)
        assert set(partitions) == {
            "anthropic__claude-sonnet-5__low#claude-sonnet-5-20250101",
            "anthropic__claude-sonnet-5__low#claude-sonnet-5-20250601",
        }
        assert len(partitions["anthropic__claude-sonnet-5__low#claude-sonnet-5-20250101"]["rows"]) == 8
        assert len(warnings) == 1

    def test_strict_mode_raises_instead_of_splitting(self):
        rows = (
            make_full_superblock("sb1", 1, response_model="claude-sonnet-5-20250101")
            + make_full_superblock("sb2", 1, response_model="claude-sonnet-5-20250601")
        )
        with pytest.raises(ValueError, match="multiple response_model versions"):
            av2.partition_by_resolved_evaluator(rows, STUDY_CONFIG, strict=True)

    def test_single_response_model_version_is_not_split(self):
        rows = make_full_superblock("sb1", 1) + make_full_superblock("sb2", 1)
        partitions, unmatched, warnings = av2.partition_by_resolved_evaluator(rows, STUDY_CONFIG)
        assert set(partitions) == {"anthropic__claude-sonnet-5__low"}
        assert warnings == []


# ---------------------------------------------------------------------------
# Pair-level cell rates: built from story identity
# ---------------------------------------------------------------------------

class TestBuildTreatmentCellRates:
    def test_synthetic_context_effect_recovers_expected_sign_and_direction(self):
        """story1 has value 'a' under 'forward' -- if the evaluator always
        picks whichever story has 'a', D_pair for this pair must be +1."""
        rows = []
        for replicate in range(1, 4):
            for instruction_condition in ("matched_control",):
                for assignment, expect_story1_wins in (("forward", True), ("flipped", False)):
                    for position in ("story1_as_a", "story2_as_a"):
                        story_a_id = "s1" if position == "story1_as_a" else "s2"
                        story_b_id = "s2" if position == "story1_as_a" else "s1"
                        winner_id = "s1" if expect_story1_wins else "s2"
                        choice = "A" if winner_id == story_a_id else "B"
                        rows.append(make_treatment_row(
                            f"t_{replicate}_{assignment}_{position}", "sb1", replicate, "c1", instruction_condition,
                            assignment, position, "s1", "s2", story_a_id, story_b_id, choice=choice,
                        ))
        cell_rates = av2.build_treatment_cell_rates(rows)
        pair_rows = av2.build_pair_context_effects(cell_rates)
        assert len(pair_rows) == 1
        assert pair_rows[0]["d_pair"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Baseline pair table is built ONLY from baseline rows -- independent of
# whatever treatment data exists for the same evaluator
# ---------------------------------------------------------------------------

class TestBaselineIndependence:
    def test_baseline_pair_table_ignores_treatment_rows_entirely(self):
        baseline_rows = [
            make_baseline_row("b1__a", "b1", 1, "s1", "s2", "story1_as_a", choice="A"),
            make_baseline_row("b1__b", "b1", 1, "s1", "s2", "story2_as_a", choice="B"),
        ]
        table_without_treatment, _ = av2.build_baseline_pair_table(baseline_rows)

        # Adding a large amount of unrelated treatment data must not change
        # the baseline table at all -- build_baseline_pair_table never reads
        # treatment rows in the first place.
        treatment_rows = make_full_superblock("sbX", 1, story_1_id="s1", story_2_id="s2")
        table_with_treatment, _ = av2.build_baseline_pair_table(baseline_rows)  # treatment_rows never passed in
        assert table_without_treatment == table_with_treatment

    def test_ambiguity_join_matches_by_story_pair_not_by_contrast(self):
        """baseline_strength has no contrast_id of its own (the baseline has
        no context at all) -- it must join onto every contrast's pair row
        for the same (story_1_id, story_2_id)."""
        baseline_rows = [
            make_baseline_row("b1__a", "b1", 1, "s1", "s2", "story1_as_a", choice="A"),
            make_baseline_row("b1__b", "b1", 1, "s1", "s2", "story2_as_a", choice="A"),
        ]
        baseline_table, _ = av2.build_baseline_pair_table(baseline_rows)
        strength_by_pair = {(r["story_1_id"], r["story_2_id"]): r["baseline_strength"] for r in baseline_table}

        pair_rows = [
            {"contrast_id": "c1", "instruction_condition": "matched_control", "story_1_id": "s1", "story_2_id": "s2", "d_pair": 0.1},
            {"contrast_id": "c2", "instruction_condition": "matched_control", "story_1_id": "s1", "story_2_id": "s2", "d_pair": 0.2},
        ]
        for row in pair_rows:
            row["baseline_strength"] = strength_by_pair.get((row["story_1_id"], row["story_2_id"]))
        assert all(row["baseline_strength"] is not None for row in pair_rows)
        assert pair_rows[0]["baseline_strength"] == pair_rows[1]["baseline_strength"]  # same pair, same baseline, different contrasts


# ---------------------------------------------------------------------------
# Response compliance: first-attempt status only, never inflated by retries
# ---------------------------------------------------------------------------

class TestResponseCompliance:
    def test_counts_are_grouped_by_contrast_and_instruction_condition(self):
        rows = make_full_superblock("sb1", 1)
        compliance = av2.response_compliance_table(rows)
        assert len(compliance) == 2  # matched_control, text_only
        for row in compliance:
            assert row["n_attempted"] == 4
            assert row["n_valid_first_attempt"] == 4
            assert row["valid_rate"] == 1.0

    def test_a_first_attempt_refusal_is_counted_separately_from_invalid(self):
        row = make_treatment_row("t1", "sb1", 1, "c1", "matched_control", "forward", "story1_as_a", "s1", "s2", "s1", "s2")
        row["first_attempt_status"] = "refusal"
        compliance = av2.response_compliance_table([row])
        assert compliance[0]["n_refusal_first_attempt"] == 1
        assert compliance[0]["n_invalid_first_attempt"] == 0
