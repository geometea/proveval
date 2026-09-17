"""Tests for controllability_v2_execution.py: randomised execution order,
--limit unit-aware selection, bounded-retry attempt bookkeeping, and
evaluator identity.
"""

import pytest

import controllability_v2_execution as ex


def make_superblock(superblock_id, contrast_id="c1", pair="s1_vs_s2"):
    cells = []
    for instruction_condition in ("matched_control", "text_only"):
        for assignment in ("forward", "flipped"):
            for position in ("story1_as_a", "story2_as_a"):
                cells.append({
                    "trial_id": f"{superblock_id}__{instruction_condition}__{assignment}__{position}",
                    "superblock_id": superblock_id, "contrast_id": contrast_id,
                    "instruction_condition": instruction_condition, "assignment": assignment, "position": position,
                    "prompt": "p",
                })
    return cells


def make_baseline_unit(block_id):
    return [
        {"trial_id": f"{block_id}__story1_as_a", "block_id": block_id, "position": "story1_as_a", "prompt": "p"},
        {"trial_id": f"{block_id}__story2_as_a", "block_id": block_id, "position": "story2_as_a", "prompt": "p"},
    ]


# ---------------------------------------------------------------------------
# Randomised execution order
# ---------------------------------------------------------------------------

class TestPlanTreatmentExecutionOrder:
    def test_plan_length_is_superblocks_times_cells_times_replicates(self):
        trials = make_superblock("sb1") + make_superblock("sb2")
        plan = ex.plan_treatment_execution_order(trials, replicate_count=3, seed=1)
        assert len(plan) == 2 * 8 * 3

    def test_reproducible_from_the_same_seed(self):
        trials = make_superblock("sb1") + make_superblock("sb2") + make_superblock("sb3")
        plan_a = ex.plan_treatment_execution_order(trials, replicate_count=2, seed=42)
        plan_b = ex.plan_treatment_execution_order(trials, replicate_count=2, seed=42)
        assert [p["trial_id"] for p in plan_a] == [p["trial_id"] for p in plan_b]

    def test_different_seeds_produce_different_orders(self):
        trials = make_superblock("sb1") + make_superblock("sb2") + make_superblock("sb3")
        plan_a = ex.plan_treatment_execution_order(trials, replicate_count=2, seed=1)
        plan_b = ex.plan_treatment_execution_order(trials, replicate_count=2, seed=2)
        assert [p["trial_id"] for p in plan_a] != [p["trial_id"] for p in plan_b]

    def test_does_not_preserve_a_fixed_cell_sequence(self):
        """Repeated calls (here: replicates of the whole plan) must not run
        as all replicates of one cell followed by all replicates of
        another -- the global shuffle must interleave superblocks AND
        replicates, never emit one superblock-replicate's 8 cells, then the
        next, in the original construction order."""
        superblock_ids = [f"sb{i}" for i in range(20)]
        trials = [cell for sid in superblock_ids for cell in make_superblock(sid)]
        plan = ex.plan_treatment_execution_order(trials, replicate_count=5, seed=7)
        superblock_sequence = [p["superblock_id"] for p in plan]
        # count contiguous "runs" of the same superblock id
        runs = 1 + sum(1 for a, b in zip(superblock_sequence, superblock_sequence[1:]) if a != b)
        # 20 superblocks x 5 replicates = 100 units; grouped/unshuffled
        # construction order would produce exactly 20 runs (one per
        # superblock, its 5 replicates adjacent) -- a global shuffle must
        # produce far more, smaller runs
        assert runs > 50

    def test_replicate_number_is_recorded_and_bounded(self):
        trials = make_superblock("sb1")
        plan = ex.plan_treatment_execution_order(trials, replicate_count=3, seed=1)
        replicate_numbers = {p["replicate_number"] for p in plan}
        assert replicate_numbers == {1, 2, 3}

    def test_execution_order_index_and_seed_are_recorded_on_every_entry(self):
        trials = make_superblock("sb1")
        plan = ex.plan_treatment_execution_order(trials, replicate_count=1, seed=99)
        assert [p["execution_order_index"] for p in plan] == list(range(1, 9))
        assert all(p["random_seed"] == 99 for p in plan)

    def test_within_superblock_cells_are_also_shuffled_not_fixed_order(self):
        trials = make_superblock("sb1")
        original_order = [c["trial_id"] for c in trials]
        plan = ex.plan_treatment_execution_order(trials, replicate_count=1, seed=1234)
        assert [p["trial_id"] for p in plan] != original_order


class TestPlanBaselineExecutionOrder:
    def test_plan_length_and_reproducibility(self):
        trials = make_baseline_unit("b1") + make_baseline_unit("b2")
        plan_a = ex.plan_baseline_execution_order(trials, replicate_count=2, seed=5)
        plan_b = ex.plan_baseline_execution_order(trials, replicate_count=2, seed=5)
        assert len(plan_a) == 2 * 2 * 2
        assert [p["trial_id"] for p in plan_a] == [p["trial_id"] for p in plan_b]


# ---------------------------------------------------------------------------
# --limit cannot orphan a unit
# ---------------------------------------------------------------------------

class TestLimitToFirstNUnits:
    def test_limit_selects_whole_superblocks(self):
        trials = make_superblock("sb1") + make_superblock("sb2") + make_superblock("sb3")
        limited = ex.limit_to_first_n_units(trials, "superblock_id", 2)
        assert len(limited) == 16
        assert {t["superblock_id"] for t in limited} == {"sb1", "sb2"}

    def test_limit_none_is_a_no_op(self):
        trials = make_superblock("sb1") + make_superblock("sb2")
        assert ex.limit_to_first_n_units(trials, "superblock_id", None) == trials

    def test_limit_selects_whole_baseline_units(self):
        trials = make_baseline_unit("b1") + make_baseline_unit("b2") + make_baseline_unit("b3")
        limited = ex.limit_to_first_n_units(trials, "block_id", 1)
        assert len(limited) == 2
        assert {t["block_id"] for t in limited} == {"b1"}


# ---------------------------------------------------------------------------
# Bounded retry handling
# ---------------------------------------------------------------------------

class TestClassifyAttempt:
    def test_valid(self):
        assert ex.classify_attempt({"overall_quality": "A"}, None) == "valid"

    def test_refusal(self):
        assert ex.classify_attempt(None, "Response reads as ambiguous/hedged (contains 'tie'), not an unambiguous A or B") == "refusal"

    def test_invalid(self):
        assert ex.classify_attempt(None, "Response must be an unambiguous \"A\" or \"B\"") == "invalid"

    def test_api_error(self):
        assert ex.classify_attempt(None, "API call failed: 500 oops") == "api_error"


class TestRunOneObservation:
    def test_stops_retrying_once_valid(self):
        calls = {"n": 0}

        def call_fn(trial):
            calls["n"] += 1
            return {"response_text": "A"}

        result = ex.run_one_observation({"trial_id": "t1"}, call_fn, retry_limit=5)
        assert calls["n"] == 1
        assert result["total_attempts"] == 1
        assert result["first_attempt_status"] == "valid"
        assert result["first_valid_response"] == {"overall_quality": "A"}
        assert result["parsing_status"] == "resolved"
        assert result["refusal_status"] is False
        assert result["api_error_status"] is False

    def test_retries_up_to_the_limit_and_records_every_attempt(self):
        def call_fn(trial):
            return {"response_text": "I really can't decide"}

        result = ex.run_one_observation({"trial_id": "t1"}, call_fn, retry_limit=2)
        assert result["total_attempts"] == 3  # 1 initial + 2 retries, never more
        assert len(result["attempts"]) == 3
        assert result["first_attempt_status"] == "refusal"
        assert result["refusal_status"] is True
        assert result["parsing_status"] == "unresolved"
        assert result["first_valid_response"] is None

    def test_recovers_on_a_later_attempt_but_first_attempt_status_still_reflects_the_first(self):
        calls = {"n": 0}

        def call_fn(trial):
            calls["n"] += 1
            return {"response_text": "hard to say"} if calls["n"] == 1 else {"response_text": "B"}

        result = ex.run_one_observation({"trial_id": "t1"}, call_fn, retry_limit=3)
        assert result["total_attempts"] == 2
        assert result["first_attempt_status"] == "refusal"  # first attempt was a refusal...
        assert result["parsing_status"] == "resolved"  # ...but a later attempt did resolve it
        assert result["first_valid_response"] == {"overall_quality": "B"}

    def test_api_error_path_never_raises(self):
        def call_fn(trial):
            raise RuntimeError("connection reset")

        result = ex.run_one_observation({"trial_id": "t1"}, call_fn, retry_limit=1)
        assert result["total_attempts"] == 2
        assert result["api_error_status"] is True
        assert result["parsing_status"] == "unresolved"

    def test_retries_never_increase_beyond_the_configured_limit(self):
        calls = {"n": 0}

        def call_fn(trial):
            calls["n"] += 1
            return {"response_text": "still ambiguous"}

        ex.run_one_observation({"trial_id": "t1"}, call_fn, retry_limit=0)
        assert calls["n"] == 1  # retry_limit=0 -- exactly one attempt, no retries


# ---------------------------------------------------------------------------
# Evaluator identity
# ---------------------------------------------------------------------------

def test_build_evaluator_identity_has_all_required_fields():
    identity = ex.build_evaluator_identity(
        provider="anthropic", requested_model="claude-sonnet-5", reasoning_profile="low",
        provider_reasoning_settings={"thinking": {"type": "adaptive"}}, sampling_settings={}, run_id="wave1",
    )
    for field in ("provider", "requested_model", "reasoning_profile", "provider_reasoning_settings",
                  "sampling_settings", "response_model", "model_version", "run_id"):
        assert field in identity
    assert identity["provider"] == "anthropic"
    assert identity["run_id"] == "wave1"
