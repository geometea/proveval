"""v3 execution logic: deterministic interleaved randomisation, bounded
retries with latency/error capture, valid-only resume, and
frozen-config-authoritative settings. No API calls anywhere."""

import json

import pytest

import controllability_v3_execution as ex
from controllability_v3_study_config import build_study_config


def cell(family, block, trial, **extra):
    base = {"trial_id": trial, "block_id": block, "family": family, "pair_id": "s1_vs_s2", "story_1_id": "s1", "story_2_id": "s2",
            "story_a_id": "s1", "story_b_id": "s2", "cue_id": "c", "cue_family": "f", "assignment": "forward", "position": "story1_as_a",
            "intervention_id": "I0", "instruction_id": "I0", "context_present": True}
    base.update(extra)
    return base


def make_trials():
    trials = []
    for b in range(3):
        for c in range(4):
            trials.append(cell("primary_context", f"v3::primary_context::p::c::I{b}", f"v3::primary_context::p::c::a::pos{c}::I{b}::ctx"))
    for b in range(2):
        for c in range(2):
            trials.append(cell("primary_nocontext", f"v3::primary_nocontext::p::I{b}", f"v3::primary_nocontext::p::none::none::pos{c}::I{b}::noctx", context_present=False))
    return trials


class TestPlanning:
    def test_plan_covers_every_block_replicate_once_and_is_seed_deterministic(self):
        trials = make_trials()
        plan = ex.attach_observation_ids(ex.plan_execution_order(trials, 3, seed=42))
        assert len(plan) == len(trials) * 3
        ids = [p["planned_observation_id"] for p in plan]
        assert len(set(ids)) == len(ids)
        assert [p["execution_order_index"] for p in plan] == list(range(1, len(plan) + 1))
        again = ex.attach_observation_ids(ex.plan_execution_order(trials, 3, seed=42))
        assert [p["planned_observation_id"] for p in again] == ids
        other = ex.attach_observation_ids(ex.plan_execution_order(trials, 3, seed=43))
        assert [p["planned_observation_id"] for p in other] != ids

    def test_families_are_interleaved_and_units_stay_contiguous(self):
        plan = ex.plan_execution_order(make_trials(), 2, seed=7)
        families = [p["family"] for p in plan]
        assert families != sorted(families)  # not all context first
        # every unit's cells are consecutive
        i = 0
        while i < len(plan):
            unit = (plan[i]["block_id"], plan[i]["replicate_number"])
            size = 4 if plan[i]["family"] == "primary_context" else 2
            assert all((p["block_id"], p["replicate_number"]) == unit for p in plan[i:i + size])
            i += size

    def test_within_unit_order_varies_across_units(self):
        plan = ex.plan_execution_order(make_trials(), 4, seed=1)
        orders = set()
        i = 0
        while i < len(plan):
            size = 4 if plan[i]["family"] == "primary_context" else 2
            orders.add(tuple(p["trial_id"].split("::")[5] for p in plan[i:i + size]))
            i += size
        assert len(orders) > 2

    def test_limit_keeps_whole_blocks(self):
        trials = make_trials()
        limited = ex.limit_to_first_n_units(trials, 2)
        assert len(limited) == 8 and len({t["block_id"] for t in limited}) == 2
        assert ex.limit_to_first_n_units(trials, None) is trials

    def test_v2_prefixed_ids_can_never_be_planned(self):
        trials = [cell("primary_context", "b", "context_controllability_v2::x")]
        with pytest.raises(ValueError):
            ex.attach_observation_ids(ex.plan_execution_order(trials, 1, 0))


class TestRunOneObservation:
    def test_valid_first_attempt(self):
        clock = iter([0.0, 0.5])
        result = ex.run_one_observation({}, "prompt", lambda p: {"response_text": "A", "input_tokens": 3, "output_tokens": 1, "prompt_cache_hit_tokens": 2}, retry_limit=3, clock=lambda: next(clock))
        assert result["parsing_status"] == "resolved" and result["parsed_choice"] == "A" and result["total_attempts"] == 1
        assert result["attempts"][0]["latency_seconds"] == 0.5 and result["attempts"][0]["cached_input_tokens"] == 2
        assert result["error"] is None and result["first_attempt_status"] == "valid"

    def test_invalid_then_valid_is_resolved_with_two_attempts(self):
        responses = iter(["I can't decide", "Passage B"])
        result = ex.run_one_observation({}, "p", lambda p: {"response_text": next(responses)}, retry_limit=3)
        assert result["parsed_choice"] == "B" and result["total_attempts"] == 2
        assert result["first_attempt_status"] == "refusal" and result["attempts"][0]["error"]

    def test_api_error_is_recorded_not_raised(self):
        calls = {"n": 0}

        def flaky(p):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return {"response_text": "A"}
        result = ex.run_one_observation({}, "p", flaky, retry_limit=2)
        assert result["attempts"][0]["status"] == "api_error" and "boom" in result["attempts"][0]["error"]
        assert result["parsing_status"] == "resolved"

    def test_exhausted_retries_is_unresolved_with_error(self):
        result = ex.run_one_observation({}, "p", lambda p: {"response_text": "both are fine"}, retry_limit=2)
        assert result["parsing_status"] == "unresolved" and result["parsed_choice"] is None
        assert result["total_attempts"] == 3 and result["error"]

    def test_retry_never_adds_a_second_opinion_after_a_valid_answer(self):
        calls = {"n": 0}

        def f(p):
            calls["n"] += 1
            return {"response_text": "A"}
        ex.run_one_observation({}, "p", f, retry_limit=5)
        assert calls["n"] == 1


def write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestResume:
    def test_only_genuinely_valid_rows_count_as_completed(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        write_rows(path, [
            {"planned_observation_id": "v3::a", "parsing_status": "resolved", "raw_response": "A", "parsed_choice": "A", "family": "primary_context"},
            {"planned_observation_id": "v3::b", "parsing_status": "unresolved", "raw_response": "hmm", "parsed_choice": None, "family": "primary_context"},
            {"planned_observation_id": "v3::c", "parsing_status": "resolved", "raw_response": "neither", "parsed_choice": "A", "family": "primary_context"},  # claims valid, doesn't re-parse
            {"planned_observation_id": "v3::d", "parsing_status": "resolved", "raw_response": "B", "parsed_choice": "A", "family": "primary_context"},  # mismatch
        ])
        assert ex.load_valid_completed_ids(str(path)) == {"v3::a"}
        assert ex.load_valid_completed_ids(str(tmp_path / "missing.jsonl")) == set()

    def test_duplicate_valid_and_foreign_ids_are_found(self, tmp_path):
        path = tmp_path / "raw.jsonl"
        write_rows(path, [
            {"planned_observation_id": "v3::a", "parsing_status": "resolved", "family": "primary_context"},
            {"planned_observation_id": "v3::a", "parsing_status": "resolved", "family": "primary_context"},
            {"planned_observation_id": "v3::p", "parsing_status": "resolved", "family": "pilot"},
            {"planned_observation_id": "context_controllability_v2::x", "parsing_status": "resolved", "family": "primary_context"},
        ])
        assert ex.find_duplicate_valid_ids(str(path)) == ["v3::a@deepseek_flash_low"]  # keyed by evaluation id; legacy rows = primary profile
        assert ex.find_foreign_ids(str(path), ("primary_context", "primary_nocontext")) == ["v3::p", "context_controllability_v2::x"]
        assert ex.count_rows(str(path)) == 4

    def test_filter_unresumed_plan(self):
        plan = ex.attach_observation_ids(ex.plan_execution_order(make_trials(), 1, 0))
        done = {plan[0]["planned_observation_id"], plan[3]["planned_observation_id"]}
        remaining = ex.filter_unresumed_plan(plan, done)
        assert len(remaining) == len(plan) - 2 and not any(p["planned_observation_id"] in done for p in remaining)


class TestSettingsAndRows:
    def test_frozen_settings_are_authoritative(self):
        config = build_study_config()
        s = ex.resolve_production_settings({}, config, "primary")
        assert (s["provider"], s["model"], s["reasoning_profile"], s["replicates"], s["max_output_tokens"]) == ("deepseek", "deepseek-flash", "low", 10, 4096)
        assert ex.resolve_production_settings({}, config, "pilot")["replicates"] == 1
        assert ex.resolve_production_settings({}, config, "holdout")["replicates"] == 10
        with pytest.raises(ValueError, match="replicates"):
            ex.resolve_production_settings({"replicates": 3}, config, "primary")
        with pytest.raises(ValueError, match="model"):
            ex.resolve_production_settings({"model": "other"}, config, "primary")
        with pytest.raises(ValueError):
            ex.resolve_production_settings({}, config, "bogus")

    def test_result_row_carries_every_required_field(self):
        plan = ex.attach_observation_ids(ex.plan_execution_order(make_trials(), 1, 0))
        entry = plan[0]
        identity = ex.build_evaluator_identity("deepseek", "deepseek-flash", "low", {"reasoning_effort": "low"}, {}, run_id="r", max_output_tokens=4096, retry_limit=3)
        observation = ex.run_one_observation({}, "p", lambda p: {"response_text": "B", "input_tokens": 10, "output_tokens": 2, "reasoning_tokens": 5, "prompt_cache_hit_tokens": 4}, 1)
        row = ex.build_result_row(entry, identity, observation, "sha", "r", "production", "msha")
        required = ["planned_observation_id", "experiment", "story_pair", "story_a", "story_b", "cue", "context_assignment", "display_position",
                    "intervention_id", "context_present", "replicate", "provider", "model", "reasoning_effort", "max_output_tokens", "raw_response",
                    "parsed_choice", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "latency_seconds", "timestamp",
                    "run_id", "attempts", "total_attempts", "error", "family", "run_config_id"]
        assert all(k in row for k in required)
        assert row["experiment"] == "v3" and row["parsed_choice"] == "B" and row["cached_input_tokens"] == 4 and row["reasoning_effort"] == "low"
        assert row["planned_observation_id"].startswith("v3::")
