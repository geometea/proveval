"""Tests for run_controllability_v2.py's Wave 2 recovery CLI surface:
recovery-preflight, recovery-run, validation-run, baseline-text-only, and
merge. model_providers.call_model is always monkeypatched -- these tests
never make a real API call, and never touch the real
results/controllability_v2/production/ directory or the real frozen study
config's manifests (a tiny synthetic design is used throughout)."""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

import model_providers
import run_controllability_v2 as rv2
import controllability_v2_recovery as rec
from controllability_v2_execution import make_observation_id

EVALUATOR_ID = "deepseek__deepseek-flash__low"
EXPERIMENT_ID = "context_controllability_v2"


def _treatment_trial(trial_id, block_id, superblock_id, position="story1_as_a", assignment="forward",
                      instruction_condition="matched_control", contrast_id="c1", s1="s1", s2="s2"):
    story_a_id, story_b_id = (s1, s2) if position == "story1_as_a" else (s2, s1)
    return {
        "trial_id": trial_id, "block_id": block_id, "superblock_id": superblock_id,
        "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab",
        "contrast_id": contrast_id, "dimension": "d", "expected_direction": "a",
        "instruction_condition": instruction_condition, "assignment": assignment, "position": position,
        "story_1_id": s1, "story_2_id": s2, "story_a_id": story_a_id, "story_b_id": story_b_id,
        "prompt": f"prompt {trial_id}", "prompt_sha256": "x",
    }


def _baseline_trial(trial_id, block_id, position="story1_as_a", instruction_condition="matched_control", s1="s1", s2="s2"):
    return {
        "trial_id": trial_id, "block_id": block_id, "type": "context_pairwise", "choice_mode": "forced",
        "response_format": "plain_ab", "instruction_condition": instruction_condition,
        "story_1_id": s1, "story_2_id": s2, "position": position, "prompt": f"prompt {trial_id}", "prompt_sha256": "x",
    }


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def tiny_study(tmp_path):
    treatment_trials = [
        _treatment_trial("t1", "b1", "sb1"),
        _treatment_trial("t2", "b1", "sb1", position="story2_as_a"),
    ]
    baseline_trials = [_baseline_trial("base1", "bb1"), _baseline_trial("base2", "bb2")]
    treatment_path = tmp_path / "treatment.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(treatment_path, treatment_trials)
    _write_jsonl(baseline_path, baseline_trials)

    study_config = {
        "treatment_manifest_file": str(treatment_path), "baseline_manifest_file": str(baseline_path),
        "primary_evaluator": {"evaluator_id": EVALUATOR_ID, "provider": "deepseek",
                               "requested_model": "deepseek-flash", "reasoning_profile": "low"},
        "treatment_replicate_count": 2, "baseline_replicate_count": 1, "retry_limit": 1, "random_seed": 0,
        "max_output_tokens": 512,
    }
    return {"study_config": study_config, "treatment_trials": treatment_trials, "baseline_trials": baseline_trials}


def _oid(trial_id, replicate_number):
    return make_observation_id(EXPERIMENT_ID, EVALUATOR_ID, trial_id, replicate_number)


def _resolved_row(oid, trial_id, response="A"):
    return {"observation_id": oid, "trial_id": trial_id, "parsing_status": "resolved",
            "first_valid_response": {"overall_quality": response},
            "attempts": [{"response_text": response, "reasoning_tokens": 3}]}


@pytest.fixture
def fake_call(monkeypatch):
    calls = []

    def call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
        calls.append({"provider": provider, "model": model, "max_output_tokens": max_output_tokens})
        return {"response_text": "A", "response_model": model, "request_id": "r1",
                "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 1}

    monkeypatch.setattr(model_providers, "call_model", call_model)
    return calls


class TestRecoveryPreflight:
    def test_reports_exact_wave1_valid_and_unresolved_counts(self, tiny_study, tmp_path):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        assert len(planned_index) == 6  # 2 treatment trials * 2 replicates + 2 baseline trials * 1 replicate

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        ids = sorted(planned_index)
        valid_ids, unresolved_ids = ids[:4], ids[4:]
        rows_by_kind = {"treatment": [], "baseline": []}
        for oid in valid_ids:
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"]))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])

        ok, checks, context = rv2.run_recovery_preflight(study_config, str(source_dir), allow_peak_pricing=False)
        assert context["n_planned"] == 6
        assert context["n_valid"] == 4
        assert context["n_unresolved"] == 2
        assert set(context["classification"]["unresolved_ids"]) == set(unresolved_ids)

    def test_fails_when_source_results_are_missing(self, tiny_study, tmp_path):
        ok, checks, context = rv2.run_recovery_preflight(tiny_study["study_config"], str(tmp_path / "nope"), allow_peak_pricing=False)
        assert ok is False
        assert any(name == "source_results_present" and not c_ok for name, c_ok, _detail in checks)

    def test_fails_on_duplicate_valid_answers(self, tiny_study, tmp_path):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        oid = _oid("t1", 1)
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [_resolved_row(oid, "t1", "A"), _resolved_row(oid, "t1", "B")])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        ok, checks, context = rv2.run_recovery_preflight(study_config, str(source_dir), allow_peak_pricing=False)
        assert ok is False
        assert any(name == "wave1_internally_consistent" and not c_ok for name, c_ok, _detail in checks)

    def test_makes_no_api_calls(self, tiny_study, tmp_path, monkeypatch):
        def must_not_be_called(*a, **k):
            raise AssertionError("recovery-preflight must never call the model API")
        monkeypatch.setattr(model_providers, "call_model", must_not_be_called)

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        rv2.run_recovery_preflight(tiny_study["study_config"], str(source_dir), allow_peak_pricing=False)

    def test_print_summary_matches_the_required_format(self, tiny_study, tmp_path, capsys):
        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        ok, checks, context = rv2.run_recovery_preflight(study_config, str(source_dir), allow_peak_pricing=False)
        rv2.print_recovery_preflight_summary(context)
        out = capsys.readouterr().out

        assert "Source run ID: 35355521243" in out
        assert f"Wave 1 planned observations: {context['n_planned']}" in out
        assert f"Wave 1 valid retained: {context['n_valid']}" in out
        assert f"Wave 1 unresolved: {context['n_unresolved']}" in out
        assert f"Recovery observations planned: {context['n_unresolved']}" in out
        assert f"New baseline_text_only observations: {context['n_new_baseline_text_only']}" in out
        assert f"Validation duplicates planned: {context['n_validation']}" in out
        assert "Total Wave 2 model calls planned:" in out
        assert "Model: deepseek-flash" in out
        assert "Reasoning effort: low" in out
        assert "Wave 1 max output tokens: 512" in out
        assert "Wave 2 max output tokens: 4096" in out
        assert "Peak-pricing override: disabled" in out

    def test_wave1_max_output_tokens_is_derived_from_the_real_study_config_not_hardcoded(self, tiny_study, tmp_path, capsys):
        study_config = dict(tiny_study["study_config"])
        study_config["max_output_tokens"] = 999  # a deliberately non-standard value
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        ok, checks, context = rv2.run_recovery_preflight(study_config, str(source_dir), allow_peak_pricing=False)
        rv2.print_recovery_preflight_summary(context)
        out = capsys.readouterr().out
        assert "Wave 1 max output tokens: 999" in out


class TestRecoveryRun:
    def test_only_unresolved_observations_are_dispatched_and_tagged(self, tiny_study, tmp_path, monkeypatch, fake_call):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        ids = sorted(planned_index)
        valid_ids, unresolved_ids = ids[:4], ids[4:]

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        rows_by_kind = {"treatment": [], "baseline": []}
        for oid in valid_ids:
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"]))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])

        recovery_path = tmp_path / "recovery_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_recovery_run(args)

        written = [json.loads(line) for line in recovery_path.read_text().splitlines()]
        assert {row["observation_id"] for row in written} == set(unresolved_ids)
        assert all(row["collection_wave"] == "wave2_recovery" for row in written)
        assert all(row["max_output_tokens"] == 4096 for row in written)
        assert all(row["source_run_id"] == "35355521243" for row in written)
        assert all(call["max_output_tokens"] == 4096 for call in fake_call)

    def test_dry_run_makes_no_calls_and_writes_nothing(self, tiny_study, tmp_path, monkeypatch, fake_call):
        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        recovery_path = tmp_path / "recovery_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=True)
        rv2.cmd_recovery_run(args)
        assert fake_call == []
        assert not recovery_path.exists()

    def test_refuses_to_run_against_an_unfrozen_design(self, tiny_study, tmp_path, monkeypatch, fake_call):
        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (False, "not frozen"))
        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        with pytest.raises(SystemExit):
            rv2.cmd_recovery_run(args)
        assert fake_call == []

    def test_recovery_creates_no_new_replicates_end_to_end(self, tiny_study, tmp_path, monkeypatch, fake_call):
        """Every recovery observation_id written is one of the ORIGINAL
        planned ids -- recovery can never manufacture a new replicate."""
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        recovery_path = tmp_path / "recovery_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=2, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_recovery_run(args)

        written = [json.loads(line) for line in recovery_path.read_text().splitlines()]
        assert len(written) == 6  # everything was unresolved
        assert all(row["observation_id"] in planned_index for row in written)
        assert len({row["observation_id"] for row in written}) == 6  # no duplicates


class TestRecoveryResumeSemantics:
    """Wave 2 resume semantics (controllability_v2_recovery.
    load_valid_completed_observation_ids): only a genuinely VALID prior
    answer skips re-dispatch on a later invocation of the SAME command.
    Every kind of failure -- an API error (402, 429, ...), a malformed/
    no-answer response, or an exhausted-retries non-answer at the 4096-token
    cap -- must remain eligible for that rerun. This is the opposite of
    Wave 1's own (unchanged) "any terminal row is done" semantics."""

    def _run_recovery_with_preexisting_row(self, tiny_study, tmp_path, monkeypatch, preexisting_row, response_text="A"):
        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])  # everything unresolved from Wave 1's own perspective

        recovery_path = tmp_path / "recovery_raw.jsonl"
        _write_jsonl(recovery_path, [preexisting_row])

        calls = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls.append(prompt)
            return {"response_text": response_text, "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 1}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_recovery_run(args)

        written = [json.loads(line) for line in recovery_path.read_text().splitlines()]
        return calls, written

    def test_a_prior_valid_answer_is_skipped_on_resume(self, tiny_study, tmp_path, monkeypatch):
        oid = _oid("t1", 1)
        preexisting = _resolved_row(oid, "t1", "A")
        calls, written = self._run_recovery_with_preexisting_row(tiny_study, tmp_path, monkeypatch, preexisting)
        rows_for_id = [row for row in written if row.get("observation_id") == oid]
        assert len(rows_for_id) == 1  # no new row appended for the already-valid id
        assert len(calls) == 5  # only the other 5 unresolved ids were dispatched

    def test_a_prior_402_style_api_error_is_retried(self, tiny_study, tmp_path, monkeypatch):
        oid = _oid("t1", 1)
        preexisting = {
            "observation_id": oid, "trial_id": "t1", "parsing_status": "unresolved",
            "first_valid_response": None, "api_error_status": True,
            "attempts": [{"response_text": None, "validation_error": "API call failed: 402 Payment Required", "reasoning_tokens": None}],
        }
        calls, written = self._run_recovery_with_preexisting_row(tiny_study, tmp_path, monkeypatch, preexisting)
        assert len(calls) == 6  # every one of the 6 unresolved ids was dispatched, INCLUDING t1/r1
        rows_for_id = [row for row in written if row.get("observation_id") == oid]
        assert len(rows_for_id) == 2  # the old failed row PLUS the new (successful) attempt
        assert rows_for_id[-1]["parsing_status"] == "resolved"

    def test_a_prior_429_style_api_error_is_retried(self, tiny_study, tmp_path, monkeypatch):
        oid = _oid("base1", 1)
        preexisting = {
            "observation_id": oid, "trial_id": "base1", "parsing_status": "unresolved",
            "first_valid_response": None, "api_error_status": True,
            "attempts": [{"response_text": None, "validation_error": "API call failed: 429 Too Many Requests", "reasoning_tokens": None}],
        }
        calls, written = self._run_recovery_with_preexisting_row(tiny_study, tmp_path, monkeypatch, preexisting)
        assert len(calls) == 6
        rows_for_id = [row for row in written if row.get("observation_id") == oid]
        assert len(rows_for_id) == 2
        assert rows_for_id[-1]["parsing_status"] == "resolved"

    def test_a_prior_malformed_no_answer_response_is_retried(self, tiny_study, tmp_path, monkeypatch):
        oid = _oid("t2", 1)
        preexisting = {
            "observation_id": oid, "trial_id": "t2", "parsing_status": "unresolved",
            "first_valid_response": None, "api_error_status": False,
            "attempts": [{"response_text": "I think both are good, hard to say", "reasoning_tokens": 50,
                          "validation_error": "Response reads as ambiguous/hedged"}],
        }
        calls, written = self._run_recovery_with_preexisting_row(tiny_study, tmp_path, monkeypatch, preexisting)
        assert len(calls) == 6
        rows_for_id = [row for row in written if row.get("observation_id") == oid]
        assert len(rows_for_id) == 2
        assert rows_for_id[-1]["parsing_status"] == "resolved"

    def test_a_prior_response_exhausting_the_4096_token_cap_without_an_answer_is_retried(self, tiny_study, tmp_path, monkeypatch):
        oid = _oid("t2", 2)
        preexisting = {
            "observation_id": oid, "trial_id": "t2", "parsing_status": "unresolved",
            "first_valid_response": None, "api_error_status": False, "max_output_tokens": 4096,
            "attempts": [{"response_text": "Let me think through this very carefully... " * 80,
                          "reasoning_tokens": 4090, "output_tokens": 4096,
                          "validation_error": "no unambiguous A or B found"}],
        }
        calls, written = self._run_recovery_with_preexisting_row(tiny_study, tmp_path, monkeypatch, preexisting)
        assert len(calls) == 6
        rows_for_id = [row for row in written if row.get("observation_id") == oid]
        assert len(rows_for_id) == 2
        assert rows_for_id[-1]["parsing_status"] == "resolved"

    def test_no_duplicate_valid_observation_is_created_across_two_resumed_invocations(self, tiny_study, tmp_path, monkeypatch):
        """Running recovery-run twice in a row (simulating a resumed job)
        must never create a second VALID row for an id that already has
        one from the first invocation."""
        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        recovery_path = tmp_path / "recovery_raw.jsonl"

        calls = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls.append(prompt)
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 1}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_recovery_run(args)  # first invocation: resolves all 6
        assert len(calls) == 6

        rv2.cmd_recovery_run(args)  # second invocation (a resumed job)
        assert len(calls) == 6  # no NEW calls made -- everything already has a valid answer

        written = [json.loads(line) for line in recovery_path.read_text().splitlines()]
        assert len(written) == 6  # still exactly one row per id, no duplicates
        by_id = {}
        for row in written:
            by_id.setdefault(row["observation_id"], []).append(row)
        assert all(len(rows) == 1 for rows in by_id.values())


class TestRecoveryOffPeakGuard:
    """The existing DeepSeek off-peak pricing guard (controllability_v2_deepseek_pricing.py,
    unchanged by this recovery work) must also gate every Wave 2 recovery
    dispatch -- never only the original treatment/baseline commands."""

    def _install_fake_clock(self, monkeypatch, start):
        import controllability_v2_deepseek_pricing as pricing

        clock = {"now": start}

        def now_fn():
            return clock["now"]

        def sleep_fn(seconds):
            clock["now"] = clock["now"] + timedelta(seconds=seconds)

        def gate_factory(allow_peak=False):
            return pricing.DeepSeekPricingGate(allow_peak=allow_peak, now_fn=now_fn, sleep_fn=sleep_fn)

        monkeypatch.setattr(rv2, "DeepSeekPricingGate", gate_factory)
        return clock

    def test_recovery_run_dispatch_blocks_during_peak_and_resumes_after_the_boundary(self, tiny_study, tmp_path, monkeypatch):
        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)  # Monday, second peak window
        clock = self._install_fake_clock(monkeypatch, peak_start)

        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])  # everything unresolved -> full recovery plan

        call_times = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            call_times.append(clock["now"])
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        recovery_path = tmp_path / "recovery_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=False, dry_run=False)
        rv2.cmd_recovery_run(args)

        assert len(call_times) == 6
        resume_at = datetime(2026, 9, 21, 10, 1, tzinfo=timezone.utc)  # 10:00 boundary + 60s safety buffer
        assert all(t >= resume_at for t in call_times)

    def test_allow_peak_pricing_bypasses_the_guard_for_recovery_dispatch(self, tiny_study, tmp_path, monkeypatch):
        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
        clock = self._install_fake_clock(monkeypatch, peak_start)

        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])

        call_times = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            call_times.append(clock["now"])
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        recovery_path = tmp_path / "recovery_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "RECOVERY_RESULTS_FILE", str(recovery_path))

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_recovery_run(args)

        assert len(call_times) == 6
        assert all(t == peak_start for t in call_times)  # dispatched immediately, clock never advanced


class TestValidationRun:
    def test_validation_ids_never_collide_with_real_planned_ids_and_agreement_is_recorded(self, tiny_study, tmp_path, monkeypatch, fake_call):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        ids = sorted(planned_index)

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        rows_by_kind = {"treatment": [], "baseline": []}
        # fake_call always answers "A" -- give t1/r1 an original answer of "A" (agrees) and t1/r2 "B" (disagrees)
        for oid, response in zip(ids, ["A", "B", "A", "A", "A", "A"]):
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"], response))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])

        validation_path = tmp_path / "validation_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "VALIDATION_RESULTS_FILE", str(validation_path))
        monkeypatch.setattr(rec, "VALIDATION_SAMPLE_SIZE", 6)  # sample everything valid in this tiny design

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_validation_run(args)

        written = [json.loads(line) for line in validation_path.read_text().splitlines()]
        assert len(written) == 6
        for row in written:
            assert row["validation_observation_id"] not in planned_index
            assert row["original_observation_id"] in planned_index
            assert row["collection_wave"] == "wave2_validation"
            assert row["max_output_tokens"] == 4096
        agreements = {row["original_observation_id"]: row["agrees_with_original"] for row in written}
        assert agreements[ids[0]] is True   # original "A", rerun "A"
        assert agreements[ids[1]] is False  # original "B", rerun "A"

    def test_dry_run_makes_no_calls(self, tiny_study, tmp_path, monkeypatch, fake_call):
        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [])
        validation_path = tmp_path / "validation_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "VALIDATION_RESULTS_FILE", str(validation_path))
        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=True)
        rv2.cmd_validation_run(args)
        assert fake_call == []

    def test_a_failed_validation_attempt_is_retried_on_resume_never_a_valid_one(self, tiny_study, tmp_path, monkeypatch):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        ids = sorted(planned_index)

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        rows_by_kind = {"treatment": [], "baseline": []}
        for oid in ids:
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"], "A"))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])

        validation_path = tmp_path / "validation_raw.jsonl"
        # Pre-seed one FAILED validation attempt and one VALID one.
        failed_vid = f"{ids[0]}::validation_dup"
        valid_vid = f"{ids[1]}::validation_dup"
        _write_jsonl(validation_path, [
            {"observation_id": failed_vid, "validation_observation_id": failed_vid, "original_observation_id": ids[0],
             "parsing_status": "unresolved", "trial_id": planned_index[ids[0]]["trial_id"]},
            {"observation_id": valid_vid, "validation_observation_id": valid_vid, "original_observation_id": ids[1],
             "parsing_status": "resolved", "first_valid_response": {"overall_quality": "A"},
             "trial_id": planned_index[ids[1]]["trial_id"]},
        ])

        calls = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls.append(prompt)
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 1}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "VALIDATION_RESULTS_FILE", str(validation_path))
        monkeypatch.setattr(rec, "VALIDATION_SAMPLE_SIZE", 6)

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=True, dry_run=False)
        rv2.cmd_validation_run(args)

        # 6 planned, 1 already valid (ids[1]) -> 5 new calls (including a retry for ids[0]).
        assert len(calls) == 5
        written = [json.loads(line) for line in validation_path.read_text().splitlines()]
        rows_for_failed = [row for row in written if row.get("observation_id") == failed_vid]
        assert len(rows_for_failed) == 2  # old failed row + new (successful) attempt
        rows_for_valid = [row for row in written if row.get("observation_id") == valid_vid]
        assert len(rows_for_valid) == 1  # never re-attempted


class TestValidationOffPeakGuard:
    def _install_fake_clock(self, monkeypatch, start):
        import controllability_v2_deepseek_pricing as pricing

        clock = {"now": start}

        def now_fn():
            return clock["now"]

        def sleep_fn(seconds):
            clock["now"] = clock["now"] + timedelta(seconds=seconds)

        def gate_factory(allow_peak=False):
            return pricing.DeepSeekPricingGate(allow_peak=allow_peak, now_fn=now_fn, sleep_fn=sleep_fn)

        monkeypatch.setattr(rv2, "DeepSeekPricingGate", gate_factory)
        return clock

    def test_validation_run_dispatch_blocks_during_peak_and_resumes_after_the_boundary(self, tiny_study, tmp_path, monkeypatch):
        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
        clock = self._install_fake_clock(monkeypatch, peak_start)

        study_config = tiny_study["study_config"]
        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        _write_jsonl(source_dir / "treatment_raw.jsonl", [
            _resolved_row(_oid("t1", 1), "t1"), _resolved_row(_oid("t1", 2), "t1"),
            _resolved_row(_oid("t2", 1), "t2"), _resolved_row(_oid("t2", 2), "t2"),
        ])
        _write_jsonl(source_dir / "baseline_raw.jsonl", [
            _resolved_row(_oid("base1", 1), "base1"), _resolved_row(_oid("base2", 1), "base2"),
        ])

        call_times = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            call_times.append(clock["now"])
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        validation_path = tmp_path / "validation_raw.jsonl"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "verify_frozen", lambda *a, **k: (True, None))
        monkeypatch.setattr(rv2, "VALIDATION_RESULTS_FILE", str(validation_path))
        monkeypatch.setattr(rec, "VALIDATION_SAMPLE_SIZE", 6)

        args = argparse.Namespace(source_results=str(source_dir), concurrency=1, allow_peak_pricing=False, dry_run=False)
        rv2.cmd_validation_run(args)

        assert len(call_times) == 6
        resume_at = datetime(2026, 9, 21, 10, 1, tzinfo=timezone.utc)
        assert all(t >= resume_at for t in call_times)


class TestBaselineTextOnlyRun:
    def test_exploratory_run_writes_to_its_own_results_file_and_tags_text_only(self, tmp_path, monkeypatch, fake_call):
        baseline_text_only_trials = [
            _baseline_trial("bt1", "bblk1", "story1_as_a", instruction_condition="text_only"),
            _baseline_trial("bt2", "bblk1", "story2_as_a", instruction_condition="text_only"),
        ]
        trials_path = tmp_path / "baseline_text_only.jsonl"
        _write_jsonl(trials_path, baseline_text_only_trials)
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_TRIALS_FILE", str(trials_path))

        results_path = tmp_path / "exploratory_baseline_text_only.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_BASELINE_TEXT_ONLY_RESULTS_FILE", str(results_path))

        args = argparse.Namespace(
            provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=1, seed=0,
            retry_limit=1, run_id=None, limit=None, concurrency=1, production=False, allow_unfrozen=True,
            allow_peak_pricing=True, dry_run=False,
        )
        rv2.cmd_baseline_text_only(args)

        written = [json.loads(line) for line in results_path.read_text().splitlines()]
        assert len(written) == 2
        assert all(row["trial_meta"]["instruction_condition"] == "text_only" for row in written)
        assert all(row["block_id"] == "bblk1" for row in written)

    def test_baseline_text_only_calls_use_the_4096_recovery_cap_not_the_historical_512(self, tmp_path, monkeypatch, fake_call):
        baseline_text_only_trials = [_baseline_trial("bt1", "bblk1", "story1_as_a", instruction_condition="text_only")]
        trials_path = tmp_path / "baseline_text_only.jsonl"
        _write_jsonl(trials_path, baseline_text_only_trials)
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_TRIALS_FILE", str(trials_path))
        results_path = tmp_path / "exploratory_baseline_text_only.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_BASELINE_TEXT_ONLY_RESULTS_FILE", str(results_path))

        args = argparse.Namespace(
            provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=1, seed=0,
            retry_limit=1, run_id=None, limit=None, concurrency=1, production=False, allow_unfrozen=True,
            allow_peak_pricing=True, dry_run=False,
        )
        rv2.cmd_baseline_text_only(args)

        assert len(fake_call) == 1
        assert fake_call[0]["max_output_tokens"] == rec.RECOVERY_MAX_OUTPUT_TOKENS == 4096
        written = [json.loads(line) for line in results_path.read_text().splitlines()]
        assert written[0]["evaluator"]["max_output_tokens"] == 4096

    def test_results_live_under_wave2_recovery_not_production(self):
        """Item 5: baseline_text_only's production raw file must live under
        RECOVERY_DIR so the GitHub Actions recovery job's own cache restore/
        save (which only covers wave2_recovery/) actually preserves its
        progress across an interrupted/resumed recovery run."""
        assert rv2.BASELINE_TEXT_ONLY_RESULTS_FILE.startswith(rv2.RECOVERY_DIR)
        assert not rv2.BASELINE_TEXT_ONLY_RESULTS_FILE.startswith(rv2.PRODUCTION_DIR)

    def test_a_failed_baseline_text_only_attempt_is_retried_on_resume_never_a_valid_one(self, tmp_path, monkeypatch):
        baseline_text_only_trials = [
            _baseline_trial("bt1", "bblk1", "story1_as_a", instruction_condition="text_only"),
            _baseline_trial("bt2", "bblk2", "story1_as_a", instruction_condition="text_only"),
        ]
        trials_path = tmp_path / "baseline_text_only.jsonl"
        _write_jsonl(trials_path, baseline_text_only_trials)
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_TRIALS_FILE", str(trials_path))
        results_path = tmp_path / "exploratory_baseline_text_only.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_BASELINE_TEXT_ONLY_RESULTS_FILE", str(results_path))

        failed_oid = "context_controllability_v2::deepseek__deepseek-flash__low::bt1::r1"
        valid_oid = "context_controllability_v2::deepseek__deepseek-flash__low::bt2::r1"
        _write_jsonl(results_path, [
            {"observation_id": failed_oid, "trial_id": "bt1", "block_id": "bblk1", "parsing_status": "unresolved",
             "first_valid_response": None},
            {"observation_id": valid_oid, "trial_id": "bt2", "block_id": "bblk2", "parsing_status": "resolved",
             "first_valid_response": {"overall_quality": "A"}},
        ])

        calls = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls.append(prompt)
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 1}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        args = argparse.Namespace(
            provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=1, seed=0,
            retry_limit=1, run_id=None, limit=None, concurrency=1, production=False, allow_unfrozen=True,
            allow_peak_pricing=True, dry_run=False,
        )
        rv2.cmd_baseline_text_only(args)

        assert len(calls) == 1  # only bt1/r1 (the failed one) was retried -- bt2/r1 (valid) was skipped
        written = [json.loads(line) for line in results_path.read_text().splitlines()]
        rows_for_failed = [row for row in written if row["observation_id"] == failed_oid]
        rows_for_valid = [row for row in written if row["observation_id"] == valid_oid]
        assert len(rows_for_failed) == 2  # old failed row + new successful attempt
        assert rows_for_failed[-1]["parsing_status"] == "resolved"
        assert len(rows_for_valid) == 1  # never re-attempted

    def test_off_peak_guard_applies_to_baseline_text_only_dispatch(self, tmp_path, monkeypatch):
        import controllability_v2_deepseek_pricing as pricing

        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
        clock = {"now": peak_start}

        def now_fn():
            return clock["now"]

        def sleep_fn(seconds):
            clock["now"] = clock["now"] + timedelta(seconds=seconds)

        monkeypatch.setattr(rv2, "DeepSeekPricingGate",
                             lambda allow_peak=False: pricing.DeepSeekPricingGate(allow_peak=allow_peak, now_fn=now_fn, sleep_fn=sleep_fn))

        baseline_text_only_trials = [_baseline_trial("bt1", "bblk1", "story1_as_a", instruction_condition="text_only")]
        trials_path = tmp_path / "baseline_text_only.jsonl"
        _write_jsonl(trials_path, baseline_text_only_trials)
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_TRIALS_FILE", str(trials_path))
        results_path = tmp_path / "exploratory_baseline_text_only.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_BASELINE_TEXT_ONLY_RESULTS_FILE", str(results_path))

        call_times = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            call_times.append(clock["now"])
            return {"response_text": "A", "response_model": model, "request_id": "r",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 1}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        args = argparse.Namespace(
            provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=1, seed=0,
            retry_limit=1, run_id=None, limit=None, concurrency=1, production=False, allow_unfrozen=True,
            allow_peak_pricing=False, dry_run=False,
        )
        rv2.cmd_baseline_text_only(args)

        assert len(call_times) == 1
        resume_at = datetime(2026, 9, 21, 10, 1, tzinfo=timezone.utc)
        assert call_times[0] >= resume_at


class TestValidationReport:
    def test_computes_and_writes_agreement_diagnostics_using_the_existing_functions(self, tmp_path, monkeypatch):
        validation_rows = [
            {"validation_observation_id": "v1", "original_observation_id": "o1", "parsing_status": "resolved",
             "agrees_with_original": True, "original_reasoning_tokens": 10,
             "original_first_valid_response": {"overall_quality": "A"}, "first_valid_response": {"overall_quality": "A"}},
            {"validation_observation_id": "v2", "original_observation_id": "o2", "parsing_status": "resolved",
             "agrees_with_original": False, "original_reasoning_tokens": 600,
             "original_first_valid_response": {"overall_quality": "A"}, "first_valid_response": {"overall_quality": "B"}},
            {"validation_observation_id": "v3", "original_observation_id": "o3", "parsing_status": "unresolved",
             "agrees_with_original": None},
        ]
        validation_path = tmp_path / "validation_raw.jsonl"
        _write_jsonl(validation_path, validation_rows)

        diagnostics_path = tmp_path / "validation_diagnostics.json"
        bin_path = tmp_path / "validation_agreement_by_bin.csv"
        context_path = tmp_path / "validation_context_effect_comparison.csv"
        monkeypatch.setattr(rv2, "RECOVERY_DIR", str(tmp_path))
        monkeypatch.setattr(rv2, "VALIDATION_DIAGNOSTICS_FILE", str(diagnostics_path))
        monkeypatch.setattr(rv2, "VALIDATION_AGREEMENT_BY_BIN_FILE", str(bin_path))
        monkeypatch.setattr(rv2, "VALIDATION_CONTEXT_EFFECT_FILE", str(context_path))

        args = argparse.Namespace(validation_results=str(validation_path))
        rv2.cmd_validation_report(args)

        assert diagnostics_path.exists()
        summary = json.loads(diagnostics_path.read_text())
        assert summary["n_validation_attempted"] == 3
        assert summary["n_validation_resolved"] == 2
        assert summary["n_validation_failed"] == 1
        assert summary["agreement_rate"] == 0.5
        assert summary == {**summary, **rec.compute_validation_agreement(validation_rows)}  # reuses the shared function verbatim

        assert bin_path.exists()
        bin_text = bin_path.read_text()
        assert "1-99" in bin_text and "500+" in bin_text

    def test_validation_report_never_touches_the_primary_merged_dataset(self, tmp_path, monkeypatch):
        """cmd_validation_report only ever reads/writes files under the
        validation-specific paths -- it must not import or touch anything
        related to MERGED_TREATMENT_RESULTS_FILE/MERGED_BASELINE_RESULTS_FILE."""
        import inspect
        source = inspect.getsource(rv2.cmd_validation_report)
        assert "MERGED_TREATMENT_RESULTS_FILE" not in source
        assert "MERGED_BASELINE_RESULTS_FILE" not in source

    def test_missing_validation_file_produces_an_empty_but_valid_report(self, tmp_path, monkeypatch):
        diagnostics_path = tmp_path / "validation_diagnostics.json"
        monkeypatch.setattr(rv2, "RECOVERY_DIR", str(tmp_path))
        monkeypatch.setattr(rv2, "VALIDATION_DIAGNOSTICS_FILE", str(diagnostics_path))
        args = argparse.Namespace(validation_results=str(tmp_path / "does_not_exist.jsonl"))
        rv2.cmd_validation_report(args)  # must not raise
        summary = json.loads(diagnostics_path.read_text())
        assert summary["n_validation_attempted"] == 0
        assert summary["agreement_rate"] is None


class TestMerge:
    def _patch_merge_paths(self, monkeypatch, tmp_path, study_config):
        merged_treatment = tmp_path / "merged_treatment.jsonl"
        merged_baseline = tmp_path / "merged_baseline.jsonl"
        diagnostics_path = tmp_path / "merge_diagnostics.json"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "MERGED_TREATMENT_RESULTS_FILE", str(merged_treatment))
        monkeypatch.setattr(rv2, "MERGED_BASELINE_RESULTS_FILE", str(merged_baseline))
        monkeypatch.setattr(rv2, "MERGE_DIAGNOSTICS_FILE", str(diagnostics_path))
        monkeypatch.setattr(rv2, "RECOVERY_DIR", str(tmp_path))
        # This tiny synthetic design is not the real 27720/1320-observation
        # frozen design -- rescale the hard-gate's known-size constants to
        # match it, exactly the way a real run checks against the REAL
        # frozen constants (27720/1320/132) instead.
        monkeypatch.setattr(rv2, "EXPECTED_WAVE1_PLANNED_TOTAL", 6)
        monkeypatch.setattr(rv2, "EXPECTED_BASELINE_TEXT_ONLY_TOTAL", 1)
        monkeypatch.setattr(rv2, "EXPECTED_BASELINE_TEXT_ONLY_UNIQUE_CELLS", 1)
        return merged_treatment, merged_baseline, diagnostics_path

    def _write_complete_baseline_text_only(self, tmp_path, monkeypatch, study_config):
        """A tiny (1 cell x 2 replicates = 2 observations) but FULLY VALID
        baseline_text_only dataset, so the merge gate's baseline_text_only
        check passes alongside the main recovery check."""
        bto_trial = _baseline_trial("bto1", "bto_blk1", "story1_as_a", instruction_condition="text_only")
        bto_trials_path = tmp_path / "baseline_text_only.jsonl"
        _write_jsonl(bto_trials_path, [bto_trial])
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_TRIALS_FILE", str(bto_trials_path))

        # cmd_merge reuses study_config["baseline_replicate_count"] (1, from
        # the tiny_study fixture) for the new condition too -- 1 trial x 1
        # replicate = 1 planned observation, matching
        # EXPECTED_BASELINE_TEXT_ONLY_TOTAL/UNIQUE_CELLS=1 set above.
        bto_index = rec.enumerate_planned_ids(
            [bto_trial], study_config["baseline_replicate_count"],
            study_config["primary_evaluator"]["evaluator_id"], "baseline_text_only",
        )
        bto_results_path = tmp_path / "baseline_text_only_raw.jsonl"
        _write_jsonl(bto_results_path, [_resolved_row(oid, "bto1") for oid in bto_index])
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_RESULTS_FILE", str(bto_results_path))

    def test_merge_writes_split_files_and_passes_the_gate_when_everything_is_complete(self, tiny_study, tmp_path, monkeypatch):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        ids = sorted(planned_index)

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        rows_by_kind = {"treatment": [], "baseline": []}
        for oid in ids[:3]:
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"]))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])

        recovery_path = tmp_path / "recovery_raw.jsonl"
        recovery_rows = []
        for oid in ids[3:]:
            desc = planned_index[oid]
            recovery_rows.append({
                "observation_id": oid, "trial_id": desc["trial_id"], "parsing_status": "resolved",
                "first_valid_response": {"overall_quality": "B"}, "collection_wave": "wave2_recovery",
                "max_output_tokens": 4096, "source_wave": "wave1", "source_run_id": "35355521243",
            })
        _write_jsonl(recovery_path, recovery_rows)

        merged_treatment, merged_baseline, diagnostics_path = self._patch_merge_paths(monkeypatch, tmp_path, study_config)
        self._write_complete_baseline_text_only(tmp_path, monkeypatch, study_config)

        args = argparse.Namespace(source_results=str(source_dir), recovery_results=str(recovery_path))
        rv2.cmd_merge(args)  # must NOT raise -- everything is complete

        all_merged_ids = set()
        for path in (merged_treatment, merged_baseline):
            for line in path.read_text().splitlines():
                all_merged_ids.add(json.loads(line)["observation_id"])
        assert all_merged_ids == set(ids)  # all 6 resolved between wave1 + recovery

        diagnostics = json.loads(diagnostics_path.read_text())
        assert diagnostics["wave1_planned_total"] == 6
        assert diagnostics["wave1_valid_retained"] == 3
        assert diagnostics["recovery_successfully_resolved"] == 3
        assert diagnostics["still_unresolved_after_recovery"] == 0

    def test_merge_hard_fails_and_preserves_outputs_when_recovery_is_incomplete(self, tiny_study, tmp_path, monkeypatch):
        """One of the 6 planned observations is never resolved by either
        Wave 1 or recovery -- the merge gate must refuse to let analysis
        proceed, while still writing the merged/diagnostics files so the
        job can be rerun and inspected."""
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        ids = sorted(planned_index)

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        rows_by_kind = {"treatment": [], "baseline": []}
        for oid in ids[:3]:
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"]))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])

        # Recovery only resolves 2 of the remaining 3 -- ids[5] stays unresolved.
        recovery_path = tmp_path / "recovery_raw.jsonl"
        recovery_rows = []
        for oid in ids[3:5]:
            desc = planned_index[oid]
            recovery_rows.append({
                "observation_id": oid, "trial_id": desc["trial_id"], "parsing_status": "resolved",
                "first_valid_response": {"overall_quality": "B"}, "collection_wave": "wave2_recovery",
                "max_output_tokens": 4096, "source_wave": "wave1", "source_run_id": "35355521243",
            })
        _write_jsonl(recovery_path, recovery_rows)

        merged_treatment, merged_baseline, diagnostics_path = self._patch_merge_paths(monkeypatch, tmp_path, study_config)
        self._write_complete_baseline_text_only(tmp_path, monkeypatch, study_config)

        args = argparse.Namespace(source_results=str(source_dir), recovery_results=str(recovery_path))
        with pytest.raises(SystemExit, match="INCOMPLETE"):
            rv2.cmd_merge(args)

        # The merged/diagnostics outputs are still written and preserved --
        # a failing gate never deletes anything, so the job is rerunnable.
        assert merged_treatment.exists()
        assert merged_baseline.exists()
        diagnostics = json.loads(diagnostics_path.read_text())
        assert diagnostics["still_unresolved_after_recovery"] == 1
        assert diagnostics["final_completeness"] < 1.0

    def test_merge_hard_fails_when_baseline_text_only_is_incomplete(self, tiny_study, tmp_path, monkeypatch):
        study_config = tiny_study["study_config"]
        _tt, _bt, planned_index, _tbi = rv2._load_wave1_planned_index(study_config)
        ids = sorted(planned_index)

        source_dir = tmp_path / "wave1"
        source_dir.mkdir()
        rows_by_kind = {"treatment": [], "baseline": []}
        for oid in ids:
            desc = planned_index[oid]
            rows_by_kind[desc["unit_kind"]].append(_resolved_row(oid, desc["trial_id"]))
        _write_jsonl(source_dir / "treatment_raw.jsonl", rows_by_kind["treatment"])
        _write_jsonl(source_dir / "baseline_raw.jsonl", rows_by_kind["baseline"])
        recovery_path = tmp_path / "recovery_raw.jsonl"
        _write_jsonl(recovery_path, [])  # nothing needed -- wave1 alone is fully valid

        self._patch_merge_paths(monkeypatch, tmp_path, study_config)
        # Deliberately do NOT write a complete baseline_text_only dataset --
        # point at an empty one instead.
        bto_trial = _baseline_trial("bto1", "bto_blk1", "story1_as_a", instruction_condition="text_only")
        bto_trials_path = tmp_path / "baseline_text_only.jsonl"
        _write_jsonl(bto_trials_path, [bto_trial])
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_TRIALS_FILE", str(bto_trials_path))
        # 1 trial x study_config["baseline_replicate_count"] (1) = 1 planned
        # observation, matching EXPECTED_BASELINE_TEXT_ONLY_TOTAL/UNIQUE_CELLS=1 --
        # but the results file is left empty, so it's 0/1 valid: incomplete.
        monkeypatch.setattr(rv2, "BASELINE_TEXT_ONLY_RESULTS_FILE", str(tmp_path / "empty_bto.jsonl"))
        _write_jsonl(tmp_path / "empty_bto.jsonl", [])

        args = argparse.Namespace(source_results=str(source_dir), recovery_results=str(recovery_path))
        with pytest.raises(SystemExit, match="INCOMPLETE or invalid"):
            rv2.cmd_merge(args)
