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


class TestMerge:
    def test_merge_writes_split_files_and_diagnostics(self, tiny_study, tmp_path, monkeypatch):
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

        merged_treatment = tmp_path / "merged_treatment.jsonl"
        merged_baseline = tmp_path / "merged_baseline.jsonl"
        diagnostics_path = tmp_path / "merge_diagnostics.json"
        monkeypatch.setattr(rv2, "load_study_config", lambda *a, **k: study_config)
        monkeypatch.setattr(rv2, "MERGED_TREATMENT_RESULTS_FILE", str(merged_treatment))
        monkeypatch.setattr(rv2, "MERGED_BASELINE_RESULTS_FILE", str(merged_baseline))
        monkeypatch.setattr(rv2, "MERGE_DIAGNOSTICS_FILE", str(diagnostics_path))
        monkeypatch.setattr(rv2, "RECOVERY_DIR", str(tmp_path))

        args = argparse.Namespace(source_results=str(source_dir), recovery_results=str(recovery_path))
        rv2.cmd_merge(args)

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
