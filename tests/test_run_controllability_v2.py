"""Tests for run_controllability_v2.py: production/exploratory output
separation, bounded concurrency (valid JSONL, preserved observation ids),
and the production preflight (malformed-manifest rejection, exact planned
counts).

Real API calls are never made: model_providers.call_model is always
monkeypatched. Real result directories are never touched: every test
redirects the module's file-path constants at tmp_path first.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import pytest

import model_providers
import run_controllability_v2 as rv2
from controllability_v2_execution import plan_baseline_execution_order, plan_treatment_execution_order
from controllability_v2_trials import PRIMARY_INSTRUCTION_CONDITIONS


def make_superblock_trials(superblock_id, contrast_id="c1", story_1_id="s1", story_2_id="s2"):
    cells = []
    for instruction_condition in PRIMARY_INSTRUCTION_CONDITIONS:
        for assignment in ("forward", "flipped"):
            for position in ("story1_as_a", "story2_as_a"):
                story_a_id = story_1_id if position == "story1_as_a" else story_2_id
                story_b_id = story_2_id if position == "story1_as_a" else story_1_id
                cells.append({
                    "trial_id": f"{superblock_id}__{instruction_condition}__{assignment}__{position}",
                    "block_id": f"{superblock_id}__{instruction_condition}", "superblock_id": superblock_id,
                    "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab",
                    "instruction_condition": instruction_condition, "contrast_id": contrast_id,
                    "story_1_id": story_1_id, "story_2_id": story_2_id,
                    "story_a_id": story_a_id, "story_b_id": story_b_id,
                    "assignment": assignment, "position": position, "prompt": f"prompt for {superblock_id}",
                })
    return cells


def make_baseline_trials(block_id, story_1_id="s1", story_2_id="s2"):
    return [
        {"trial_id": f"{block_id}__story1_as_a", "block_id": block_id, "type": "context_pairwise",
         "choice_mode": "forced", "response_format": "plain_ab", "instruction_condition": "matched_control",
         "story_1_id": story_1_id, "story_2_id": story_2_id, "story_a_id": story_1_id, "story_b_id": story_2_id,
         "position": "story1_as_a", "prompt": f"baseline prompt for {block_id} A"},
        {"trial_id": f"{block_id}__story2_as_a", "block_id": block_id, "type": "context_pairwise",
         "choice_mode": "forced", "response_format": "plain_ab", "instruction_condition": "matched_control",
         "story_1_id": story_1_id, "story_2_id": story_2_id, "story_a_id": story_2_id, "story_b_id": story_1_id,
         "position": "story2_as_a", "prompt": f"baseline prompt for {block_id} B"},
    ]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def fake_call(monkeypatch):
    calls = {"n": 0}

    def call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
        calls["n"] += 1
        return {"response_text": "A", "response_model": "deepseek-flash-test", "request_id": "req",
                "input_tokens": 10, "output_tokens": 1, "reasoning_tokens": 0,
                "system_fingerprint": "fp_test", "prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 5}

    monkeypatch.setattr(model_providers, "call_model", call_model)
    return calls


def base_args(**overrides):
    defaults = dict(provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=2, seed=1,
                     retry_limit=1, run_id="wave_test", limit=None, concurrency=4, production=False,
                     allow_unfrozen=True, allow_peak_pricing=True, dry_run=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Production / exploratory output separation (item 5)
# ---------------------------------------------------------------------------

class TestOutputSeparation:
    def test_exploratory_run_writes_only_to_the_exploratory_path(self, tmp_path, monkeypatch, fake_call):
        exploratory_path = tmp_path / "exploratory" / "treatment_raw.jsonl"
        production_path = tmp_path / "production" / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(exploratory_path))
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(production_path))

        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        rv2._run(base_args(allow_unfrozen=True), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        assert exploratory_path.exists()
        assert not production_path.exists()

    def test_analysis_default_paths_point_at_production_never_exploratory(self):
        assert rv2.TREATMENT_RESULTS_FILE != rv2.EXPLORATORY_TREATMENT_RESULTS_FILE
        assert rv2.BASELINE_RESULTS_FILE != rv2.EXPLORATORY_BASELINE_RESULTS_FILE
        assert "production" in rv2.TREATMENT_RESULTS_FILE
        assert "exploratory" in rv2.EXPLORATORY_TREATMENT_RESULTS_FILE

    def test_analyze_controllability_v2_defaults_to_the_production_paths(self):
        import analyze_controllability_v2 as av2

        assert av2.TREATMENT_RESULTS_FILE == rv2.TREATMENT_RESULTS_FILE
        assert av2.BASELINE_RESULTS_FILE == rv2.BASELINE_RESULTS_FILE


# ---------------------------------------------------------------------------
# Bounded concurrency (item 6)
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_writes_produce_valid_jsonl_with_every_planned_id_present_once(self, tmp_path, monkeypatch, fake_call):
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))

        trials_file = tmp_path / "treatment.jsonl"
        trials = make_superblock_trials("sb1") + make_superblock_trials("sb2") + make_superblock_trials("sb3")
        write_jsonl(trials_file, trials)

        rv2._run(base_args(concurrency=8, replicates=3), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        lines = results_path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]  # raises if any line is not valid JSON
        assert len(rows) == 3 * 8 * 3  # 3 superblocks x 8 cells x 3 replicates
        ids = [r["observation_id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert fake_call["n"] == len(rows)

    def test_concurrency_preserves_the_planned_execution_order_index_values(self, tmp_path, monkeypatch, fake_call):
        """Concurrency changes WHEN each call runs, never the planned
        execution_order_index recorded on the row -- the set of indices
        written must be exactly 1..N regardless of concurrency."""
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))

        trials_file = tmp_path / "treatment.jsonl"
        trials = make_superblock_trials("sb1") + make_superblock_trials("sb2")
        write_jsonl(trials_file, trials)

        rv2._run(base_args(concurrency=8, replicates=1), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        indices = sorted(r["execution_order_index"] for r in rows)
        assert indices == list(range(1, len(rows) + 1))

    def test_sequential_and_concurrent_execution_write_the_same_set_of_observation_ids(self, tmp_path, monkeypatch, fake_call):
        trials_file = tmp_path / "treatment.jsonl"
        trials = make_superblock_trials("sb1") + make_superblock_trials("sb2")
        write_jsonl(trials_file, trials)

        seq_path = tmp_path / "seq.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(seq_path))
        rv2._run(base_args(concurrency=1, replicates=1), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        conc_path = tmp_path / "conc.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(conc_path))
        rv2._run(base_args(concurrency=8, replicates=1), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        seq_ids = {json.loads(line)["observation_id"] for line in seq_path.read_text(encoding="utf-8").splitlines()}
        conc_ids = {json.loads(line)["observation_id"] for line in conc_path.read_text(encoding="utf-8").splitlines()}
        assert seq_ids == conc_ids


# ---------------------------------------------------------------------------
# Resumable execution at the CLI level (item 4) -- see also
# test_controllability_v2_execution.py's pure-function tests
# ---------------------------------------------------------------------------

class TestResumeAtCliLevel:
    def test_clean_initial_run_executes_every_planned_observation(self, tmp_path, monkeypatch, fake_call):
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        rv2._run(base_args(replicates=2), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")
        assert fake_call["n"] == 16  # 8 cells x 2 replicates
        assert len(results_path.read_text(encoding="utf-8").splitlines()) == 16

    def test_interrupted_then_resumed_run_only_executes_whats_missing(self, tmp_path, monkeypatch, fake_call):
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        # "interrupted": manually write results for only 3 of the 8 planned observations
        plan = plan_treatment_execution_order(make_superblock_trials("sb1"), replicate_count=1, seed=1)
        from controllability_v2_execution import attach_observation_ids
        attach_observation_ids(plan, "context_controllability_v2", "deepseek__deepseek-flash__low")
        partial_rows = [{"observation_id": e["observation_id"], "parsing_status": "resolved"} for e in plan[:3]]
        write_jsonl(results_path, partial_rows)

        rv2._run(base_args(replicates=1, seed=1), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        assert fake_call["n"] == 5  # only the 5 missing observations were executed
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 8  # 3 pre-existing + 5 new
        ids = [r["observation_id"] for r in rows]
        assert len(ids) == len(set(ids))  # never duplicated

    def test_fully_resumed_run_makes_no_further_calls(self, tmp_path, monkeypatch, fake_call):
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        rv2._run(base_args(replicates=1), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")
        first_call_count = fake_call["n"]
        rv2._run(base_args(replicates=1), str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")
        assert fake_call["n"] == first_call_count  # resume added nothing


# ---------------------------------------------------------------------------
# Production preflight (item 11)
# ---------------------------------------------------------------------------

def make_study_config(treatment_file, baseline_file, **overrides):
    config = {
        "design_version": "v2", "experiment_id": "context_controllability_v2", "corpus_id": "c",
        "primary_evaluator": {"evaluator_id": "deepseek__deepseek-flash__low", "provider": "deepseek",
                               "requested_model": "deepseek-flash", "reasoning_profile": "low"},
        "replication_evaluators": [],
        "treatment_replicate_count": 10, "baseline_replicate_count": 10,
        "random_seed": 20260917, "retry_limit": 3, "max_output_tokens": 512,
        "treatment_manifest_file": treatment_file, "baseline_manifest_file": baseline_file,
        "status": "draft",
    }
    config.update(overrides)
    return config


class TestPreflight:
    def test_reports_exact_planned_production_observation_counts(self, monkeypatch, tmp_path):
        # use the real, generated v2 manifests (already validated at exactly 2640/132)
        from controllability_v2_trials import BASELINE_TRIALS_FILE, TRIALS_FILE

        config = make_study_config(TRIALS_FILE, BASELINE_TRIALS_FILE)
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(tmp_path / "t.jsonl"))
        monkeypatch.setattr(rv2, "BASELINE_RESULTS_FILE", str(tmp_path / "b.jsonl"))

        ok, checks, context = rv2.run_preflight(config, {}, {}, concurrency=32)
        treatment_check = dict((n, (o, d)) for n, o, d in checks)["treatment_manifest_has_2640_unique_cells"]
        baseline_check = dict((n, (o, d)) for n, o, d in checks)["baseline_manifest_has_132_unique_cells"]
        assert treatment_check[0] is True
        assert baseline_check[0] is True

        planned_treatment = len(context["treatment_trials"]) * config["treatment_replicate_count"]
        planned_baseline = len(context["baseline_trials"]) * config["baseline_replicate_count"]
        assert planned_treatment == 26400
        assert planned_baseline == 1320

    def test_rejects_a_malformed_treatment_manifest_missing_a_cell(self, monkeypatch, tmp_path):
        from controllability_v2_trials import BASELINE_TRIALS_FILE

        broken_treatment = tmp_path / "broken_treatment.jsonl"
        write_jsonl(broken_treatment, make_superblock_trials("sb1")[:-1])  # 7/8 cells -- incomplete superblock

        config = make_study_config(str(broken_treatment), BASELINE_TRIALS_FILE)
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(tmp_path / "t.jsonl"))
        monkeypatch.setattr(rv2, "BASELINE_RESULTS_FILE", str(tmp_path / "b.jsonl"))

        ok, checks, context = rv2.run_preflight(config, {}, {}, concurrency=32)
        checks_by_name = {n: o for n, o, d in checks}
        assert checks_by_name["every_treatment_superblock_structurally_complete"] is False
        assert ok is False

    def test_rejects_a_baseline_manifest_with_the_wrong_unique_cell_count(self, monkeypatch, tmp_path):
        from controllability_v2_trials import TRIALS_FILE

        small_baseline = tmp_path / "small_baseline.jsonl"
        write_jsonl(small_baseline, make_baseline_trials("b1"))  # only 2 cells, not 132

        config = make_study_config(TRIALS_FILE, str(small_baseline))
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(tmp_path / "t.jsonl"))
        monkeypatch.setattr(rv2, "BASELINE_RESULTS_FILE", str(tmp_path / "b.jsonl"))

        ok, checks, context = rv2.run_preflight(config, {}, {}, concurrency=32)
        checks_by_name = {n: o for n, o, d in checks}
        assert checks_by_name["baseline_manifest_has_132_unique_cells"] is False
        assert ok is False

    def test_rejects_mismatched_cli_settings(self, monkeypatch, tmp_path):
        from controllability_v2_trials import BASELINE_TRIALS_FILE, TRIALS_FILE

        config = make_study_config(TRIALS_FILE, BASELINE_TRIALS_FILE)
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(tmp_path / "t.jsonl"))
        monkeypatch.setattr(rv2, "BASELINE_RESULTS_FILE", str(tmp_path / "b.jsonl"))

        ok, checks, context = rv2.run_preflight(config, {"replicates": 3}, {}, concurrency=32)
        checks_by_name = {n: o for n, o, d in checks}
        assert checks_by_name["cli_settings_match_frozen_evaluator_and_replicate_counts"] is False
        assert ok is False

    def test_detects_duplicate_observation_ids_in_existing_production_output(self, monkeypatch, tmp_path):
        from controllability_v2_trials import BASELINE_TRIALS_FILE, TRIALS_FILE

        config = make_study_config(TRIALS_FILE, BASELINE_TRIALS_FILE)
        dup_path = tmp_path / "t.jsonl"
        write_jsonl(dup_path, [{"observation_id": "dup"}, {"observation_id": "dup"}])
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(dup_path))
        monkeypatch.setattr(rv2, "BASELINE_RESULTS_FILE", str(tmp_path / "b.jsonl"))

        ok, checks, context = rv2.run_preflight(config, {}, {}, concurrency=32)
        checks_by_name = {n: o for n, o, d in checks}
        assert checks_by_name["production_output_has_no_duplicate_observation_ids"] is False

    def test_preflight_reports_pricing_state_and_makes_no_api_calls_either_way(self, monkeypatch, tmp_path):
        from controllability_v2_trials import BASELINE_TRIALS_FILE, TRIALS_FILE

        def must_not_be_called(*a, **k):
            pytest.fail("preflight must never call a provider SDK")

        monkeypatch.setattr(model_providers, "call_model", must_not_be_called)
        config = make_study_config(TRIALS_FILE, BASELINE_TRIALS_FILE)
        monkeypatch.setattr(rv2, "TREATMENT_RESULTS_FILE", str(tmp_path / "t.jsonl"))
        monkeypatch.setattr(rv2, "BASELINE_RESULTS_FILE", str(tmp_path / "b.jsonl"))

        for allow_peak in (False, True):
            ok, checks, context = rv2.run_preflight(config, {}, {}, concurrency=32, allow_peak_pricing=allow_peak)
            assert context["allow_peak_pricing"] is allow_peak
            assert isinstance(context["pricing_window_off_peak"], bool)
            if not context["pricing_window_off_peak"]:
                assert context["pricing_window_next_off_peak"] is not None
            # pricing state must never gate the overall preflight result
            checks_by_name = {n: o for n, o, d in checks}
            assert "pricing" not in "".join(checks_by_name).lower() or True  # no pricing entry in `checks` at all
            assert not any("pricing" in name for name in checks_by_name)


# ---------------------------------------------------------------------------
# DeepSeek off-peak pricing guard wiring (production dispatch guard)
# ---------------------------------------------------------------------------

class RecordingGate:
    def __init__(self, allow_peak=False):
        self.allow_peak = allow_peak
        self.wait_calls = 0

    def wait_until_dispatch_allowed(self):
        self.wait_calls += 1


class TestMakeCallFnPricingGateWiring:
    def test_deepseek_checks_the_gate_before_every_dispatch(self, monkeypatch):
        order = []
        gate = RecordingGate()

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            order.append("dispatched")
            return {"response_text": "A"}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        original_wait = gate.wait_until_dispatch_allowed

        def recording_wait():
            order.append("waited")
            original_wait()

        gate.wait_until_dispatch_allowed = recording_wait
        call_fn = rv2.make_call_fn("deepseek", "deepseek-flash", "low", 512, {}, gate)
        call_fn({"prompt": "p"})

        assert order == ["waited", "dispatched"]  # gate consulted BEFORE dispatch
        assert gate.wait_calls == 1

    def test_other_providers_never_consult_the_gate(self, monkeypatch):
        gate = RecordingGate()

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            return {"response_text": "A"}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        for provider in ("anthropic", "openai", "gemini"):
            call_fn = rv2.make_call_fn(provider, "some-model", "low", 512, {}, gate)
            call_fn({"prompt": "p"})
        assert gate.wait_calls == 0

    def test_gate_is_checked_on_every_retry_attempt(self, monkeypatch):
        from controllability_v2_execution import run_one_observation

        gate = RecordingGate()

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            return {"response_text": "A"}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        call_fn = rv2.make_call_fn("deepseek", "deepseek-flash", "low", 512, {}, gate)
        # force 3 attempts by making the first two invalid
        responses = iter(["not a valid answer", "still not valid", "A"])

        def flaky_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            return {"response_text": next(responses)}

        monkeypatch.setattr(model_providers, "call_model", flaky_call_model)
        run_one_observation({"prompt": "p", "trial_id": "t1"}, call_fn, retry_limit=2)
        assert gate.wait_calls == 3  # once per attempt, never skipped on a retry

    def test_run_uses_a_single_shared_gate_instance_for_the_whole_plan(self, tmp_path, monkeypatch, fake_call):
        """Every worker in the concurrent pool must consult the SAME gate --
        otherwise a peak-window transition wouldn't correctly pause every
        queued worker, only whichever ones happened to share an instance."""
        created = []

        class CountingGate(RecordingGate):
            def __init__(self, allow_peak=False):
                super().__init__(allow_peak)
                created.append(self)

        monkeypatch.setattr(rv2, "DeepSeekPricingGate", CountingGate)
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1") + make_superblock_trials("sb2"))

        rv2._run(base_args(concurrency=8, replicates=1, allow_peak_pricing=False),
                  str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        assert len(created) == 1  # exactly one gate constructed for the entire run
        assert created[0].wait_calls == 16  # every one of the 16 dispatches consulted it


class TestDeepSeekPeakGuardIntegration:
    """End-to-end: a real DeepSeekPricingGate wired into _run() with an
    injected fake clock, so these never actually sleep or depend on the
    real wall-clock time."""

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

    def test_production_dispatch_blocks_during_peak_and_resumes_after_the_boundary(self, tmp_path, monkeypatch):
        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)  # Monday, second peak window
        clock = self._install_fake_clock(monkeypatch, peak_start)

        call_times = []

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            call_times.append(clock["now"])
            return {"response_text": "A", "response_model": "deepseek-flash-test", "request_id": "req",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        rv2._run(base_args(concurrency=4, replicates=1, allow_peak_pricing=False),
                  str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        assert len(call_times) == 8
        resume_at = datetime(2026, 9, 21, 10, 1, tzinfo=timezone.utc)  # 10:00 boundary + 60s buffer
        assert all(t >= resume_at for t in call_times)

    def test_allow_peak_pricing_bypasses_the_guard_and_dispatches_immediately(self, tmp_path, monkeypatch):
        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
        clock = self._install_fake_clock(monkeypatch, peak_start)

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            return {"response_text": "A", "response_model": "deepseek-flash-test", "request_id": "req",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        rv2._run(base_args(concurrency=4, replicates=1, allow_peak_pricing=True),
                  str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        assert clock["now"] == peak_start  # never advanced -- no waiting occurred

    def test_non_deepseek_provider_is_unaffected_even_during_a_simulated_deepseek_peak(self, tmp_path, monkeypatch):
        peak_start = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
        clock = self._install_fake_clock(monkeypatch, peak_start)

        def fake_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            return {"response_text": "A", "response_model": "claude-sonnet-5-test", "request_id": "req",
                    "input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0}

        monkeypatch.setattr(model_providers, "call_model", fake_call_model)
        results_path = tmp_path / "treatment_raw.jsonl"
        monkeypatch.setattr(rv2, "EXPLORATORY_TREATMENT_RESULTS_FILE", str(results_path))
        trials_file = tmp_path / "treatment.jsonl"
        write_jsonl(trials_file, make_superblock_trials("sb1"))

        rv2._run(base_args(provider="anthropic", model="claude-sonnet-5", concurrency=4, replicates=1, allow_peak_pricing=False),
                  str(trials_file), "superblock_id", plan_treatment_execution_order, "treatment")

        assert clock["now"] == peak_start  # the gate was constructed but never consulted for a non-DeepSeek provider
