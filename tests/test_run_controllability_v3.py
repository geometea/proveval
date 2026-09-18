"""run_controllability_v3.py: preflight (zero API calls, every failure
mode), dry runs, exploratory/pilot/production output separation, append-only
valid-only resume, concurrency, the time budget, and the pilot report.
model_providers.call_model is always monkeypatched; every path constant is
redirected to tmp_path so no real results directory is ever touched."""

import argparse
import json
import os

import pytest

import model_providers
import run_controllability_v3 as rv3
from controllability_v3_execution import iter_rows, load_valid_completed_ids


@pytest.fixture
def redirect(tmp_path, monkeypatch):
    paths = {
        "PRIMARY_RESULTS_FILE": tmp_path / "production" / "primary_raw.jsonl",
        "HOLDOUT_RESULTS_FILE": tmp_path / "production" / "holdout_raw.jsonl",
        "PILOT_RESULTS_FILE": tmp_path / "pilot" / "pilot_raw.jsonl",
        "EXPLORATORY_PRIMARY_RESULTS_FILE": tmp_path / "exploratory" / "primary_raw.jsonl",
        "EXPLORATORY_HOLDOUT_RESULTS_FILE": tmp_path / "exploratory" / "holdout_raw.jsonl",
        "EXPLORATORY_PILOT_RESULTS_FILE": tmp_path / "exploratory" / "pilot_raw.jsonl",
    }
    for name, path in paths.items():
        monkeypatch.setattr(rv3, name, str(path))
    monkeypatch.setattr(rv3, "PRODUCTION_DIR", str(tmp_path / "production"))
    monkeypatch.setattr(rv3, "RUN_KIND_FILES", {
        "primary": {"families": ("primary_context", "primary_nocontext"), "production": str(paths["PRIMARY_RESULTS_FILE"]), "exploratory": str(paths["EXPLORATORY_PRIMARY_RESULTS_FILE"])},
        "holdout": {"families": ("holdout",), "production": str(paths["HOLDOUT_RESULTS_FILE"]), "exploratory": str(paths["EXPLORATORY_HOLDOUT_RESULTS_FILE"])},
        "pilot": {"families": ("pilot",), "production": str(paths["PILOT_RESULTS_FILE"]), "exploratory": str(paths["EXPLORATORY_PILOT_RESULTS_FILE"])},
    })
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    return {k: str(v) for k, v in paths.items()}


@pytest.fixture
def fake_call(monkeypatch):
    state = {"n": 0, "responses": None}

    def call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
        state["n"] += 1
        text = "A" if state["responses"] is None else state["responses"](state["n"], prompt)
        return {"response_text": text, "response_model": "deepseek-flash-test", "request_id": "req", "input_tokens": 10,
                "output_tokens": 1, "reasoning_tokens": 0, "system_fingerprint": "fp", "prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 5}
    monkeypatch.setattr(model_providers, "call_model", call_model)
    return state


@pytest.fixture
def forbid_calls(monkeypatch):
    def call_model(*a, **k):
        raise AssertionError("a model API call was attempted")
    monkeypatch.setattr(model_providers, "call_model", call_model)


def args(**overrides):
    defaults = dict(provider=None, model=None, reasoning_profile=None, replicates=None, seed=None, retry_limit=None, run_id="t",
                    limit=2, concurrency=4, time_budget_minutes=None, production=False, allow_unfrozen=False, allow_peak_pricing=True, dry_run=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def exploratory(**overrides):
    return args(allow_unfrozen=True, provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=1, seed=1, retry_limit=0, **overrides)


class TestDesignArithmetic:
    def test_exact_expected_counts(self):
        a = rv3.design_arithmetic()
        assert (a["stories"], a["story_pairs"], a["cues"], a["interventions"]) == (12, 66, 5, 8)
        assert a["context_unique_cells"] == 10560 and a["context_judgments"] == 105600
        assert a["nocontext_unique_cells"] == 1056 and a["nocontext_judgments"] == 10560
        assert a["primary_total_judgments"] == 116160
        assert a["holdout_unique_cells"] == 1320 and a["holdout_judgments"] == 13200


class TestPreflight:
    def test_preflight_passes_on_the_real_design_and_makes_no_api_calls(self, redirect, forbid_calls, capsys):
        rv3.cmd_preflight(argparse.Namespace(concurrency=32, allow_peak_pricing=False, skip_credential_check=False))
        out = capsys.readouterr().out
        assert "v3 preflight PASSED. No API calls were made." in out
        for line in ("Stories: 12", "Story pairs: 66", "Cues: 5", "Interventions: 8", "Context-present unique cells: 10,560",
                     "Context-present judgments @ 10 reps: 105,600", "No-context unique cells: 1,056",
                     "No-context judgments @ 10 reps: 10,560", "Primary total judgments: 116,160"):
            assert line in out
        assert "[FAIL]" not in out

    def test_preflight_fails_without_credential_unless_skipped(self, redirect, forbid_calls, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY")
        with pytest.raises(SystemExit):
            rv3.cmd_preflight(argparse.Namespace(concurrency=32, allow_peak_pricing=False, skip_credential_check=False))
        rv3.cmd_preflight(argparse.Namespace(concurrency=32, allow_peak_pricing=False, skip_credential_check=True))

    def _failing(self, checks, name):
        return [c for c in checks if c[0] == name and not c[1]]

    def test_manifest_mutation_fails(self, redirect, forbid_calls, monkeypatch):
        monkeypatch.setattr(rv3, "verify_frozen", lambda: (False, "Frozen v3 files no longer match the lock: ['context_manifest_sha256']"))
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "v3_study_lock_exists_and_hashes_validate")

    def test_v2_modification_fails(self, redirect, forbid_calls, monkeypatch):
        monkeypatch.setattr(rv3, "verify_v2_unchanged", lambda: (False, "v2 files modified: ['data/controllability_v2_trials.jsonl']"))
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "v2_files_unchanged_and_v2_lock_validates")

    def test_missing_cells_and_duplicate_ids_fail(self, redirect, forbid_calls, monkeypatch):
        real = rv3.load_manifest

        def truncated(path):
            rows = real(path)
            return rows[:-1] if "context_trials" in path and "nocontext" not in path else rows
        monkeypatch.setattr(rv3, "load_manifest", truncated)
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "context_manifest_has_10560_unique_cells")
        assert self._failing(checks, "every_context_block_counterbalanced_and_all_interventions_present")

        def duplicated(path):
            rows = real(path)
            return rows + rows[:1] if "nocontext" in path else rows
        monkeypatch.setattr(rv3, "load_manifest", duplicated)
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "nocontext_manifest_has_1056_unique_cells")

    def test_counterbalancing_failure_fails(self, redirect, forbid_calls, monkeypatch):
        real = rv3.load_manifest

        def swapped(path):
            rows = real(path)
            if "context_trials" in path and "nocontext" not in path:
                rows[0] = dict(rows[0], assignment="flipped")  # two flipped/story1_as_a cells in one block
            return rows
        monkeypatch.setattr(rv3, "load_manifest", swapped)
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert self._failing(checks, "every_context_block_counterbalanced_and_all_interventions_present")

    def test_unexpected_replicate_count_fails(self, redirect, forbid_calls):
        config = rv3.load_study_config()
        config["replicate_count"] = 3
        ok, checks, _ = rv3.run_preflight(config)
        assert not ok and self._failing(checks, "replicate_counts_as_designed")

    def test_incorrect_instruction_mapping_fails(self, redirect, forbid_calls):
        config = rv3.load_study_config()
        config["instruction_strings"]["I4"] = "different wording"
        ok, checks, _ = rv3.run_preflight(config)
        assert not ok and self._failing(checks, "instruction_mapping_matches_frozen_wording")

    def test_unknown_story_fails(self, redirect, forbid_calls, monkeypatch):
        real = rv3.load_story_texts
        monkeypatch.setattr(rv3, "load_story_texts", lambda: {k: v for k, v in real().items() if k != "gilbert"})
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "prompt_hashes_rebuild_from_story_files_and_all_stories_known")

    def test_token_ceiling_below_recovery_era_fails(self, redirect, forbid_calls):
        config = rv3.load_study_config()
        config["max_output_tokens"] = 512
        ok, checks, _ = rv3.run_preflight(config)
        assert not ok and self._failing(checks, "cli_settings_match_frozen_evaluator_replicates_and_token_ceiling")

    def test_duplicate_valid_or_pilot_rows_in_production_output_fail(self, redirect, forbid_calls):
        os.makedirs(os.path.dirname(redirect["PRIMARY_RESULTS_FILE"]))
        row = {"planned_observation_id": "v3::primary_context::x", "parsing_status": "resolved", "family": "primary_context"}
        with open(redirect["PRIMARY_RESULTS_FILE"], "w") as f:
            f.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "production_and_pilot_outputs_have_no_duplicate_valid_or_foreign_ids")
        with open(redirect["PRIMARY_RESULTS_FILE"], "w") as f:
            f.write(json.dumps({"planned_observation_id": "v3::pilot::x", "parsing_status": "resolved", "family": "pilot"}) + "\n")
        ok, checks, _ = rv3.run_preflight(rv3.load_study_config())
        assert not ok and self._failing(checks, "production_and_pilot_outputs_have_no_duplicate_valid_or_foreign_ids")


class TestRuns:
    def test_dry_run_makes_no_calls_and_writes_nothing(self, redirect, forbid_calls, capsys):
        rv3._run(args(dry_run=True), "primary")
        assert "Dry run: no network calls made." in capsys.readouterr().out
        assert not os.path.exists(redirect["PRIMARY_RESULTS_FILE"])

    def test_real_run_requires_exactly_one_mode(self, redirect, forbid_calls):
        with pytest.raises(SystemExit):
            rv3._run(args(), "primary")
        with pytest.raises(SystemExit):
            rv3._run(args(production=True, allow_unfrozen=True), "primary")

    def test_exploratory_run_writes_only_to_the_exploratory_path(self, redirect, fake_call):
        rv3._run(exploratory(), "pilot")
        assert os.path.exists(redirect["EXPLORATORY_PILOT_RESULTS_FILE"]) and not os.path.exists(redirect["PILOT_RESULTS_FILE"])
        rows = list(iter_rows(redirect["EXPLORATORY_PILOT_RESULTS_FILE"]))
        assert len(rows) == 8 == fake_call["n"] and all(r["collection"] == "exploratory" for r in rows)

    def test_production_pilot_run_writes_pilot_family_rows_with_frozen_settings(self, redirect, fake_call):
        rv3._run(args(production=True, limit=3), "pilot")
        rows = list(iter_rows(redirect["PILOT_RESULTS_FILE"]))
        assert len(rows) == 12 and all(r["family"] == "pilot" and r["collection"] == "pilot" for r in rows)
        assert all(r["max_output_tokens"] == 4096 and r["reasoning_effort"] == "low" and r["model"] == "deepseek-flash" for r in rows)
        assert all(r["planned_observation_id"].startswith("v3::pilot::") for r in rows)
        assert rows[0]["prompt_sha256"] and rows[0]["latency_seconds"] is not None

    def test_production_primary_run_interleaves_context_and_nocontext(self, redirect, fake_call, monkeypatch):
        # limit selects the first N blocks of context+nocontext concatenation; use a larger limit via a custom loader
        real = rv3._load_trials_for

        def small(run_kind, config, stress_config=None):
            trials = real(run_kind, config)
            ctx = [t for t in trials if t["family"] == "primary_context"][:8]
            noctx = [t for t in trials if t["family"] == "primary_nocontext"][:4]
            return ctx + noctx
        monkeypatch.setattr(rv3, "_load_trials_for", small)
        rv3._run(args(production=True, limit=None), "primary")
        rows = list(iter_rows(redirect["PRIMARY_RESULTS_FILE"]))
        assert {r["family"] for r in rows} == {"primary_context", "primary_nocontext"}
        assert len(rows) == 12 * 10  # 10 frozen replicates

    def test_production_refuses_when_preflight_fails(self, redirect, forbid_calls, monkeypatch):
        monkeypatch.setattr(rv3, "verify_frozen", lambda: (False, "nope"))
        with pytest.raises(SystemExit, match="preflight FAILED"):
            rv3._run(args(production=True), "primary")
        assert not os.path.exists(redirect["PRIMARY_RESULTS_FILE"])

    def test_resume_reruns_only_failed_observations_and_never_overwrites(self, redirect, fake_call):
        fake_call["responses"] = lambda n, prompt: "I can't decide" if n <= 3 else "A"
        rv3._run(exploratory(concurrency=1), "pilot")
        path = redirect["EXPLORATORY_PILOT_RESULTS_FILE"]
        first_pass = open(path).read()
        rows = list(iter_rows(path))
        assert len(rows) == 8 and sum(r["parsing_status"] == "unresolved" for r in rows) == 3
        assert len(load_valid_completed_ids(path)) == 5

        fake_call["responses"] = None
        fake_call["n"] = 0
        rv3._run(exploratory(concurrency=1), "pilot")
        assert fake_call["n"] == 3  # only the three failures were re-run
        after = open(path).read()
        assert after.startswith(first_pass)  # append-only: the original lines are untouched
        assert len(load_valid_completed_ids(path)) == 8
        rows = list(iter_rows(path))
        assert len(rows) == 11

        fake_call["n"] = 0
        rv3._run(exploratory(concurrency=1), "pilot")
        assert fake_call["n"] == 0  # fully resumed: nothing to do

    def test_concurrent_run_yields_valid_jsonl_with_every_id_once(self, redirect, fake_call):
        rv3._run(exploratory(concurrency=8, limit=6), "primary")
        rows = list(iter_rows(redirect["EXPLORATORY_PRIMARY_RESULTS_FILE"]))
        ids = [r["planned_observation_id"] for r in rows]
        assert len(ids) == len(set(ids)) == 24
        assert sorted(r["execution_order_index"] for r in rows) == list(range(1, 25))

    def test_time_budget_stops_submission_without_calls(self, redirect, fake_call, capsys):
        rv3._run(exploratory(time_budget_minutes=-1), "pilot")
        assert fake_call["n"] == 0
        assert "Not submitted (time budget): 8" in capsys.readouterr().out

    def test_run_plan_deadline_returns_unsubmitted_count(self):
        import time
        done = []
        n = rv3.run_plan([1, 2, 3], done.append, concurrency=2, deadline=time.time() - 1)
        assert n == 3 and done == []
        n = rv3.run_plan([1, 2, 3], done.append, concurrency=2, deadline=None)
        assert n == 0 and sorted(done) == [1, 2, 3]


class TestPilotReport:
    def make_rows(self, intervention, context_present, n, choice="A", status="valid", tokens=20):
        return [{"family": "pilot", "intervention_id": intervention, "context_present": context_present, "parsing_status": "resolved" if status == "valid" else "unresolved",
                 "first_attempt_status": status, "parsed_choice": choice if status == "valid" else None, "output_tokens": tokens, "reasoning_tokens": 100,
                 "latency_seconds": 1.0} for _ in range(n)]

    def test_report_flags_pathologies_and_reports_no_effect_sizes(self):
        config = rv3.load_study_config()
        rows = self.make_rows("I0", True, 30, "A") + self.make_rows("I1", True, 15, "A") + self.make_rows("I1", True, 15, "B") \
            + self.make_rows("I2", False, 10, status="invalid") + self.make_rows("I3", True, 10, tokens=4096)
        report = rv3.build_pilot_report(rows, config)
        flags = " ".join(report["flags"])
        assert "I0 ctx=True: P(A)=1.00" in flags and "I2 ctx=False: resolved rate 0.00" in flags and "4096-token ceiling" in flags
        assert not any("I1" in f for f in report["flags"])
        keys = {k for entry in report["per_condition"] for k in entry}
        assert not any("effect" in k or "suppression" in k or "drift" in k for k in keys)

    def test_non_pilot_rows_are_flagged(self):
        rows = self.make_rows("I0", True, 5)
        rows[0]["family"] = "primary_context"
        report = rv3.build_pilot_report(rows, rv3.load_study_config())
        assert any("non-pilot rows" in f for f in report["flags"])
