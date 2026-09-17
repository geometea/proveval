"""Tests for batch_run.py: submit / status / collect.

Reuses run_batch.py's real selection/completion functions (imported, not
duplicated) against small synthetic trials/results files. Provider network
calls are never made: model_providers.submit_batch/batch_status/
collect_batch are monkeypatched directly (batch_run.py's own job is request/
response bookkeeping and trial-selection reuse, not provider protocol
translation -- that's tested against real payload shapes in
test_model_providers.py).
"""

import hashlib
import json
import sys

import pytest

import batch_run
import model_providers


def make_trial(trial_id, block_id, assignment, position, replicate_hint="s1_vs_s2"):
    prompt = f"prompt for {trial_id}"
    return {
        "trial_id": trial_id,
        "block_id": block_id,
        "type": "context_pairwise",
        "choice_mode": "forced",
        "response_format": "plain_ab",
        "evaluation_regime": "naturalistic",
        "contrast_id": "provenance_human_vs_llm",
        "story_1_id": "s1",
        "story_2_id": "s2",
        "assignment": assignment,
        "position": position,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "experiment_id": "context_controllability_v1",
    }


def make_block(block_id):
    return [
        make_trial(f"{block_id}__forward__story1_as_a", block_id, "forward", "story1_as_a"),
        make_trial(f"{block_id}__forward__story2_as_a", block_id, "forward", "story2_as_a"),
        make_trial(f"{block_id}__flipped__story1_as_a", block_id, "flipped", "story1_as_a"),
        make_trial(f"{block_id}__flipped__story2_as_a", block_id, "flipped", "story2_as_a"),
    ]


@pytest.fixture
def trials_file(tmp_path):
    trials = make_block("block_a") + make_block("block_b")
    path = tmp_path / "trials.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")
    return str(path), trials


@pytest.fixture
def results_file(tmp_path):
    return str(tmp_path / "results.jsonl")


def run_cli(monkeypatch, tmp_path, argv):
    job_dir = tmp_path / "batch_jobs"
    monkeypatch.setattr(batch_run, "JOB_DIR", str(job_dir))
    monkeypatch.setattr(sys, "argv", ["batch_run.py"] + argv)
    batch_run.main()
    return job_dir


# ---------------------------------------------------------------------------
# submit: dry-run (no network calls), block-aware --limit, job file contents
# ---------------------------------------------------------------------------

class TestSubmitDryRun:
    def test_makes_no_network_calls_and_reports_the_right_count(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, _ = trials_file
        monkeypatch.setattr(model_providers, "submit_batch", lambda *a, **k: pytest.fail("submit_batch must not be called during --dry-run"))
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5", "--reasoning-profile", "low",
            "--trials-file", trials_path, "--results-file", results_file, "--limit", "1", "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 4" in out
        assert "Constructed 4 provider-native request payload(s)" in out

    def test_limit_selects_whole_blocks_not_partial_ones(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, _ = trials_file
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5",
            "--trials-file", trials_path, "--results-file", results_file, "--limit", "2", "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 8" in out  # both blocks, 4 cells each

    def test_deepseek_submit_fails_clearly_and_points_at_concurrent_execution(self, monkeypatch, tmp_path, trials_file, results_file):
        trials_path, _ = trials_file
        with pytest.raises(SystemExit, match="concurrency"):
            run_cli(monkeypatch, tmp_path, [
                "submit", "--provider", "deepseek", "--model", "deepseek-v4-pro",
                "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
            ])


# ---------------------------------------------------------------------------
# submit: completion identity -- prompt hash / provider / model / reasoning
# profile mismatch must never count as already-satisfied.
# ---------------------------------------------------------------------------

def write_result(results_file, trial, replicate_id=1, provider="anthropic", model="claude-sonnet-5", reasoning_profile="low",
                  prompt_sha256=None, valid=True):
    row = {
        "trial_id": trial["trial_id"], "model": model, "requested_model": model, "provider": provider,
        "reasoning_profile": reasoning_profile, "replicate_id": replicate_id, "attempt_id": 1,
        "sampling_regime": "low_variance_primary",
        "parsed_response": {"overall_quality": "A"} if valid else None,
        "validation_error": None if valid else "invalid",
        "trial_meta": {**{k: v for k, v in trial.items() if k != "prompt"}, "prompt_sha256": prompt_sha256 or trial["prompt_sha256"]},
    }
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


class TestSubmitCompletionIdentity:
    """No --limit here (which block --limit picks is an unrelated shuffle
    detail) -- these check whether one already-saved result, out of all 8
    cells across both blocks, is correctly counted as satisfying its trial."""

    def test_exact_match_is_skipped(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, trials = trials_file
        write_result(results_file, trials[0])
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5",
            "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 7" in out  # one of the 8 cells already done

    def test_prompt_hash_mismatch_does_not_count_as_complete(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, trials = trials_file
        write_result(results_file, trials[0], prompt_sha256="stale_hash_from_before_wording_changed")
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5",
            "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 8" in out  # nothing counted as done

    def test_provider_mismatch_does_not_count_as_complete(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, trials = trials_file
        write_result(results_file, trials[0], provider="openai")
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5",
            "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 8" in out

    def test_model_mismatch_does_not_count_as_complete(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, trials = trials_file
        write_result(results_file, trials[0], model="claude-opus-5")
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5",
            "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 8" in out

    def test_reasoning_profile_mismatch_does_not_count_as_complete(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, trials = trials_file
        write_result(results_file, trials[0], reasoning_profile="high")
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5", "--reasoning-profile", "low",
            "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 8" in out

    def test_pre_existing_result_with_no_identity_metadata_still_counts_as_complete(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        """Backward compatibility: a result saved before multi-provider
        support existed has no provider/requested_model/reasoning_profile
        fields at all -- it must still satisfy the trial."""
        trials_path, trials = trials_file
        row = {
            "trial_id": trials[0]["trial_id"], "model": "claude-sonnet-5", "replicate_id": 1, "attempt_id": 1,
            "sampling_regime": "low_variance_primary",
            "parsed_response": {"overall_quality": "A"}, "validation_error": None,
            "trial_meta": {k: v for k, v in trials[0].items() if k != "prompt"},
        }
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5",
            "--trials-file", trials_path, "--results-file", results_file, "--dry-run",
        ])
        out = capsys.readouterr().out
        assert "Eligible requests (not already completed): 7" in out


# ---------------------------------------------------------------------------
# submit (real, not dry-run): job file written with required fields
# ---------------------------------------------------------------------------

class TestSubmitJobFile:
    def test_job_file_has_the_required_fields(self, monkeypatch, tmp_path, trials_file, results_file):
        trials_path, trials = trials_file
        monkeypatch.setattr(model_providers, "submit_batch", lambda provider, model, requests, reasoning_profile, max_output_tokens: {
            "provider_batch_id": "batch_xyz", "raw": {}
        })
        job_dir = run_cli(monkeypatch, tmp_path, [
            "submit", "--provider", "anthropic", "--model", "claude-sonnet-5", "--reasoning-profile", "low",
            "--trials-file", trials_path, "--results-file", results_file, "--limit", "1",
        ])
        job_files = list(job_dir.glob("*.json"))
        assert len(job_files) == 1
        job = json.loads(job_files[0].read_text(encoding="utf-8"))
        for field in ("provider", "provider_batch_id", "requested_model", "reasoning_profile",
                      "provider_reasoning_settings", "created_at", "trials_file", "results_file",
                      "sampling_regime", "request_count", "requests"):
            assert field in job
        assert job["provider_batch_id"] == "batch_xyz"
        assert job["request_count"] == 4
        assert job["collected_request_keys"] == []
        # request-key -> trial_id/replicate_id/prompt_sha256 mapping
        one_key = next(iter(job["requests"]))
        assert set(job["requests"][one_key]) == {"trial_id", "replicate_id", "prompt_sha256"}


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def write_job_file(job_dir, job):
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "test_job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return str(path)


class TestStatus:
    def test_prints_normalized_status(self, monkeypatch, tmp_path, capsys):
        job_dir = tmp_path / "batch_jobs"
        job_path = write_job_file(job_dir, {
            "provider": "anthropic", "provider_batch_id": "batch_xyz", "request_count": 4,
            "collected_request_keys": [],
        })
        monkeypatch.setattr(model_providers, "batch_status", lambda provider, batch_id: {"status": "completed", "raw": {"foo": "bar"}})
        monkeypatch.setattr(sys, "argv", ["batch_run.py", "status", "--job-file", job_path])
        batch_run.main()
        out = capsys.readouterr().out
        assert "Status: completed" in out
        assert "Collected so far: 0 / 4" in out


# ---------------------------------------------------------------------------
# collect: matched by request key (never order), idempotent, preserves
# failures, replicate ids survive the round trip.
# ---------------------------------------------------------------------------

class TestCollect:
    def _job(self, tmp_path, trials_path, results_path, trials):
        job_dir = tmp_path / "batch_jobs"
        requests = {
            batch_run.request_key(t["trial_id"], 1): {"trial_id": t["trial_id"], "replicate_id": 1, "prompt_sha256": t["prompt_sha256"]}
            for t in trials
        }
        job = {
            "provider": "anthropic", "provider_batch_id": "batch_xyz", "requested_model": "claude-sonnet-5",
            "reasoning_profile": "low", "provider_reasoning_settings": {}, "created_at": "now",
            "trials_file": trials_path, "results_file": results_path, "sampling_regime": "low_variance_primary",
            "request_count": len(requests), "requests": requests, "collected_request_keys": [],
        }
        return write_job_file(job_dir, job)

    def test_matches_results_by_key_even_when_returned_out_of_order(self, monkeypatch, tmp_path, trials_file, results_file):
        trials_path, trials = trials_file
        block_a = [t for t in trials if t["block_id"] == "block_a"]
        job_path = self._job(tmp_path, trials_path, results_file, block_a)

        def fake_collect(provider, batch_id, request_keys):
            keys = sorted(request_keys, reverse=True)  # deliberately not submission order
            return {k: model_providers.normalize_response("anthropic", "claude-sonnet-5", "A") for k in keys}

        monkeypatch.setattr(model_providers, "collect_batch", fake_collect)
        monkeypatch.setattr(sys, "argv", ["batch_run.py", "collect", "--job-file", job_path])
        batch_run.main()

        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert len(rows) == 4
        by_trial = {r["trial_id"]: r for r in rows}
        for t in block_a:
            assert by_trial[t["trial_id"]]["parsed_response"] == {"overall_quality": "A"}
            assert by_trial[t["trial_id"]]["execution_mode"] == "batch"

    def test_replicate_ids_survive_the_round_trip(self, monkeypatch, tmp_path, trials_file, results_file):
        trials_path, trials = trials_file
        block_a = [trials[0]]
        job_dir = tmp_path / "batch_jobs"
        key = batch_run.request_key(block_a[0]["trial_id"], 7)
        job = {
            "provider": "anthropic", "provider_batch_id": "batch_xyz", "requested_model": "claude-sonnet-5",
            "reasoning_profile": "low", "provider_reasoning_settings": {}, "created_at": "now",
            "trials_file": trials_path, "results_file": results_file, "sampling_regime": "low_variance_primary",
            "request_count": 1, "requests": {key: {"trial_id": block_a[0]["trial_id"], "replicate_id": 7, "prompt_sha256": block_a[0]["prompt_sha256"]}},
            "collected_request_keys": [],
        }
        job_path = write_job_file(job_dir, job)
        monkeypatch.setattr(model_providers, "collect_batch", lambda p, b, keys: {key: model_providers.normalize_response("anthropic", "claude-sonnet-5", "B")})
        monkeypatch.setattr(sys, "argv", ["batch_run.py", "collect", "--job-file", job_path])
        batch_run.main()

        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert rows[0]["replicate_id"] == 7

    def test_preserves_a_failed_provider_request_as_an_api_error(self, monkeypatch, tmp_path, trials_file, results_file):
        trials_path, trials = trials_file
        block_a = [trials[0]]
        job_path = self._job(tmp_path, trials_path, results_file, block_a)
        key = batch_run.request_key(block_a[0]["trial_id"], 1)

        monkeypatch.setattr(model_providers, "collect_batch", lambda p, b, keys: {key: {"error": "rate_limited"}})
        monkeypatch.setattr(sys, "argv", ["batch_run.py", "collect", "--job-file", job_path])
        batch_run.main()

        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert rows[0]["parsed_response"] is None
        assert "API call failed" in rows[0]["validation_error"]
        assert rows[0]["response_text"] is None

    def test_collect_is_idempotent_no_duplicate_rows(self, monkeypatch, tmp_path, trials_file, results_file, capsys):
        trials_path, trials = trials_file
        block_a = [trials[0]]
        job_path = self._job(tmp_path, trials_path, results_file, block_a)
        key = batch_run.request_key(block_a[0]["trial_id"], 1)

        call_count = {"n": 0}

        def fake_collect(p, b, keys):
            call_count["n"] += 1
            return {key: model_providers.normalize_response("anthropic", "claude-sonnet-5", "A")}

        monkeypatch.setattr(model_providers, "collect_batch", fake_collect)
        monkeypatch.setattr(sys, "argv", ["batch_run.py", "collect", "--job-file", job_path])
        batch_run.main()
        batch_run.main()  # second call: nothing new to collect

        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert len(rows) == 1  # never duplicated
        out = capsys.readouterr().out
        assert "Nothing new to collect" in out

    def test_runs_the_real_forced_choice_validator_on_collected_text(self, monkeypatch, tmp_path, trials_file, results_file):
        """collect must run the ordinary parser/validator, not fabricate a
        valid parsed_response for whatever text comes back."""
        trials_path, trials = trials_file
        block_a = [trials[0]]
        job_path = self._job(tmp_path, trials_path, results_file, block_a)
        key = batch_run.request_key(block_a[0]["trial_id"], 1)

        monkeypatch.setattr(model_providers, "collect_batch", lambda p, b, keys: {key: model_providers.normalize_response("anthropic", "claude-sonnet-5", "I really can't decide")})
        monkeypatch.setattr(sys, "argv", ["batch_run.py", "collect", "--job-file", job_path])
        batch_run.main()

        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert rows[0]["parsed_response"] is None
        assert rows[0]["validation_error"] is not None
