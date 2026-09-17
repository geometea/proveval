"""Tests for run_batch.py's trial/retry selection logic.

Covers the block-level "orphan cell" fix (item H of the offline
verification) and regression-tests the sampling_regime identity bug: a
valid result recorded under one sampling regime must never cause the same
trial/replicate to be skipped when subsequently run under a different
sampling regime into the same results file.
"""

import json
import os

import pytest

import run_batch as rb


# ---------------------------------------------------------------------------
# Block-level unit grouping (--limit never orphans a context_pairwise cell)
# ---------------------------------------------------------------------------

def make_block_trials(block_id, n=4):
    return [
        {"trial_id": f"{block_id}__{i}", "type": "context_pairwise", "block_id": block_id, "assignment": a, "position": p}
        for i, (a, p) in enumerate(
            [("forward", "story1_as_a"), ("forward", "story2_as_a"), ("flipped", "story1_as_a"), ("flipped", "story2_as_a")]
        )
    ][:n]


def test_group_trials_into_units_groups_full_blocks_together():
    trials = make_block_trials("block_0") + make_block_trials("block_1")
    units = rb.group_trials_into_units(trials)
    assert len(units) == 2
    assert all(len(u) == 4 for u in units)


def test_group_trials_into_units_singleton_for_non_block_trials():
    trials = [{"trial_id": f"single_{i}", "type": "context_single"} for i in range(5)]
    units = rb.group_trials_into_units(trials)
    assert len(units) == 5
    assert all(len(u) == 1 for u in units)


def test_limit_never_selects_a_partial_block():
    trials = make_block_trials("block_0") + make_block_trials("block_1") + make_block_trials("block_2")
    selected = rb.select_trials(trials, None, None, None, None, None, limit=2)
    block_ids = set(t["block_id"] for t in selected)
    assert len(block_ids) == 2
    for bid in block_ids:
        cells = [t for t in selected if t["block_id"] == bid]
        assert len(cells) == 4, f"block {bid} was only partially selected: {cells}"


def test_limit_is_unaffected_for_non_block_trial_types():
    trials = [{"trial_id": f"single_{i}", "type": "context_single"} for i in range(10)]
    selected = rb.select_trials(trials, None, None, None, None, None, limit=3)
    assert len(selected) == 3


def test_group_failed_candidates_groups_by_block_and_replicate():
    candidates = [
        ({"block_id": "b0"}, 1),
        ({"block_id": "b0"}, 1),  # same block+replicate -> same unit
        ({"block_id": "b0"}, 2),  # same block, different replicate -> different unit
        ({"block_id": "b1"}, 1),
        ({}, 1),  # no block_id -> singleton
    ]
    units = rb.group_failed_candidates_into_units(candidates)
    assert len(units) == 4
    sizes = sorted(len(u) for u in units)
    assert sizes == [1, 1, 1, 2]


# ---------------------------------------------------------------------------
# Replicate count resolution
# ---------------------------------------------------------------------------

def test_resolve_replicate_counts_defaults_neutral_to_treatment():
    treatment, neutral = rb.resolve_replicate_counts(5, None, None)
    assert treatment == 5 and neutral == 5


def test_resolve_replicate_counts_neutral_can_exceed_treatment():
    treatment, neutral = rb.resolve_replicate_counts(1, 2, 6)
    assert treatment == 2 and neutral == 6


def test_resolve_replicate_counts_rejects_neutral_below_treatment():
    with pytest.raises(ValueError):
        rb.resolve_replicate_counts(1, 5, 2)


def test_is_neutral_single_trial():
    assert rb.is_neutral_single_trial({"type": "context_single", "condition_id": "neutral"})
    assert not rb.is_neutral_single_trial({"type": "context_single", "condition_id": "provenance__ai_claude"})
    assert not rb.is_neutral_single_trial({"type": "context_pairwise", "condition_id": "neutral"})


# ---------------------------------------------------------------------------
# sampling_regime identity (regression test for the fixed bug)
# ---------------------------------------------------------------------------

def test_result_sampling_regime_is_none_for_v01_trial_types():
    assert rb.result_sampling_regime("single", "low_variance_primary") is None
    assert rb.result_sampling_regime("comparison", "provider_default_secondary") is None


def test_result_sampling_regime_passes_through_for_v02_trial_types():
    assert rb.result_sampling_regime("context_single", "low_variance_primary") == "low_variance_primary"
    assert rb.result_sampling_regime("context_pairwise", "provider_default_secondary") == "provider_default_secondary"


def test_load_existing_results_keys_include_sampling_regime(tmp_path):
    results_file = tmp_path / "results.jsonl"
    results_file.write_text(
        json.dumps(
            {
                "trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
                "response_text": "{}", "parsed_response": {"x": 1}, "validation_error": None,
                "sampling_regime": "low_variance_primary",
            }
        )
        + "\n"
    )
    existing = rb.load_existing_results(str(results_file))
    assert ("t1", "m", 1, "low_variance_primary") in existing
    assert ("t1", "m", 1, "provider_default_secondary") not in existing


def test_a_valid_result_under_one_sampling_regime_does_not_block_the_other(tmp_path):
    """Regression test: this is the exact bug reported -- a valid result
    saved under low_variance_primary must not cause a subsequent run under
    provider_default_secondary (same trial/model/replicate, same results
    file) to be silently skipped as already-completed."""
    results_file = tmp_path / "results.jsonl"
    results_file.write_text(
        json.dumps(
            {
                "trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
                "response_text": "{}", "parsed_response": {"x": 1}, "validation_error": None,
                "sampling_regime": "low_variance_primary",
            }
        )
        + "\n"
    )
    existing = rb.load_existing_results(str(results_file))
    trial = {"trial_id": "t1", "type": "context_single"}

    key_for_primary_run = ("t1", "m", 1, rb.result_sampling_regime(trial["type"], "low_variance_primary"))
    key_for_secondary_run = ("t1", "m", 1, rb.result_sampling_regime(trial["type"], "provider_default_secondary"))

    assert key_for_primary_run in existing and rb.is_completed(existing[key_for_primary_run])
    assert key_for_secondary_run not in existing  # must run, not be skipped


def test_select_failed_observations_does_not_cross_sampling_regimes(tmp_path):
    """A failed attempt recorded under one sampling regime must not be
    "retried" by a --retry-failed run under a different sampling regime."""
    results_file = tmp_path / "results.jsonl"
    with open(results_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "trial_id": "t1", "model": "m", "replicate_id": 1, "attempt_id": 1,
                    "response_text": None, "parsed_response": None, "validation_error": "API call failed",
                    "sampling_regime": "low_variance_primary",
                }
            )
            + "\n"
        )
    existing = rb.load_existing_results(str(results_file))
    trials_by_id = {"t1": {"trial_id": "t1", "type": "context_single", "condition_id": "neutral"}}

    retry_under_primary = rb.select_failed_observations(
        existing, trials_by_id, "m", "low_variance_primary", None, None, None, None, None, None
    )
    retry_under_secondary = rb.select_failed_observations(
        existing, trials_by_id, "m", "provider_default_secondary", None, None, None, None, None, None
    )
    assert len(retry_under_primary) == 1
    assert len(retry_under_secondary) == 0


# ---------------------------------------------------------------------------
# Multi-provider result identity: provider/requested_model/reasoning_profile
# (see model_providers.py), backward compatible with pre-existing results
# that carry none of this metadata.
# ---------------------------------------------------------------------------

class TestIsCompletedProviderIdentity:
    def _valid_record(self, **overrides):
        record = {"parsed_response": {"overall_quality": "A"}, "validation_error": None}
        record.update(overrides)
        return record

    def test_matching_provider_model_and_profile_counts_as_complete(self):
        record = self._valid_record(provider="anthropic", requested_model="claude-sonnet-5", reasoning_profile="low")
        assert rb.is_completed([record], expected_provider="anthropic", expected_model="claude-sonnet-5", expected_reasoning_profile="low")

    def test_different_provider_does_not_count_as_complete(self):
        record = self._valid_record(provider="openai", requested_model="claude-sonnet-5", reasoning_profile="low")
        assert not rb.is_completed([record], expected_provider="anthropic", expected_model="claude-sonnet-5", expected_reasoning_profile="low")

    def test_different_requested_model_does_not_count_as_complete(self):
        record = self._valid_record(provider="anthropic", requested_model="claude-opus-5", reasoning_profile="low")
        assert not rb.is_completed([record], expected_provider="anthropic", expected_model="claude-sonnet-5", expected_reasoning_profile="low")

    def test_different_reasoning_profile_does_not_count_as_complete(self):
        record = self._valid_record(provider="anthropic", requested_model="claude-sonnet-5", reasoning_profile="high")
        assert not rb.is_completed([record], expected_provider="anthropic", expected_model="claude-sonnet-5", expected_reasoning_profile="low")

    def test_a_record_predating_multi_provider_support_still_counts_as_complete(self):
        """No provider/requested_model/reasoning_profile fields at all --
        absent is "unknown, assume compatible", never a mismatch."""
        record = self._valid_record()
        assert rb.is_completed([record], expected_provider="anthropic", expected_model="claude-sonnet-5", expected_reasoning_profile="low")


# ---------------------------------------------------------------------------
# Concurrent (--concurrency) execution: retry/backoff on transient errors,
# and --dry-run makes no network/provider calls even with --concurrency set.
# ---------------------------------------------------------------------------

class TestConcurrentRetryBackoff:
    def test_is_retryable_error_recognizes_429_5xx_and_timeout(self):
        assert rb.is_retryable_error(Exception("HTTP 429 Too Many Requests"))
        assert rb.is_retryable_error(Exception("503 Service Unavailable"))
        assert rb.is_retryable_error(Exception("Request timed out"))
        assert not rb.is_retryable_error(Exception("400 Bad Request: invalid schema"))

    def test_run_one_retries_a_transient_error_and_then_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rb.time, "sleep", lambda seconds: None)  # don't actually wait in tests
        calls = {"n": 0}

        def flaky_call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("503 Service Unavailable")
            return {"response_text": "A", "response_model": "m-1", "stop_reason": "end_turn", "input_tokens": 1, "output_tokens": 1}

        monkeypatch.setattr(rb.model_providers, "call_model", flaky_call_model)
        trial = {"trial_id": "t1", "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab", "prompt": "p"}
        results_file = str(tmp_path / "results.jsonl")

        status = rb.run_one(trial, 1, "deepseek-v4-pro", 1, results_file, "low_variance_primary",
                             provider="deepseek", max_transient_retries=4)

        assert status == "valid"
        assert calls["n"] == 3
        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert len(rows) == 1  # only the final successful attempt is saved, not the transient failures

    def test_run_one_gives_up_after_exhausting_retries_and_saves_one_error_row(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rb.time, "sleep", lambda seconds: None)
        calls = {"n": 0}

        def always_fails(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls["n"] += 1
            raise Exception("429 rate limited")

        monkeypatch.setattr(rb.model_providers, "call_model", always_fails)
        trial = {"trial_id": "t1", "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab", "prompt": "p"}
        results_file = str(tmp_path / "results.jsonl")

        status = rb.run_one(trial, 1, "deepseek-v4-pro", 1, results_file, "low_variance_primary",
                             provider="deepseek", max_transient_retries=2)

        assert status == "error"
        assert calls["n"] == 3  # initial attempt + 2 retries
        rows = [json.loads(line) for line in open(results_file, encoding="utf-8")]
        assert len(rows) == 1
        assert "API call failed" in rows[0]["validation_error"]

    def test_non_retryable_error_is_not_retried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rb.time, "sleep", lambda seconds: None)
        calls = {"n": 0}

        def bad_request(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
            calls["n"] += 1
            raise Exception("400 Bad Request")

        monkeypatch.setattr(rb.model_providers, "call_model", bad_request)
        trial = {"trial_id": "t1", "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab", "prompt": "p"}
        results_file = str(tmp_path / "results.jsonl")

        status = rb.run_one(trial, 1, "deepseek-v4-pro", 1, results_file, "low_variance_primary",
                             provider="deepseek", max_transient_retries=4)

        assert status == "error"
        assert calls["n"] == 1  # never retried

    def test_concurrency_dry_run_makes_no_provider_calls(self, tmp_path, monkeypatch, capsys):
        def must_not_be_called(*args, **kwargs):
            pytest.fail("model_providers.call_model must never be invoked during --dry-run")

        monkeypatch.setattr(rb.model_providers, "call_model", must_not_be_called)

        trials = []
        for block in ("block_a", "block_b"):
            for assignment, position in [("forward", "story1_as_a"), ("forward", "story2_as_a"),
                                          ("flipped", "story1_as_a"), ("flipped", "story2_as_a")]:
                trials.append({"trial_id": f"{block}__{assignment}__{position}", "block_id": block,
                                "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab",
                                "prompt": "p"})
        trials_file = tmp_path / "trials.jsonl"
        with open(trials_file, "w", encoding="utf-8") as f:
            for t in trials:
                f.write(json.dumps(t) + "\n")

        monkeypatch.setattr(
            "sys.argv",
            ["run_batch.py", "--trials-file", str(trials_file), "--results-file", str(tmp_path / "results.jsonl"),
             "--provider", "deepseek", "--model", "deepseek-v4-pro", "--concurrency", "4", "--limit", "2", "--dry-run"],
        )
        rb.main()
        out = capsys.readouterr().out
        assert "planned" in out
        assert not (tmp_path / "results.jsonl").exists()
