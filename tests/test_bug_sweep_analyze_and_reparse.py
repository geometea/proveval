"""Regression tests for the 2026-09-16 bug sweep of analyze.py and
reparse_results.py.
"""

import json

import analyze
import context_analysis_io as io
from reparse_results import reparse

RATINGS = {"plot_structure": 3, "prose_style": 3, "characterization": 3, "originality": 3, "overall_quality": 3}


class TestAnalyzeCollapseAttemptsMissingReplicateId:
    """A row saved by run_trial.py's single-trial CLI (README.md's
    `python3 run_trial.py <trial_id>`) carries no replicate_id at all --
    collapse_attempts must default it to 1 rather than raising KeyError."""

    def test_row_with_no_replicate_id_does_not_crash(self):
        rows = [{"trial_id": "single__gilbert__neutral__critique", "model": "m",
                  "parsed_response": RATINGS, "validation_error": None}]
        observations, unresolved, absorbed = analyze.collapse_attempts(rows)
        assert len(observations) == 1
        assert absorbed == 0

    def test_missing_and_explicit_replicate_1_collapse_to_the_same_key(self):
        rows = [
            {"trial_id": "single__gilbert__neutral__critique", "model": "m",
             "parsed_response": RATINGS, "validation_error": "bad", "attempt_id": 1},
            {"trial_id": "single__gilbert__neutral__critique", "model": "m", "replicate_id": 1,
             "parsed_response": RATINGS, "validation_error": None, "attempt_id": 2},
        ]
        observations, unresolved, absorbed = analyze.collapse_attempts(rows)
        assert len(observations) == 1
        assert absorbed == 1  # the one genuine failed attempt


class TestAnalyzeAbsorbedFailedAttemptsCount:
    """Only failed attempts before an eventual success count as "absorbed" --
    a second genuinely successful attempt must never be miscounted as one."""

    TRIAL_ID = "single__gilbert__neutral__critique"

    def test_two_successful_attempts_are_not_counted_as_absorbed_failures(self):
        rows = [
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": RATINGS, "validation_error": None},
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": RATINGS, "validation_error": None},
        ]
        observations, unresolved, absorbed = analyze.collapse_attempts(rows)
        assert absorbed == 0

    def test_one_failure_then_one_success_counts_as_one_absorbed(self):
        rows = [
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": None, "validation_error": "bad", "attempt_id": 1},
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": RATINGS, "validation_error": None, "attempt_id": 2},
        ]
        observations, unresolved, absorbed = analyze.collapse_attempts(rows)
        assert absorbed == 1

    def test_two_failures_then_two_successes_counts_only_the_failures(self):
        rows = [
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": None, "validation_error": "bad", "attempt_id": 1},
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": None, "validation_error": "bad", "attempt_id": 2},
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": RATINGS, "validation_error": None, "attempt_id": 3},
            {"trial_id": self.TRIAL_ID, "model": "m", "replicate_id": 1, "parsed_response": RATINGS, "validation_error": None, "attempt_id": 4},
        ]
        observations, unresolved, absorbed = analyze.collapse_attempts(rows)
        assert absorbed == 2  # not 3


class TestContextAnalysisIoSameFixes:
    """The v0.2 collapse_attempts (context_analysis_io.py) had the identical
    two bugs -- same fixes, same tests."""

    def test_row_with_no_replicate_id_does_not_crash(self):
        rows = [{"trial_id": "t1", "model": "m", "sampling_regime": "low_variance_primary",
                  "parsed_response": {"x": 1}, "validation_error": None}]
        trials_by_id = {"t1": {"trial_id": "t1", "type": "context_single"}}
        observations, unresolved, absorbed, missing = io.collapse_attempts(rows, trials_by_id)
        assert len(observations) == 1
        assert absorbed == 0

    def test_two_successful_attempts_are_not_counted_as_absorbed_failures(self):
        rows = [
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "sampling_regime": "low_variance_primary",
             "parsed_response": {"x": 1}, "validation_error": None},
            {"trial_id": "t1", "model": "m", "replicate_id": 1, "sampling_regime": "low_variance_primary",
             "parsed_response": {"x": 2}, "validation_error": None},
        ]
        trials_by_id = {"t1": {"trial_id": "t1", "type": "context_single"}}
        observations, unresolved, absorbed, missing = io.collapse_attempts(rows, trials_by_id)
        assert absorbed == 0


class TestReparseHandlesFailedAndStaleRows:
    """reparse() must not crash on a row with no response_text (API call
    failure) or an unrecognized trial_id (stale manifest) -- both are
    passed through unchanged instead of aborting the whole run."""

    def test_none_response_text_is_passed_through_unchanged(self):
        result = {"trial_id": "t1", "response_text": None, "parsed_response": None, "validation_error": "API call failed: timeout"}
        updated, status = reparse(result, {"t1": {"type": "single"}})
        assert status == "no_response_text"
        assert updated == result

    def test_unknown_trial_id_is_passed_through_unchanged(self):
        result = {"trial_id": "ghost_trial", "response_text": json.dumps(RATINGS)}
        updated, status = reparse(result, {})
        assert status == "unknown_trial_id"
        assert updated == result

    def test_normal_row_is_still_reparsed(self):
        result = {"trial_id": "t1", "response_text": json.dumps(RATINGS)}
        updated, status = reparse(result, {"t1": {"type": "single"}})
        assert status == "reparsed"
        assert updated["validation_error"] is None
        assert updated["parsed_response"] == RATINGS

    def test_reparse_forced_choice_pairwise_respects_choice_mode(self):
        payload = {"plot_structure": "A", "prose_style": "B", "characterization": "A", "originality": "tie", "overall_quality": "B"}
        result = {"trial_id": "p1", "response_text": json.dumps(payload)}
        trial = {"type": "context_pairwise", "choice_mode": "forced"}
        updated, status = reparse(result, {"p1": trial})
        assert status == "reparsed"
        assert updated["validation_error"] is not None  # tie rejected under forced choice_mode
