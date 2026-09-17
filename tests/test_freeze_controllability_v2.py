"""Tests for freeze_controllability_v2.py: freezing writes a lock file and
flips the study config's status, verify_frozen() only passes when every
hash matches, and a real (non-dry-run) run refuses to execute in a
draft/mismatched state.

Everything here operates on a temp directory's own copies of the study
config/contrasts/corpus/manifests -- the real repo's draft
data/controllability_v2_study_config.json is never touched or frozen by
these tests.
"""

import json
import os

import pytest

import freeze_controllability_v2 as fz


@pytest.fixture
def scratch_design(tmp_path):
    (tmp_path / "data").mkdir()
    contrasts_path = tmp_path / "data" / "contrasts.jsonl"
    contrasts_path.write_text('{"contrast_id": "c1", "a": "x", "b": "y"}\n', encoding="utf-8")

    treatment_path = tmp_path / "data" / "treatment.jsonl"
    treatment_path.write_text('{"trial_id": "t1"}\n', encoding="utf-8")

    baseline_path = tmp_path / "data" / "baseline.jsonl"
    baseline_path.write_text('{"trial_id": "b1"}\n', encoding="utf-8")

    corpus_path = tmp_path / "data" / "corpus.json"
    corpus_path.write_text(json.dumps({"corpus_id": "c", "stories": []}), encoding="utf-8")

    config_path = tmp_path / "data" / "study_config.json"
    config = {
        "design_version": "v2", "experiment_id": "context_controllability_v2",
        "contrast_file": str(contrasts_path), "treatment_manifest_file": str(treatment_path),
        "baseline_manifest_file": str(baseline_path), "status": "draft",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    lock_path = tmp_path / "data" / "lock.json"
    return {
        "config_path": str(config_path), "corpus_path": str(corpus_path), "lock_path": str(lock_path),
        "contrasts_path": str(contrasts_path), "treatment_path": str(treatment_path), "baseline_path": str(baseline_path),
    }


class TestVerifyFrozen:
    def test_draft_config_fails_verification(self, scratch_design):
        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is False
        assert "draft" in reason

    def test_missing_lock_file_fails_verification_even_if_frozen(self, scratch_design):
        config = json.loads(open(scratch_design["config_path"]).read())
        config["status"] = "frozen"
        with open(scratch_design["config_path"], "w") as f:
            json.dump(config, f)
        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is False
        assert "lock" in reason.lower()

    def test_freeze_then_verify_passes(self, scratch_design):
        lock = fz.freeze(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert os.path.exists(scratch_design["lock_path"])
        config = json.loads(open(scratch_design["config_path"]).read())
        assert config["status"] == "frozen"

        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is True
        assert reason is None
        assert set(lock["hashes"]) == {
            "study_config_sha256", "contrasts_sha256", "corpus_sha256",
            "baseline_manifest_sha256", "treatment_manifest_sha256",
        }

    def test_tampering_with_contrasts_after_freeze_is_detected(self, scratch_design):
        fz.freeze(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        with open(scratch_design["contrasts_path"], "a") as f:
            f.write('{"contrast_id": "c2", "a": "x", "b": "z"}\n')
        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is False
        assert "contrasts_sha256" in reason

    def test_tampering_with_the_treatment_manifest_after_freeze_is_detected(self, scratch_design):
        fz.freeze(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        with open(scratch_design["treatment_path"], "a") as f:
            f.write('{"trial_id": "t2"}\n')
        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is False
        assert "treatment_manifest_sha256" in reason

    def test_tampering_with_the_study_config_itself_after_freeze_is_detected(self, scratch_design):
        fz.freeze(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        config = json.loads(open(scratch_design["config_path"]).read())
        config["random_seed"] = 999999
        with open(scratch_design["config_path"], "w") as f:
            json.dump(config, f)
        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is False
        assert "study_config_sha256" in reason

    def test_reformatting_the_config_without_changing_content_does_not_break_verification(self, scratch_design):
        """sha256_of_json_obj is canonical (sorted keys, no whitespace
        sensitivity) -- re-saving the exact same config with different
        formatting must not produce a spurious mismatch."""
        fz.freeze(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        config = json.loads(open(scratch_design["config_path"]).read())
        with open(scratch_design["config_path"], "w") as f:
            json.dump(config, f, indent=4, sort_keys=False)  # different formatting, same content
        ok, reason = fz.verify_frozen(scratch_design["config_path"], scratch_design["corpus_path"], scratch_design["lock_path"])
        assert ok is True


def test_the_real_repo_draft_config_was_never_frozen_by_these_tests():
    """Guard against a test accidentally freezing the real on-disk design --
    freezing must remain an explicit, never-automatic action."""
    from controllability_v2_study_config import STUDY_CONFIG_FILE, load_study_config

    config = load_study_config(STUDY_CONFIG_FILE)
    assert config["status"] == "draft"
    assert not os.path.exists(fz.LOCK_FILE)
