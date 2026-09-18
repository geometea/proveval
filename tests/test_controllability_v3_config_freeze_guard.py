"""Study config (model configuration identical to v2's recovery-era
production), the v3 freeze/lock, and the v2 immutability guard."""

import json
import os
import shutil
import subprocess

import pytest

import controllability_v3_study_config as cfg
import controllability_v3_v2_guard as guard
import freeze_controllability_v3 as fz
from controllability_v2_recovery import RECOVERY_MAX_OUTPUT_TOKENS
from controllability_v2_study_config import load_study_config as load_v2_config
from controllability_v3_design import INTERVENTION_SENTENCES, all_instruction_sentences


class TestStudyConfig:
    def test_model_configuration_matches_v2_primary_evaluator_and_recovery_ceiling(self):
        config = cfg.build_study_config()
        v2 = load_v2_config()
        assert config["primary_evaluator"] == v2["primary_evaluator"]
        assert config["primary_evaluator"]["provider"] == "deepseek"
        assert config["primary_evaluator"]["reasoning_profile"] == "low"
        assert config["max_output_tokens"] == RECOVERY_MAX_OUTPUT_TOKENS == 4096
        assert config["max_output_tokens"] != 512
        assert config["sampling"] == "provider_default"

    def test_wording_is_read_from_the_design_never_retyped(self):
        config = cfg.build_study_config()
        assert config["instruction_strings"] == INTERVENTION_SENTENCES
        assert set(config["holdout_instruction_strings"]) == set(all_instruction_sentences()) - set(INTERVENTION_SENTENCES)

    def test_replicates_and_counts(self):
        config = cfg.build_study_config()
        assert config["replicate_count"] == 10 and config["holdout_replicate_count"] == 10 and config["pilot_replicate_count"] == 1
        assert config["expected_unique_cells"] == {"primary_context": 10560, "primary_nocontext": 1056, "holdout": 1320}
        assert config["random_seed"] != load_v2_config()["random_seed"]

    def test_on_disk_config_is_frozen_and_matches_the_builder(self):
        on_disk = cfg.load_study_config()
        assert on_disk["status"] == "frozen"
        assert {k: v for k, v in on_disk.items() if k != "status"} == {k: v for k, v in cfg.build_study_config().items() if k != "status"}


@pytest.fixture
def scratch(tmp_path):
    """Copies of the real v3 design files in a temp tree (the real files
    are never touched)."""
    (tmp_path / "data").mkdir()
    for name in ("controllability_v3_contrasts.jsonl", "controllability_v3_corpus.json", "controllability_v3_context_trials.jsonl",
                 "controllability_v3_nocontext_trials.jsonl", "controllability_v3_holdout_trials.jsonl", "controllability_v3_v2_snapshot.json"):
        shutil.copy(os.path.join("data", name), tmp_path / "data" / name)
    config = cfg.build_study_config()
    for key in ("contrast_file", "context_manifest_file", "nocontext_manifest_file", "holdout_manifest_file"):
        config[key] = str(tmp_path / config[key])
    config_path = tmp_path / "data" / "study_config.json"
    cfg.save_study_config(config, str(config_path))
    return {
        "config": str(config_path), "corpus": str(tmp_path / "data" / "controllability_v3_corpus.json"),
        "lock": str(tmp_path / "data" / "lock.json"), "snapshot": str(tmp_path / "data" / "controllability_v3_v2_snapshot.json"),
        "context": config["context_manifest_file"], "holdout": config["holdout_manifest_file"],
    }


class TestFreeze:
    def test_draft_fails_then_freeze_passes(self, scratch):
        ok, reason = fz.verify_frozen(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"])
        assert not ok and "draft" in reason
        lock = fz.freeze(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"])
        assert set(lock["hashes"]) == {"study_config_sha256", "contrasts_sha256", "corpus_sha256", "context_manifest_sha256",
                                       "nocontext_manifest_sha256", "holdout_manifest_sha256", "v2_snapshot_sha256"}
        assert fz.verify_frozen(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"]) == (True, None)

    @pytest.mark.parametrize("target,key", [("context", "context_manifest_sha256"), ("holdout", "holdout_manifest_sha256"),
                                             ("snapshot", "v2_snapshot_sha256"), ("corpus", "corpus_sha256")])
    def test_tampering_after_freeze_is_detected(self, scratch, target, key):
        fz.freeze(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"])
        with open(scratch[target], "a") as f:
            f.write("\n")
        ok, reason = fz.verify_frozen(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"])
        assert not ok and key in reason

    def test_editing_wording_in_the_config_after_freeze_is_detected(self, scratch):
        fz.freeze(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"])
        config = json.load(open(scratch["config"]))
        config["instruction_strings"]["I4"] += " Really."
        cfg.save_study_config(config, scratch["config"])
        ok, reason = fz.verify_frozen(scratch["config"], scratch["corpus"], scratch["lock"], scratch["snapshot"])
        assert not ok and "study_config_sha256" in reason

    def test_the_real_design_is_frozen_and_verifies(self):
        assert fz.verify_frozen() == (True, None)
        lock = fz.load_lock()
        assert lock["experiment_id"] == "context_controllability_v3"
        assert lock["hashes"]["contrasts_sha256"] == json.load(open("data/controllability_v2_frozen_lock.json"))["hashes"]["contrasts_sha256"]


class TestV2Guard:
    def test_real_repo_passes(self):
        assert guard.verify_v2_unchanged() == (True, None)

    def test_snapshot_covers_every_tracked_v2_and_story_file(self):
        try:
            tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git not available")
        expected = {p for p in tracked if "controllability_v2" in p or p.startswith("data/stories/") or p == "data/items.jsonl" or p == ".github/workflows/deepseek-v2.yml"}
        expected -= {p for p in tracked if p.startswith("tests/")}
        assert expected <= set(guard.V2_PROTECTED_FILES)
        assert set(json.load(open(guard.SNAPSHOT_FILE))["files"]) == set(guard.V2_PROTECTED_FILES)

    def test_modified_v2_file_is_detected(self, tmp_path):
        root = tmp_path
        for path in guard.V2_PROTECTED_FILES[:3]:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, root / path)
        snapshot = {"files": guard.compute_snapshot(guard.V2_PROTECTED_FILES[:3], root=str(root))}
        (root / "snap.json").write_text(json.dumps(snapshot))
        assert guard.verify_v2_unchanged("snap.json", root=str(root), check_v2_lock=False) == (True, None)
        with open(root / guard.V2_PROTECTED_FILES[0], "a") as f:
            f.write("x")
        ok, reason = guard.verify_v2_unchanged("snap.json", root=str(root), check_v2_lock=False)
        assert not ok and guard.V2_PROTECTED_FILES[0] in reason
        os.remove(root / guard.V2_PROTECTED_FILES[1])
        ok, reason = guard.verify_v2_unchanged("snap.json", root=str(root), check_v2_lock=False)
        assert not ok and "missing" in reason
