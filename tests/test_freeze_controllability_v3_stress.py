"""Staged stress lock: design stage on tmp copies (tamper detection for
split / dose manifest / config), attacks stage requires design, real
on-disk design stage verifies, primary lock untouched."""

import json
import os
import shutil

import pytest

import freeze_controllability_v3_stress as fzs
from controllability_v3_stress_config import build_stress_config, save_stress_config


@pytest.fixture
def scratch(tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("controllability_v3_stress_adversarial_split.json", "controllability_v3_stress_dose_trials.jsonl", "controllability_v3_frozen_lock.json",
                 "controllability_v3_v2_snapshot.json"):
        shutil.copy(os.path.join("data", name), tmp_path / "data" / name)
    cfg = build_stress_config()
    cfg["adversarial"]["split_file"] = str(tmp_path / "data" / "controllability_v3_stress_adversarial_split.json")
    cfg["dose_response"]["manifest_file"] = str(tmp_path / "data" / "controllability_v3_stress_dose_trials.jsonl")
    for key in ("candidates_file", "dev_manifest_file", "selected_attacks_file", "eval_manifest_file"):
        cfg["adversarial"][key] = str(tmp_path / "data" / f"{key}.json")
        (tmp_path / "data" / f"{key}.json").write_text("{}")
    cfg_path = tmp_path / "data" / "stress_config.json"
    save_stress_config(cfg, str(cfg_path))
    return {"cfg": str(cfg_path), "lock": str(tmp_path / "lock.json"), "primary_lock": str(tmp_path / "data" / "controllability_v3_frozen_lock.json"),
            "snapshot": str(tmp_path / "data" / "controllability_v3_v2_snapshot.json"), "split": cfg["adversarial"]["split_file"],
            "dose": cfg["dose_response"]["manifest_file"], "candidates": cfg["adversarial"]["candidates_file"]}


def verify(s, stage):
    return fzs.verify_frozen(stage, s["cfg"], s["lock"], s["primary_lock"], s["snapshot"])


class TestStages:
    def test_design_freeze_and_verify(self, scratch):
        assert not verify(scratch, "design")[0]
        lock = fzs.freeze("design", scratch["cfg"], scratch["lock"], scratch["primary_lock"], scratch["snapshot"])
        assert set(lock["stages"]["design"]["hashes"]) == {"stress_config_sha256", "split_sha256", "dose_manifest_sha256", "primary_lock_sha256", "v2_snapshot_sha256"}
        assert verify(scratch, "design") == (True, None)
        ok, reason = verify(scratch, "attacks")
        assert not ok and "not been frozen" in reason

    @pytest.mark.parametrize("target,key", [("split", "split_sha256"), ("dose", "dose_manifest_sha256"), ("primary_lock", "primary_lock_sha256")])
    def test_tampering_detected(self, scratch, target, key):
        fzs.freeze("design", scratch["cfg"], scratch["lock"], scratch["primary_lock"], scratch["snapshot"])
        with open(scratch[target], "a") as f:
            f.write("\n")
        ok, reason = verify(scratch, "design")
        assert not ok and key in reason

    def test_config_edit_after_freeze_detected(self, scratch):
        fzs.freeze("design", scratch["cfg"], scratch["lock"], scratch["primary_lock"], scratch["snapshot"])
        cfg = json.load(open(scratch["cfg"]))
        cfg["dose_response"]["replicates"] = 1
        save_stress_config(cfg, scratch["cfg"])
        ok, reason = verify(scratch, "design")
        assert not ok and "stress_config_sha256" in reason

    def test_attacks_stage_requires_design_and_detects_candidate_tampering(self, scratch):
        with pytest.raises(ValueError, match="design stage"):
            fzs.freeze("attacks", scratch["cfg"], scratch["lock"], scratch["primary_lock"], scratch["snapshot"])
        fzs.freeze("design", scratch["cfg"], scratch["lock"], scratch["primary_lock"], scratch["snapshot"])
        fzs.freeze("attacks", scratch["cfg"], scratch["lock"], scratch["primary_lock"], scratch["snapshot"])
        assert verify(scratch, "attacks") == (True, None)
        with open(scratch["candidates"], "a") as f:
            f.write("\n")
        ok, reason = verify(scratch, "attacks")
        assert not ok and "candidates_sha256" in reason
        assert verify(scratch, "design") == (True, None)  # design stage unaffected

    def test_unknown_stage(self, scratch):
        with pytest.raises(ValueError):
            fzs.freeze("bogus", scratch["cfg"], scratch["lock"])


def test_real_design_stage_is_frozen_and_verifies():
    assert fzs.verify_frozen("design") == (True, None)
    ok, reason = fzs.verify_frozen("attacks")
    assert not ok and "not been frozen" in reason  # attacks are generated/selected later
    lock = fzs.load_lock()
    assert lock["stages"]["design"]["hashes"]["primary_lock_sha256"] == fzs.sha256_of_file("data/controllability_v3_frozen_lock.json")
