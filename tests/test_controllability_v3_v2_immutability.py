"""v3 never modifies v2. Dynamic: hash every v2-owned file, exercise every
v3 path (manifest generation, freeze, preflight, a monkeypatched run,
analysis), re-hash. Static: no v3 module opens a v2 path for writing."""

import argparse
import glob
import hashlib
import os
import re
import sys

import pytest

import model_providers
import controllability_v3_trials as tr
import controllability_v3_v2_guard as guard
import freeze_controllability_v3 as fz
import run_controllability_v3 as rv3

V3_SOURCES = sorted(glob.glob("*_v3*.py") + ["freeze_controllability_v3.py", "run_controllability_v3.py", "analyze_controllability_v3.py"])


def hashes():
    return {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in guard.V2_PROTECTED_FILES}


def test_v3_pipeline_leaves_every_v2_file_byte_identical(tmp_path, monkeypatch):
    before = hashes()
    # manifest generation into tmp paths
    for name in ("CONTEXT_TRIALS_FILE", "NOCONTEXT_TRIALS_FILE", "HOLDOUT_TRIALS_FILE", "PILOT_TRIALS_FILE"):
        monkeypatch.setattr(tr, name, str(tmp_path / f"{name}.jsonl"))
    monkeypatch.setattr(sys, "argv", ["controllability_v3_trials.py"])
    tr.main()
    # freeze into tmp (copies of the config/lock)
    import shutil
    cfg = tmp_path / "cfg.json"
    shutil.copy("data/controllability_v3_study_config.json", cfg)
    fz.freeze(str(cfg), lock_path=str(tmp_path / "lock.json"))
    # preflight + monkeypatched run into tmp
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setattr(model_providers, "call_model", lambda *a, **k: {"response_text": "A"})
    monkeypatch.setattr(rv3, "EXPLORATORY_PILOT_RESULTS_FILE", str(tmp_path / "pilot.jsonl"))
    monkeypatch.setattr(rv3, "RUN_KIND_FILES", {**rv3.RUN_KIND_FILES, "pilot": {"families": ("pilot",), "production": str(tmp_path / "p.jsonl"), "exploratory": str(tmp_path / "pilot.jsonl")}})
    rv3.cmd_preflight(argparse.Namespace(concurrency=1, allow_peak_pricing=True, skip_credential_check=True))
    rv3._run(argparse.Namespace(provider="deepseek", model="deepseek-flash", reasoning_profile="low", replicates=1, seed=0, retry_limit=0, run_id="t",
                                limit=1, concurrency=1, time_budget_minutes=None, production=False, allow_unfrozen=True, allow_peak_pricing=True, dry_run=False), "pilot")
    import analyze_controllability_v3 as an
    an.run_analysis(str(tmp_path / "pilot.jsonl"), None, bootstrap_draws=5, analysis_dir=str(tmp_path / "analysis"))
    assert hashes() == before
    assert guard.verify_v2_unchanged() == (True, None)


def test_no_v3_module_opens_a_v2_path_for_writing():
    v2_path = re.compile(r"controllability_v2_[a-z_]*\.(jsonl|json)|data/stories|deepseek-v2\.yml")
    for path in V3_SOURCES:
        for lineno, line in enumerate(open(path, encoding="utf-8"), start=1):
            if v2_path.search(line) and "open(" in line:
                assert '"w"' not in line and '"a"' not in line and "'w'" not in line and "'a'" not in line, f"{path}:{lineno}: {line.strip()}"
    assert "results/controllability_v2" not in open("run_controllability_v3.py", encoding="utf-8").read().replace('V2_RESULT_DIRS = ("results/controllability_v2",)', "")


def test_v3_result_and_analysis_dirs_are_outside_v2():
    for d in (rv3.RESULTS_DIR, rv3.PRODUCTION_DIR, rv3.PILOT_DIR, rv3.ANALYSIS_DIR):
        assert d.startswith("results/controllability_v3")
        assert not d.startswith("results/controllability_v2")
