"""Workflow isolation for .github/workflows/proveval-v3-selective-suppression.yml:
it can never invoke v2, has no schedule, and no mode other than the exact
strings 'pilot'/'production' can reach a paid-call step. The v2 workflow
file itself is unchanged."""

import hashlib
import json

import yaml

V3 = ".github/workflows/proveval-v3-selective-suppression.yml"
V2 = ".github/workflows/deepseek-v2.yml"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def steps():
    return load(V3)["jobs"]["experiment"]["steps"]


class TestIsolation:
    def test_name_and_manual_only_trigger(self):
        w = load(V3)
        assert w["name"] == "Proveval v3 selective suppression experiment"
        assert set(w[True]) == {"workflow_dispatch"}  # no schedule, ever
        assert w[True]["workflow_dispatch"]["inputs"]["mode"]["options"] == ["preflight", "pilot", "production", "analysis-only"]
        assert w[True]["workflow_dispatch"]["inputs"]["mode"]["default"] == "preflight"

    def test_never_references_v2(self):
        text = "\n".join(line for line in open(V3, encoding="utf-8").read().splitlines() if not line.strip().startswith("#"))
        for forbidden in ("run_controllability_v2", "analyze_controllability_v2", "results/controllability_v2", "proveval-v2", "freeze_controllability_v2"):
            assert forbidden not in text

    def test_concurrency_group_and_cache_keys_are_v3_specific(self):
        w = load(V3)
        assert w["concurrency"]["group"] == "proveval-v3-production" != load(V2)["concurrency"]["group"]
        for s in steps():
            if s.get("uses", "").startswith("actions/cache"):
                assert s["with"]["key"].startswith("proveval-v3-") and s["with"]["path"] == "results/controllability_v3/production"
            if s.get("uses", "").startswith("actions/upload-artifact"):
                assert s["with"]["name"].startswith("proveval-v3-")

    def test_every_api_calling_step_is_gated_on_an_exact_paid_mode(self):
        api_commands = ("py pilot --production", "py production --production", "py holdout --production")
        found = 0
        for s in steps():
            run = s.get("run", "")
            if any(cmd in run for cmd in api_commands):
                found += 1
                cond = s["if"]
                assert "inputs.mode == 'pilot'" in cond or "inputs.mode == 'production'" in cond
                assert "preflight" not in cond and "analysis-only" not in cond
                assert "!=" not in cond
        assert found == 3

    def test_preflight_and_analysis_only_modes_reach_no_api_step(self):
        for mode in ("preflight", "analysis-only"):
            for s in steps():
                run = s.get("run", "")
                cond = s.get("if", "")
                reachable = (not cond) or (f"inputs.mode == '{mode}'" in cond) or (cond.strip() == f"${{{{ inputs.mode != 'analysis-only' }}}}" and mode != "analysis-only")
                if reachable:
                    assert "--production" not in run, f"mode {mode} can reach {s['name']}"
                    assert "pilot --" not in run and "production --" not in run and "holdout --" not in run

    def test_preflight_step_runs_only_the_preflight_command(self):
        by_name = {s["name"]: s for s in steps()}
        assert by_name["v3 preflight (zero API calls)"]["run"].strip() == "python3 run_controllability_v3.py preflight --concurrency 32"

    def test_production_mode_is_resumable_with_cache_restore_and_save(self):
        by_name = {s["name"]: s for s in steps()}
        assert "restore-keys" in by_name["Restore previous v3 production state"]["with"]
        assert "always()" in by_name["Save resumable v3 production state"]["if"]
        assert "--time-budget-minutes" in by_name["Run v3 primary experiment (context + no-context, interleaved)"]["run"]

    def test_v2_workflow_is_unchanged(self):
        snapshot = json.load(open("data/controllability_v3_v2_snapshot.json"))["files"]
        assert hashlib.sha256(open(V2, "rb").read()).hexdigest() == snapshot[V2]
