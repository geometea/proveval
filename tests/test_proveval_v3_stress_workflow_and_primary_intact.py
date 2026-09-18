"""(1) Workflow isolation for proveval-v3-stress.yml; (2) the frozen v3
primary and the v2 study are byte-identical after the stress additions;
(3) the primary analysis stratifies by evaluator profile and writes the
cross-evaluator tables; (4) no stress module opens a primary manifest for
writing."""

import glob
import hashlib
import json
import os
import re
import subprocess

import pytest
import yaml

import analyze_controllability_v3 as an
import controllability_v3_trials as tr
import controllability_v3_v2_guard as guard
import freeze_controllability_v3 as fz
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID, profile_ids

STRESS_WF = ".github/workflows/proveval-v3-stress.yml"
PRIMARY_WF = ".github/workflows/proveval-v3-selective-suppression.yml"


def load(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


def steps():
    return load(STRESS_WF)["jobs"]["stress"]["steps"]


class TestStressWorkflowIsolation:
    def test_name_manual_only_modes_and_profile_input(self):
        w = load(STRESS_WF)
        assert w["name"] == "Proveval v3 stress experiments"
        assert set(w[True]) == {"workflow_dispatch"}
        inputs = w[True]["workflow_dispatch"]["inputs"]
        assert inputs["mode"]["options"] == ["preflight", "adversarial-generate", "adversarial-search", "adversarial-eval", "dose-response", "analysis-only"]
        assert set(inputs["evaluator_profile"]["options"]) == {p for p in profile_ids() if not p.startswith("attacker_")}
        assert inputs["evaluator_profile"]["default"] == PRIMARY_PROFILE_ID
        assert w["concurrency"]["group"] == "proveval-v3-stress" != load(PRIMARY_WF)["concurrency"]["group"]

    def test_never_references_v2_or_runs_primary_commands(self):
        text = "\n".join(l for l in open(STRESS_WF, encoding="utf-8").read().splitlines() if not l.strip().startswith("#"))
        for forbidden in ("run_controllability_v2", "analyze_controllability_v2", "results/controllability_v2", "proveval-v2"):
            assert forbidden not in text
        for s in steps():
            run = s.get("run", "")
            assert "py production --production" not in run and "py holdout --production" not in run and "py pilot --production" not in run

    def test_only_stress_directories_are_saved_or_uploaded(self):
        for s in steps():
            uses = s.get("uses", "")
            if uses.startswith("actions/cache/save") or uses.startswith("actions/upload-artifact"):
                path = s["with"]["path"]
                assert "results/controllability_v3/production" not in path
                assert all(p.strip().startswith(("results/controllability_v3/stress_", "results/controllability_v3/analysis/stress", "data/controllability_v3_stress_")) for p in path.split() if p.strip())
            if uses.startswith("actions/cache/save"):
                assert s["with"]["key"].startswith("proveval-v3-stress-")

    def test_paid_steps_gated_on_exact_modes(self):
        paid = {"stress-adversarial-generate": "adversarial-generate", "stress-adversarial-search --production": "adversarial-search",
                "stress-adversarial-run --production": "adversarial-eval", "stress-dose-run --production": "dose-response"}
        found = set()
        for s in steps():
            run = s.get("run", "")
            for cmd, mode in paid.items():
                if cmd in run:
                    found.add(cmd)
                    assert s["if"].strip() == f"${{{{ inputs.mode == '{mode}' }}}}", (cmd, s["if"])
        assert found == set(paid)

    def test_preflight_and_analysis_only_reach_no_paid_step(self):
        for mode in ("preflight", "analysis-only"):
            for s in steps():
                cond = s.get("if", "")
                reachable = (not cond) or (f"'{mode}'" in cond and "==" in cond and "!=" not in cond) or ("!=" in cond and f"'{mode}'" not in cond)
                if reachable:
                    assert "--production" not in s.get("run", "") and "stress-adversarial-generate" not in s.get("run", ""), (mode, s["name"])

    def test_primary_workflow_is_unchanged_from_git_head(self):
        try:
            head = subprocess.run(["git", "show", f"HEAD:{PRIMARY_WF}"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git not available")
        assert head == open(PRIMARY_WF, encoding="utf-8").read()


class TestPrimaryAndV2Intact:
    def test_primary_lock_verifies_and_manifests_hash_to_the_lock(self):
        assert fz.verify_frozen() == (True, None)
        lock = fz.load_lock()["hashes"]
        assert fz.sha256_of_file(tr.CONTEXT_TRIALS_FILE) == lock["context_manifest_sha256"]
        assert fz.sha256_of_file(tr.NOCONTEXT_TRIALS_FILE) == lock["nocontext_manifest_sha256"]
        assert fz.sha256_of_file(tr.HOLDOUT_TRIALS_FILE) == lock["holdout_manifest_sha256"]

    def test_primary_manifests_are_byte_identical_to_regeneration(self, tmp_path):
        context, nocontext, holdout = tr.generate_frozen_manifests()
        for rows, path in ((context, tr.CONTEXT_TRIALS_FILE), (nocontext, tr.NOCONTEXT_TRIALS_FILE), (holdout, tr.HOLDOUT_TRIALS_FILE)):
            out = tmp_path / os.path.basename(path)
            tr.write_manifest(rows, str(out))
            assert out.read_bytes() == open(path, "rb").read()

    def test_primary_prompt_hashes_unchanged(self):
        texts = tr.load_story_texts()
        tr.verify_prompt_hashes(tr.load_manifest(tr.NOCONTEXT_TRIALS_FILE), texts)
        tr.verify_prompt_hashes(tr.load_manifest(tr.CONTEXT_TRIALS_FILE)[:2000], texts)

    def test_v2_guard_still_passes(self):
        assert guard.verify_v2_unchanged() == (True, None)

    def test_no_stress_module_writes_a_primary_manifest(self):
        primary = re.compile(r"controllability_v3_(context|nocontext|holdout|pilot)_trials|controllability_v3_study_config\.json|controllability_v3_frozen_lock")
        for path in glob.glob("*stress*.py") + ["controllability_v3_evaluator_profiles.py"]:
            for lineno, line in enumerate(open(path, encoding="utf-8"), start=1):
                if primary.search(line) and "open(" in line:
                    assert '"w"' not in line and '"a"' not in line, f"{path}:{lineno}"
        text = open("run_controllability_v3.py", encoding="utf-8").read()
        for name in ("CONTEXT_TRIALS_FILE", "NOCONTEXT_TRIALS_FILE", "HOLDOUT_TRIALS_FILE"):
            assert f"write_manifest(" not in text or name not in text.split("write_manifest(")[1][:80]


def synth_primary(pid, bias):
    """Minimal complete no-context + context rows for two interventions."""
    from controllability_v3_evaluator_profiles import PROFILES
    from controllability_v3_trials import make_planned_observation_id, make_trial_id
    p = PROFILES[pid]
    ev = {"provider": p["provider"], "requested_model": p["model"], "reasoning_profile": p["reasoning_profile"], "max_output_tokens": p["max_output_tokens"]}
    rows = []
    pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("a", "d"), ("b", "d"), ("c", "d")]
    for i, pair in enumerate(pairs):
        base = 0.3 + 0.05 * i
        for iid in ("I0", "I1"):
            for position in ("story1_as_a", "story2_as_a"):
                for r in range(1, 21):
                    rows.append(_row(pair, None, None, position, iid, r, r <= round(base * 20), pid, ev, "primary_nocontext"))
            for assignment in ("forward", "flipped"):
                for position in ("story1_as_a", "story2_as_a"):
                    for r in range(1, 21):
                        rate = base + (bias[iid] if assignment == "forward" else -bias[iid])
                        rows.append(_row(pair, "provenance_human_vs_llm", assignment, position, iid, r, r <= round(rate * 20), pid, ev, "primary_context"))
    return rows


def _row(pair, cue, assignment, position, iid, r, chosen1, pid, ev, family):
    from controllability_v3_trials import make_planned_observation_id, make_trial_id
    s1, s2 = pair
    sa, sb = (s1, s2) if position == "story1_as_a" else (s2, s1)
    choice = ("A" if chosen1 else "B") if position == "story1_as_a" else ("B" if chosen1 else "A")
    tid = make_trial_id(family, f"{s1}_vs_{s2}", cue, assignment, position, iid, cue is not None)
    return {"planned_observation_id": make_planned_observation_id(tid, r), "evaluator_profile_id": pid, "family": family, "collection": "production", "trial_id": tid,
            "block_id": "::".join(["v3", family, f"{s1}_vs_{s2}"] + ([cue] if cue else []) + [iid]), "story_1_id": s1, "story_2_id": s2, "story_a": sa, "story_b": sb,
            "cue": cue, "context_assignment": assignment, "display_position": position, "intervention_id": iid, "context_present": cue is not None, "replicate": r,
            "evaluator": ev, "raw_response": choice, "parsed_choice": choice, "parsing_status": "resolved", "first_attempt_status": "valid",
            "attempts": [{"input_tokens": 1, "output_tokens": 1}], "total_attempts": 1, "output_tokens": 1, "latency_seconds": 0.1}


class TestPrimaryAnalysisStratification:
    def test_profiles_are_analysed_separately_with_cross_evaluator_tables(self, tmp_path):
        from controllability_v3_study_config import build_study_config
        cfg = build_study_config(status="frozen")
        low = synth_primary(PRIMARY_PROFILE_ID, {"I0": 0.2, "I1": 0.1})
        high = synth_primary("deepseek_flash_high", {"I0": 0.1, "I1": 0.0})
        with open(tmp_path / "p.jsonl", "w") as f:
            for r in low + high:
                f.write(json.dumps(r) + "\n")
        res_low = an.run_analysis(str(tmp_path / "p.jsonl"), None, bootstrap_draws=30, analysis_dir=str(tmp_path / "out"), study_config_file="data/controllability_v3_study_config.json")
        sup = {r["intervention_id"]: r for r in res_low["suppression_by_intervention"]}
        assert sup["I1"]["CE_control"] == pytest.approx(0.4) and sup["I1"]["residual_context_effect"] == pytest.approx(0.2)
        assert os.path.exists(tmp_path / "out" / "profiles" / "deepseek_flash_high" / "suppression_by_intervention.csv")
        for name in ("cross_evaluator_suppression.csv", "cross_evaluator_drift.csv", "cross_evaluator_baseline_disagreement.csv"):
            assert os.path.exists(tmp_path / "out" / name), name
        text = open(tmp_path / "out" / "cross_evaluator_suppression.csv").read()
        assert "deepseek_flash_high" in text and PRIMARY_PROFILE_ID in text
        # the other profile's rows never entered the primary profile's estimate
        alone = an.analyze(low, [], cfg, True, 30, 1, profile_id=PRIMARY_PROFILE_ID)
        assert alone["suppression_by_intervention"][1]["CE_control"] == sup["I1"]["CE_control"]
