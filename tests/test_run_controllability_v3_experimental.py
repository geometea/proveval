"""CLI-level tests for the experimental families with fake providers and
tmp paths: adaptive rounds (resume reconstructs identical decisions, hard
budget, replay mismatch refused), iterative pipeline (dev-only scoring,
held-out inaccessible until freeze, ancestry, deterministic selection),
capability sweep under two profiles (identical scientific cells, planted
differences recovered), analyses, workflow isolation, primary/stress/v2
intactness after everything."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess

import pytest
import yaml

import analyze_controllability_v3_experimental as ax
import controllability_v3_adaptive_dose as ad
import controllability_v3_capability_sweep as cap
import controllability_v3_iterative_attack as it
import controllability_v3_stress_design as sd
import controllability_v3_v2_guard as guard
import freeze_controllability_v3 as fz
import freeze_controllability_v3_stress as fzs
import model_providers
import run_controllability_v3 as rv3
from controllability_v3_design import all_instruction_sentences
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID
from controllability_v3_execution import iter_rows, load_valid_completed_ids
from controllability_v3_experimental_config import build_config
from controllability_v3_trials import load_contrasts, load_story_texts

TEXTS = load_story_texts()
T2I = {v: k for k, v in TEXTS.items()}
SENT = {v: k for k, v in all_instruction_sentences().items()}
FAV_CLAUSES = [c["a_clause"] for c in load_contrasts()]
QUAL = {s: random.Random(7).gauss(0, 1) for s in sorted(TEXTS)}
SLOPE = {"I0": 0.6, "I1": 0.4, "I4": 0.08, "I5": 0.1, "I7": 0.15, "I2": 0.2, "I3": 0.25, "I6": 0.3}
BIAS = {"I0": 1.2, "I1": 0.8, "I4": 0.1, "I5": 0.2, "I7": 0.3, "I2": 0.4, "I3": 0.5, "I6": 0.6}


def sig(x):
    return 1 / (1 + math.exp(-x))


def fake_judge(prompt, scale):
    paras = prompt.split("\n\n")
    intro, iid = paras[0], SENT[paras[1]]
    m = re.match(r"Passage A:\n(.*)\n\nPassage B:\n(.*)", "\n\n".join(paras[2:-1]), re.S)
    sa, sb = T2I[m.group(1)], T2I[m.group(2)]
    bias = 0.0
    dm = re.search(r"(\d+) out of 100 readers preferred Passage ([AB])", intro)
    if dm:
        bias = SLOPE[iid] * math.log(int(dm.group(1)) / (100 - int(dm.group(1)))) * (1 if dm.group(2) == "A" else -1) * scale
    elif "Passage A" in intro:
        mk = re.search(r"Passage ([AB]) \[\[FAV\]\]", intro)
        if mk:
            strength, fav_a = float(re.search(r"str=([0-9.]+)", intro).group(1)), mk.group(1) == "A"
        else:
            mm = re.match(r"I'd like your view on two passages\. Passage A (.*), while Passage B (.*)\.", intro)
            fav_a, strength = bool(mm) and mm.group(1) in FAV_CLAUSES, 0.0
        b = BIAS[iid] * (1 + 1.5 * strength) * scale
        bias = b if fav_a else -b
    p = sig(QUAL[sa] - QUAL[sb] + bias + 0.1)
    seed = int(hashlib.sha256((prompt + "|" + str(random.random())).encode()).hexdigest()[:12], 16)
    return "A" if random.Random(seed).random() < p else "B"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Tmp copies of every mutable experimental path; frozen manifests stay real."""
    cfg = build_config()
    ic = cfg["iterative_attack"]
    ic["population_file"], ic["selected_file"], ic["eval_manifest_file"] = str(tmp_path / "pop.jsonl"), str(tmp_path / "sel.json"), str(tmp_path / "eval.jsonl")
    ic["policy"].update({"cues": ["reception_positive_vs_negative"], "interventions": ["I0", "I4"], "generations": 1, "population_size": 3, "min_dev_pairs_for_score": 2})
    ap = cfg["adaptive_dose"]["policy"]
    ap.update({"interventions": ["I0"], "max_rounds_per_unit": 6, "n_strata": 1})
    ap["global_max_judgments"] = ad.design_arithmetic(ap)["max_judgments_hard_budget"]
    cfg["capability_sweep"]["design"]["replicates"] = 1
    cfg["bootstrap_draws"] = 60
    monkeypatch.setattr(rv3, "load_experimental_config", lambda path=None: json.loads(json.dumps(cfg)))
    stages = {"adaptive_design": (True, None), "iterative_policy": (True, None), "iterative_attacks": (False, "not frozen"), "capability_design": (True, None)}
    monkeypatch.setattr(rv3, "verify_experimental_frozen", lambda stage: stages[stage])
    import freeze_controllability_v3_experimental as fx
    monkeypatch.setattr(fx, "freeze", lambda stage: stages.__setitem__(stage, (True, None)) or {"stages": {stage: {"frozen_at": "t"}}})
    for name in ("ADAPTIVE_RESULTS_FILE", "ADAPTIVE_SCHEDULE_FILE", "ADAPTIVE_STATE_FILE", "ITERATIVE_DEV_RESULTS_FILE", "ITERATIVE_EVAL_RESULTS_FILE", "CAPABILITY_RESULTS_FILE", "PRIMARY_RESULTS_FILE", "DOSE_RESULTS_FILE"):
        monkeypatch.setattr(rv3, name, str(tmp_path / f"{name.lower()}.jsonl"))
    kinds = {k: dict(v) for k, v in rv3.RUN_KIND_FILES.items()}
    kinds["primary"]["production"] = rv3.PRIMARY_RESULTS_FILE
    for k in kinds:
        kinds[k]["exploratory"] = str(tmp_path / "expl" / f"{k}.jsonl")
    monkeypatch.setattr(rv3, "RUN_KIND_FILES", kinds)
    monkeypatch.setattr(rv3, "run_preflight", lambda *a, **k: (True, [], {}))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    return {"cfg": cfg, "tmp": tmp_path, "stages": stages}


@pytest.fixture
def fake_models(monkeypatch):
    state = {"judge": 0, "attacker": 0, "attacker_prompts": []}

    def attacker(prompt):
        state["attacker"] += 1
        state["attacker_prompts"].append(prompt)
        base = float(m.group(1)) if (m := re.search(r"str=([0-9.]+)", prompt)) else 0.0
        return json.dumps({"candidates": [{"family": f"f{i}", "template": f"I'd like your view on two passages. Readers reviewed both: Passage {{FAV}} [[FAV]] was praised and Passage {{UNFAV}} [[UNFAV]] was panned, v{state['attacker']}-{i}. str={min(1.0, base + 0.3 + 0.1 * i):.2f}", "rationale": ""} for i in range(4)]})

    def call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
        if "red-team assistant" in prompt:
            return {"response_text": attacker(prompt), "response_model": "atk-sim"}
        state["judge"] += 1
        return {"response_text": fake_judge(prompt, 1.0 if reasoning_profile == "low" else 0.5), "response_model": f"sim-{reasoning_profile}", "input_tokens": 100, "output_tokens": 2}
    monkeypatch.setattr(model_providers, "call_model", call_model)
    return state


def run(argv):
    import sys
    sys.argv = ["x"] + argv
    rv3.main()


COMMON = ["--production", "--concurrency", "4", "--run-id", "t"]


class TestAdaptiveCLI:
    def test_rounds_resume_replay_and_budget(self, env, fake_models, capsys):
        run(["adaptive-dose-run"] + COMMON)
        out = capsys.readouterr().out
        assert "Every adaptive unit has stopped" in out
        schedule = rv3.load_schedule(rv3.ADAPTIVE_SCHEDULE_FILE)
        doses = [e["dose"] for e in schedule if not e.get("stop")]
        assert doses[:4] == [51, 70, 90, 99] and len(doses) <= 6 and schedule[-1]["stop"]
        assert all("state_before" in e and "candidates" in e and "reason" in e for e in schedule)
        state = json.load(open(rv3.ADAPTIVE_STATE_FILE))
        assert state["strata_basis"].startswith("single pooled") and set(state["strata"].values()) == {"all"}
        rows = list(iter_rows(rv3.ADAPTIVE_RESULTS_FILE))
        assert len(rows) == len(doses) * 66 * 4 and all(r["family"] == ad.FAMILY and r["planned_observation_id"].startswith("v3a::") for r in rows)
        # resume: identical decisions re-derived, zero calls
        before = fake_models["judge"]
        run(["adaptive-dose-run"] + COMMON)
        assert fake_models["judge"] == before and len(rv3.load_schedule(rv3.ADAPTIVE_SCHEDULE_FILE)) == len(schedule)
        # a tampered schedule is refused
        tampered = dict(schedule[1], dose=55)
        with open(rv3.ADAPTIVE_SCHEDULE_FILE, "w") as f:
            for e in [schedule[0], tampered] + schedule[2:]:
                f.write(json.dumps(e) + "\n")
        with pytest.raises(SystemExit, match="replay mismatch"):
            run(["adaptive-dose-run"] + COMMON)

    def test_partial_round_is_completed_before_a_new_decision(self, env, fake_models):
        fake_models["judge"] = 0
        run(["adaptive-dose-run"] + COMMON + ["--time-budget-minutes", "-1"])   # deadline already passed: plan first round, submit nothing
        schedule = rv3.load_schedule(rv3.ADAPTIVE_SCHEDULE_FILE)
        assert len(schedule) == 1 and fake_models["judge"] == 0
        run(["adaptive-dose-run"] + COMMON)
        schedule2 = rv3.load_schedule(rv3.ADAPTIVE_SCHEDULE_FILE)
        assert schedule2[0] == schedule[0] and len(schedule2) > 1

    def test_analysis_outputs(self, env, fake_models, tmp_path):
        run(["adaptive-dose-run"] + COMMON)
        run(["adaptive-dose-analysis", "--analysis-dir", str(tmp_path / "an"), "--bootstrap-draws", "30"])
        for name in ("adaptive_thresholds.csv", "adaptive_slopes.csv", "adaptive_stopping_reasons.csv", "adaptive_decisions.csv"):
            assert os.path.exists(tmp_path / "an" / name), name
        import csv
        t = list(csv.DictReader(open(tmp_path / "an" / "adaptive_thresholds.csv")))[0]
        assert float(t["beta"]) > 0.3 and 55 < float(t["d10_dose"]) < 80


class TestIterativeCLI:
    def test_pipeline_ancestry_dev_only_and_heldout_gating(self, env, fake_models, tmp_path):
        run(["production"] + COMMON)  # ordinary reference for scoring (frozen replicate count; small limit)
        run(["iterative-attack-generate"])
        pop = it.load_population(env["cfg"]["iterative_attack"]["population_file"])
        assert {m["origin"] for m in it.members(pop)} == {"ordinary", "fresh"}
        run(["iterative-attack-evaluate-dev"] + COMMON)
        dev_rows = list(iter_rows(rv3.ITERATIVE_DEV_RESULTS_FILE))
        split = sd.load_split()
        assert dev_rows and all(r["story_pair"] in set(split["attack_development_pairs"]) for r in dev_rows)
        before = fake_models["judge"]
        run(["iterative-attack-evaluate-dev"] + COMMON)
        assert fake_models["judge"] == before  # resumed
        run(["iterative-attack-next-generation"])
        pop = it.load_population(env["cfg"]["iterative_attack"]["population_file"])
        gen1 = [m for m in it.members(pop) if m["generation"] == 1]
        assert gen1 and all(m["parent_ids"] for m in gen1) and all(m["origin"] in ("mutation", "recombination") for m in gen1)
        parents = {m["attack_id"] for m in it.members(pop) if m["generation"] == 0}
        assert all(set(m["parent_ids"]) <= parents for m in gen1)
        for prompt in fake_models["attacker_prompts"]:
            assert "held-out" not in prompt.lower() and not any(p in prompt for p in split["attack_evaluation_pairs"] + split["attack_development_pairs"])
        run(["iterative-attack-evaluate-dev"] + COMMON)
        # held-out run refused before freeze, zero calls
        before = fake_models["judge"]
        with pytest.raises(SystemExit, match="not frozen"):
            run(["iterative-attack-run-heldout"] + COMMON)
        assert fake_models["judge"] == before
        run(["iterative-attack-freeze"])
        sel = json.load(open(env["cfg"]["iterative_attack"]["selected_file"]))
        assert sel["rule"].startswith("top final_top_k") and all(r["rank"] in (1, 2) for r in sel["selected"])
        run(["iterative-attack-freeze"])  # deterministic: same selection
        assert json.load(open(env["cfg"]["iterative_attack"]["selected_file"]))["selected"] == sel["selected"]
        run(["iterative-attack-run-heldout"] + COMMON + ["--limit", "3"])
        ev = list(iter_rows(rv3.ITERATIVE_EVAL_RESULTS_FILE))
        assert ev and all(r["story_pair"] in set(split["attack_evaluation_pairs"]) and r["family"] == it.FAMILY_EVAL for r in ev)
        with pytest.raises(SystemExit, match="never be revised"):
            run(["iterative-attack-freeze"])
        run(["iterative-attack-analysis", "--analysis-dir", str(tmp_path / "an"), "--bootstrap-draws", "20"])
        import csv
        curve = list(csv.DictReader(open(tmp_path / "an" / "iterative_attack_learning_curve.csv")))
        assert {r["intervention_id"] for r in curve} == {"I0", "I4"} and all(r["queries_to_break"] != "" or r["broken"] == "False" for r in curve)
        summary = list(csv.DictReader(open(tmp_path / "an" / "iterative_attack_generation_summary.csv")))
        i0 = {int(r["generation"]): float(r["best_dev_score"]) for r in summary if r["intervention_id"] == "I0" and r["best_dev_score"]}
        assert i0[1] >= i0[0] - 0.05   # the fake attacker's known improvement rule shows on dev


class TestCapabilityCLI:
    def test_two_profiles_same_cells_planted_difference_recovered(self, env, fake_models, tmp_path):
        run(["capability-run"] + COMMON + ["--limit", "60"])
        run(["capability-run"] + COMMON + ["--limit", "60", "--evaluator-profile", "deepseek_flash_medium"])
        low = list(iter_rows(rv3.CAPABILITY_RESULTS_FILE))
        med_path = rv3.results_path_for_profile(rv3.CAPABILITY_RESULTS_FILE, "deepseek_flash_medium")
        med = list(iter_rows(med_path))
        assert sorted(r["planned_observation_id"] for r in low) == sorted(r["planned_observation_id"] for r in med)
        assert {r["evaluation_observation_id"] for r in low}.isdisjoint({r["evaluation_observation_id"] for r in med})
        assert all(r["planned_observation_id"].startswith("v3c::capability_sweep::") for r in low)
        assert {r["condition"] for r in low} == {"nocontext", "ordinary", "dose"}
        run(["capability-analysis", "--analysis-dir", str(tmp_path / "an"), "--bootstrap-draws", "20", "--profiles", PRIMARY_PROFILE_ID, "deepseek_flash_medium"])
        import csv
        per = list(csv.DictReader(open(tmp_path / "an" / "capability_per_profile.csv")))
        assert {r["evaluator_profile_id"] for r in per} == {PRIMARY_PROFILE_ID, "deepseek_flash_medium"}
        between = list(csv.DictReader(open(tmp_path / "an" / "capability_between_profiles.csv")))
        assert between and "delta_suppression" in between[0] and "blind_preference_disagreement_rms" in between[0]

    def test_preflight_and_dry_run_make_no_calls(self, env, monkeypatch, capsys):
        monkeypatch.setattr(model_providers, "call_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("API call")))
        rv3.cmd_capability_preflight(argparse.Namespace(evaluator_profile=PRIMARY_PROFILE_ID, profiles=None, skip_credential_check=True, throughput_jps=None, throughput_from_results=None))
        assert "Capability preflight PASSED" in capsys.readouterr().out
        rv3.cmd_adaptive_dose_preflight(argparse.Namespace(evaluator_profile=PRIMARY_PROFILE_ID, profiles=None, skip_credential_check=True, throughput_jps=1.4, throughput_from_results=None))
        out = capsys.readouterr().out
        assert "Adaptive-dose preflight PASSED" in out and "1.40 j/s" in out
        run(["capability-run", "--dry-run"])
        assert "Dry run" in capsys.readouterr().out


class TestPlannedDifferencesInAnalysis:
    """Analysis-level recovery with synthetic rows: capability planted profile differences."""

    def test_capability_analysis_recovers_planted_profile_differences(self):
        from controllability_v3_evaluator_profiles import PROFILES
        design = {**cap.DESIGN, "interventions": ["I0", "I1"], "cues": ["reception_positive_vs_negative"], "doses": [51, 99]}
        pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("a", "d"), ("b", "d"), ("c", "d")]
        rows = {}
        for pid, scale in ((PRIMARY_PROFILE_ID, 1.0), ("deepseek_flash_high", 0.5)):
            p = PROFILES[pid]
            ev = {"provider": p["provider"], "requested_model": p["model"], "reasoning_profile": p["reasoning_profile"], "max_output_tokens": p["max_output_tokens"]}
            out = []
            for i, (s1, s2) in enumerate(pairs):
                base = 0.3 + 0.05 * i
                for iid, bias in (("I0", 0.2), ("I1", 0.1)):
                    conds = [("nocontext", None, None, base, "nocontext", False)] 
                    for asg in ("forward", "flipped"):
                        sign = 1 if asg == "forward" else -1
                        conds.append(("ordinary", "reception_positive_vs_negative", asg, base + sign * bias * scale, "ordinary", True))
                        for n in (51, 99):
                            conds.append((sd.dose_id(n), "reader_consensus_numeric", asg, base + sign * (0.02 if n == 51 else 0.2) * scale, "dose", True))
                    for variant, cue, asg, rate, cond, ctx in conds:
                        for pos in ("story1_as_a", "story2_as_a"):
                            for r in range(1, 21):
                                chosen1 = r <= round(rate * 20)
                                sa, sb = (s1, s2) if pos == "story1_as_a" else (s2, s1)
                                choice = ("A" if chosen1 else "B") if pos == "story1_as_a" else ("B" if chosen1 else "A")
                                tid = cap.make_trial_id(f"{s1}_vs_{s2}", cue, asg, pos, iid, variant, ctx)
                                out.append({"planned_observation_id": cap.make_observation_id(tid, r), "evaluator_profile_id": pid, "family": cap.FAMILY, "collection": "production",
                                            "trial_id": tid, "block_id": f"v3c::capability_sweep::{s1}_vs_{s2}::{cue}::{iid}::{variant}", "story_1_id": s1, "story_2_id": s2, "story_a": sa, "story_b": sb,
                                            "cue": cue, "context_assignment": asg, "display_position": pos, "intervention_id": iid, "context_present": ctx, "replicate": r, "condition": cond, "variant": variant,
                                            "evaluator": ev, "raw_response": choice, "parsed_choice": choice, "parsing_status": "resolved", "first_attempt_status": "valid", "attempts": [{}], "total_attempts": 1})
            rows[pid] = out
        from controllability_v3_study_config import build_study_config
        res = ax.analyze_capability(rows, design, build_study_config(status="frozen"), n_draws=50, seed=1)
        per = {(r["evaluator_profile_id"], r["intervention_id"]): r for r in res["capability_per_profile"]}
        assert per[(PRIMARY_PROFILE_ID, "I0")]["CE_control"] == pytest.approx(0.4) and per[("deepseek_flash_high", "I0")]["CE_control"] == pytest.approx(0.2)
        assert per[(PRIMARY_PROFILE_ID, "I1")]["signed_suppression"] == pytest.approx(0.2) and per[("deepseek_flash_high", "I1")]["signed_suppression"] == pytest.approx(0.1)
        assert per[(PRIMARY_PROFILE_ID, "I0")]["dose_slope"] > per[("deepseek_flash_high", "I0")]["dose_slope"] > 0
        b = {r["intervention_id"]: r for r in res["capability_between_profiles"]}
        sign = 1 if b["I0"]["profile_a"] == PRIMARY_PROFILE_ID else -1   # deltas are profile_a minus profile_b (alphabetical)
        assert sign * b["I0"]["delta_context_susceptibility"] == pytest.approx(0.2) and sign * b["I1"]["delta_suppression"] == pytest.approx(0.1)
        assert b["I0"]["blind_preference_disagreement_rms"] == 0.0   # identical blind preferences were planted


class TestWorkflowAndIntactness:
    WF = ".github/workflows/proveval-v3-experimental.yml"

    def test_workflow_isolation(self):
        w = yaml.safe_load(open(self.WF))
        assert set(w[True]) == {"workflow_dispatch"} and w["concurrency"]["group"] == "proveval-v3-experimental"
        text = "\n".join(l for l in open(self.WF).read().splitlines() if not l.strip().startswith("#"))
        for forbidden in ("run_controllability_v2", "results/controllability_v2", "py production --production", "py holdout --production", "stress-dose-run", "stress-adversarial-run"):
            assert forbidden not in text
        assert "8 judgments" not in text   # the rejected assumption may be mentioned in comments only
        paid = {"adaptive-dose-run --production": "adaptive-dose", "iterative-attack-generate": "iterative-generate", "iterative-attack-evaluate-dev --production": "iterative-step",
                "iterative-attack-next-generation": "iterative-step", "iterative-attack-run-heldout --production": "iterative-heldout", "capability-run --production": "capability-sweep"}
        seen = set()
        for s in w["jobs"]["experimental"]["steps"]:
            run_text = s.get("run", "")
            for cmd, mode in paid.items():
                if cmd in run_text:
                    seen.add(cmd)
                    assert s["if"].strip() == f"${{{{ inputs.mode == '{mode}' }}}}", (cmd, s["if"])
            if s.get("uses", "").startswith(("actions/cache/save", "actions/upload-artifact")):
                assert "results/controllability_v3/production" not in s["with"]["path"]
        assert seen == set(paid)

    def test_frozen_primary_stress_and_v2_are_intact(self):
        assert fz.verify_frozen() == (True, None) and fzs.verify_frozen("design") == (True, None) and guard.verify_v2_unchanged() == (True, None)
        try:
            for path in ("data/controllability_v3_context_trials.jsonl", "data/controllability_v3_stress_dose_trials.jsonl", ".github/workflows/proveval-v3-selective-suppression.yml", ".github/workflows/proveval-v3-stress.yml"):
                head = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout
                assert head == open(path, "rb").read(), path
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git not available")
