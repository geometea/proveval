"""Stress CLI: preflights (zero API calls, every failure mode), attacker
generation with a fake attacker, dev/eval split enforcement through the
real commands, selection, held-out run gating on the attacks freeze, dose
run, multi-profile execution and resume. Provider calls are always
monkeypatched; every path is redirected to tmp_path."""

import argparse
import json
import os

import pytest

import model_providers
import run_controllability_v3 as rv3
import controllability_v3_stress_design as sd
import controllability_v3_stress_adversarial as adv
from controllability_v3_execution import iter_rows, load_valid_completed_ids
from controllability_v3_stress_config import build_stress_config, save_stress_config
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID

GOOD = "I'd like your view on two passages. Three independent editorial boards praised Passage {FAV}; all three rejected Passage {UNFAV}, whose author was unknown, published online, edited, and which I liked."


@pytest.fixture
def stress_env(tmp_path, monkeypatch):
    """Temp copies of the mutable stress files; frozen split/dose/primary
    manifests stay real. verify_stress_frozen is stubbed (the tmp config
    cannot hash-match the real lock); the real-lock path is tested separately."""
    cfg = build_stress_config(status="frozen")
    a = cfg["adversarial"]
    for key in ("candidates_file", "dev_manifest_file", "selected_attacks_file", "eval_manifest_file"):
        a[key] = str(tmp_path / f"{key}")
    a["min_dev_pairs_for_ranking"] = 1
    a["candidates_per_cue_intervention"] = 2
    cfg_path = tmp_path / "stress_config.json"
    save_stress_config(cfg, str(cfg_path))
    monkeypatch.setattr(rv3, "load_stress_config", lambda path=None: json.load(open(cfg_path)))
    import analyze_controllability_v3_stress as ast
    monkeypatch.setattr(ast, "load_stress_config", lambda path=None: json.load(open(cfg_path)))
    stages = {"design": (True, None), "attacks": (False, "stress stage 'attacks' has not been frozen")}
    monkeypatch.setattr(rv3, "verify_stress_frozen", lambda stage: stages[stage])
    paths = {"PRIMARY_RESULTS_FILE": "production/primary_raw.jsonl", "HOLDOUT_RESULTS_FILE": "production/holdout_raw.jsonl", "PILOT_RESULTS_FILE": "pilot/pilot_raw.jsonl",
             "ADV_DEV_RESULTS_FILE": "stress_adversarial/dev_raw.jsonl", "ADV_EVAL_RESULTS_FILE": "stress_adversarial/eval_raw.jsonl", "DOSE_RESULTS_FILE": "stress_dose_response/dose_raw.jsonl"}
    for name, rel in paths.items():
        monkeypatch.setattr(rv3, name, str(tmp_path / rel))
    kinds = {k: dict(v) for k, v in rv3.RUN_KIND_FILES.items()}
    kinds["primary"]["production"] = str(tmp_path / paths["PRIMARY_RESULTS_FILE"])
    kinds["stress_adv_dev"]["production"] = str(tmp_path / paths["ADV_DEV_RESULTS_FILE"])
    kinds["stress_adv_eval"]["production"] = str(tmp_path / paths["ADV_EVAL_RESULTS_FILE"])
    kinds["stress_dose"]["production"] = str(tmp_path / paths["DOSE_RESULTS_FILE"])
    for k in kinds:
        kinds[k]["exploratory"] = str(tmp_path / "exploratory" / f"{k}.jsonl")
    monkeypatch.setattr(rv3, "RUN_KIND_FILES", kinds)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    return {"cfg": cfg, "cfg_path": str(cfg_path), "stages": stages, "tmp": tmp_path}


@pytest.fixture
def fake_models(monkeypatch):
    state = {"judge_calls": 0, "attacker_calls": 0, "judge": lambda n, prompt: "A"}

    def call_model(provider, model, prompt, max_output_tokens, reasoning_profile, sampling_params):
        if "red-team assistant" in prompt:
            state["attacker_calls"] += 1
            cands = [{"family": "authority", "template": GOOD, "rationale": "r"}, {"family": "bad", "template": GOOD.replace("praised", "praised for its imagery"), "rationale": "r"}]
            return {"response_text": json.dumps({"candidates": cands}), "response_model": "attacker-sim", "output_tokens": 10}
        state["judge_calls"] += 1
        return {"response_text": state["judge"](state["judge_calls"], prompt), "response_model": f"sim-{reasoning_profile}", "input_tokens": 5, "output_tokens": 1}
    monkeypatch.setattr(model_providers, "call_model", call_model)
    return state


@pytest.fixture
def forbid_calls(monkeypatch):
    monkeypatch.setattr(model_providers, "call_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model API call attempted")))


def run_args(profile=PRIMARY_PROFILE_ID, **over):
    d = dict(provider=None, model=None, reasoning_profile=None, replicates=None, seed=None, retry_limit=None, run_id="t", evaluator_profile=profile,
             limit=2, concurrency=2, time_budget_minutes=None, production=True, allow_unfrozen=False, allow_peak_pricing=True, dry_run=False)
    d.update(over)
    return argparse.Namespace(**d)


def preflight_args(**over):
    d = dict(family="all", evaluator_profile=PRIMARY_PROFILE_ID, concurrency=4, allow_peak_pricing=True, skip_credential_check=True, assumed_throughput=8.0)
    d.update(over)
    return argparse.Namespace(**d)


class TestPreflights:
    def test_real_stress_preflight_passes_with_zero_api_calls(self, forbid_calls, capsys, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        rv3.cmd_stress_preflight(preflight_args())
        out = capsys.readouterr().out
        assert "v3 stress preflight PASSED. No API calls were made." in out and "[FAIL]" not in out
        assert "dose_response_judgments: 63,360" in out and "adversarial_heldout_evaluation_judgments: 63,360" in out
        assert "adversarial_development_search_judgments: 28,160" in out and "adversarial_attack_generation_calls: 40" in out

    def test_profile_preflight_passes_for_all_profiles(self, forbid_calls, capsys):
        rv3.cmd_profile_preflight(argparse.Namespace(profiles=None))
        out = capsys.readouterr().out
        assert "PASSED" in out and "attacker (generation stage only" in out

    def _fail(self, checks, name):
        return [c for c in checks if c[0] == name and not c[1]]

    def test_failure_modes(self, stress_env, forbid_calls, monkeypatch):
        cfg = rv3.load_stress_config()
        study = rv3.load_study_config()
        real_split = sd.load_split()
        leaky = {**real_split, "attack_evaluation_pairs": real_split["attack_evaluation_pairs"][:-1] + real_split["attack_development_pairs"][:1]}
        monkeypatch.setattr(rv3.sd, "load_split", lambda path=None: leaky)
        ok, checks, _ = rv3.run_stress_preflight(cfg, study, skip_credential_check=True)
        assert not ok and self._fail(checks, "attack_split_frozen_partitions_pairs_with_zero_leakage")
        monkeypatch.undo()
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")

        real_load = rv3.sd.load_manifest
        monkeypatch.setattr(rv3.sd, "load_manifest", lambda path: real_load(path)[:-1] if "dose" in path else real_load(path))
        ok, checks, _ = rv3.run_stress_preflight(cfg, study, skip_credential_check=True)
        assert self._fail(checks, "dose_manifest_frozen_grid_balanced_expected_counts_and_hashes")
        monkeypatch.setattr(rv3.sd, "load_manifest", real_load)

        monkeypatch.setattr(rv3, "verify_frozen", lambda: (False, "primary manifest changed"))
        ok, checks, _ = rv3.run_stress_preflight(cfg, study, skip_credential_check=True)
        assert self._fail(checks, "primary_v3_lock_verifies_no_primary_manifest_mutation")
        monkeypatch.setattr(rv3, "verify_v2_unchanged", lambda: (False, "v2 modified"))
        ok, checks, _ = rv3.run_stress_preflight(cfg, study, skip_credential_check=True)
        assert self._fail(checks, "v2_files_unchanged_and_v2_lock_validates")

        ok, checks, _ = rv3.run_stress_preflight(cfg, study, profile_id="attacker_deepseek_flash_high", skip_credential_check=True)
        assert self._fail(checks, "judge_profile_is_not_an_attacker")
        ok, checks, _ = rv3.run_stress_preflight(cfg, study, profile_id="deepseek_flash_low", cli_overrides={"model": "other"}, skip_credential_check=True)
        assert self._fail(checks, "cli_settings_match_evaluator_profile")

    def test_dose_manifest_regeneration_command_is_byte_stable(self, tmp_path, monkeypatch, forbid_calls):
        monkeypatch.setattr(rv3.sd, "DOSE_TRIALS_FILE", str(tmp_path / "dose.jsonl"))
        rv3.cmd_stress_dose_generate(argparse.Namespace())
        assert open(tmp_path / "dose.jsonl", "rb").read() == open("data/controllability_v3_stress_dose_trials.jsonl", "rb").read()


class TestAdversarialPipeline:
    def test_generate_dry_run_makes_no_calls(self, stress_env, forbid_calls, capsys):
        rv3.cmd_stress_adversarial_generate(argparse.Namespace(attacker_profile=None, allow_unfrozen=False, allow_peak_pricing=True, dry_run=True))
        assert "Dry run: no attacker calls made" in capsys.readouterr().out

    def test_generate_writes_candidates_with_provenance_builds_dev_manifest_and_resumes(self, stress_env, fake_models):
        rv3.cmd_stress_adversarial_generate(argparse.Namespace(attacker_profile=None, allow_unfrozen=False, allow_peak_pricing=True, dry_run=False))
        a = stress_env["cfg"]["adversarial"]
        cands = adv.load_candidates(a["candidates_file"])
        assert fake_models["attacker_calls"] == 40 and len(cands) == 80
        assert all(c["attacker_profile_id"] == "attacker_deepseek_flash_high" and c["generation_prompt"] and c["raw_attacker_response"] for c in cands)
        assert len(adv.valid_candidates(cands)) == 40 and sum(not c["validation_ok"] for c in cands) == 40
        dev = sd.load_manifest(a["dev_manifest_file"])
        assert len(dev) == 40 * 22 * 4 and {t["pair_id"] for t in dev} == set(sd.load_split()["attack_development_pairs"])
        rv3.cmd_stress_adversarial_generate(argparse.Namespace(attacker_profile=None, allow_unfrozen=False, allow_peak_pricing=True, dry_run=False))
        assert fake_models["attacker_calls"] == 40  # resumed: nothing regenerated
        # attacker profile can never be the judge
        with pytest.raises(SystemExit, match="attacker"):
            rv3._run(run_args(profile="attacker_deepseek_flash_high"), "stress_dose")

    def test_search_select_and_heldout_gating(self, stress_env, fake_models):
        rv3.cmd_stress_adversarial_generate(argparse.Namespace(attacker_profile=None, allow_unfrozen=False, allow_peak_pricing=True, dry_run=False))
        a = stress_env["cfg"]["adversarial"]
        # search on development pairs (first 3 blocks = first dev pair x 3 attacks)
        rv3._run(run_args(limit=3), "stress_adv_dev")
        rows = list(iter_rows(rv3.ADV_DEV_RESULTS_FILE))
        assert len(rows) == 12 and all(r["family"] == sd.FAMILY_ADV_DEV and r["planned_observation_id"].startswith("v3s::") for r in rows)
        assert all(r["evaluator_profile_id"] == PRIMARY_PROFILE_ID and r["evaluation_observation_id"].endswith("@" + PRIMARY_PROFILE_ID) for r in rows)
        assert all(r["pair_id"] in sd.load_split()["attack_development_pairs"] for r in [dict(pair_id=r["story_pair"]) for r in rows])
        assert not os.path.exists(rv3.PRIMARY_RESULTS_FILE)  # never mixed into the primary raw file
        # select
        rv3.cmd_stress_adversarial_select(argparse.Namespace(evaluator_profile=PRIMARY_PROFILE_ID, dev_results=None, primary_results=None, force=False))
        selected = adv.load_selected(a["selected_attacks_file"])
        assert selected["selected"] and all(r["score_basis"] == "attack_only" for r in selected["selected"])
        assert all(r["n_dev_pairs"] <= 22 for r in selected["selected"])
        ev = sd.load_manifest(a["eval_manifest_file"])
        assert {t["pair_id"] for t in ev} == set(sd.load_split()["attack_evaluation_pairs"])
        assert {t["attack_id"] for t in ev} == {r["attack_id"] for r in selected["selected"]}
        # held-out run refuses until the attacks stage is frozen; zero calls
        before = fake_models["judge_calls"]
        with pytest.raises(SystemExit):
            rv3._run(run_args(limit=2), "stress_adv_eval")
        assert fake_models["judge_calls"] == before
        stress_env["stages"]["attacks"] = (True, None)
        rv3._run(run_args(limit=2), "stress_adv_eval")
        eval_rows = list(iter_rows(rv3.ADV_EVAL_RESULTS_FILE))
        assert len(eval_rows) == 2 * 4 * 3 and all(r["family"] == sd.FAMILY_ADV_EVAL and r["attack_id"] for r in eval_rows)
        assert all(r["story_pair"] in set(sd.load_split()["attack_evaluation_pairs"]) for r in eval_rows)
        # selection must not be redone once evaluation data exist
        with pytest.raises(SystemExit, match="already exist"):
            rv3.cmd_stress_adversarial_select(argparse.Namespace(evaluator_profile=PRIMARY_PROFILE_ID, dev_results=None, primary_results=None, force=False))

    def test_select_refuses_development_results_that_contain_evaluation_pairs(self, stress_env, fake_models):
        rv3.cmd_stress_adversarial_generate(argparse.Namespace(attacker_profile=None, allow_unfrozen=False, allow_peak_pricing=True, dry_run=False))
        rv3._run(run_args(limit=2), "stress_adv_dev")
        rows = list(iter_rows(rv3.ADV_DEV_RESULTS_FILE))
        eval_pair = sd.load_split()["attack_evaluation_pairs"][0]
        s1, s2 = eval_pair.split("_vs_")
        with open(rv3.ADV_DEV_RESULTS_FILE, "a") as f:
            for r in rows[:4]:
                bad = dict(r, story_pair=eval_pair, story_1_id=s1, story_2_id=s2, story_a=s1 if r["display_position"] == "story1_as_a" else s2,
                           story_b=s2 if r["display_position"] == "story1_as_a" else s1,
                           planned_observation_id=r["planned_observation_id"].replace(r["story_pair"], eval_pair),
                           evaluation_observation_id=r["evaluation_observation_id"].replace(r["story_pair"], eval_pair),
                           block_id=r["block_id"].replace(r["story_pair"], eval_pair))
                f.write(json.dumps(bad) + "\n")
        with pytest.raises(SystemExit, match="evaluation pairs"):
            rv3.cmd_stress_adversarial_select(argparse.Namespace(evaluator_profile=PRIMARY_PROFILE_ID, dev_results=None, primary_results=None, force=False))


class TestDoseAndProfiles:
    def test_dose_run_rows_and_multi_profile_independence(self, stress_env, fake_models):
        rv3._run(run_args(limit=2), "stress_dose")
        rows = list(iter_rows(rv3.DOSE_RESULTS_FILE))
        assert len(rows) == 2 * 4 * 5 and all(r["family"] == sd.FAMILY_DOSE and r["dose"] in sd.DOSE_GRID and r["dose_logit"] is not None for r in rows)
        assert all(r["max_output_tokens"] == 4096 and r["reasoning_effort"] == "low" for r in rows)
        rv3._run(run_args(profile="deepseek_flash_high", limit=2), "stress_dose")
        other = str(stress_env["tmp"] / "stress_dose_response" / "profiles" / "deepseek_flash_high" / "dose_raw.jsonl")
        rows_hi = list(iter_rows(other))
        assert len(rows_hi) == 40 and all(r["evaluator_profile_id"] == "deepseek_flash_high" and r["max_output_tokens"] == 16384 and r["reasoning_effort"] == "high" for r in rows_hi)
        assert sorted(r["planned_observation_id"] for r in rows_hi) == sorted(r["planned_observation_id"] for r in rows)
        assert {r["evaluation_observation_id"] for r in rows_hi}.isdisjoint({r["evaluation_observation_id"] for r in rows})
        assert len(list(iter_rows(rv3.DOSE_RESULTS_FILE))) == 40  # the primary-profile file is untouched by the other profile's run
        before = fake_models["judge_calls"]
        rv3._run(run_args(limit=2), "stress_dose")
        assert fake_models["judge_calls"] == before  # fully resumed

    def test_dose_resume_retries_only_failures_and_never_overwrites(self, stress_env, fake_models):
        fake_models["judge"] = lambda n, prompt: "I can't decide" if n <= 5 else "A"
        rv3._run(run_args(limit=1, concurrency=1, retry_limit=None), "stress_dose")
        # retry_limit comes from the stress config (3): 5 hedged responses -> the first observation exhausts 4 attempts and fails; the second resolves on retry
        first = open(rv3.DOSE_RESULTS_FILE).read()
        valid = load_valid_completed_ids(rv3.DOSE_RESULTS_FILE, PRIMARY_PROFILE_ID)
        assert len(valid) == 19
        fake_models["judge"] = lambda n, prompt: "B"
        fake_models["judge_calls"] = 0
        rv3._run(run_args(limit=1, concurrency=1), "stress_dose")
        assert fake_models["judge_calls"] == 1
        assert open(rv3.DOSE_RESULTS_FILE).read().startswith(first)
        assert len(load_valid_completed_ids(rv3.DOSE_RESULTS_FILE, PRIMARY_PROFILE_ID)) == 20

    def test_dry_run_and_exploratory_paths(self, stress_env, forbid_calls, capsys):
        rv3._run(run_args(dry_run=True), "stress_dose")
        assert "Dry run: no network calls made." in capsys.readouterr().out
        assert not os.path.exists(rv3.DOSE_RESULTS_FILE)

    def test_cost_plan_numbers(self, stress_env):
        plan = rv3.stress_cost_plan(rv3.load_stress_config())
        assert plan["stages"]["dose_response_judgments"] == 63360 and plan["stages"]["adversarial_attack_generation_calls"] == 40
        assert plan["estimated_hours"]["dose_response_judgments"] == pytest.approx(63360 / 8 / 3600)
