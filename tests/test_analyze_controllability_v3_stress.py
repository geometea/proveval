"""Stress analysis on synthetic rows with planted truth: dose slopes and
thresholds, attack recovery on held-out pairs only, per-profile
stratification (never pooled), cross-evaluator tables, leakage refusal."""

import json
import os

import pytest

import analyze_controllability_v3_stress as ast
import controllability_v3_stress_design as sd
import controllability_v3_stress_stats as ss
from controllability_v3_design import INTERVENTION_IDS
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID, PROFILES
from controllability_v3_stress_config import build_stress_config
from controllability_v3_study_config import build_study_config
from controllability_v3_trials import make_planned_observation_id, make_trial_id

STORIES = ["a", "b", "c", "d"]
PAIRS = [(x, y) for i, x in enumerate(STORIES) for y in STORIES[i + 1:]]
CUE = "reception_positive_vs_negative"
REPS = 20
BASE = {p: 0.3 + 0.05 * i for i, p in enumerate(PAIRS)}
DOSE_SLOPE = {"I0": 0.09, "I1": 0.06, "I4": 0.01}
ATTACK_BIAS = {"I0": 0.25, "I1": 0.15, "I4": 0.0}       # ordinary cue half-width per intervention
RECOVERY = {"atk_x": 0.15, "atk_y": 0.05}                # extra half-width under attack
PROFILE_SCALE = {PRIMARY_PROFILE_ID: 1.0, "deepseek_flash_high": 0.5}


def realized(rate, r):
    rate = min(0.999, max(0.001, rate))
    return r <= round(rate * REPS)


def evaluator(pid):
    p = PROFILES[pid]
    return {"provider": p["provider"], "requested_model": p["model"], "reasoning_profile": p["reasoning_profile"], "max_output_tokens": p["max_output_tokens"]}


def row(family, pair, cue, assignment, position, iid, variant, r, story1_chosen, pid=PRIMARY_PROFILE_ID, **extra):
    s1, s2 = pair
    story_a, story_b = (s1, s2) if position == "story1_as_a" else (s2, s1)
    choice = ("A" if story1_chosen else "B") if position == "story1_as_a" else ("B" if story1_chosen else "A")
    if family.startswith("stress"):
        tid = sd.make_stress_trial_id(family, f"{s1}_vs_{s2}", cue, assignment, position, iid, variant)
        oid = sd.make_stress_observation_id(tid, r)
        block = "::".join(["v3s", family, f"{s1}_vs_{s2}", cue, iid, variant])
    else:
        tid = make_trial_id(family, f"{s1}_vs_{s2}", cue, assignment, position, iid, cue is not None)
        oid = make_planned_observation_id(tid, r)
        block = "::".join(["v3", family, f"{s1}_vs_{s2}"] + ([cue] if cue else []) + [iid])
    return {"planned_observation_id": oid, "evaluation_observation_id": f"{oid}@{pid}", "evaluator_profile_id": pid, "family": family, "collection": "production",
            "trial_id": tid, "block_id": block, "story_pair": f"{s1}_vs_{s2}", "story_1_id": s1, "story_2_id": s2, "story_a": story_a, "story_b": story_b,
            "cue": cue, "context_assignment": assignment, "display_position": position, "intervention_id": iid, "instruction_id": iid, "variant": variant,
            "context_present": cue is not None, "replicate": r, "evaluator": evaluator(pid), "provider": evaluator(pid)["provider"], "model": evaluator(pid)["requested_model"],
            "reasoning_effort": evaluator(pid)["reasoning_profile"], "max_output_tokens": evaluator(pid)["max_output_tokens"],
            "raw_response": choice, "parsed_choice": choice, "parsing_status": "resolved", "first_attempt_status": "valid",
            "attempts": [{"input_tokens": 10, "output_tokens": 2, "reasoning_tokens": 1}], "total_attempts": 1, "output_tokens": 2, "reasoning_tokens": 1,
            "latency_seconds": 0.5, "input_tokens": 10, "cached_input_tokens": 0, **extra}


def synth(pid=PRIMARY_PROFILE_ID, split=None):
    scale = PROFILE_SCALE[pid]
    primary, dev, ev, dose = [], [], [], []
    for pair in PAIRS:
        for iid in ATTACK_BIAS:
            for position in ("story1_as_a", "story2_as_a"):
                for r in range(1, REPS + 1):
                    primary.append(row("primary_nocontext", pair, None, None, position, iid, None, r, realized(BASE[pair], r), pid))
            for assignment in ("forward", "flipped"):
                sign = 1 if assignment == "forward" else -1
                for position in ("story1_as_a", "story2_as_a"):
                    for r in range(1, REPS + 1):
                        primary.append(row("primary_context", pair, CUE, assignment, position, iid, None, r, realized(BASE[pair] + sign * ATTACK_BIAS[iid] * scale, r), pid))
                        for atk, extra_bias in RECOVERY.items():
                            fam = sd.FAMILY_ADV_EVAL if (split is None or f"{pair[0]}_vs_{pair[1]}" in split["attack_evaluation_pairs"]) else sd.FAMILY_ADV_DEV
                            ev_rate = BASE[pair] + sign * (ATTACK_BIAS[iid] + extra_bias) * scale
                            (ev if fam == sd.FAMILY_ADV_EVAL else dev).append(row(fam, pair, CUE, assignment, position, iid, atk, r, realized(ev_rate, r), pid, attack_id=atk))
                    for n in sd.DOSE_GRID:
                        for r in range(1, REPS + 1):
                            rate = BASE[pair] + sign * DOSE_SLOPE[iid] * ss.dose_x_of(n) * scale
                            dose.append(row(sd.FAMILY_DOSE, pair, "reader_consensus_numeric", assignment, position, iid, sd.dose_id(n), r, realized(rate, r), pid, dose=n, dose_logit=ss.dose_x_of(n)))
    return primary, dev, ev, dose


SPLIT = {"attack_development_pairs": ["a_vs_b", "a_vs_c"], "attack_evaluation_pairs": ["a_vs_d", "b_vs_c", "b_vs_d", "c_vs_d"]}
SELECTED = {"selected": [{"attack_id": a, "cue_id": CUE, "intervention_id": i, "rank": 1 if a == "atk_x" else 2, "dev_score": RECOVERY[a], "attack_family": "f"} for a in RECOVERY for i in ATTACK_BIAS],
            "ranked": [{"attack_id": a, "cue_id": CUE, "intervention_id": i, "dev_score": RECOVERY[a], "rank": 1, "template": "t", "eligible": True} for a in RECOVERY for i in ATTACK_BIAS]}


@pytest.fixture(scope="module")
def configs():
    study = build_study_config(status="frozen")
    stress = build_stress_config(status="frozen")
    stress["bootstrap_draws"] = 100
    return study, stress


@pytest.fixture(scope="module")
def results(configs):
    study, stress = configs
    primary, dev, ev, dose = synth(split=SPLIT)
    return ast.analyze_profile(PRIMARY_PROFILE_ID, primary, dev, ev, dose, study, stress, 100, 1, SPLIT, SELECTED)


class TestDose:
    def test_slopes_recover_planted_ordering_with_cis(self, results):
        slopes = {r["intervention_id"]: r for r in results["dose_susceptibility_slopes"]}
        assert slopes["I0"]["slope"] > slopes["I1"]["slope"] > slopes["I4"]["slope"]
        assert slopes["I0"]["slope"] == pytest.approx(2 * DOSE_SLOPE["I0"], abs=0.03)   # D_pair = 2 x half-width
        assert slopes["I0"]["slope_ci_95_lo"] <= slopes["I0"]["slope"] <= slopes["I0"]["slope_ci_95_hi"]
        assert slopes["I0"]["dose_scale"] == ss.DOSE_SCALE_LABEL

    def test_thresholds_have_uncertainty_and_flags(self, results):
        t = {r["intervention_id"]: r for r in results["dose_reversal_thresholds"]}
        assert t["I0"]["plus_gain_threshold_dose"] is not None and t["I0"]["plus_gain_threshold_ci_95_lo"] is not None
        assert t["I0"]["reversal_dose_p50"] is not None and "reversal_dose_p50_extrapolated" in t["I0"]
        assert t["I0"]["beta_dose"] > t["I4"]["beta_dose"]

    def test_interactions_and_known_random_contrasts(self, results):
        inter = {(r["intervention_id"], r["interaction"]): r for r in results["dose_intervention_interactions"]}
        assert inter[("I4", "suppression_slope_vs_I0")]["suppression_slope"] > inter[("I1", "suppression_slope_vs_I0")]["suppression_slope"] > 0
        assert ("I0", "ambiguous_minus_strong_slope") in inter
        kr = [r for r in results["known_random_high_dose_contrasts"] if r["contrast"] == "I1_vs_I4" and r["dose"] == 99][0]
        assert kr["CE_difference"] > 0 and kr["high_dose"]
        curve = [r for r in results["dose_response_by_intervention"] if r["intervention_id"] == "I0"]
        assert [r["dose"] for r in curve] == list(sd.DOSE_GRID) and curve[-1]["CE"] > curve[0]["CE"]
        assert any(r["pair_strength_group"] == "ambiguous_half" for r in results["dose_response_by_pair_strength"])


class TestAdversarial:
    def test_recovery_measured_on_heldout_pairs_only(self, results):
        rec = {(r["intervention_id"], r["attack_id"]): r for r in results["adversarial_attack_recovery"]}
        assert rec[("I0", "atk_x")]["n_story_pairs"] == 4  # the four evaluation pairs, never the two development pairs
        assert rec[("I0", "atk_x")]["attack_recovery"] == pytest.approx(2 * RECOVERY["atk_x"], abs=0.06)
        assert rec[("I0", "atk_y")]["attack_recovery"] == pytest.approx(2 * RECOVERY["atk_y"], abs=0.06)
        assert rec[("I0", "atk_x")]["attack_recovery"] > rec[("I0", "atk_y")]["attack_recovery"]

    def test_by_intervention_sets_success_and_robust_suppression(self, results):
        by = {(r["intervention_id"], r["attack_set"]): r for r in results["adversarial_effects_by_intervention"]}
        top1, mean = by[("I1", "top1_by_dev_score")], by[("I1", "mean_of_selected")]
        assert top1["n_attacks"] == 1 and mean["n_attacks"] == 2 and top1["CE_attack"] >= mean["CE_attack"]
        assert 0 <= mean["attack_success_rate_point"] <= 1 and mean["choice_flip_rate"] is not None
        assert by[("I1", "mean_of_selected")]["robust_suppression_reported"]
        assert by[("I1", "mean_of_selected")]["robust_suppression"] < by[("I1", "mean_of_selected")]["ordinary_suppression"]

    def test_generalization_loso_rankings_and_cue_table(self, results):
        per_attack = [r for r in results["attack_generalization"] if r["attack_id"] != "SUMMARY"]
        assert per_attack and all(r["dev_score"] is not None and r["eval_attack_recovery"] is not None for r in per_attack)
        assert not [r for r in results["attack_generalization"] if r["attack_id"] == "SUMMARY"]  # a correlation needs >= 3 attacks; this fixture has 2
        assert {r["excluded_story_id"] for r in results["adversarial_leave_one_story_out"]} == set(STORIES)
        assert len(results["attack_rankings"]) == len(SELECTED["ranked"])
        assert any(r["cue_id"] == CUE for r in results["adversarial_effects_by_cue"])

    def test_evaluation_rows_on_development_pairs_are_refused(self, configs):
        study, stress = configs
        primary, dev, ev, dose = synth(split=SPLIT)
        leaked = ev + [dict(r, family=sd.FAMILY_ADV_EVAL, planned_observation_id=r["planned_observation_id"].replace("stress_adversarial_dev", "stress_adversarial_eval"),
                            evaluation_observation_id=r["evaluation_observation_id"].replace("stress_adversarial_dev", "stress_adversarial_eval"),
                            trial_id=r["trial_id"].replace("stress_adversarial_dev", "stress_adversarial_eval"), block_id=r["block_id"].replace("stress_adversarial_dev", "stress_adversarial_eval")) for r in dev]
        with pytest.raises(ValueError, match="non-evaluation pairs"):
            ast.analyze_profile(PRIMARY_PROFILE_ID, primary, [], leaked, [], study, stress, 20, 1, SPLIT, SELECTED)


class TestProfiles:
    def test_profiles_are_never_pooled_and_cross_tables_are_written(self, configs, tmp_path):
        study, stress = configs
        p1, d1, e1, o1 = synth(PRIMARY_PROFILE_ID, SPLIT)
        p2, d2, e2, o2 = synth("deepseek_flash_high", SPLIT)
        alone = ast.analyze_profile(PRIMARY_PROFILE_ID, p1, d1, e1, o1, study, stress, 50, 1, SPLIT, SELECTED)
        mixed = ast.analyze_profile(PRIMARY_PROFILE_ID, p1 + p2, d1 + d2, e1 + e2, o1 + o2, study, stress, 50, 1, SPLIT, SELECTED)
        assert [r["slope"] for r in alone["dose_susceptibility_slopes"]] == [r["slope"] for r in mixed["dose_susceptibility_slopes"]]
        assert any("evaluator_profile_mismatch:deepseek_flash_high" in e["reason"] for e in mixed["exclusions"])
        high = ast.analyze_profile("deepseek_flash_high", p1 + p2, d1 + d2, e1 + e2, o1 + o2, study, stress, 50, 1, SPLIT, SELECTED)
        s_low = {r["intervention_id"]: r["slope"] for r in alone["dose_susceptibility_slopes"]}
        s_high = {r["intervention_id"]: r["slope"] for r in high["dose_susceptibility_slopes"]}
        assert s_high["I0"] < s_low["I0"]  # the planted 'different judge' (half the dose sensitivity)
        written = ast.write_cross_evaluator_outputs({PRIMARY_PROFILE_ID: alone, "deepseek_flash_high": high}, str(tmp_path))
        assert set(written) == {"cross_evaluator_attack_robustness.csv", "cross_evaluator_dose_response.csv"}
        text = open(tmp_path / "cross_evaluator_dose_response.csv").read()
        assert "deepseek_flash_high" in text and PRIMARY_PROFILE_ID in text

    def test_run_stress_analysis_end_to_end_writes_per_profile_dirs(self, tmp_path, monkeypatch, configs):
        study, stress = configs
        monkeypatch.setattr(ast, "load_stress_config", lambda path=None: stress)
        monkeypatch.setattr(ast, "load_study_config", lambda path=None: study)
        monkeypatch.setattr(ast, "verify_stress_frozen", lambda stage: (True, None))
        monkeypatch.setattr(ast.sd, "load_split", lambda path=None: SPLIT)
        monkeypatch.setattr(os.path, "exists", lambda p: True if str(p).endswith("split.json") else os.path.lexists(p))
        files = {}
        for pid in (PRIMARY_PROFILE_ID, "deepseek_flash_high"):
            p, d, e, o = synth(pid, SPLIT)
            for name, rows in (("primary", p), ("dev", d), ("eval", e), ("dose", o)):
                path = tmp_path / f"{pid}_{name}.jsonl"
                with open(path, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                files.setdefault(pid, {})[name] = str(path)
        f1, f2 = files[PRIMARY_PROFILE_ID], files["deepseek_flash_high"]
        per = ast.run_stress_analysis(f1["primary"], f1["dev"], f1["eval"], f1["dose"], str(tmp_path / "out"), 30, extra_profile_files=[f2])
        assert set(per) == {PRIMARY_PROFILE_ID, "deepseek_flash_high"}
        assert os.path.exists(tmp_path / "out" / "dose_susceptibility_slopes.csv")
        assert os.path.exists(tmp_path / "out" / "profiles" / "deepseek_flash_high" / "dose_susceptibility_slopes.csv")
        assert os.path.exists(tmp_path / "out" / "cross_evaluator_dose_response.csv")
        for name in ast.STRESS_OUTPUT_TABLES:
            if name not in ("stress_incomplete_units.csv", "stress_exclusions.csv", "attack_rankings.csv"):
                assert os.path.exists(tmp_path / "out" / name), name
