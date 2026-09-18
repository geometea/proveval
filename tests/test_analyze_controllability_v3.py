"""Analysis orchestration on synthetic v3 result rows with KNOWN truth:
inclusion/exclusion, dedupe, complete-unit filtering, exact recovery of
context effects/suppression/drift/holdout transfer from deterministic
responses, baseline strength from the no-context I0 condition only, and
the full output set."""

import csv
import json
import os

import pytest

import analyze_controllability_v3 as an
from controllability_v3_design import CUE_ORDER, INTERVENTION_IDS, holdout_instruction_id
from controllability_v3_study_config import build_study_config
from controllability_v3_trials import make_planned_observation_id, make_trial_id

STORIES = ["a", "b", "c", "d"]
PAIRS = [(x, y) for i, x in enumerate(STORIES) for y in STORIES[i + 1:]]
CUES = list(CUE_ORDER[:2])
INTERVENTIONS = ["I0", "I1", "I4", "I5"]
REPS = 20
EVALUATOR = {"provider": "deepseek", "requested_model": "deepseek-flash", "reasoning_profile": "low"}

BASE = {p: 0.3 + 0.05 * i for i, p in enumerate(PAIRS)}         # no-context story1 preference (0.3..0.55); every planted rate is a multiple of 1/REPS inside [0, 1]
BIAS = {"I0": 0.2, "I1": 0.1, "I4": 0.0, "I5": 0.0}                 # context effect half-width: D_pair = 2*bias
HOLDOUT_BIAS = 0.15
SHIFT = {"I0": 0.0, "I1": 0.0, "I4": 0.0, "I5": -0.2}               # no-context drift


def realized(rate, replicate):
    """Deterministic response sequence realising `rate` exactly over REPS replicates."""
    return replicate <= round(rate * REPS)


def make_row(family, pair, cue, assignment, position, intervention, context_present, replicate, story1_chosen, valid=True,
             evaluator=EVALUATOR, collection="production", instruction=None):
    s1, s2 = pair
    story_a, story_b = (s1, s2) if position == "story1_as_a" else (s2, s1)
    choice = ("A" if story1_chosen else "B") if position == "story1_as_a" else ("B" if story1_chosen else "A")
    instruction = instruction or intervention
    trial_id = make_trial_id(family, f"{s1}_vs_{s2}", cue, assignment, position, instruction, context_present)
    block = "::".join(["v3", family, f"{s1}_vs_{s2}"] + ([cue] if cue else []) + [instruction])
    return {
        "planned_observation_id": make_planned_observation_id(trial_id, replicate), "family": family, "collection": collection,
        "trial_id": trial_id, "block_id": block, "story_pair": f"{s1}_vs_{s2}", "story_1_id": s1, "story_2_id": s2, "story_a": story_a, "story_b": story_b,
        "cue": cue, "cue_family": None, "context_assignment": assignment, "display_position": position, "intervention_id": instruction,
        "instruction_id": instruction, "context_present": context_present, "replicate": replicate, "evaluator": dict(evaluator),
        "raw_response": choice if valid else "both are good", "parsed_choice": choice if valid else None,
        "parsing_status": "resolved" if valid else "unresolved", "first_attempt_status": "valid" if valid else "invalid",
        "attempts": [{"input_tokens": 100, "output_tokens": 5, "reasoning_tokens": 50, "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20}],
        "total_attempts": 1, "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 5, "reasoning_tokens": 50, "latency_seconds": 1.0,
        "response_model": "deepseek-flash-x",
    }


def synth():
    primary, holdout = [], []
    for pair in PAIRS:
        for i in INTERVENTIONS:
            for cue in CUES:
                for assignment in ("forward", "flipped"):
                    rate = BASE[pair] + (BIAS[i] if assignment == "forward" else -BIAS[i])
                    for position in ("story1_as_a", "story2_as_a"):
                        for r in range(1, REPS + 1):
                            primary.append(make_row("primary_context", pair, cue, assignment, position, i, True, r, realized(rate, r)))
            for position in ("story1_as_a", "story2_as_a"):
                for r in range(1, REPS + 1):
                    primary.append(make_row("primary_nocontext", pair, None, None, position, i, False, r, realized(BASE[pair] + SHIFT[i], r)))
        for cue in CUES:
            hid = holdout_instruction_id("I2", cue)
            for assignment in ("forward", "flipped"):
                rate = BASE[pair] + (HOLDOUT_BIAS if assignment == "forward" else -HOLDOUT_BIAS)
                for position in ("story1_as_a", "story2_as_a"):
                    for r in range(1, REPS + 1):
                        holdout.append(make_row("holdout", pair, cue, assignment, position, hid, True, r, realized(rate, r), instruction=hid))
    return primary, holdout


@pytest.fixture(scope="module")
def config():
    cfg = build_study_config(status="frozen")
    cfg["bootstrap_draws"] = 100
    return cfg


@pytest.fixture(scope="module")
def results(config):
    primary, holdout = synth()
    # add I2 context rows so the holdout comparison has a full-enumeration reference
    for pair in PAIRS:
        for cue in CUES:
            for assignment in ("forward", "flipped"):
                rate = BASE[pair] + (0.05 if assignment == "forward" else -0.05)
                for position in ("story1_as_a", "story2_as_a"):
                    for r in range(1, REPS + 1):
                        primary.append(make_row("primary_context", pair, cue, assignment, position, "I2", True, r, realized(rate, r)))
    return an.analyze(primary, holdout, config, True, 100, 1)


def by_intervention(rows, key="intervention_id"):
    return {r[key]: r for r in rows}


class TestRecovery:
    def test_context_effects_are_recovered_exactly(self, results):
        ce = {(r["intervention_id"], r["cue_id"]): r["context_effect"] for r in results["context_effects_by_intervention"]}
        for i in INTERVENTIONS:
            assert ce[(i, "all")] == pytest.approx(2 * BIAS[i], abs=1e-9)
            for cue in CUES:
                assert ce[(i, cue)] == pytest.approx(2 * BIAS[i], abs=1e-9)

    def test_suppression_is_control_minus_treated(self, results):
        sup = by_intervention(results["suppression_by_intervention"])
        assert sup["I1"]["signed_suppression"] == pytest.approx(0.2) and sup["I4"]["magnitude_suppression"] == pytest.approx(0.4)
        assert sup["I0"]["signed_suppression"] == pytest.approx(0.0) and sup["I1"]["relative_suppression"] == pytest.approx(0.5)
        assert sup["I1"]["residual_context_effect"] == pytest.approx(0.2)

    def test_drift_is_zero_without_shift_and_present_with_it(self, results):
        drift = by_intervention(results["no_context_drift_by_intervention"])
        assert drift["I0"]["abs_prob_shift"] == 0.0 and drift["I0"]["rms_prob_shift_noise_corrected"] == 0.0
        assert drift["I1"]["abs_prob_shift"] == pytest.approx(0.0) and drift["I1"]["rms_prob_shift_noise_corrected"] == 0.0
        assert drift["I5"]["signed_prob_shift"] == pytest.approx(-0.2) and drift["I5"]["rms_prob_shift_noise_corrected"] > 0.1
        assert drift["I5"]["pair_preference_flip"] > 0

    def test_frontier_flags_and_axes(self, results):
        frontier = by_intervention(results["suppression_distortion_frontier"])
        assert frontier["I4"]["pareto_efficient"] and not frontier["I5"]["pareto_efficient"] and "I4" in frontier["I5"]["dominated_by"]
        assert frontier["I4"]["magnitude_suppression"] == pytest.approx(0.4) and frontier["I4"]["drift_rms_prob_shift_noise_corrected"] == 0.0

    def test_headline_contrasts_present_for_available_pairs(self, results):
        contrasts = {r["contrast"]: r for r in results["headline_contrasts"]}
        assert set(contrasts) == {"I1_vs_I0", "I4_vs_I0", "I4_vs_I1", "I5_vs_I1", "I4_vs_I5"}
        assert contrasts["I4_vs_I1"]["residual_ce_difference"] == pytest.approx(-0.2)
        assert contrasts["I4_vs_I5"]["drift_mean_squared_shift_difference"] < 0

    def test_holdout_transfer(self, results):
        rows = {r["cue_id"]: r for r in results["held_out_cue_generalization"]}
        assert set(rows) == set(CUES)
        r = rows[CUES[0]]
        assert r["CE_control"] == pytest.approx(0.4) and r["CE_full"] == pytest.approx(0.1) and r["CE_holdout"] == pytest.approx(0.3)
        assert r["suppression_full_enumeration"] == pytest.approx(0.3) and r["suppression_holdout"] == pytest.approx(0.1)
        assert r["transfer_gap_full_minus_holdout"] == pytest.approx(0.2) and r["transfer_fraction"] == pytest.approx(1 / 3)

    def test_baseline_strength_comes_only_from_nocontext_i0(self, results, config):
        table = {(r["story_1_id"], r["story_2_id"]): r for r in results["baseline_pair_strength"]}
        assert len(table) == len(PAIRS)
        wins = 2 * round(BASE[PAIRS[0]] * REPS)
        assert table[PAIRS[0]]["baseline_n"] == 2 * REPS and table[PAIRS[0]]["baseline_p"] == pytest.approx((wins + 0.5) / (2 * REPS + 1))
        primary, holdout = synth()
        shifted = [dict(r, parsed_choice=("B" if r["parsed_choice"] == "A" else "A"), raw_response=("B" if r["parsed_choice"] == "A" else "A"))
                   if r["family"] == "primary_nocontext" and r["intervention_id"] == "I1" else r for r in primary]
        again = an.analyze(shifted, [], config, True, 20, 1)
        assert [r["baseline_strength"] for r in again["baseline_pair_strength"]] == [r["baseline_strength"] for r in results["baseline_pair_strength"]]

    def test_ambiguity_and_position_and_loso_and_stability_tables(self, results):
        amb = results["suppression_by_ambiguity"]
        assert any(r["analysis"] == "regression_signed_suppression_on_baseline_strength" and r["intervention_id"] == "I1" and r["cue_id"] == "all" for r in amb)
        assert any(r["analysis"].startswith("tertile_") for r in amb) and any("median_split" in r["analysis"] for r in amb)
        assert not any(r["analysis"] == "regression_signed_suppression_on_baseline_strength" and r["intervention_id"] == "I0" for r in amb)
        pos = results["position_diagnostics"]
        assert {r["family"] for r in pos} == {"primary_context", "primary_nocontext"}
        loso = results["leave_one_story_out"]
        assert {r["excluded_story_id"] for r in loso} == set(STORIES)
        stab = results["replicate_stability"]
        assert all(abs(r["estimate_odd_replicates"] - r["estimate_even_replicates"]) < 0.25 for r in stab if r["family"] == "primary_context")
        inter = results["intervention_by_cue_interaction"]
        assert all(abs(r["interaction"]) < 1e-9 for r in inter)  # planted effects are cue-homogeneous


class TestInclusion:
    def test_pilot_foreign_and_mismatched_rows_are_excluded_and_counted(self, config):
        primary, _ = synth()
        pilot = dict(primary[0], family="pilot", planned_observation_id=primary[0]["planned_observation_id"].replace("v3::primary_context", "v3::pilot::primary_context"))
        other = dict(primary[1], evaluator={"provider": "openai", "requested_model": "gpt-5.6", "reasoning_profile": "low"}, planned_observation_id="v3::primary_context::zz::r1")
        v2 = dict(primary[2], planned_observation_id="context_controllability_v2::x")
        by_id, exclusions = an.include_rows(primary + [pilot, other, v2], an.PRIMARY_FAMILIES, config)
        reasons = {e["reason"]: e["n_rows"] for e in exclusions}
        assert reasons == {"family_not_allowed:pilot": 1, "evaluator_mismatch": 1, "not_a_v3_id": 1}
        assert len(by_id) == len(primary)

    def test_dedupe_prefers_the_valid_row_and_keeps_the_first_for_compliance(self, config):
        primary, _ = synth()
        row = primary[0]
        failed = dict(row, parsing_status="unresolved", parsed_choice=None, raw_response="hmm", first_attempt_status="invalid")
        by_id = an.dedupe_rows([failed, row, dict(row, raw_response="B")])
        entry = by_id[row["planned_observation_id"]]
        assert entry["first"] is failed and entry["valid"] is row and entry["n_rows"] == 3
        comp = an.response_compliance(by_id)
        assert comp[0]["n_first_attempt_invalid"] == 1 and comp[0]["n_resolved"] == 1

    def test_incomplete_units_are_excluded_and_diagnosed(self, config):
        primary, _ = synth()
        # drop one cell of one context block (never attempted) and fail another block's cell (attempted)
        drop_id = primary[0]["planned_observation_id"]
        rows = [r for r in primary if r["planned_observation_id"] != drop_id]
        fail_target = next(r for r in rows if r["family"] == "primary_context" and r["replicate"] == 2 and r["intervention_id"] == "I1")
        rows = [dict(r, parsing_status="unresolved", parsed_choice=None, raw_response="neither") if r is fail_target else r for r in rows]
        res = an.analyze(rows, [], config, True, 20, 1)
        c = res["completeness"]
        assert c["n_incomplete_context_units"] == 2
        reasons = {d["reason"] for d in res["incomplete_units"]}
        assert reasons == {"temporarily_incomplete", "terminally_incomplete"}
        n_cells = sum(r["n_resolved_cells_in_complete_units"] for r in res["cell_counts"] if r["family"] == "primary_context")
        assert n_cells == len([r for r in primary if r["family"] == "primary_context"]) - 8
        assert not c["primary_complete"]


class TestOutputs:
    def test_every_table_json_and_plot_is_written_to_the_analysis_dir_only(self, results, config, tmp_path):
        out = tmp_path / "analysis"
        written = an.write_outputs(results, config, True, str(out))
        for name in an.OUTPUT_TABLES:
            if name in ("incomplete_units.csv", "exclusions.csv"):
                continue  # legitimately empty on complete, clean data (nothing to report)
            assert name in written, name
        assert {"headline_results.json", "summary.json", "suppression_distortion_frontier.svg"} <= set(written)
        headline = json.load(open(out / "headline_results.json"))
        assert "interpretation_note" in headline and headline["completeness"]["primary_complete"] is False
        assert set(os.listdir(tmp_path)) == {"analysis"}
        rows = list(csv.DictReader(open(out / "suppression_distortion_frontier.csv")))
        assert {r["intervention_id"] for r in rows} == set(INTERVENTIONS)  # I2 has no no-context rows in this fixture, so no frontier point

    def test_run_analysis_end_to_end(self, tmp_path, monkeypatch):
        primary, holdout = synth()
        p, h = tmp_path / "primary_raw.jsonl", tmp_path / "holdout_raw.jsonl"
        for path, rows in ((p, primary), (h, holdout)):
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        res = an.run_analysis(str(p), str(h), bootstrap_draws=50, analysis_dir=str(tmp_path / "out"))
        assert os.path.exists(tmp_path / "out" / "headline_results.json")
        assert res["completeness"]["n_valid_primary"] == len(primary)
