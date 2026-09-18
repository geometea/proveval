"""Stress-study config (data/controllability_v3_stress_config.json):
every pre-specified number and definition for the two stress families,
read from the code modules (never retyped). Frozen by
freeze_controllability_v3_stress.py (stage "design"); attacks are frozen
later (stage "attacks") once generated and selected."""

import json

import controllability_v3_stress_adversarial as adv
import controllability_v3_stress_design as sd
import controllability_v3_stress_dose as dose
from controllability_v3_design import EXPERIMENT_ID, INTERVENTION_IDS
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID

STRESS_CONFIG_FILE = sd.STRESS_CONFIG_FILE

DOSE_HEADLINE_CONTRASTS = [["I1", "I4"], ["I4", "I0"], ["I1", "I0"]]
DOSE_HIGH_DOSES = [90, 99]
REVERSAL_TARGET_GAIN_PP = 10          # "dose at which the context-favoured passage gains +10 percentage points"


def build_stress_config(status="draft"):
    return {
        "design_version": "v3_stress",
        "parent_experiment_id": EXPERIMENT_ID,
        "experiment_id": "context_controllability_v3_stress",
        "default_evaluator_profile_id": PRIMARY_PROFILE_ID,
        "random_seed": 20260919,
        "bootstrap_draws": 5000,
        "retry_limit": 3,
        "relative_min_denominator": 0.02,
        "adversarial": {
            "split_file": sd.SPLIT_FILE, "split_seed": sd.SPLIT_SEED, "n_development_pairs": sd.N_DEV_PAIRS, "n_evaluation_pairs": sd.N_EVAL_PAIRS,
            "candidates_per_cue_intervention": adv.N_CANDIDATES_PER_CUE_INTERVENTION, "generation_attempts": adv.GENERATION_ATTEMPTS,
            "top_k": adv.TOP_K, "min_dev_pairs_for_ranking": adv.MIN_DEV_PAIRS_FOR_RANKING,
            "dev_replicates": adv.DEV_REPLICATES, "eval_replicates": adv.EVAL_REPLICATES,
            "default_attacker_profile_id": "attacker_deepseek_flash_high",
            "attacker_system_prompt_sha256": adv.sha256_hex(adv.ATTACKER_SYSTEM_PROMPT),
            "style_dimensions_offered": list(adv.ATTACK_STYLE_DIMENSIONS),
            "score_definition": adv.SCORE_DEFINITION,
            "candidates_file": sd.CANDIDATES_FILE, "dev_manifest_file": sd.DEV_TRIALS_FILE,
            "selected_attacks_file": sd.SELECTED_ATTACKS_FILE, "eval_manifest_file": sd.EVAL_TRIALS_FILE,
            "interventions": list(INTERVENTION_IDS),
        },
        "dose_response": {
            "manifest_file": sd.DOSE_TRIALS_FILE, "cue_id": dose.DOSE_CUE_ID, "panel_size": sd.DOSE_PANEL_SIZE, "grid": list(sd.DOSE_GRID),
            "intro_template": sd.DOSE_INTRO_TEMPLATE, "dose_scale": "logit(n / (panel - n))", "replicates": dose.DOSE_REPLICATES,
            "unique_cells": dose.DOSE_UNIQUE_CELLS, "zero_dose_reference": "primary_nocontext (same intervention) from the primary experiment",
            "headline_contrasts": DOSE_HEADLINE_CONTRASTS, "high_doses": DOSE_HIGH_DOSES, "reversal_target_gain_pp": REVERSAL_TARGET_GAIN_PP,
            "interventions": list(INTERVENTION_IDS),
        },
        "status": status,
    }


def load_stress_config(path=STRESS_CONFIG_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_stress_config(config, path=STRESS_CONFIG_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def main():
    config = build_stress_config()
    save_stress_config(config)
    print(f"Wrote {STRESS_CONFIG_FILE} (status={config['status']})")


if __name__ == "__main__":
    main()
