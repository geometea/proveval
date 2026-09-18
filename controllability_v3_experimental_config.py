"""Config for the three EXPERIMENTAL v3 families (adaptive dose, iterative
adversary, capability sweep): every policy number read from the modules,
written to data/controllability_v3_experimental_config.json and frozen in
stages by freeze_controllability_v3_experimental.py."""

import json

import controllability_v3_adaptive_dose as ad
import controllability_v3_capability_sweep as cap
import controllability_v3_iterative_attack as it
import controllability_v3_stress_design as sd
from controllability_v3_evaluator_profiles import PRIMARY_PROFILE_ID

CONFIG_FILE = "data/controllability_v3_experimental_config.json"


def build_config(status=None):
    adaptive = dict(ad.DEFAULT_POLICY)
    arith = ad.design_arithmetic(adaptive)
    adaptive["global_max_judgments"] = arith["max_judgments_hard_budget"]
    return {
        "design_version": "v3_experimental",
        "experiment_id": "context_controllability_v3_experimental",
        "default_evaluator_profile_id": PRIMARY_PROFILE_ID,
        "random_seed": 20260921,
        "bootstrap_draws": 5000,
        "retry_limit": 3,
        "adaptive_dose": {"policy": adaptive, "arithmetic": arith, "split_file": None,
                          "schedule_file": "results/controllability_v3/stress_dose_adaptive/adaptive_dose_schedule.jsonl",
                          "state_file": "results/controllability_v3/stress_dose_adaptive/adaptive_dose_state.json",
                          "status": (status or {}).get("adaptive_design", "draft")},
        "iterative_attack": {"policy": dict(it.DEFAULT_POLICY), "arithmetic": it.design_arithmetic(it.DEFAULT_POLICY), "split_file": sd.SPLIT_FILE,
                             "population_file": it.POPULATION_FILE, "selected_file": it.SELECTED_FILE, "eval_manifest_file": it.EVAL_TRIALS_FILE,
                             "default_attacker_profile_id": "attacker_deepseek_flash_high",
                             "status": (status or {}).get("iterative_policy", "draft")},
        "capability_sweep": {"design": dict(cap.DESIGN), "arithmetic_without_attacks": cap.design_arithmetic(cap.DESIGN, False),
                             "arithmetic_with_attacks": cap.design_arithmetic(cap.DESIGN, True), "manifest_file": cap.TRIALS_FILE, "meta_file": cap.META_FILE,
                             "status": (status or {}).get("capability_design", "draft")},
    }


def load_config(path=CONFIG_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path=CONFIG_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def main():
    save_config(build_config())
    print(f"Wrote {CONFIG_FILE}")


if __name__ == "__main__":
    main()
