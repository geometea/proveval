"""Staged freeze for the experimental families
(data/controllability_v3_experimental_frozen_lock.json). Never touches the
primary, stress, or v2 locks; binds to them by hash.

  adaptive_design    adaptive policy section, allowed grid, stopping rule,
                     budget (NOT future observations)
  iterative_policy   split, search policy, population settings, feedback
                     list, validator (attacker system prompt hash), final
                     selection rule
  iterative_attacks  population file, selected attacks, held-out manifest
  capability_design  capability manifest + meta
"""

import hashlib
import json
from datetime import datetime, timezone

from controllability_v3_experimental_config import CONFIG_FILE, load_config, save_config
from controllability_v3_v2_guard import SNAPSHOT_FILE
from freeze_controllability_v3 import LOCK_FILE as PRIMARY_LOCK_FILE
from freeze_controllability_v3_stress import STRESS_LOCK_FILE

LOCK_FILE = "data/controllability_v3_experimental_frozen_lock.json"
STAGES = ("adaptive_design", "iterative_policy", "iterative_attacks", "capability_design")
_STATUS_KEY = {"adaptive_design": "adaptive_dose", "iterative_policy": "iterative_attack", "capability_design": "capability_sweep"}


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compute_stage_hashes(stage, config_path=CONFIG_FILE):
    config = load_config(config_path)
    common = {"primary_lock_sha256": sha_file(PRIMARY_LOCK_FILE), "stress_lock_sha256": sha_file(STRESS_LOCK_FILE), "v2_snapshot_sha256": sha_file(SNAPSHOT_FILE)}
    if stage == "adaptive_design":
        return {**common, "adaptive_policy_sha256": sha_obj(config["adaptive_dose"]["policy"])}
    if stage == "iterative_policy":
        it = config["iterative_attack"]
        return {**common, "iterative_policy_sha256": sha_obj(it["policy"]), "split_sha256": sha_file(it["split_file"])}
    if stage == "iterative_attacks":
        it = config["iterative_attack"]
        return {"population_sha256": sha_file(it["population_file"]), "selected_sha256": sha_file(it["selected_file"]), "eval_manifest_sha256": sha_file(it["eval_manifest_file"])}
    if stage == "capability_design":
        c = config["capability_sweep"]
        return {**common, "capability_design_sha256": sha_obj(c["design"]), "capability_manifest_sha256": sha_file(c["manifest_file"]), "capability_meta_sha256": sha_file(c["meta_file"])}
    raise ValueError(stage)


def load_lock(lock_path=LOCK_FILE):
    with open(lock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def freeze(stage, config_path=CONFIG_FILE, lock_path=LOCK_FILE):
    if stage not in STAGES:
        raise ValueError(stage)
    config = load_config(config_path)
    if stage == "iterative_attacks":
        ok, reason = verify_frozen("iterative_policy", config_path, lock_path)
        if not ok:
            raise ValueError(f"cannot freeze attacks before the iterative policy verifies: {reason}")
    if stage in _STATUS_KEY:
        config[_STATUS_KEY[stage]]["status"] = "frozen"
        save_config(config, config_path)
    try:
        lock = load_lock(lock_path)
    except FileNotFoundError:
        lock = {"experiment_id": config["experiment_id"], "stages": {}}
    lock["stages"][stage] = {"frozen_at": datetime.now(timezone.utc).isoformat(), "hashes": compute_stage_hashes(stage, config_path)}
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")
    return lock


def verify_frozen(stage, config_path=CONFIG_FILE, lock_path=LOCK_FILE):
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return False, f"experimental config not found: {config_path}"
    if stage in _STATUS_KEY and config[_STATUS_KEY[stage]].get("status") != "frozen":
        return False, f"{stage}: config status is {config[_STATUS_KEY[stage]].get('status')!r}, not 'frozen'"
    try:
        lock = load_lock(lock_path)
    except FileNotFoundError:
        return False, f"no experimental lock at {lock_path}"
    if stage not in lock.get("stages", {}):
        return False, f"stage {stage!r} has not been frozen"
    try:
        current = compute_stage_hashes(stage, config_path)
    except FileNotFoundError as e:
        return False, f"a locked file is missing: {e}"
    expected = lock["stages"][stage]["hashes"]
    mismatches = sorted(k for k in current if current[k] != expected.get(k))
    if mismatches:
        return False, f"frozen files no longer match the lock ({stage}): {mismatches}"
    return True, None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=STAGES)
    a = p.parse_args()
    lock = freeze(a.stage)
    print(f"Experimental stage {a.stage!r} frozen at {lock['stages'][a.stage]['frozen_at']}")
    for k, v in lock["stages"][a.stage]["hashes"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
