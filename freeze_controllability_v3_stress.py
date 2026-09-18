"""Staged freeze/lock for the v3 stress families
(data/controllability_v3_stress_frozen_lock.json). Never touches the
primary v3 lock, which it references by hash so a stress freeze is bound
to one specific primary design.

  stage "design"  : stress config, adversarial split, dose manifest,
                    primary lock file, v2 snapshot.
  stage "attacks" : candidates file, development manifest, selected
                    attacks, held-out evaluation manifest.

verify_frozen("design") is required before any dose run or attack search;
verify_frozen("attacks") before the held-out attack evaluation.
"""

import hashlib
import json
from datetime import datetime, timezone

import controllability_v3_stress_design as sd
from controllability_v3_stress_config import STRESS_CONFIG_FILE, load_stress_config, save_stress_config
from controllability_v3_v2_guard import SNAPSHOT_FILE
from freeze_controllability_v3 import LOCK_FILE as PRIMARY_LOCK_FILE

STRESS_LOCK_FILE = "data/controllability_v3_stress_frozen_lock.json"
STAGES = ("design", "attacks")


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_of_json_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compute_stage_hashes(stage, config_path=STRESS_CONFIG_FILE, primary_lock_path=PRIMARY_LOCK_FILE, snapshot_path=SNAPSHOT_FILE):
    config = load_stress_config(config_path)
    if stage == "design":
        return {
            "stress_config_sha256": sha256_of_json_obj(config),
            "split_sha256": sha256_of_file(config["adversarial"]["split_file"]),
            "dose_manifest_sha256": sha256_of_file(config["dose_response"]["manifest_file"]),
            "primary_lock_sha256": sha256_of_file(primary_lock_path),
            "v2_snapshot_sha256": sha256_of_file(snapshot_path),
        }
    if stage == "attacks":
        a = config["adversarial"]
        return {
            "candidates_sha256": sha256_of_file(a["candidates_file"]),
            "dev_manifest_sha256": sha256_of_file(a["dev_manifest_file"]),
            "selected_attacks_sha256": sha256_of_file(a["selected_attacks_file"]),
            "eval_manifest_sha256": sha256_of_file(a["eval_manifest_file"]),
        }
    raise ValueError(f"unknown stage {stage!r}")


def load_lock(lock_path=STRESS_LOCK_FILE):
    with open(lock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def freeze(stage, config_path=STRESS_CONFIG_FILE, lock_path=STRESS_LOCK_FILE, primary_lock_path=PRIMARY_LOCK_FILE, snapshot_path=SNAPSHOT_FILE):
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    config = load_stress_config(config_path)
    if stage == "design":
        config["status"] = "frozen"
        save_stress_config(config, config_path)
    try:
        lock = load_lock(lock_path)
    except FileNotFoundError:
        lock = {"experiment_id": config["experiment_id"], "stages": {}}
    if stage == "attacks":
        ok, reason = verify_frozen("design", config_path, lock_path, primary_lock_path, snapshot_path)
        if not ok:
            raise ValueError(f"cannot freeze attacks before the design stage verifies: {reason}")
    lock["stages"][stage] = {"frozen_at": datetime.now(timezone.utc).isoformat(),
                             "hashes": compute_stage_hashes(stage, config_path, primary_lock_path, snapshot_path)}
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")
    return lock


def verify_frozen(stage, config_path=STRESS_CONFIG_FILE, lock_path=STRESS_LOCK_FILE, primary_lock_path=PRIMARY_LOCK_FILE, snapshot_path=SNAPSHOT_FILE):
    try:
        config = load_stress_config(config_path)
    except FileNotFoundError:
        return False, f"stress config not found: {config_path}"
    if config.get("status") != "frozen":
        return False, f"stress config status is {config.get('status')!r}, not 'frozen' -- run freeze_controllability_v3_stress.py design"
    try:
        lock = load_lock(lock_path)
    except FileNotFoundError:
        return False, f"no stress lock at {lock_path}"
    if stage not in lock.get("stages", {}):
        return False, f"stress stage {stage!r} has not been frozen"
    try:
        current = compute_stage_hashes(stage, config_path, primary_lock_path, snapshot_path)
    except FileNotFoundError as e:
        return False, f"a locked file is missing: {e}"
    expected = lock["stages"][stage]["hashes"]
    mismatches = sorted(k for k in current if current[k] != expected.get(k))
    if mismatches:
        return False, f"frozen stress files no longer match the lock ({stage}): {mismatches}"
    return True, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Freeze a v3 stress stage.")
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    lock = freeze(args.stage)
    print(f"Stress stage {args.stage!r} frozen at {lock['stages'][args.stage]['frozen_at']}")
    for k, v in lock["stages"][args.stage]["hashes"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
