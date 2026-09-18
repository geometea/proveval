"""Freeze the v3 study design and write its lock file
(data/controllability_v3_frozen_lock.json). Entirely separate from v2's
freeze/lock, which it never touches.

Deliberate, explicit action (never run by generation/analysis). Hashes:
the (now-frozen) study config (canonical JSON), the contrasts file, the
corpus metadata, the three frozen manifests (context, no-context, holdout),
and the v2 snapshot file. verify_frozen() recomputes all of them and
refuses on any drift. The pilot manifest is deliberately NOT locked (it is
a regenerable, non-primary subset) -- its hash is recorded in pilot result
metadata instead.

Design rule: once frozen, this design must not change. A wording change
after the pilot means regenerating the manifests, re-freezing, and
documenting the change (STUDY_PROTOCOL_V3.md) BEFORE production.
"""

import hashlib
import json
from datetime import datetime, timezone

from controllability_v3_corpus import CORPUS_FILE
from controllability_v3_study_config import STUDY_CONFIG_FILE, load_study_config, save_study_config
from controllability_v3_v2_guard import SNAPSHOT_FILE

LOCK_FILE = "data/controllability_v3_frozen_lock.json"


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_of_json_obj(obj):
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_lock_hashes(study_config_path=STUDY_CONFIG_FILE, corpus_path=CORPUS_FILE, snapshot_path=SNAPSHOT_FILE):
    config = load_study_config(study_config_path)
    return {
        "study_config_sha256": sha256_of_json_obj(config),
        "contrasts_sha256": sha256_of_file(config["contrast_file"]),
        "corpus_sha256": sha256_of_file(corpus_path),
        "context_manifest_sha256": sha256_of_file(config["context_manifest_file"]),
        "nocontext_manifest_sha256": sha256_of_file(config["nocontext_manifest_file"]),
        "holdout_manifest_sha256": sha256_of_file(config["holdout_manifest_file"]),
        "v2_snapshot_sha256": sha256_of_file(snapshot_path),
    }


def freeze(study_config_path=STUDY_CONFIG_FILE, corpus_path=CORPUS_FILE, lock_path=LOCK_FILE, snapshot_path=SNAPSHOT_FILE):
    config = load_study_config(study_config_path)
    config["status"] = "frozen"
    save_study_config(config, study_config_path)
    lock = {
        "design_version": config["design_version"],
        "experiment_id": config["experiment_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "hashes": compute_lock_hashes(study_config_path, corpus_path, snapshot_path),
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")
    return lock


def load_lock(lock_path=LOCK_FILE):
    with open(lock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_frozen(study_config_path=STUDY_CONFIG_FILE, corpus_path=CORPUS_FILE, lock_path=LOCK_FILE, snapshot_path=SNAPSHOT_FILE):
    """(ok, reason): ok only if the config is frozen, the lock exists, and
    every recomputed hash matches it."""
    try:
        config = load_study_config(study_config_path)
    except FileNotFoundError:
        return False, f"Study config not found: {study_config_path}"
    if config.get("status") != "frozen":
        return False, f"Study config status is {config.get('status')!r}, not 'frozen' -- run freeze_controllability_v3.py first"
    try:
        lock = load_lock(lock_path)
    except FileNotFoundError:
        return False, f"No v3 lock file at {lock_path} -- run freeze_controllability_v3.py first"
    if lock.get("experiment_id") != config.get("experiment_id"):
        return False, "Lock file belongs to a different experiment_id"
    try:
        current = compute_lock_hashes(study_config_path, corpus_path, snapshot_path)
    except FileNotFoundError as e:
        return False, f"A locked file is missing: {e}"
    mismatches = sorted(k for k in current if current[k] != lock.get("hashes", {}).get(k))
    if mismatches:
        return False, f"Frozen v3 files no longer match the lock: {mismatches}"
    return True, None


def main():
    lock = freeze()
    print("v3 study design frozen.")
    print(f"Experiment id: {lock['experiment_id']}  Frozen at: {lock['frozen_at']}")
    for key, value in lock["hashes"].items():
        print(f"  {key}: {value}")
    print(f"Lock file: {LOCK_FILE}")


if __name__ == "__main__":
    main()
