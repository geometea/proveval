"""Freeze the v2 study design and write the frozen-design lock file.

Freezing is a DELIBERATE, EXPLICIT action -- it is never run automatically
by manifest generation, analysis, or planning. It:

  1. Loads the current study config, contrast file, corpus metadata, and the
     generated treatment/baseline manifests.
  2. Sets the study config's status to "frozen" and re-saves it.
  3. Hashes all five inputs (the now-frozen study config, the contrasts
     file, the corpus metadata file, the baseline manifest, the treatment
     manifest) and writes those hashes into the lock file
     (data/controllability_v2_frozen_lock.json).

Once frozen, `verify_frozen()` recomputes the same five hashes from the
files currently on disk and compares them against the lock -- any drift
(edited contrast wording, a changed story file, a regenerated manifest,
an edited study config) is reported as a mismatch. A production run
(run_controllability_v2.py without --dry-run) calls verify_frozen() first
and refuses to execute on any mismatch, or if the design was never frozen
at all.

Design rule (see STUDY_PROTOCOL_V2.md): once frozen, this design's config/
contrasts/manifests must not change. A change in scope (e.g. more
confirmatory replicates after inspecting results) must go into a NEW
study config / experiment_id / wave, never edited into this one -- a
lock-file mismatch makes any such silent edit impossible to run against
undetected.
"""

import hashlib
import json
from datetime import datetime, timezone

from controllability_v2_corpus import CORPUS_FILE
from controllability_v2_study_config import STUDY_CONFIG_FILE, load_study_config, save_study_config

LOCK_FILE = "data/controllability_v2_frozen_lock.json"


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_of_json_obj(obj):
    """Canonical (sorted-key, compact) JSON hash -- independent of
    formatting/key order, so re-saving a semantically-identical file never
    produces a spurious mismatch."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_lock_hashes(study_config_path=STUDY_CONFIG_FILE, corpus_path=CORPUS_FILE):
    """Hash the 5 required inputs from whatever is currently on disk. Used
    both by freeze() (to write the lock) and verify_frozen() (to check
    against it) -- the exact same hashing logic, so freezing and verifying
    can never drift apart."""
    config = load_study_config(study_config_path)
    return {
        "study_config_sha256": sha256_of_json_obj(config),
        "contrasts_sha256": sha256_of_file(config["contrast_file"]),
        "corpus_sha256": sha256_of_file(corpus_path),
        "baseline_manifest_sha256": sha256_of_file(config["baseline_manifest_file"]),
        "treatment_manifest_sha256": sha256_of_file(config["treatment_manifest_file"]),
    }


def freeze(study_config_path=STUDY_CONFIG_FILE, corpus_path=CORPUS_FILE, lock_path=LOCK_FILE):
    """Flip the study config to status="frozen", save it, then hash it (and
    the contrasts/corpus/manifest files it points to) into the lock file.
    Returns the lock dict that was written."""
    config = load_study_config(study_config_path)
    config["status"] = "frozen"
    save_study_config(config, study_config_path)

    hashes = compute_lock_hashes(study_config_path, corpus_path)
    lock = {
        "design_version": config["design_version"],
        "experiment_id": config["experiment_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "hashes": hashes,
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")
    return lock


def load_lock(lock_path=LOCK_FILE):
    with open(lock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_frozen(study_config_path=STUDY_CONFIG_FILE, corpus_path=CORPUS_FILE, lock_path=LOCK_FILE):
    """Returns (ok, reason). ok is True only if: a lock file exists, the
    study config's status is "frozen", and every one of the 5 recomputed
    hashes exactly matches the lock. `reason` is a human-readable
    explanation whenever ok is False (used to refuse a production run)."""
    try:
        config = load_study_config(study_config_path)
    except FileNotFoundError:
        return False, f"Study config not found: {study_config_path}"

    if config.get("status") != "frozen":
        return False, f"Study config status is {config.get('status')!r}, not 'frozen' -- run freeze_controllability_v2.py first"

    try:
        lock = load_lock(lock_path)
    except FileNotFoundError:
        return False, f"No frozen-design lock file found at {lock_path} -- run freeze_controllability_v2.py first"

    current_hashes = compute_lock_hashes(study_config_path, corpus_path)
    mismatches = [key for key in current_hashes if current_hashes[key] != lock.get("hashes", {}).get(key)]
    if mismatches:
        return False, f"Frozen files no longer match the lock: {sorted(mismatches)}"

    return True, None


def main():
    lock = freeze()
    print("Study design frozen.")
    print(f"Experiment id: {lock['experiment_id']}")
    print(f"Frozen at: {lock['frozen_at']}")
    for key, value in lock["hashes"].items():
        print(f"  {key}: {value}")
    print(f"Lock file: {LOCK_FILE}")


if __name__ == "__main__":
    main()
