"""v2 immutability guard for the v3 experiment.

data/controllability_v3_v2_snapshot.json records the sha256 of every
git-tracked v2 design/data/code file (and the shared story corpus) as it
stood when v3 was set up. verify_v2_unchanged() re-hashes those files and
also re-runs v2's own frozen-lock verification (freeze_controllability_v2.
verify_frozen, read-only). Any difference fails the v3 preflight -- v3 must
never modify, regenerate, or otherwise contaminate anything v2 owns.

Only ever READS the listed files. Writing the snapshot itself is an
explicit, one-off action (main()).
"""

import hashlib
import json
import os

SNAPSHOT_FILE = "data/controllability_v3_v2_snapshot.json"

# Every v2-owned path, plus the shared corpus inputs. Code files are
# included too: v3 must not edit v2's generator/analysis either.
V2_PROTECTED_FILES = (
    "data/controllability_v2_baseline_text_only_trials.jsonl",
    "data/controllability_v2_baseline_trials.jsonl",
    "data/controllability_v2_contrasts.jsonl",
    "data/controllability_v2_corpus.json",
    "data/controllability_v2_frozen_lock.json",
    "data/controllability_v2_minimal_trials.jsonl",
    "data/controllability_v2_study_config.json",
    "data/controllability_v2_trials.jsonl",
    "data/items.jsonl",
    "data/stories/afterlife.txt",
    "data/stories/buddy.txt",
    "data/stories/charlotte_train.txt",
    "data/stories/dana_brownies.txt",
    "data/stories/dunnest_smoke.txt",
    "data/stories/eyecut_lowway.txt",
    "data/stories/gilbert.txt",
    "data/stories/morrow_transport.txt",
    "data/stories/prophet.txt",
    "data/stories/qual_panic.txt",
    "data/stories/santa.txt",
    "data/stories/saturn.txt",
    "analyze_controllability_v2.py",
    "controllability_v2_corpus.py",
    "controllability_v2_cost.py",
    "controllability_v2_deepseek_pricing.py",
    "controllability_v2_execution.py",
    "controllability_v2_plots.py",
    "controllability_v2_recovery.py",
    "controllability_v2_stats.py",
    "controllability_v2_study_config.py",
    "controllability_v2_trials.py",
    "freeze_controllability_v2.py",
    "plan_controllability_v2.py",
    "run_controllability_v2.py",
    ".github/workflows/deepseek-v2.yml",
    "STUDY_PROTOCOL_V2.md",
)

# v2 result directories v3 must never write into.
V2_RESULT_DIRS = ("results/controllability_v2",)


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_snapshot(files=V2_PROTECTED_FILES, root="."):
    return {path: sha256_of_file(os.path.join(root, path)) for path in files}


def write_snapshot(path=SNAPSHOT_FILE, root="."):
    snapshot = {"description": "sha256 of every v2-owned file at v3 setup time; v3 preflight fails if any differs",
                "files": compute_snapshot(root=root)}
    with open(os.path.join(root, path), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")
    return snapshot


def load_snapshot(path=SNAPSHOT_FILE, root="."):
    with open(os.path.join(root, path), "r", encoding="utf-8") as f:
        return json.load(f)


def verify_v2_unchanged(snapshot_path=SNAPSHOT_FILE, root=".", check_v2_lock=True):
    """Returns (ok, reason). ok only if every snapshotted file still hashes
    identically AND (when check_v2_lock) v2's own frozen lock still
    verifies."""
    try:
        snapshot = load_snapshot(snapshot_path, root)
    except FileNotFoundError:
        return False, f"v2 snapshot not found at {snapshot_path}"
    changed, missing = [], []
    for path, expected in snapshot["files"].items():
        full = os.path.join(root, path)
        if not os.path.exists(full):
            missing.append(path)
        elif sha256_of_file(full) != expected:
            changed.append(path)
    if changed or missing:
        return False, f"v2 files modified: {changed}; missing: {missing}"
    if check_v2_lock:
        from freeze_controllability_v2 import verify_frozen as verify_v2_frozen
        ok, reason = verify_v2_frozen()
        if not ok:
            return False, f"v2 frozen lock no longer verifies: {reason}"
    return True, None


def main():
    snapshot = write_snapshot()
    print(f"Snapshotted {len(snapshot['files'])} v2-owned file(s) into {SNAPSHOT_FILE}")


if __name__ == "__main__":
    main()
