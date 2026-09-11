"""Re-run parse_and_validate() over existing saved results, without calling the API.

Useful after fixing a bug in parse_and_validate() (e.g. handling Markdown
code fences) so old results/raw.jsonl entries get re-checked with the
current logic instead of staying stuck with their old, possibly wrong,
parsed_response/validation_error.
"""

import json
import shutil

from run_trial import parse_and_validate, load_trials, TRIALS_FILE

RESULTS_FILE = "results/raw.jsonl"
BACKUP_FILE = "results/raw.jsonl.bak"


def load_results(path):
    """Read all results from the .jsonl file into a list of dicts."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def reparse(result, trial_types):
    """Return a copy of result with parsed_response and validation_error updated."""
    trial_type = trial_types[result["trial_id"]]
    parsed_response, validation_error = parse_and_validate(result["response_text"], trial_type)
    updated = dict(result)
    updated["parsed_response"] = parsed_response
    updated["validation_error"] = validation_error
    return updated


def main():
    results = load_results(RESULTS_FILE)
    trial_types = {trial["trial_id"]: trial["type"] for trial in load_trials(TRIALS_FILE)}

    shutil.copyfile(RESULTS_FILE, BACKUP_FILE)
    print(f"Backed up {RESULTS_FILE} to {BACKUP_FILE}")

    updated_results = [reparse(result, trial_types) for result in results]

    with open(RESULTS_FILE, "w") as f:
        for result in updated_results:
            f.write(json.dumps(result) + "\n")

    num_valid = sum(1 for r in updated_results if r["validation_error"] is None)
    print(f"Reparsed {len(updated_results)} results.")
    print(f"Now valid: {num_valid} / {len(updated_results)}")


if __name__ == "__main__":
    main()
