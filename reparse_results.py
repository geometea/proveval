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


def reparse(result, trials_by_id):
    """Return a copy of result with parsed_response and validation_error
    updated, or an unchanged copy if there's nothing to reparse.

    Two cases pass through unchanged rather than crashing the whole run:
    - response_text is None (the API call itself failed -- there is no
      text to re-parse; parse_and_validate would otherwise raise
      AttributeError trying to strip() a None).
    - trial_id has no match in the current trials manifest (e.g. after
      regenerating data/trials.jsonl or data/context_trials.jsonl with a
      different trial set) -- an unhandled KeyError here would otherwise
      abort before any other row got reparsed.
    """
    if result.get("response_text") is None:
        return dict(result), "no_response_text"
    trial = trials_by_id.get(result["trial_id"])
    if trial is None:
        return dict(result), "unknown_trial_id"
    parsed_response, validation_error = parse_and_validate(
        result["response_text"], trial["type"], trial.get("choice_mode")
    )
    updated = dict(result)
    updated["parsed_response"] = parsed_response
    updated["validation_error"] = validation_error
    return updated, "reparsed"


def main():
    results = load_results(RESULTS_FILE)
    trials_by_id = {trial["trial_id"]: trial for trial in load_trials(TRIALS_FILE)}

    shutil.copyfile(RESULTS_FILE, BACKUP_FILE)
    print(f"Backed up {RESULTS_FILE} to {BACKUP_FILE}")

    updated_results = []
    skip_counts = {"no_response_text": 0, "unknown_trial_id": 0}
    for result in results:
        updated, status = reparse(result, trials_by_id)
        updated_results.append(updated)
        if status != "reparsed":
            skip_counts[status] += 1

    with open(RESULTS_FILE, "w") as f:
        for result in updated_results:
            f.write(json.dumps(result) + "\n")

    num_valid = sum(1 for r in updated_results if r["validation_error"] is None)
    print(f"Reparsed {len(updated_results)} results.")
    print(f"Now valid: {num_valid} / {len(updated_results)}")
    if skip_counts["no_response_text"]:
        print(f"Skipped {skip_counts['no_response_text']} row(s) with no response_text (API call failures) -- left unchanged.")
    if skip_counts["unknown_trial_id"]:
        print(f"Skipped {skip_counts['unknown_trial_id']} row(s) whose trial_id is not in {TRIALS_FILE} -- left unchanged.")


if __name__ == "__main__":
    main()
