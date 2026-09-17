"""Re-run parse_and_validate() over existing saved results, without calling the API.

Useful after fixing a bug in parse_and_validate() (e.g. handling Markdown
code fences, or the plain_ab response format) so old results entries get
re-checked with the current logic instead of staying stuck with their old,
possibly wrong, parsed_response/validation_error.

Defaults to the v0.1 pilot's data/trials.jsonl + results/raw.jsonl, exactly
as before; pass --trials-file/--results-file to reparse any other
experiment's results (e.g. the standalone context-controllability
experiment's data/controllability_trials.jsonl + results/controllability/
raw.jsonl) without another API call.
"""

import argparse
import json
import shutil

from context_analysis_io import get_trial_meta
from run_trial import parse_and_validate, load_trials, TRIALS_FILE

RESULTS_FILE = "results/raw.jsonl"


def load_results(path):
    """Read all results from the .jsonl file into a list of dicts."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def reparse(result, trials_by_id):
    """Return a copy of result with parsed_response and validation_error
    updated, or an unchanged copy if there's nothing to reparse.

    Metadata (type, choice_mode, rubric, response_format) is read the same
    way collapse_attempts does (context_analysis_io.get_trial_meta): prefer
    the result row's own embedded trial_meta, else join the trials manifest
    by trial_id. This means a result already carries everything the CURRENT
    validator needs (including response_format="plain_ab" for the
    standalone context-controllability experiment) even if the trials file
    passed in doesn't match, and reparsing never needs another API call.

    Two cases pass through unchanged rather than crashing the whole run:
    - response_text is None (the API call itself failed -- there is no
      text to re-parse; parse_and_validate would otherwise raise
      AttributeError trying to strip() a None).
    - no trial metadata is available at all (neither an embedded
      trial_meta nor a match in the current trials manifest, e.g. after
      regenerating a trials file with a different trial set) -- an
      unhandled KeyError here would otherwise abort before any other row
      got reparsed.
    """
    if result.get("response_text") is None:
        return dict(result), "no_response_text"
    meta = get_trial_meta(result, trials_by_id)
    if meta is None:
        return dict(result), "unknown_trial_id"
    parsed_response, validation_error = parse_and_validate(
        result["response_text"], meta["type"], meta.get("choice_mode"), meta.get("rubric", "full"), meta.get("response_format")
    )
    updated = dict(result)
    updated["parsed_response"] = parsed_response
    updated["validation_error"] = validation_error
    return updated, "reparsed"


def main():
    parser = argparse.ArgumentParser(description="Re-run parse_and_validate() over existing saved results, offline.")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Trials manifest to join by trial_id (default: {TRIALS_FILE})")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Results file to reparse in place (default: {RESULTS_FILE})")
    args = parser.parse_args()

    results = load_results(args.results_file)
    trials_by_id = {trial["trial_id"]: trial for trial in load_trials(args.trials_file)}

    backup_file = args.results_file + ".bak"
    shutil.copyfile(args.results_file, backup_file)
    print(f"Backed up {args.results_file} to {backup_file}")

    updated_results = []
    skip_counts = {"no_response_text": 0, "unknown_trial_id": 0}
    for result in results:
        updated, status = reparse(result, trials_by_id)
        updated_results.append(updated)
        if status != "reparsed":
            skip_counts[status] += 1

    with open(args.results_file, "w", encoding="utf-8") as f:
        for result in updated_results:
            f.write(json.dumps(result) + "\n")

    num_valid = sum(1 for r in updated_results if r["validation_error"] is None)
    print(f"Reparsed {len(updated_results)} results.")
    print(f"Now valid: {num_valid} / {len(updated_results)}")
    if skip_counts["no_response_text"]:
        print(f"Skipped {skip_counts['no_response_text']} row(s) with no response_text (API call failures) -- left unchanged.")
    if skip_counts["unknown_trial_id"]:
        print(f"Skipped {skip_counts['unknown_trial_id']} row(s) with no trial metadata available (no embedded "
              f"trial_meta and no match in {args.trials_file}) -- left unchanged.")


if __name__ == "__main__":
    main()
