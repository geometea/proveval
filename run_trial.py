"""Run exactly one trial from data/trials.jsonl through the Claude API.

Usage:
    python3 run_trial.py <trial_id>            # calls the API and saves the result
    python3 run_trial.py <trial_id> --dry-run  # just prints what would be sent
"""

import argparse
import json
import os

TRIALS_FILE = "data/trials.jsonl"
RESULTS_FILE = "results/raw.jsonl"
DEFAULT_MODEL = "claude-sonnet-5"


def load_trials(path):
    """Read all trials from the .jsonl file into a list of dicts."""
    trials = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def find_trial(trials, trial_id):
    """Return the trial dict matching trial_id, or None if not found."""
    for trial in trials:
        if trial["trial_id"] == trial_id:
            return trial
    return None


def get_api_key():
    """Read ANTHROPIC_API_KEY from the environment, stripped of whitespace."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return api_key


def call_claude(prompt, model):
    """Send the prompt as a single user message and return the reply text."""
    import anthropic  # imported here so --dry-run works without the package installed

    client = anthropic.Anthropic(api_key=get_api_key())
    response = client.messages.create(
        model=model,
        max_tokens=1024,  # required by the API; not a sampling parameter
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        block_types = [block.type for block in response.content]
        raise ValueError(f"No text blocks in response. Block types were: {block_types}")

    return "\n".join(text_blocks)


def save_result(trial_id, model, response_text):
    """Append one JSON result to results/raw.jsonl, creating the folder if needed."""
    os.makedirs("results", exist_ok=True)
    result = {
        "trial_id": trial_id,
        "model": model,
        "response_text": response_text,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run one trial through the Claude API.")
    parser.add_argument("trial_id", help="The trial_id to run, e.g. single__gilbert__ai__critique")
    parser.add_argument("--dry-run", action="store_true", help="Print the trial without calling the API")
    args = parser.parse_args()

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)

    trials = load_trials(TRIALS_FILE)
    trial = find_trial(trials, args.trial_id)
    if trial is None:
        print(f"No trial found with trial_id: {args.trial_id}")
        return

    if args.dry_run:
        print(f"Model: {model}")
        print("Trial:")
        print(json.dumps(trial, indent=2))
        return

    response_text = call_claude(trial["prompt"], model)
    save_result(trial["trial_id"], model, response_text)
    print(f"Saved result for {trial['trial_id']} to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
