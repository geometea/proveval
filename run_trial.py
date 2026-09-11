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

RATING_FIELDS = [
    "plot_structure",
    "prose_style",
    "characterization",
    "originality",
    "overall_quality",
]


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
        max_tokens=4096,  # required by the API; not a sampling parameter
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        block_types = [block.type for block in response.content]
        raise ValueError(f"No text blocks in response. Block types were: {block_types}")

    return {
        "response_text": "\n".join(text_blocks),
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def is_valid_ratings(ratings):
    """Check that ratings has exactly the 5 rating fields, each an int from 1 to 5."""
    if not isinstance(ratings, dict) or set(ratings.keys()) != set(RATING_FIELDS):
        return False
    for field in RATING_FIELDS:
        value = ratings[field]
        if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 5):
            return False
    return True


def validate_single_response(parsed):
    """Return an error string, or None if a single-story response is valid."""
    if not is_valid_ratings(parsed):
        return "Response must contain exactly the 5 rating fields, each an integer 1-5"
    return None


def validate_comparison_response(parsed):
    """Return an error string, or None if a comparison response is valid."""
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"story_a", "story_b", "preference"}:
        return "Response must contain exactly story_a, story_b, and preference"
    if not is_valid_ratings(parsed["story_a"]):
        return "story_a must contain exactly the 5 rating fields, each an integer 1-5"
    if not is_valid_ratings(parsed["story_b"]):
        return "story_b must contain exactly the 5 rating fields, each an integer 1-5"
    preference = parsed["preference"]
    if not isinstance(preference, int) or isinstance(preference, bool) or not (-2 <= preference <= 2):
        return "preference must be an integer from -2 to 2"
    return None


def strip_code_fence(text):
    """Remove a surrounding Markdown code fence (```json ... ``` or ``` ... ```), if present."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    lines = lines[1:]  # drop the opening ``` or ```json line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_and_validate(response_text, trial_type):
    """Try to parse response_text as JSON and check it matches the expected shape.

    Returns (parsed_response, validation_error). On any failure, parsed_response
    is None and validation_error explains why. response_text is unwrapped from a
    Markdown code fence first, if the model added one.
    """
    try:
        parsed = json.loads(strip_code_fence(response_text))
    except json.JSONDecodeError:
        return None, "Response is not valid JSON"

    if trial_type == "single":
        error = validate_single_response(parsed)
    else:
        error = validate_comparison_response(parsed)

    if error:
        return None, error
    return parsed, None


def save_result(result):
    """Append one JSON result to results/raw.jsonl, creating the folder if needed."""
    os.makedirs("results", exist_ok=True)
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

    api_result = call_claude(trial["prompt"], model)
    response_text = api_result["response_text"]
    parsed_response, validation_error = parse_and_validate(response_text, trial["type"])

    if validation_error:
        print(f"Warning: invalid response for {trial['trial_id']}: {validation_error}")

    save_result(
        {
            "trial_id": trial["trial_id"],
            "model": model,
            "stop_reason": api_result["stop_reason"],
            "input_tokens": api_result["input_tokens"],
            "output_tokens": api_result["output_tokens"],
            "response_text": response_text,
            "parsed_response": parsed_response,
            "validation_error": validation_error,
        }
    )
    print(f"Saved result for {trial['trial_id']} to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
