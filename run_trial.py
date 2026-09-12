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


# Two named v0.2 sampling regimes (see resolve_sampling_params). The primary
# regime pins temperature=0 for the lowest-variance inference settings the
# API supports for this request type -- NOT an assumption that this is
# literally deterministic (repeated replicates still matter; see
# run_batch.py's --replicates* flags), just the cleanest setting for
# measuring a causal/context-sensitivity effect. The secondary regime is
# provider defaults, for a later robustness check under more naturalistic
# usage -- supported here, but never run by anything in this repo.
SAMPLING_REGIMES = {
    "low_variance_primary": {"temperature": 0},
    "provider_default_secondary": {},
}


def resolve_sampling_params(trial_type, regime_name):
    """Only v0.2 context trial types carry an explicit sampling regime. v0.1
    trial types ("single", "comparison", "comparison_control") return None,
    meaning: no extra API parameter is sent and no sampling metadata is
    added to the saved row, regardless of --sampling-regime -- so the
    completed v0.1 pilot stays reproducible exactly as before, independent
    of this flag's default.
    """
    if trial_type not in CONTEXT_TRIAL_TYPES:
        return None
    return SAMPLING_REGIMES[regime_name]


def call_claude(prompt, model, sampling_params=None):
    """Send the prompt as a single user message and return the reply text.

    sampling_params (e.g. {"temperature": 0}) are passed through to the API
    verbatim when truthy; None or {} (both mean "no override") send nothing
    extra, so the v0.1 pilot's original call shape is unchanged when this is
    omitted.
    """
    import anthropic  # imported here so --dry-run works without the package installed

    client = anthropic.Anthropic(api_key=get_api_key())
    create_kwargs = dict(
        model=model,
        max_tokens=4096,  # required by the API; not a sampling parameter
        messages=[{"role": "user", "content": prompt}],
    )
    if sampling_params:
        create_kwargs.update(sampling_params)
    response = client.messages.create(**create_kwargs)

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
    """Return an error string, or None if a single-story response is valid.

    This is the v0.1 pilot's original integer 1-5 schema, unchanged and used
    only for trial_type "single" -- see validate_context_single_response for
    v0.2's separate 1.0-10.0 decimal schema.
    """
    if not is_valid_ratings(parsed):
        return "Response must contain exactly the 5 rating fields, each an integer 1-5"
    return None


# v0.2 single-text rating scale: 1.0-10.0, at most one decimal place (see
# data/context_tasks.jsonl). A distinct schema from the v0.1 pilot's integer
# 1-5 scale above -- kept as a separate function/task file rather than
# changed in place, so re-running the completed v0.1 pilot still reproduces
# its original integer-scale results exactly.
def has_at_most_one_decimal_place(value):
    scaled = value * 10
    return abs(scaled - round(scaled)) < 1e-6


def is_valid_context_rating(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not (1.0 <= value <= 10.0):
        return False
    return has_at_most_one_decimal_place(value)


def is_valid_context_ratings(ratings):
    """Check that ratings has exactly the 5 rating fields, each 1.0-10.0 with <=1 decimal place."""
    if not isinstance(ratings, dict) or set(ratings.keys()) != set(RATING_FIELDS):
        return False
    return all(is_valid_context_rating(ratings[field]) for field in RATING_FIELDS)


def validate_context_single_response(parsed):
    """Return an error string, or None if a v0.2 context_single response is valid."""
    if not is_valid_context_ratings(parsed):
        return (
            "Response must contain exactly the 5 rating fields, each a number "
            "from 1.0 to 10.0 with at most one decimal place"
        )
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


# context_comparisons.py's prompt (see that file) asks the model to return
# "characterisation" (British spelling) in its example JSON, so a compliant
# response will use that spelling. We accept either spelling on the way in,
# but standardize on "characterization" (matching RATING_FIELDS) for anything
# we actually store, so downstream analysis never has to handle both.
AB_TIE_FIELDS = ["plot_structure", "prose_style", "characterization", "originality", "overall_quality"]
AB_TIE_VALUES = {"A", "B", "tie"}


def normalize_ab_tie_keys(parsed):
    """Return a copy of parsed with "characterisation" renamed to "characterization"."""
    if not isinstance(parsed, dict):
        return parsed
    normalized = dict(parsed)
    if "characterisation" in normalized and "characterization" not in normalized:
        normalized["characterization"] = normalized.pop("characterisation")
    return normalized


def validate_pairwise_context_response(parsed):
    """Check a context_pairwise/context_prompt response: 5 fields, each "A"/"B"/"tie".

    Normalizes "characterisation" -> "characterization" first (see
    normalize_ab_tie_keys), so either spelling validates the same way.
    Returns (normalized_parsed_or_None, error_or_None).
    """
    parsed = normalize_ab_tie_keys(parsed)
    if not isinstance(parsed, dict) or set(parsed.keys()) != set(AB_TIE_FIELDS):
        return None, (
            "Response must contain exactly plot_structure, prose_style, characterization "
            "(or characterisation), originality, and overall_quality"
        )
    for field in AB_TIE_FIELDS:
        if parsed[field] not in AB_TIE_VALUES:
            return None, f"{field} must be exactly \"A\", \"B\", or \"tie\""
    return parsed, None


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

    Dispatch by trial_type, preserving the exact old behavior for the
    original v0.1 types:
      "single"          -> integer 1-5 ratings (unchanged v0.1 schema)
      "context_single"  -> v0.2's own 1.0-10.0 decimal ratings (a distinct
        schema -- see validate_context_single_response)
      "context_pairwise" / "context_prompt" / "context_pairwise_same"
        -> A/B/tie per category (new; parsed_response is normalized to use
        "characterization"). "context_pairwise_same" is the OPTIONAL,
        never-run same-context trial family (context_trials.py); it reuses
        this schema since it uses the same A/B/tie prompt template.
      anything else (e.g. "comparison", "comparison_control", and any future
        or unrecognized type) -> the original story_a/story_b/preference
        schema, exactly as before this function grew a dispatch at all.
    """
    try:
        parsed = json.loads(strip_code_fence(response_text))
    except json.JSONDecodeError:
        return None, "Response is not valid JSON"

    if trial_type == "single":
        error = validate_single_response(parsed)
        return (None, error) if error else (parsed, None)

    if trial_type == "context_single":
        error = validate_context_single_response(parsed)
        return (None, error) if error else (parsed, None)

    if trial_type in ("context_pairwise", "context_prompt", "context_pairwise_same"):
        return validate_pairwise_context_response(parsed)

    error = validate_comparison_response(parsed)
    return (None, error) if error else (parsed, None)


def save_result(result, results_file=RESULTS_FILE):
    """Append one JSON result to results_file, creating its folder if needed."""
    folder = os.path.dirname(results_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def resolve_model(cli_model):
    """--model on the command line wins; otherwise ANTHROPIC_MODEL, otherwise DEFAULT_MODEL."""
    return cli_model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


# Trial types introduced for the v0.2 context benchmark (context_trials.py).
# Their trial dicts already carry structured metadata (story ids, dimension,
# contrast_id, etc.) instead of relying on parsing it out of trial_id, so we
# copy that metadata into the saved result -- excluding the prompt, which
# would just duplicate what's already in the trials file. Old trial types
# ("single", "comparison", "comparison_control") are left exactly as before:
# their saved rows keep the same shape they've always had.
# "context_pairwise_same" is the OPTIONAL same-context trial family; it's
# listed here for forward-compatibility (so it would be handled correctly
# *if* ever run), not because it's part of any required run.
CONTEXT_TRIAL_TYPES = ("context_single", "context_pairwise", "context_prompt", "context_pairwise_same")


def trial_metadata(trial):
    """Trial fields worth copying into a result row, excluding the prompt itself."""
    return {k: v for k, v in trial.items() if k != "prompt"}


def main():
    parser = argparse.ArgumentParser(description="Run one trial through the Claude API.")
    parser.add_argument("trial_id", help="The trial_id to run, e.g. single__gilbert__ai__critique")
    parser.add_argument("--trials-file", default=TRIALS_FILE, help=f"Trials manifest to read (default: {TRIALS_FILE})")
    parser.add_argument("--results-file", default=RESULTS_FILE, help=f"Results file to append to (default: {RESULTS_FILE})")
    parser.add_argument("--model", help="Override the model (default: $ANTHROPIC_MODEL, else claude-sonnet-5)")
    parser.add_argument(
        "--sampling-regime",
        choices=list(SAMPLING_REGIMES),
        default="low_variance_primary",
        help="v0.2 context trial types only (ignored, and not recorded, for v0.1 trial types): "
        "low_variance_primary (default; temperature=0) or provider_default_secondary",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the trial without calling the API")
    args = parser.parse_args()

    model = resolve_model(args.model)

    trials = load_trials(args.trials_file)
    trial = find_trial(trials, args.trial_id)
    if trial is None:
        print(f"No trial found with trial_id: {args.trial_id}")
        return

    sampling_params = resolve_sampling_params(trial["type"], args.sampling_regime)

    if args.dry_run:
        print(f"Model: {model}")
        if sampling_params is not None:
            print(f"Sampling regime: {args.sampling_regime}  (params: {sampling_params})")
        print("Trial:")
        print(json.dumps(trial, indent=2))
        return

    api_result = call_claude(trial["prompt"], model, sampling_params)
    response_text = api_result["response_text"]
    parsed_response, validation_error = parse_and_validate(response_text, trial["type"])

    if validation_error:
        print(f"Warning: invalid response for {trial['trial_id']}: {validation_error}")

    result = {
        "trial_id": trial["trial_id"],
        "model": model,
        "stop_reason": api_result["stop_reason"],
        "input_tokens": api_result["input_tokens"],
        "output_tokens": api_result["output_tokens"],
        "response_text": response_text,
        "parsed_response": parsed_response,
        "validation_error": validation_error,
    }
    if trial["type"] in CONTEXT_TRIAL_TYPES:
        result["trial_meta"] = trial_metadata(trial)
    if sampling_params is not None:
        result["sampling_regime"] = args.sampling_regime
        result["sampling_params"] = sampling_params

    save_result(result, args.results_file)
    print(f"Saved result for {trial['trial_id']} to {args.results_file}")


if __name__ == "__main__":
    main()
