"""Run exactly one trial from data/trials.jsonl through the Claude API.

Usage:
    python3 run_trial.py <trial_id>            # calls the API and saves the result
    python3 run_trial.py <trial_id> --dry-run  # just prints what would be sent
"""

import argparse
import json
import os
import re

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
    with open(path, "r", encoding="utf-8") as f:
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


DEFAULT_MAX_TOKENS = 4096

# response_format="plain_ab" trials (see controllability_trials.RESPONSE_FORMAT)
# only ever need a few tokens of output ("A", "Passage B", etc.), so they use
# a much smaller cap -- every other trial type keeps DEFAULT_MAX_TOKENS.
PLAIN_AB_MAX_TOKENS = 64


def max_tokens_for(response_format):
    return PLAIN_AB_MAX_TOKENS if response_format == "plain_ab" else DEFAULT_MAX_TOKENS


def call_claude(prompt, model, sampling_params=None, max_tokens=DEFAULT_MAX_TOKENS):
    """Send the prompt as a single user message and return the reply text.

    sampling_params (e.g. {"temperature": 0}) are passed through to the API
    verbatim when truthy; None or {} (both mean "no override") send nothing
    extra, so the v0.1 pilot's original call shape is unchanged when this is
    omitted. max_tokens defaults to the original 4096 for every existing
    trial type; see max_tokens_for/PLAIN_AB_MAX_TOKENS for the smaller cap
    used by the plain A/B response format.
    """
    import anthropic  # imported here so --dry-run works without the package installed

    client = anthropic.Anthropic(api_key=get_api_key())
    create_kwargs = dict(
        model=model,
        max_tokens=max_tokens,  # required by the API; not a sampling parameter
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
        # The actual model version the API resolved the request to -- may
        # differ from the requested alias (e.g. "claude-sonnet-5"). See
        # trial/result field "response_model"; analysis must not silently
        # pool different returned model versions together.
        "response_model": getattr(response, "model", None),
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

# RUBRIC_FIELDS keeps the rubric-keyed lookup used by _validate_ab_tie_shape
# below general (an "excerpt" rubric used to live here for the standalone
# context-controllability experiment; that experiment now uses its own
# dedicated plain-A/B response format -- see
# controllability_trials.RESPONSE_FORMAT/parse_plain_ab_response -- and no
# longer goes through this AB/tie schema at all, so "excerpt" was removed as
# dead code). "full" remains every other experiment's only rubric.
RUBRIC_FIELDS = {"full": AB_TIE_FIELDS}

# PRIMARY forced-choice schema: "tie" is not an allowed value. A model that
# answers "tie" on a forced-choice trial fails validation -- it is never
# silently coerced into "A" or "B" (see validate_pairwise_context_response_forced).
FORCED_CHOICE_VALUES = {"A", "B"}

# SECONDARY tie-allowed schema, used only for the hedging/indifference
# diagnostic (choice_mode="tie_allowed", see context_comparisons.py and
# context_trials.build_tie_allowed_pairwise_trials). Never used for the
# primary context_pairwise task.
TIE_ALLOWED_VALUES = {"A", "B", "tie"}

# Kept for backward-compatible naming; equivalent to TIE_ALLOWED_VALUES.
AB_TIE_VALUES = TIE_ALLOWED_VALUES


def normalize_ab_tie_keys(parsed):
    """Return a copy of parsed with "characterisation" renamed to "characterization"."""
    if not isinstance(parsed, dict):
        return parsed
    normalized = dict(parsed)
    if "characterisation" in normalized and "characterization" not in normalized:
        normalized["characterization"] = normalized.pop("characterisation")
    return normalized


def _validate_ab_tie_shape(parsed, allowed_values, allowed_values_label, rubric="full"):
    fields = RUBRIC_FIELDS[rubric]
    parsed = normalize_ab_tie_keys(parsed)
    if not isinstance(parsed, dict) or set(parsed.keys()) != set(fields):
        return None, (
            f"Response must contain exactly {', '.join(fields[:-1])} "
            "(characterization may also be spelled characterisation), "
            f"and {fields[-1]}"
        )
    for field in fields:
        if parsed[field] not in allowed_values:
            return None, f"{field} must be exactly {allowed_values_label}"
    return parsed, None


def validate_pairwise_context_response_forced(parsed, rubric="full"):
    """Check a PRIMARY forced-choice context_pairwise/context_prompt response.

    5 fields, each exactly "A" or "B" -- "tie" fails validation here rather
    than being coerced into either letter (requirement: never silently treat
    a declined choice as a real preference). Normalizes "characterisation"
    -> "characterization" first (see normalize_ab_tie_keys). rubric selects
    which 5 fields are expected ("full", the default, or "excerpt" -- see
    RUBRIC_FIELDS). Returns (normalized_parsed_or_None, error_or_None).
    """
    return _validate_ab_tie_shape(parsed, FORCED_CHOICE_VALUES, '"A" or "B"', rubric)


def validate_pairwise_context_response_tie_allowed(parsed, rubric="full"):
    """Check a SECONDARY tie-allowed hedging-diagnostic response.

    5 fields, each "A"/"B"/"tie". Used only for choice_mode="tie_allowed"
    trials (context_trials.build_tie_allowed_pairwise_trials) and for the
    legacy "context_pairwise_same" optional family. Returns
    (normalized_parsed_or_None, error_or_None).
    """
    return _validate_ab_tie_shape(parsed, TIE_ALLOWED_VALUES, '"A", "B", or "tie"', rubric)


def validate_pairwise_context_response(parsed):
    """Backward-compatible alias for the tie-allowed validator.

    Kept only so any external caller still importing this exact name keeps
    working; new code should call validate_pairwise_context_response_forced
    or validate_pairwise_context_response_tie_allowed explicitly instead of
    relying on this name's behavior.
    """
    return validate_pairwise_context_response_tie_allowed(parsed)


_CODE_FENCE_OPEN = re.compile(r"^```[ \t]*[a-zA-Z0-9_+-]*[ \t]*\n?")


def strip_code_fence(text):
    """Remove a surrounding Markdown code fence (```json ... ``` or ``` ... ```), if present.

    Handles the fence entirely on its own lines (the common case) AND a
    fence collapsed onto a single line with no internal newline (e.g.
    '```json{"a": 1}```') -- a line-based split used to drop the whole
    first "line" as the opening fence, which silently discarded the entire
    response when it was all on one line, turning a valid answer into a
    JSON-decode failure.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text

    match = _CODE_FENCE_OPEN.match(text)
    text = text[match.end():] if match else text[3:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


# ---------------------------------------------------------------------------
# "plain_ab" response format: the standalone context-controllability
# experiment's single-outcome A/B judgment (see
# controllability_trials.RESPONSE_FORMAT), normalized internally to
# {"overall_quality": "A"} / {"overall_quality": "B"} so the existing
# pairwise analysis (context_analysis_pairwise.py, with
# categories=["overall_quality"]) can be reused unmodified. This is
# conservative pattern matching, not an LLM parser: it only recognizes a
# short, closed set of unambiguous forms, and never scans an arbitrary long
# response for the first stray "A"/"B" it can find.
# ---------------------------------------------------------------------------

_PLAIN_AB_STRIP_CHARS = "*_`'\"“”‘’ \t\r\n"

# Any of these substrings appearing anywhere in the (lowercased) response
# means the model hedged, refused, or called it a tie -- never coerced into
# a letter, even if the response also happens to start with "A" or "B".
_PLAIN_AB_AMBIGUITY_MARKERS = (
    "tie", "both", "neither", "unsure", "not sure", "no preference",
    "can't decide", "cannot decide", "can't tell", "cannot tell",
    "hard to say", "hard to tell", "difficult to say", "difficult to tell",
    "difficult to choose", "hard to choose", "toss-up", "toss up",
    "roughly equal", "about equal", "equally good", "equally strong",
)

# Structural forms accepted, in order: an optional "I prefer " and/or
# "Passage " prefix, then the letter, then either the end of the string or a
# word boundary followed by anything (so "A", "A.", "Passage B", "I prefer
# Passage A", and "A -- because..." all match, but "Both" or "According to
# ny reading..." do not, since the letter must be the very first token).
_PLAIN_AB_PATTERNS = [
    re.compile(r"i\s+prefer\s+passage\s+([ab])\b.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"i\s+prefer\s+([ab])\b.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"passage\s+([ab])\b.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"([ab])\b.*", re.IGNORECASE | re.DOTALL),
]


def parse_plain_ab_response(response_text):
    """Parse a plain_ab response into ({"overall_quality": "A"|"B"}, None),
    or (None, error) if the response isn't an unambiguous A/B judgment.

    Tries a JSON fallback first ({"choice": "A"} or {"overall_quality": "A"},
    in case a model unexpectedly wraps its answer in JSON), then falls back
    to plain-text pattern matching. Never uses an LLM to parse; case
    -insensitive; ignores surrounding whitespace and simple Markdown
    emphasis/quote characters.
    """
    stripped = strip_code_fence(response_text).strip(_PLAIN_AB_STRIP_CHARS)

    try:
        candidate = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = None

    if candidate is not None:
        if not isinstance(candidate, dict):
            return None, "JSON response must be an object with a 'choice' or 'overall_quality' key"
        value = candidate.get("overall_quality", candidate.get("choice"))
        if not isinstance(value, str) or value.strip().upper() not in ("A", "B"):
            return None, "JSON response must have 'choice' or 'overall_quality' set to exactly \"A\" or \"B\""
        return {"overall_quality": value.strip().upper()}, None

    lowered = stripped.lower()
    for marker in _PLAIN_AB_AMBIGUITY_MARKERS:
        if marker in lowered:
            return None, f"Response reads as ambiguous/hedged (contains {marker!r}), not an unambiguous A or B"

    for pattern in _PLAIN_AB_PATTERNS:
        match = pattern.fullmatch(stripped)
        if match:
            return {"overall_quality": match.group(1).upper()}, None

    return None, 'Response must be an unambiguous "A" or "B" (optionally "Passage A/B" or "I prefer A/B"), got: ' + repr(
        response_text[:200]
    )


def parse_and_validate(response_text, trial_type, choice_mode=None, rubric="full", response_format=None):
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
        -> dispatches again on choice_mode:
          - choice_mode == "tie_allowed" -> A/B/tie per category (SECONDARY
            hedging diagnostic; see validate_pairwise_context_response_tie_allowed)
          - anything else (choice_mode == "forced", or missing/None) -> A/B
            only per category (PRIMARY forced-choice task; see
            validate_pairwise_context_response_forced). A response of "tie"
            fails validation here rather than being coerced into "A" or "B".
          rubric ("full", the only rubric now -- see RUBRIC_FIELDS) selects
            which 5 fields are expected; every existing call site omits it
            and gets the unchanged "full" schema.
        "context_pairwise_same" is the OPTIONAL, never-run same-context
        trial family (context_trials.py); it carries choice_mode="forced"
        and is validated the same way as the primary task.
      anything else (e.g. "comparison", "comparison_control", and any future
        or unrecognized type) -> the original story_a/story_b/preference
        schema (v0.1's integer -2..2 "preference" field), exactly as before
        this function grew a dispatch at all -- entirely separate from, and
        unaffected by, the A/B/tie string schema above.

    response_format="plain_ab" (see controllability_trials.RESPONSE_FORMAT)
    bypasses all of the above and every existing call site: it's checked
    first, dispatches straight to parse_plain_ab_response, and is the only
    thing that can turn a non-JSON response (plain "A"/"B" text) into a
    valid result. Every existing trial omits response_format (None) and is
    completely unaffected.
    """
    if response_format == "plain_ab":
        return parse_plain_ab_response(response_text)

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
        if choice_mode == "tie_allowed":
            return validate_pairwise_context_response_tie_allowed(parsed, rubric)
        return validate_pairwise_context_response_forced(parsed, rubric)

    error = validate_comparison_response(parsed)
    return (None, error) if error else (parsed, None)


def save_result(result, results_file=RESULTS_FILE):
    """Append one JSON result to results_file, creating its folder if needed."""
    folder = os.path.dirname(results_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as f:
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

    api_result = call_claude(trial["prompt"], model, sampling_params, max_tokens_for(trial.get("response_format")))
    response_text = api_result["response_text"]
    parsed_response, validation_error = parse_and_validate(
        response_text, trial["type"], trial.get("choice_mode"), trial.get("rubric", "full"), trial.get("response_format")
    )

    if validation_error:
        print(f"Warning: invalid response for {trial['trial_id']}: {validation_error}")

    result = {
        "trial_id": trial["trial_id"],
        "model": model,
        "response_model": api_result.get("response_model"),
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
