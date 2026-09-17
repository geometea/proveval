"""Shared constants for the v0.2 context-benchmark analysis modules
(context_analysis_*.py, orchestrated by analyze_context.py).

Split out on its own so every analysis module can depend on these paths and
the rating-field schema without importing from each other in a cycle.
"""

RESULTS_FILE = "results/context_raw.jsonl"
TRIALS_FILE = "data/context_trials.jsonl"
HUMAN_REFERENCE_FILE = "data/human_reference.json"
HUMAN_PAIRWISE_FILE = "data/human_pairwise.jsonl"
ANALYSIS_DIR = "results/context_analysis"
DEFAULT_SAMPLING_REGIME = "low_variance_primary"

RATING_FIELDS = ["plot_structure", "prose_style", "characterization", "originality", "overall_quality"]

# The standalone context-controllability experiment (controllability_trials.py/
# analyze_controllability.py) uses a single-category plain A/B outcome
# (["overall_quality"]) instead of a second 5-field rubric -- see
# controllability_trials.RESPONSE_FORMAT/PRIMARY_CATEGORY. The functions in
# context_analysis_pairwise.py that loop over rating categories accept an
# optional `categories` param (default RATING_FIELDS above) for exactly this
# reason, so that single-category list is passed in rather than duplicating
# any analysis logic.
