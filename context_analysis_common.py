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

# Excerpt rubric: the 12 registered corpus items are excerpts, not
# necessarily complete stories, so "plot_structure" doesn't cleanly apply to
# all of them. Used by the standalone context-controllability experiment
# (see controllability_trials.py/analyze_controllability.py) via
# context_comparisons.build_prompt's/run_trial.py's rubric="excerpt" option
# -- additive, the default "full" rubric (RATING_FIELDS above) is unchanged
# and every existing experiment keeps using it.
EXCERPT_RATING_FIELDS = ["prose_style", "characterization", "originality", "narrative_effectiveness", "overall_quality"]
