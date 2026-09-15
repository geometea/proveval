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
