"""Generate the v3 study config (data/controllability_v3_study_config.json).

Like v2's, the config is written as a DRAFT and only an explicit freeze
(freeze_controllability_v3.py) flips it to "frozen" and writes the lock.
Every wording string is read from controllability_v3_design.py, never
retyped, so the config can never drift from the generator.

Model configuration (deliberately identical to the completed v2 study's
recovery-era production configuration, for comparability):
  provider deepseek, requested model deepseek-flash, reasoning profile
  "low" (sent as reasoning_effort="low"), provider-default sampling, the
  same plain A/B parser, and max_output_tokens = 4096 (the v2 Wave 2
  recovery ceiling -- never the original 512 that truncated reasoning).
"""

import json

from controllability_v2_recovery import RECOVERY_MAX_OUTPUT_TOKENS
from controllability_v3_corpus import CORPUS_ID
from controllability_v3_design import (
    CUE_ENUMERATING_INTERVENTIONS,
    CUE_ORDER,
    DESIGN_VERSION,
    EXPERIMENT_ID,
    HEADLINE_CONTRASTS,
    INTERVENTION_IDS,
    INTERVENTION_LABELS,
    INTERVENTION_SENTENCES,
    NO_CONTEXT_INTRO,
    QUESTION,
    all_instruction_sentences,
)
from controllability_v3_trials import (
    CONTEXT_TRIALS_FILE,
    CONTEXT_UNIQUE_CELLS,
    CONTRASTS_FILE,
    HOLDOUT_TRIALS_FILE,
    HOLDOUT_UNIQUE_CELLS,
    NOCONTEXT_TRIALS_FILE,
    NOCONTEXT_UNIQUE_CELLS,
    PILOT_TRIALS_FILE,
)

STUDY_CONFIG_FILE = "data/controllability_v3_study_config.json"

PRIMARY_EVALUATOR = {
    "evaluator_id": "deepseek__deepseek-flash__low",
    "provider": "deepseek",
    "requested_model": "deepseek-flash",
    "reasoning_profile": "low",
    "role": "primary",
}

# The v2 Wave 2 recovery ceiling (4096). v2's original 512 truncated hidden
# reasoning before the visible answer; that failure mode is never reused.
MAX_OUTPUT_TOKENS = RECOVERY_MAX_OUTPUT_TOKENS

REPLICATES = 10
HOLDOUT_REPLICATES = 10
PILOT_REPLICATES = 1


def build_study_config(status="draft", evaluator=None):
    evaluator = evaluator or PRIMARY_EVALUATOR
    holdout_ids = sorted(k for k in all_instruction_sentences() if k not in INTERVENTION_IDS)
    return {
        "design_version": DESIGN_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "corpus_id": CORPUS_ID,
        "primary_evaluator": evaluator,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "sampling": "provider_default",
        "replicate_count": REPLICATES,
        "holdout_replicate_count": HOLDOUT_REPLICATES,
        "pilot_replicate_count": PILOT_REPLICATES,
        "random_seed": 20260918,
        "bootstrap_draws": 5000,
        "alpha": 0.05,
        "retry_limit": 3,
        # Relative suppression is reported only when the matched-control
        # context effect's magnitude is at least this (probability points)
        # AND its 95% CI excludes zero -- never with a near-zero denominator.
        "relative_suppression_min_denominator": 0.02,
        "question": QUESTION,
        "no_context_intro": NO_CONTEXT_INTRO,
        "intervention_ids": list(INTERVENTION_IDS),
        "intervention_labels": dict(INTERVENTION_LABELS),
        "instruction_strings": dict(INTERVENTION_SENTENCES),
        "holdout_instruction_strings": {k: all_instruction_sentences()[k] for k in holdout_ids},
        "cue_enumerating_interventions": list(CUE_ENUMERATING_INTERVENTIONS),
        "cue_order": list(CUE_ORDER),
        "headline_contrasts": [list(c) for c in HEADLINE_CONTRASTS],
        "contrast_file": CONTRASTS_FILE,
        "context_manifest_file": CONTEXT_TRIALS_FILE,
        "nocontext_manifest_file": NOCONTEXT_TRIALS_FILE,
        "holdout_manifest_file": HOLDOUT_TRIALS_FILE,
        "pilot_manifest_file": PILOT_TRIALS_FILE,
        "expected_unique_cells": {
            "primary_context": CONTEXT_UNIQUE_CELLS,
            "primary_nocontext": NOCONTEXT_UNIQUE_CELLS,
            "holdout": HOLDOUT_UNIQUE_CELLS,
        },
        "status": status,
    }


def load_study_config(path=STUDY_CONFIG_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_study_config(config, path=STUDY_CONFIG_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def main():
    config = build_study_config()
    save_study_config(config)
    print(f"Design version: {config['design_version']}  Experiment id: {config['experiment_id']}")
    print(f"Primary evaluator: {config['primary_evaluator']['evaluator_id']}  max_output_tokens={config['max_output_tokens']}")
    print(f"Replicates: primary={config['replicate_count']} holdout={config['holdout_replicate_count']} pilot={config['pilot_replicate_count']}")
    print(f"Status: {config['status']}  Output: {STUDY_CONFIG_FILE}")


if __name__ == "__main__":
    main()
