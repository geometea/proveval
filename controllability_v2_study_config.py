"""Generate the v2 frozen study config (data/controllability_v2_study_config.json).

The config's `status` field is "draft" until an explicit freeze (see
freeze_controllability_v2.py) flips it to "frozen" and writes the
frozen-design lock file. Running this file (re)writes the config as a fresh
DRAFT -- it never freezes anything itself, so re-running it after a freeze
would require re-freezing (freeze_controllability_v2.verify_frozen will then
correctly report a mismatch until that happens).

Every string value below is read from controllability_v2_trials.py, never
retyped -- so the config can never silently drift from the actual generator
wording.
"""

import json

import model_providers
from controllability_v2_corpus import CORPUS_ID
from controllability_v2_trials import (
    BASELINE_TRIALS_FILE,
    CONTRASTS_FILE,
    EXPERIMENT_ID,
    INSTRUCTION_SENTENCES,
    PRIMARY_INSTRUCTION_CONDITIONS,
    QUESTION,
    TRIALS_FILE,
)

STUDY_CONFIG_FILE = "data/controllability_v2_study_config.json"
DESIGN_VERSION = "v2"

# The evaluator roster this study is designed to run: exactly one "primary"
# evaluator plus zero or more "replication" evaluators (item 9/18). Each
# evaluator_id is the analysis-side identity key -- see
# analyze_controllability_v2.match_evaluator_config.
#
# DeepSeek Flash is the current production primary evaluator. Claude Sonnet
# 5 (the earlier primary) is kept on as a future replication evaluator
# rather than removed -- it is never run for the current production study.
DEFAULT_EVALUATORS = [
    {
        "evaluator_id": "deepseek__deepseek-flash__low",
        "provider": "deepseek",
        "requested_model": "deepseek-flash",
        "reasoning_profile": "low",
        "role": "primary",
    },
    {
        "evaluator_id": "anthropic__claude-sonnet-5__low",
        "provider": "anthropic",
        "requested_model": "claude-sonnet-5",
        "reasoning_profile": "low",
        "role": "replication",
    },
    {
        "evaluator_id": "openai__gpt-5.6__low",
        "provider": "openai",
        "requested_model": "gpt-5.6",
        "reasoning_profile": "low",
        "role": "replication",
    },
    {
        "evaluator_id": "gemini__gemini-3.8-flash__low",
        "provider": "gemini",
        "requested_model": "gemini-3.8-flash",
        "reasoning_profile": "low",
        "role": "replication",
    },
]


def build_study_config(evaluators=None, status="draft"):
    if evaluators is None:
        evaluators = DEFAULT_EVALUATORS
    primary = [e for e in evaluators if e["role"] == "primary"]
    if len(primary) != 1:
        raise ValueError(f"Study config must name exactly one primary evaluator, got {len(primary)}")

    return {
        "design_version": DESIGN_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "corpus_id": CORPUS_ID,
        "primary_evaluator": primary[0],
        "replication_evaluators": [e for e in evaluators if e["role"] == "replication"],
        "treatment_replicate_count": 10,
        "baseline_replicate_count": 10,
        "random_seed": 20260917,
        "bootstrap_draws": 10000,
        "equivalence_margin": 0.05,
        "alpha": 0.05,
        "retry_limit": 3,
        "max_output_tokens": model_providers.default_max_output_tokens(primary[0]["provider"]),
        "instruction_strings": {
            condition: INSTRUCTION_SENTENCES[condition] for condition in PRIMARY_INSTRUCTION_CONDITIONS
        },
        "question": QUESTION,
        "contrast_file": CONTRASTS_FILE,
        "treatment_manifest_file": TRIALS_FILE,
        "baseline_manifest_file": BASELINE_TRIALS_FILE,
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
    print(f"Design version: {config['design_version']}")
    print(f"Experiment id: {config['experiment_id']}")
    print(f"Primary evaluator: {config['primary_evaluator']['evaluator_id']}")
    print(f"Replication evaluators: {[e['evaluator_id'] for e in config['replication_evaluators']]}")
    print(f"Status: {config['status']}")
    print(f"Output: {STUDY_CONFIG_FILE}")


if __name__ == "__main__":
    main()
