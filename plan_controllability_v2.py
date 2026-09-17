"""Experiment-planning simulator for the v2 context-controllability design.

Simulates the EXACT v2 design (66 story pairs x 5 contrasts x 2 instruction
conditions x 4-cell counterbalance, plus the independent 66-pair x 2-position
blind baseline) under a grid of hypothetical true effect sizes and replicate
counts, using the real estimators from controllability_v2_stats.py (the same
D_pair/ATE/story-bootstrap/split-half code the real analysis uses) against
SIMULATED binomial response data -- never a real model call of any kind.

For each (treatment_replicates, baseline_replicates, true_context_effect,
baseline_pair_decisiveness, text_only_attenuation) combination, runs
`--simulations` independent synthetic datasets and reports the AVERAGE:
  - 95% story-bootstrap CI width for the context ATE (matched_control)
  - 95% story-bootstrap CI width for the instruction contrast
    (signed_instruction_difference)
  - baseline-strength split-half reliability (a stability diagnostic, not a
    CI width -- see controllability_v2_stats.split_half_baseline_reliability)
  - total model calls this configuration would need (exact, not simulated)
  - approximate total input word/token count, using the REAL per-story word
    counts recorded in data/controllability_v2_corpus.json

This is a planning approximation, not the primary analysis: CI widths are
still computed with the real story-bootstrap machinery, but a uniform
(pair-independent) true effect is assumed within one configuration, and each
configuration's report is an average over `--simulations` independent
synthetic runs, not one frozen dataset.
"""

import argparse
import csv
import itertools
import json
import random
import statistics

from context_trials import load_items, story_pairs
from controllability_v2_corpus import CORPUS_FILE
import controllability_v2_stats as st

N_CONTRASTS = 5
N_PAIRS = 66
CELLS_PER_INSTRUCTION_CONDITION = 4
INSTRUCTION_CONDITIONS = ("matched_control", "text_only")

DEFAULT_TREATMENT_REPLICATES = (1, 2, 3)
DEFAULT_BASELINE_REPLICATES = (1, 2, 3)
DEFAULT_CONTEXT_EFFECTS = (0.0, 0.05, 0.10, 0.20)
DEFAULT_BASELINE_DECISIVENESS = (0.1, 0.3, 0.5)  # spread of the per-pair "true" baseline win probability around 0.5
DEFAULT_TEXT_ONLY_ATTENUATION = (0.0, 0.5, 1.0)  # fraction of the context effect removed under text_only


def _clip(p, lo=0.02, hi=0.98):
    return max(lo, min(hi, p))


def simulate_one_dataset(story_ids, treatment_replicates, baseline_replicates, true_context_effect,
                          baseline_decisiveness, text_only_attenuation, rng):
    """One synthetic realization of ONE contrast's data (the same simulated
    numbers are reused for every contrast in a config, since the planning
    grid varies effect size, not per-contrast wording). Returns
    (pair_values_control, pair_values_text_only, baseline_pair_stats_rows,
    per_pair_replicate_choices) -- the same shapes analyze_controllability_v2
    builds from real results, so the exact same stats functions apply.
    """
    pairs = list(itertools.combinations(sorted(story_ids), 2))

    # A fixed per-pair "true" baseline decisiveness draw (reused for control
    # and text_only baseline simulation) -- some pairs are close calls, some
    # are lopsided, spread controlled by baseline_decisiveness.
    true_baseline_p = {pair: _clip(0.5 + baseline_decisiveness * rng.uniform(-1, 1)) for pair in pairs}

    baseline_rows = []
    per_pair_replicate_choices = {}
    for pair in pairs:
        p = true_baseline_p[pair]
        choices = []
        for replicate_number in range(1, baseline_replicates + 1):
            for _position in ("story1_as_a", "story2_as_a"):
                choices.append((replicate_number, rng.random() < p))
        wins1 = sum(1 for _, chosen in choices if chosen)
        wins2 = len(choices) - wins1
        baseline_rows.append({"story_1_id": pair[0], "story_2_id": pair[1], **st.baseline_pair_stats(wins1, wins2)})
        per_pair_replicate_choices[pair] = choices

    def simulate_condition(effect):
        p_forward = _clip(0.5 + effect / 2)
        p_flipped = _clip(0.5 - effect / 2)
        pair_values = {}
        for pair in pairs:
            cell_rates = {}
            for assignment, p in (("forward", p_forward), ("flipped", p_flipped)):
                for position in ("story1_as_a", "story2_as_a"):
                    wins = sum(1 for _ in range(treatment_replicates) if rng.random() < p)
                    cell_rates[(assignment, position)] = wins / treatment_replicates
            result = st.d_pair_from_cell_rates(cell_rates)
            pair_values[pair] = result["d_pair"]
        return pair_values

    pair_values_control = simulate_condition(true_context_effect)
    pair_values_text_only = simulate_condition(true_context_effect * (1 - text_only_attenuation))

    return pair_values_control, pair_values_text_only, baseline_rows, per_pair_replicate_choices


def run_one_configuration(story_ids, treatment_replicates, baseline_replicates, true_context_effect,
                           baseline_decisiveness, text_only_attenuation, n_simulations, bootstrap_draws, seed):
    rng = random.Random(seed)
    ate_ci_widths, instruction_ci_widths, split_half_rs = [], [], []

    for _ in range(n_simulations):
        control, text_only, baseline_rows, per_pair_choices = simulate_one_dataset(
            story_ids, treatment_replicates, baseline_replicates, true_context_effect,
            baseline_decisiveness, text_only_attenuation, rng,
        )

        ate_draws = [d["ate"] for d in st.story_bootstrap_joint_draws({"ate": control}, story_ids, bootstrap_draws, rng)]
        ci = st.bootstrap_ci_from_draws(ate_draws, conf_levels=(0.95,))[0.95]
        if ci[0] is not None:
            ate_ci_widths.append(ci[1] - ci[0])

        boot = st.controllability_bootstrap(control, text_only, story_ids, bootstrap_draws, rng)
        ci_instr = st.bootstrap_ci_from_draws(boot["signed_instruction_difference"], conf_levels=(0.95,))[0.95]
        if ci_instr[0] is not None:
            instruction_ci_widths.append(ci_instr[1] - ci_instr[0])

        reliability = st.split_half_baseline_reliability(per_pair_choices)
        if reliability["r_split_half"] is not None:
            split_half_rs.append(reliability["r_split_half"])

    total_model_calls = N_CONTRASTS * N_PAIRS * len(INSTRUCTION_CONDITIONS) * CELLS_PER_INSTRUCTION_CONDITION * treatment_replicates
    total_model_calls += N_PAIRS * 2 * baseline_replicates

    return {
        "treatment_replicates": treatment_replicates,
        "baseline_replicates": baseline_replicates,
        "true_context_effect": true_context_effect,
        "baseline_pair_decisiveness": baseline_decisiveness,
        "text_only_attenuation": text_only_attenuation,
        "n_simulations": n_simulations,
        "expected_ci_width_context_ate": statistics.mean(ate_ci_widths) if ate_ci_widths else None,
        "expected_ci_width_instruction_contrast": statistics.mean(instruction_ci_widths) if instruction_ci_widths else None,
        "baseline_strength_split_half_r": statistics.mean(split_half_rs) if split_half_rs else None,
        "total_model_calls": total_model_calls,
    }


def approximate_word_and_token_counts(total_model_calls_per_config):
    """Approximate per-call input size from the REAL story lengths recorded
    in data/controllability_v2_corpus.json (item 19: "approximate input
    word/token count using actual story lengths"). A treatment prompt
    contains 2 stories; a baseline prompt also contains 2. Fixed scaffolding
    (intro/instruction/question) adds a small, roughly constant word count
    on top, estimated here as ~60 words. Tokens are approximated at 1.3
    tokens/word -- a rough, provider-agnostic heuristic, not a real
    tokenizer count."""
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    word_counts = [s["word_count"] for s in corpus["stories"]]
    avg_pair_words = 2 * statistics.mean(word_counts)  # 2 stories per prompt
    avg_prompt_words = avg_pair_words + 60
    avg_prompt_tokens = avg_prompt_words * 1.3
    return {
        "avg_story_word_count": statistics.mean(word_counts),
        "avg_prompt_word_count": avg_prompt_words,
        "avg_prompt_token_count_estimate": avg_prompt_tokens,
        "total_word_count_estimate": avg_prompt_words * total_model_calls_per_config,
        "total_token_count_estimate": avg_prompt_tokens * total_model_calls_per_config,
    }


def main():
    parser = argparse.ArgumentParser(description="v2 design planning simulator -- no model API calls.")
    parser.add_argument("--treatment-replicates", type=int, nargs="+", default=list(DEFAULT_TREATMENT_REPLICATES))
    parser.add_argument("--baseline-replicates", type=int, nargs="+", default=list(DEFAULT_BASELINE_REPLICATES))
    parser.add_argument("--context-effects", type=float, nargs="+", default=list(DEFAULT_CONTEXT_EFFECTS))
    parser.add_argument("--baseline-decisiveness", type=float, nargs="+", default=list(DEFAULT_BASELINE_DECISIVENESS))
    parser.add_argument("--text-only-attenuation", type=float, nargs="+", default=list(DEFAULT_TEXT_ONLY_ATTENUATION))
    parser.add_argument("--simulations", type=int, default=20, help="Independent synthetic datasets averaged per configuration")
    parser.add_argument("--bootstrap-draws", type=int, default=500, help="Story-bootstrap draws used for each simulated dataset's CI")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-csv", default="results/controllability_v2/planning/plan_results.csv")
    parser.add_argument("--limit-configs", type=int, default=None, help="Only run the first N configurations (for a quick smoke test)")
    args = parser.parse_args()

    items = load_items()
    story_ids = sorted(item["id"] for item in items)
    assert len(story_pairs(items)) == N_PAIRS

    configs = list(itertools.product(
        args.treatment_replicates, args.baseline_replicates, args.context_effects,
        args.baseline_decisiveness, args.text_only_attenuation,
    ))
    if args.limit_configs is not None:
        configs = configs[: args.limit_configs]

    print(f"Running {len(configs)} configuration(s), {args.simulations} simulation(s) each, no model API calls.")

    rows = []
    for i, (tr, br, effect, decisiveness, attenuation) in enumerate(configs, start=1):
        result = run_one_configuration(story_ids, tr, br, effect, decisiveness, attenuation, args.simulations, args.bootstrap_draws, args.seed + i)
        result.update(approximate_word_and_token_counts(result["total_model_calls"]))
        rows.append(result)
        print(
            f"[{i}/{len(configs)}] treat_reps={tr} base_reps={br} effect={effect} decisiveness={decisiveness} "
            f"attenuation={attenuation} -> CI(ATE)={_fmt(result['expected_ci_width_context_ate'])} "
            f"CI(instr)={_fmt(result['expected_ci_width_instruction_contrast'])} "
            f"split_half_r={_fmt(result['baseline_strength_split_half_r'])} "
            f"calls={result['total_model_calls']} ~tokens={result['total_token_count_estimate']:.0f}"
        )

    import os
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if rows:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} row(s) to {args.output_csv}")


def _fmt(value):
    return f"{value:.3f}" if value is not None else "n/a"


if __name__ == "__main__":
    main()
