"""SECONDARY: agreement with the researcher reference ordering (a
single-researcher, ORDINAL-only, personalized preference ranking over the
12-story corpus -- not population ground truth, not a cardinal utility, and
not the organizing goal of this benchmark; see FINAL_DESIGN.md's
"Researcher reference ranking"). Everything in this module is secondary and
is never the headline result -- see analyze_context.py's module docstring
for the PRIMARY/SECONDARY ordering `main()` prints these in.

Two DIFFERENT reference concepts exist in this codebase and neither is
ground truth: the *uncontextualized behavioral baseline*
(context_analysis_single_text.py) is this model's own no-context answer,
used only to measure within-story context-induced change; the *researcher
reference* here is one person's fixed ordinal preference, used only for
this secondary, personalized agreement analysis.
"""

import json
import os
import statistics
from collections import defaultdict

from context_analysis_common import HUMAN_PAIRWISE_FILE, HUMAN_REFERENCE_FILE
from context_analysis_pairwise import choice_to_story_id, is_forced_choice_pairwise
from context_analysis_stats import (
    human_reference_scores,
    kendall_tau_b,
    pairwise_diagnostics_from_scores,
    spearman_tie_aware,
    tied_groups,
)


# ---------------------------------------------------------------------------
# SECONDARY: pairwise agreement with the researcher reference. DIRECT
# comparison in story identity, no derived ranking involved -- for every
# context_pairwise observation whose two displayed stories exactly match a
# judgment the researcher made, does the model's choice agree? A secondary,
# personalized comparison, not evidence about which context is "better" --
# stratified by evaluation_regime, so the more interesting reading is
# whether contextual perturbation makes this agreement more or less robust,
# not "which context wins" (see FINAL_DESIGN.md).
# ---------------------------------------------------------------------------

def analyze_pairwise_vs_human_reference(observations, human_pairs, category="overall_quality"):
    """Restricted to `category` (default overall_quality, the closest
    analogue to a single preference judgment). Reports concordant/
    discordant/model_tied counts -- never converts a model tie into a
    fabricated win or loss. Stratified by (model, evaluation_regime).
    """
    human_winner_by_pair = {frozenset((w, l)): w for w, l in human_pairs}
    rows = []
    tally = defaultdict(lambda: {"concordant": 0, "discordant": 0, "model_tied": 0})

    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        pair_key = frozenset((obs["story_a_id"], obs["story_b_id"]))
        human_winner = human_winner_by_pair.get(pair_key)
        if human_winner is None:
            continue
        chosen = choice_to_story_id(obs, category)
        if chosen is None:
            outcome = "model_tied"
        elif chosen == human_winner:
            outcome = "concordant"
        else:
            outcome = "discordant"
        tally[(obs["model"], obs["evaluation_regime"])][outcome] += 1
        rows.append(
            {
                "model": obs["model"],
                "evaluation_regime": obs["evaluation_regime"],
                "story_a_id": obs["story_a_id"],
                "story_b_id": obs["story_b_id"],
                "contrast_id": obs["contrast_id"],
                "assignment": obs["assignment"],
                "position": obs["position"],
                "replicate_id": obs["replicate_id"],
                "human_winner": human_winner,
                "model_choice": obs["parsed_response"][category],
                "model_chosen_story_id": chosen or "tie",
                "outcome": outcome,
            }
        )

    summary_rows = []
    for (model, evaluation_regime), counts in tally.items():
        n = counts["concordant"] + counts["discordant"] + counts["model_tied"]
        decided = counts["concordant"] + counts["discordant"]
        summary_rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "concordant": counts["concordant"],
                "discordant": counts["discordant"],
                "model_tied": counts["model_tied"],
                "n": n,
                "concordant_rate": round(counts["concordant"] / decided, 3) if decided else "",
            }
        )
    return rows, summary_rows


def summarize_pairwise_vs_human_reference(summary_rows):
    print("\n=== SECONDARY: direct model choices vs the researcher's pairwise judgments (no derived ranking) ===")
    print("(restricted to overall_quality, and to story pairs the researcher judged; personalized, ordinal, exploratory)")
    if not summary_rows:
        print("  No context_pairwise observations matched a researcher judgment.")
        return
    for row in sorted(summary_rows, key=lambda r: (r["model"], r["evaluation_regime"])):
        rate_text = f"{row['concordant_rate']:.3f}" if row["concordant_rate"] != "" else "unavailable"
        print(
            f"  {row['model']} | {row['evaluation_regime']}: concordant={row['concordant']} discordant={row['discordant']} "
            f"model_tied={row['model_tied']} (n={row['n']}, concordant_rate={rate_text})"
        )


# ---------------------------------------------------------------------------
# SECONDARY: single-text rankings (tie-aware) and agreement with the
# researcher reference ordering. This is a personalized, ordinal-only
# reference-agreement analysis -- not the benchmark's organizing goal, and
# not evidence that any one condition is objectively "better" (see
# FINAL_DESIGN.md's "Researcher reference ranking" and its multiple-
# comparisons caution). Scores are always kept as {story_id: (mean, n)}
# dicts -- never pre-sorted with an arbitrary tie-break -- until
# presentation time, where tied_groups() reports rank_position/tie_group_size
# without breaking ties. Stratified by evaluation_regime throughout.
# ---------------------------------------------------------------------------

def rank_from_single_text_by_condition(observations):
    """SECONDARY reference-agreement ranking: one score dict per
    (model, evaluation_regime, dimension, value), including the neutral
    baseline as its own condition. Returns
    {(model, evaluation_regime, dimension, value): {story_id: (mean_score, n)}}.
    Feeds into the researcher-reference agreement statistics below -- this
    is a secondary, personalized analysis, not the benchmark's primary
    context-sensitivity question (see
    context_analysis_single_text.analyze_treatment_vs_neutral for that).
    """
    scores = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single":
            continue
        key = (obs["model"], obs["evaluation_regime"], obs["dimension"], obs["value"])
        scores[key][obs["story_id"]].append(obs["parsed_response"]["overall_quality"])
    return {
        key: {story_id: (statistics.mean(vals), len(vals)) for story_id, vals in by_story.items()}
        for key, by_story in scores.items()
    }


def rank_from_single_text(observations):
    """DIAGNOSTIC ONLY: per (model, evaluation_regime), pooling every
    context_single condition together (still never pooling across
    evaluation_regime). Returns {(model, evaluation_regime): {story_id: (mean_score, n)}}."""
    scores = defaultdict(lambda: defaultdict(list))
    for obs in observations.values():
        if obs["type"] != "context_single":
            continue
        scores[(obs["model"], obs["evaluation_regime"])][obs["story_id"]].append(obs["parsed_response"]["overall_quality"])
    return {
        key: {story_id: (statistics.mean(vals), len(vals)) for story_id, vals in by_story.items()}
        for key, by_story in scores.items()
    }


def compute_pairwise_wins(observations):
    """DIAGNOSTIC ONLY: tally wins/losses/ties per story per
    (model, evaluation_regime), pooling ALL context_pairwise observations
    regardless of contrast/assignment/position (but never across
    evaluation_regime). See rank_from_pairwise_wins for why this isn't a
    per-condition ranking."""
    records = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}))
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        key = (obs["model"], obs["evaluation_regime"])
        a, b = obs["story_a_id"], obs["story_b_id"]
        choice = obs["parsed_response"]["overall_quality"]
        if choice == "A":
            records[key][a]["wins"] += 1
            records[key][b]["losses"] += 1
        elif choice == "B":
            records[key][b]["wins"] += 1
            records[key][a]["losses"] += 1
        else:
            records[key][a]["ties"] += 1
            records[key][b]["ties"] += 1
    return records


def rank_from_pairwise_wins(tally_by_story):
    """DIAGNOSTIC ONLY. Returns {story_id: (copeland, win_rate, games)} --
    Copeland score (wins - losses) is the scalar used for ranking/rank
    statistics; win_rate/games are informational only. Pools every
    context_pairwise observation regardless of context condition, so this
    can't be read as a ranking "under" any particular context -- see
    context_analysis_pairwise.analyze_directional_pairwise_effects for the
    primary pairwise result.
    """
    return {
        story_id: (tally["wins"] - tally["losses"], (tally["wins"] + 0.5 * tally["ties"]) / max(1, tally["wins"] + tally["losses"] + tally["ties"]), tally["wins"] + tally["losses"] + tally["ties"])
        for story_id, tally in tally_by_story.items()
    }


def load_human_reference(path=HUMAN_REFERENCE_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)["ranking"]


def load_human_pairwise(path=HUMAN_PAIRWISE_FILE):
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                pairs.append((row["winner"], row["loser"]))
    return pairs


def human_comparison_row(model, evaluation_regime, dimension, value, ranking_source, scores_dict, human_reference, human_pairs):
    """scores_dict: {story_id: (mean_score, n)} or {story_id: (scalar, ..., ...)}
    where the FIRST tuple element is always the scalar used for ranking."""
    values = {sid: v[0] for sid, v in scores_dict.items()}
    items = list(values.keys())

    tau = spearman = None
    n_common = 0
    if human_reference is not None:
        common = [it for it in items if it in human_reference]
        n_common = len(common)
        if n_common >= 2:
            human_scores = human_reference_scores(human_reference)
            tau = kendall_tau_b(values, human_scores, common)
            spearman = spearman_tie_aware(values, human_scores, common)

    concordant, discordant, model_tied, checked = pairwise_diagnostics_from_scores(values, human_pairs)
    decided = concordant + discordant
    return {
        "model": model,
        "evaluation_regime": evaluation_regime,
        "dimension": dimension,
        "value": value,
        "ranking_source": ranking_source,
        "n_common_with_human_reference": n_common,
        "kendall_tau_b": round(tau, 3) if tau is not None else "",
        "spearman_vs_human": round(spearman, 3) if spearman is not None else "",
        "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "pairwise_model_tied": model_tied,
        "pairwise_concordant_rate": round(concordant / decided, 3) if decided else "",
        "pairwise_checked_n": checked,
    }


def ranking_csv_rows(scores_with_n, extra_fields, score_field_name="mean_overall_quality"):
    """scores_with_n: {story_id: (score, n)}. rank_position/tie_group_size
    come from tied_groups() -- ties share the identical rank_position, never
    broken by story ID or any other arbitrary field. Row order (sorted by
    story_id) is for stable display only."""
    values = {sid: v[0] for sid, v in scores_with_n.items()}
    items = list(scores_with_n.keys())
    rank_by_story, group_size_by_story = {}, {}
    for g in tied_groups(values, items):
        for sid in g["story_ids"]:
            rank_by_story[sid] = g["rank_position"]
            group_size_by_story[sid] = len(g["story_ids"])
    rows = []
    for sid in sorted(items):
        score, n = scores_with_n[sid]
        rows.append(
            {
                **extra_fields,
                "story_id": sid,
                score_field_name: round(score, 3),
                "n": n,
                "rank_position": rank_by_story[sid],
                "tie_group_size": group_size_by_story[sid],
            }
        )
    return rows


def print_ranking_with_ties(scores_with_n, indent="    "):
    values = {sid: v[0] for sid, v in scores_with_n.items()}
    for g in tied_groups(values, list(scores_with_n.keys())):
        ids = ", ".join(g["story_ids"])
        tie_note = f"  [tied group of {len(g['story_ids'])}]" if len(g["story_ids"]) > 1 else ""
        print(f"{indent}rank {g['rank_position']}: {ids}  (score={g['score']:.2f}){tie_note}")
