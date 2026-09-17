"""PRIMARY pairwise context-effect analysis (choice_mode="forced" only),
its secondary raw-choice diagnostic, the position/interaction/heterogeneity/
leave-one-out decompositions, the optional tie-allowed hedging diagnostic,
and the context_prompt (extraneous prompt-level sentence) family.

Every function here that reads context_pairwise observations restricts to
`is_forced_choice_pairwise(obs)` UNLESS it's explicitly the
choice_mode="tie_allowed" diagnostic (analyze_tie_allowed_diagnostic) -- the
two choice modes are never silently pooled into one estimate.
"""

import statistics
from collections import defaultdict

from context_analysis_common import RATING_FIELDS
from context_analysis_stats import describe_attenuation


# ---------------------------------------------------------------------------
# Pairwise: SECONDARY diagnostic (raw A/B/tie "changed") and PRIMARY
# directional effect, both operating in story identity, both stratified by
# evaluation_regime.
# ---------------------------------------------------------------------------

def choice_to_story_id(obs, category):
    """Map a raw A/B/tie choice to the actual story id that was chosen, or
    None for a tie. This is what lets every analysis below operate in story
    identity instead of raw A/B letters."""
    choice = obs["parsed_response"][category]
    if choice == "A":
        return obs["story_a_id"]
    if choice == "B":
        return obs["story_b_id"]
    return None


def is_forced_choice_pairwise(obs):
    """True for a context_pairwise/context_prompt observation from the
    PRIMARY forced-choice task. choice_mode defaults to "forced" for
    observations that predate the field (there are none in this repo's own
    data, but this keeps the check total) -- never for "tie_allowed", which
    must always be excluded from every primary forced-choice analysis below
    so the two choice modes are never silently pooled into one estimate.
    """
    return obs.get("choice_mode", "forced") == "forced"


def index_pairwise_by_assignment_fixed_position(observations, position="story1_as_a"):
    index = {}
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or obs.get("position") != position or not is_forced_choice_pairwise(obs):
            continue
        key = (obs["model"], obs["evaluation_regime"], obs["story_1_id"], obs["story_2_id"], obs["contrast_id"], obs["replicate_id"])
        index.setdefault(key, {})[obs["assignment"]] = obs
    return index


def analyze_pairwise_changed_diagnostic(observations, position="story1_as_a", categories=RATING_FIELDS):
    """SECONDARY diagnostic only -- see analyze_directional_pairwise_effects
    for the PRIMARY result. Restricted to one fixed display position (default:
    story_1 shown as Story A) so this reproduces the original "did the raw
    A/B/tie choice change between forward and flipped" comparison without
    conflating display position with context assignment (see
    context_contrasts.build_contrast_block). A bare changed=True/False does
    NOT say which story or which context was preferred -- chosen story ids
    are preserved here for that, but the primary answer is the function below.
    Stratified by evaluation_regime -- naturalistic and text_only_invariance
    cells are never compared against each other here.

    categories defaults to the "full" rubric's RATING_FIELDS; pass
    context_analysis_common.EXCERPT_RATING_FIELDS (or any other rubric's
    field list) for observations using a different rubric -- see
    context_comparisons.RUBRICS. Never duplicated per-rubric logic, just a
    different set of parsed_response keys to read.
    """
    index = index_pairwise_by_assignment_fixed_position(observations, position)
    rows = []
    for (model, evaluation_regime, s1, s2, contrast_id, replicate_id), pair in index.items():
        forward, flipped = pair.get("forward"), pair.get("flipped")
        if forward is None or flipped is None:
            continue
        for category in categories:
            forward_choice = forward["parsed_response"][category]
            flipped_choice = flipped["parsed_response"][category]
            rows.append(
                {
                    "model": model,
                    "evaluation_regime": evaluation_regime,
                    "contrast_id": contrast_id,
                    "dimension": forward["dimension"],
                    "story_1_id": s1,
                    "story_2_id": s2,
                    "position": position,
                    "replicate_id": replicate_id,
                    "category": category,
                    "story1_context_forward": forward["context_a"]["value"],
                    "story1_context_flipped": flipped["context_a"]["value"],
                    "forward_choice": forward_choice,
                    "forward_chosen_story_id": choice_to_story_id(forward, category) or "tie",
                    "flipped_choice": flipped_choice,
                    "flipped_chosen_story_id": choice_to_story_id(flipped, category) or "tie",
                    "changed": forward_choice != flipped_choice,
                }
            )
    return rows


def summarize_grouped_change_rate(rows, group_key, label):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row["changed"])
    print(f"  By {label}:")
    for key, changes in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        rate = 100 * sum(changes) / len(changes)
        print(f"    {key}: n={len(changes)} changed={sum(changes)} ({rate:.1f}%)")


def summarize_pairwise_changed_diagnostic(rows):
    print("\n=== SECONDARY diagnostic: raw A/B/tie changed under forward/flipped (position=story1_as_a only) ===")
    print("(Boolean 'changed' only -- does not say which story/context was preferred. See the PRIMARY")
    print(" directional-effect analysis below for that.)")
    if not rows:
        print("  No matched forward/flipped pairs found.")
        return
    overall_changed = sum(1 for r in rows if r["changed"])
    print(f"  n={len(rows)} matched (category, forward/flipped) comparisons")
    print(f"  Overall: {overall_changed}/{len(rows)} ({100 * overall_changed / len(rows):.1f}%) changed")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["model"], "model")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["evaluation_regime"], "evaluation regime")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["contrast_id"], "contrast")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["category"], "category")


def analyze_directional_pairwise_effects(observations, categories=RATING_FIELDS):
    """PRIMARY pairwise context-effect analysis, in story identity, pooling
    over the counterbalanced display position (see
    context_contrasts.build_contrast_block) and over replicates, stratified
    by evaluation_regime. For each
    (model, evaluation_regime, contrast_id, story_1_id, story_2_id, category),
    estimates:

        P(story_1 preferred | story_1 receives contrast value "a")
      - P(story_1 preferred | story_1 receives contrast value "b")

    Positive means story_1 is favored more often when it carries value "a";
    negative means the opposite. This is directional and never collapses to
    a single changed=True/False boolean -- two scenarios with opposite signs
    are never conflated (see offline verification item D). Naturalistic and
    text_only_invariance observations are never pooled into one estimate.

    categories defaults to the "full" rubric's RATING_FIELDS; pass a
    different rubric's field list (e.g.
    context_analysis_common.EXCERPT_RATING_FIELDS) to analyze observations
    recorded under that rubric instead -- see context_comparisons.RUBRICS.
    This is the one place the set of categories is read from, so no
    per-rubric copy of this function exists.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "tie": 0, "n": 0})
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        for category in categories:
            chosen = choice_to_story_id(obs, category)
            key = (obs["model"], obs["evaluation_regime"], obs["contrast_id"], obs["story_1_id"], obs["story_2_id"], category, obs["assignment"])
            t = tallies[key]
            t["n"] += 1
            if chosen is None:
                t["tie"] += 1
            elif chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            else:
                t["story_2_preferred"] += 1

    by_pair = defaultdict(dict)
    for (model, evaluation_regime, contrast_id, s1, s2, category, assignment), t in tallies.items():
        by_pair[(model, evaluation_regime, contrast_id, s1, s2, category)][assignment] = t

    rows = []
    for (model, evaluation_regime, contrast_id, s1, s2, category), by_assignment in by_pair.items():
        fwd, flp = by_assignment.get("forward"), by_assignment.get("flipped")
        if fwd is None or flp is None:
            continue
        p_a = fwd["story_1_preferred"] / fwd["n"] if fwd["n"] else None
        p_b = flp["story_1_preferred"] / flp["n"] if flp["n"] else None
        effect = (p_a - p_b) if None not in (p_a, p_b) else None
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "contrast_id": contrast_id,
                "story_1_id": s1,
                "story_2_id": s2,
                "category": category,
                "p_story1_preferred_given_value_a": round(p_a, 3) if p_a is not None else "",
                "n_value_a": fwd["n"],
                "tie_n_value_a": fwd["tie"],
                "p_story1_preferred_given_value_b": round(p_b, 3) if p_b is not None else "",
                "n_value_b": flp["n"],
                "tie_n_value_b": flp["tie"],
                "directional_effect_a_minus_b": round(effect, 3) if effect is not None else "",
            }
        )
    return rows


def summarize_directional_effects_by_contrast(directional_rows):
    """Aggregate the per-story-pair directional effect across story pairs,
    per (model, evaluation_regime, contrast_id, category) -- the pairwise
    analogue of the single-text model x dimension x value aggregate.

    Reports heterogeneity alongside the mean (min/max/range across the
    underlying per-pair effects), not the mean alone -- with only 12
    stories/66 pairs, a single mean can obscure whether every pair actually
    behaves similarly (see analyze_per_story_context_effects and
    analyze_leave_one_story_out for finer-grained heterogeneity diagnostics).
    """
    grouped = defaultdict(list)
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        grouped[key].append(row["directional_effect_a_minus_b"])
    return [
        {
            "model": m,
            "evaluation_regime": er,
            "contrast_id": c,
            "category": cat,
            "mean_directional_effect": round(statistics.mean(effects), 3),
            "min_directional_effect": round(min(effects), 3),
            "max_directional_effect": round(max(effects), 3),
            "range_directional_effect": round(max(effects) - min(effects), 3),
            "n_story_pairs": len(effects),
        }
        for (m, er, c, cat), effects in grouped.items()
    ]


def summarize_directional_pairwise_effects(rows, by_contrast_rows):
    print("\n=== PRIMARY pairwise context-effect analysis: directional, in story identity ===")
    print('(P(story_1 preferred | value "a") - P(story_1 preferred | value "b"), position counterbalanced,')
    print(" replicates pooled into the estimate, not discarded; stratified by evaluation_regime)")
    if not rows:
        print("  No complete forward+flipped story-pair blocks found.")
        return
    print(f"  n={len(rows)} (model, evaluation_regime, contrast, story pair, category) directional estimates")
    print("\n  By (model, evaluation_regime, contrast, category), averaged across story pairs:")
    for row in sorted(by_contrast_rows, key=lambda r: (r["model"], r["evaluation_regime"], r["contrast_id"], r["category"])):
        print(
            f"    {row['model']} | {row['evaluation_regime']} | {row['contrast_id']} | {row['category']}: "
            f"mean_effect={row['mean_directional_effect']:+.3f} "
            f"range=[{row['min_directional_effect']:+.3f}, {row['max_directional_effect']:+.3f}] "
            f"(n_story_pairs={row['n_story_pairs']})"
        )


def analyze_pairwise_cell_rates(observations, categories=RATING_FIELDS):
    """Expose the four raw counterbalanced cells separately, never collapsed
    into the pooled directional estimate above.

    For each (model, evaluation_regime, contrast_id, story_1_id, story_2_id,
    category) block, reports story_1_wins/story_2_wins/n for each of the 4
    cells (forward x story1_as_a, forward x story2_as_a, flipped x
    story1_as_a, flipped x story2_as_a) -- "story2_as_a" means story_1 is
    displayed as Story B. Restricted to forced-choice observations (see
    is_forced_choice_pairwise), since ties make a "wins" count ambiguous.

    This is the shared basis for both the raw four-cell-rates output (an
    aggregate CSV is not the only way to see pairwise results) and the
    position/context-x-position analyses below, which are derived from
    exactly these four numbers per block.

    categories defaults to the "full" rubric's RATING_FIELDS; pass a
    different rubric's field list to analyze observations recorded under
    that rubric instead -- see analyze_directional_pairwise_effects.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "n": 0})
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or not is_forced_choice_pairwise(obs):
            continue
        for category in categories:
            chosen = choice_to_story_id(obs, category)
            key = (
                obs["model"], obs["evaluation_regime"], obs["contrast_id"],
                obs["story_1_id"], obs["story_2_id"], category, obs["assignment"], obs["position"],
            )
            t = tallies[key]
            t["n"] += 1
            if chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            elif chosen == obs["story_2_id"]:
                t["story_2_preferred"] += 1

    by_block = defaultdict(dict)
    for (model, evaluation_regime, contrast_id, s1, s2, category, assignment, position), t in tallies.items():
        by_block[(model, evaluation_regime, contrast_id, s1, s2, category)][(assignment, position)] = t

    cell_labels = [
        ("forward", "story1_as_a"), ("forward", "story2_as_a"),
        ("flipped", "story1_as_a"), ("flipped", "story2_as_a"),
    ]
    rows = []
    for (model, evaluation_regime, contrast_id, s1, s2, category), cells in by_block.items():
        if not all(label in cells for label in cell_labels):
            continue  # incomplete block (e.g. a partial/failed run); skip rather than guess
        row = {"model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id,
               "story_1_id": s1, "story_2_id": s2, "category": category}
        for assignment, position in cell_labels:
            t = cells[(assignment, position)]
            row[f"{assignment}_{position}_story_1_wins"] = t["story_1_preferred"]
            row[f"{assignment}_{position}_story_2_wins"] = t["story_2_preferred"]
            row[f"{assignment}_{position}_n"] = t["n"]
        rows.append(row)
    return rows


def analyze_position_and_interaction_effects(cell_rows):
    """Decompose the pooled directional context effect into a position
    effect and a context x position interaction diagnostic, from the four
    raw cells (analyze_pairwise_cell_rates) -- so a pooled context effect
    can never silently hide a large or context-dependent position effect.

    For each block, using p(cell) = story_1_wins / n:
      - context_effect_at_position_a  = p(forward, story1_as_a) - p(flipped, story1_as_a)
      - context_effect_at_position_b  = p(forward, story2_as_a) - p(flipped, story2_as_a)
      - context_x_position_interaction = context_effect_at_position_a - context_effect_at_position_b
      - position_effect_pooled = p(story_1 chosen | displayed as A) - p(story_1 chosen | displayed as B),
          pooling story_1_wins/n across assignment within each position
      - position_effect_under_forward  = p(forward, story1_as_a) - p(forward, story2_as_a)
      - position_effect_under_flipped  = p(flipped, story1_as_a) - p(flipped, story2_as_a)

    This is a descriptive decomposition of one 2x2 (assignment x position)
    table per block, not a new inferential/hierarchical model.
    """
    rows = []
    for row in cell_rows:
        def p(assignment, position):
            n = row[f"{assignment}_{position}_n"]
            return row[f"{assignment}_{position}_story_1_wins"] / n if n else None

        p_fwd_a, p_fwd_b = p("forward", "story1_as_a"), p("forward", "story2_as_a")
        p_flp_a, p_flp_b = p("flipped", "story1_as_a"), p("flipped", "story2_as_a")
        if None in (p_fwd_a, p_fwd_b, p_flp_a, p_flp_b):
            continue

        n_a = row["forward_story1_as_a_n"] + row["flipped_story1_as_a_n"]
        n_b = row["forward_story2_as_a_n"] + row["flipped_story2_as_a_n"]
        wins_a = row["forward_story1_as_a_story_1_wins"] + row["flipped_story1_as_a_story_1_wins"]
        wins_b = row["forward_story2_as_a_story_1_wins"] + row["flipped_story2_as_a_story_1_wins"]
        p_position_a = wins_a / n_a if n_a else None
        p_position_b = wins_b / n_b if n_b else None

        context_effect_at_a = p_fwd_a - p_flp_a
        context_effect_at_b = p_fwd_b - p_flp_b
        rows.append(
            {
                "model": row["model"],
                "evaluation_regime": row["evaluation_regime"],
                "contrast_id": row["contrast_id"],
                "story_1_id": row["story_1_id"],
                "story_2_id": row["story_2_id"],
                "category": row["category"],
                "context_effect_at_position_a": round(context_effect_at_a, 3),
                "context_effect_at_position_b": round(context_effect_at_b, 3),
                "context_x_position_interaction": round(context_effect_at_a - context_effect_at_b, 3),
                "position_effect_pooled": round(p_position_a - p_position_b, 3) if None not in (p_position_a, p_position_b) else "",
                "position_effect_under_forward": round(p_fwd_a - p_fwd_b, 3),
                "position_effect_under_flipped": round(p_flp_a - p_flp_b, 3),
            }
        )
    return rows


def summarize_position_and_interaction_effects(rows):
    print("\n=== PRIMARY: A/B display-position effect and context x position interaction ===")
    print("(a pooled context effect can hide a large or context-dependent position effect; this decomposes")
    print(" one 2x2 assignment x position table per block into both -- descriptive, not a new inferential model)")
    if not rows:
        print("  No complete 4-cell blocks found.")
        return
    print(f"  n={len(rows)} (model, evaluation_regime, contrast, story pair, category) blocks")
    interactions = [r["context_x_position_interaction"] for r in rows]
    positions = [r["position_effect_pooled"] for r in rows if r["position_effect_pooled"] != ""]
    print(f"  context x position interaction: mean={statistics.mean(interactions):+.3f}  "
          f"range=[{min(interactions):+.3f}, {max(interactions):+.3f}]")
    if positions:
        print(f"  position effect (pooled over assignment): mean={statistics.mean(positions):+.3f}  "
              f"range=[{min(positions):+.3f}, {max(positions):+.3f}]")


def analyze_per_story_context_effects(directional_rows):
    """Per-story summary of the directional context effect across a story's
    opponents, preserving every underlying per-pair value.

    66 unordered story pairs are NOT 66 independent samples: each of the 12
    stories appears in 11 of them, so one unusual story can create the
    appearance of a repeated effect across many pairs. This reorganizes the
    existing per-pair directional_effect_a_minus_b (analyze_directional_pairwise_effects)
    by story identity so that question is directly answerable: does a
    context value generally help whichever story holds it, or is one
    specific story (e.g. it responds unusually across most/all of its 11
    opponents) driving an apparently repeated effect? The per-pair effect is
    identical from either story's point of view in a two-outcome forced
    choice (holding value "a" either helps or doesn't, symmetrically for
    whichever story holds it) -- so grouping by story here reveals whether
    that "holding value a helps" pattern is uniform across a story's
    opponents or concentrated/absent for a particular one.
    """
    grouped = defaultdict(list)
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        base_key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        grouped[(*base_key, row["story_1_id"])].append((row["story_2_id"], row["directional_effect_a_minus_b"]))
        grouped[(*base_key, row["story_2_id"])].append((row["story_1_id"], row["directional_effect_a_minus_b"]))

    rows = []
    for (model, evaluation_regime, contrast_id, category, story_id), pair_effects in grouped.items():
        effects = [e for _, e in pair_effects]
        rows.append(
            {
                "model": model,
                "evaluation_regime": evaluation_regime,
                "contrast_id": contrast_id,
                "category": category,
                "story_id": story_id,
                "n_opponents": len(pair_effects),
                "mean_effect": round(statistics.mean(effects), 3),
                "min_effect": round(min(effects), 3),
                "max_effect": round(max(effects), 3),
                "range_effect": round(max(effects) - min(effects), 3),
                "per_opponent_effects": "; ".join(f"{opp}={e:+.3f}" for opp, e in sorted(pair_effects)),
            }
        )
    return rows


def summarize_per_story_context_effects(rows):
    print("\n=== Per-story context-effect summary (heterogeneity across a story's opponents) ===")
    print("(66 pairs are not 66 independent units -- each of 12 stories appears in 11 pairs; this checks")
    print(" whether the aggregate effect is uniform across stories or concentrated in one unusual story)")
    if not rows:
        print("  No per-story effects available.")
        return
    by_widest_range = sorted(rows, key=lambda r: -r["range_effect"])[:5]
    print("  Widest within-story range across opponents (top 5, largest heterogeneity first):")
    for row in by_widest_range:
        print(
            f"    {row['model']} | {row['evaluation_regime']} | {row['contrast_id']} | {row['category']} | "
            f"{row['story_id']}: mean={row['mean_effect']:+.3f} range=[{row['min_effect']:+.3f}, {row['max_effect']:+.3f}] "
            f"(n_opponents={row['n_opponents']})"
        )


def analyze_leave_one_story_out(directional_rows):
    """Sensitivity/robustness diagnostic: recompute each aggregate context
    effect after excluding every pair involving each story in turn.

    Descriptive only -- NOT a formal correction for the non-independence of
    story pairs (see analyze_per_story_context_effects docstring). Answers:
    is the headline aggregate (mean directional effect across all pairs for
    a model/evaluation_regime/contrast/category) carried largely by one
    unusual story, or does it hold up under removing any single story?
    """
    grouped = defaultdict(list)
    all_stories = defaultdict(set)
    for row in directional_rows:
        if row["directional_effect_a_minus_b"] == "":
            continue
        key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        grouped[key].append(row)
        all_stories[key].add(row["story_1_id"])
        all_stories[key].add(row["story_2_id"])

    rows = []
    for key, pair_rows in grouped.items():
        model, evaluation_regime, contrast_id, category = key
        full_mean = statistics.mean(r["directional_effect_a_minus_b"] for r in pair_rows)
        rows.append(
            {
                "model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id, "category": category,
                "excluded_story_id": "(none -- full aggregate)",
                "mean_directional_effect": round(full_mean, 3), "n_story_pairs": len(pair_rows),
            }
        )
        for excluded in sorted(all_stories[key]):
            remaining = [r for r in pair_rows if r["story_1_id"] != excluded and r["story_2_id"] != excluded]
            if not remaining:
                continue
            rows.append(
                {
                    "model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id, "category": category,
                    "excluded_story_id": excluded,
                    "mean_directional_effect": round(statistics.mean(r["directional_effect_a_minus_b"] for r in remaining), 3),
                    "n_story_pairs": len(remaining),
                }
            )
    return rows


def summarize_leave_one_story_out(rows):
    print("\n=== Leave-one-story-out sensitivity (descriptive robustness check, not a formal correction) ===")
    if not rows:
        print("  No leave-one-story-out results available.")
        return
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["evaluation_regime"], row["contrast_id"], row["category"])
        by_key[key][row["excluded_story_id"]] = row["mean_directional_effect"]
    for key, by_excluded in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        full = by_excluded.get("(none -- full aggregate)")
        loo_values = [v for k, v in by_excluded.items() if k != "(none -- full aggregate)"]
        if full is None or not loo_values:
            continue
        print(
            f"  {key[0]} | {key[1]} | {key[2]} | {key[3]}: full_aggregate={full:+.3f}  "
            f"leave-one-out range=[{min(loo_values):+.3f}, {max(loo_values):+.3f}]"
        )


def compare_pairwise_regimes(directional_by_contrast_rows):
    """For each (model, contrast_id, category) present under BOTH evaluation
    regimes, report the naturalistic effect, the text_only_invariance
    effect, and a descriptive attenuation label. Rows missing one regime are
    skipped (nothing to compare)."""
    by_key = defaultdict(dict)
    for row in directional_by_contrast_rows:
        key = (row["model"], row["contrast_id"], row["category"])
        by_key[key][row["evaluation_regime"]] = row["mean_directional_effect"]

    rows = []
    for (model, contrast_id, category), by_regime in by_key.items():
        if "naturalistic" not in by_regime or "text_only_invariance" not in by_regime:
            continue
        nat, inv = by_regime["naturalistic"], by_regime["text_only_invariance"]
        rows.append(
            {
                "model": model,
                "contrast_id": contrast_id,
                "category": category,
                "effect_naturalistic": nat,
                "effect_text_only_invariance": inv,
                "attenuation": describe_attenuation(nat, inv),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# SECONDARY, optional hedging/indifference diagnostic: choice_mode="tie_allowed"
# observations only (see context_trials.build_tie_allowed_pairwise_trials).
# Never pooled with the PRIMARY forced-choice observations above -- this
# section only ever reads obs with choice_mode == "tie_allowed", the primary
# analyses above only ever read is_forced_choice_pairwise(obs). This
# measures how often the model declines to state a strict preference, and
# whether context shifts that tendency -- deliberately called "tie rate" /
# "hedging rate", never "uncertainty" in a strong psychological sense.
# ---------------------------------------------------------------------------

def analyze_tie_allowed_diagnostic(observations, categories=RATING_FIELDS):
    """P(story_1 chosen) / P(story_2 chosen) / P(tie), in story identity, per
    (model, evaluation_regime, contrast_id, story_1_id, story_2_id, category,
    assignment) -- the tie-allowed analogue of analyze_pairwise_cell_rates,
    but reporting rates (incl. tie) rather than forced win counts, since ties
    are a real, informative outcome here rather than an excluded case.

    categories defaults to the "full" rubric's RATING_FIELDS -- see
    analyze_directional_pairwise_effects.
    """
    tallies = defaultdict(lambda: {"story_1_preferred": 0, "story_2_preferred": 0, "tie": 0, "n": 0})
    for obs in observations.values():
        if obs["type"] != "context_pairwise" or obs.get("choice_mode") != "tie_allowed":
            continue
        for category in categories:
            chosen = choice_to_story_id(obs, category)
            key = (obs["model"], obs["evaluation_regime"], obs["contrast_id"],
                   obs["story_1_id"], obs["story_2_id"], category, obs["assignment"])
            t = tallies[key]
            t["n"] += 1
            if chosen is None:
                t["tie"] += 1
            elif chosen == obs["story_1_id"]:
                t["story_1_preferred"] += 1
            else:
                t["story_2_preferred"] += 1

    rows = []
    for (model, evaluation_regime, contrast_id, s1, s2, category, assignment), t in tallies.items():
        n = t["n"]
        rows.append(
            {
                "model": model, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id,
                "story_1_id": s1, "story_2_id": s2, "category": category, "assignment": assignment,
                "p_story_1_chosen": round(t["story_1_preferred"] / n, 3) if n else "",
                "p_story_2_chosen": round(t["story_2_preferred"] / n, 3) if n else "",
                "p_tie": round(t["tie"] / n, 3) if n else "",
                "n": n,
            }
        )
    return rows


def summarize_tie_allowed_diagnostic(rows):
    print("\n=== SECONDARY, optional: tie-allowed hedging/indifference diagnostic ===")
    print("(choice_mode=\"tie_allowed\" observations only, never pooled with the primary forced-choice results;")
    print(" reports how often the model declines a strict preference, and whether context shifts that rate --")
    print(" \"tie rate\"/\"hedging rate\", not a claim about a psychological state of uncertainty)")
    if not rows:
        print("  No tie-allowed observations found (this diagnostic is optional and not run by default).")
        return
    tie_rates = [r["p_tie"] for r in rows if r["p_tie"] != ""]
    by_assignment = defaultdict(list)
    for r in rows:
        if r["p_tie"] != "":
            by_assignment[r["assignment"]].append(r["p_tie"])
    print(f"  n={len(rows)} (model, evaluation_regime, contrast, story pair, category, assignment) cells")
    print(f"  Overall tie rate: mean={statistics.mean(tie_rates):.3f}  range=[{min(tie_rates):.3f}, {max(tie_rates):.3f}]")
    for assignment, rates in sorted(by_assignment.items()):
        print(f"    {assignment}: mean tie rate={statistics.mean(rates):.3f} (n={len(rates)})")


# ---------------------------------------------------------------------------
# context_prompt: does a genuinely extraneous prompt-level sentence change
# the decision, holding story identity/position/story-level-context fixed?
# (Prompt-scope context is not attributed to either story, so the
# assignment x position counterbalance above doesn't apply here.) Stratified
# by evaluation_regime like everything else.
# ---------------------------------------------------------------------------

def index_prompt_by_value(observations):
    index = {}
    for obs in observations.values():
        if obs["type"] != "context_prompt":
            continue
        key = (obs["model"], obs["evaluation_regime"], obs["story_a_id"], obs["story_b_id"], obs["contrast_id"], obs["replicate_id"])
        index.setdefault(key, {})[obs["value"]] = obs
    return index


def analyze_prompt_context_effects(observations, categories=RATING_FIELDS):
    """categories defaults to the "full" rubric's RATING_FIELDS -- see
    analyze_directional_pairwise_effects."""
    index = index_prompt_by_value(observations)
    rows = []

    for (model, evaluation_regime, story_a, story_b, contrast_id, replicate_id), values in index.items():
        if len(values) < 2:
            continue
        baseline_id = next((v for v, obs in values.items() if not obs.get("prompt_context_text")), sorted(values)[0])
        baseline_obs = values[baseline_id]
        for value_id, obs in values.items():
            if value_id == baseline_id:
                continue
            for category in categories:
                baseline_choice = baseline_obs["parsed_response"][category]
                treatment_choice = obs["parsed_response"][category]
                rows.append(
                    {
                        "model": model,
                        "evaluation_regime": evaluation_regime,
                        "contrast_id": contrast_id,
                        "story_a_id": story_a,
                        "story_b_id": story_b,
                        "replicate_id": replicate_id,
                        "category": category,
                        "baseline_value": baseline_id,
                        "treatment_value": value_id,
                        "baseline_choice": baseline_choice,
                        "treatment_choice": treatment_choice,
                        "changed": baseline_choice != treatment_choice,
                    }
                )
    return rows


def summarize_prompt_context_effects(rows):
    print("\n=== Context effect from extraneous prompt-level sentences (baseline vs treatment) ===")
    if not rows:
        print("  No matched baseline/treatment prompt-context pairs found.")
        return

    overall_changed = sum(1 for r in rows if r["changed"])
    print(f"  n={len(rows)} matched (category, baseline/treatment) comparisons")
    print(f"  Overall: {overall_changed}/{len(rows)} ({100 * overall_changed / len(rows):.1f}%) changed")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["model"], "model")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["evaluation_regime"], "evaluation regime")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["contrast_id"], "contrast")
    print()
    summarize_grouped_change_rate(rows, lambda r: r["category"], "category")
