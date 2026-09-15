"""Pure-Python tie-aware rank statistics, plus the researcher-reference
scoring helper and the purely descriptive naturalistic-vs-invariance
attenuation label.

None of the rank statistics below ever use story ID, filename, insertion
order, or any other arbitrary field to break a tie -- order of the `items`
argument never affects the result, only the scores/ranks looked up by item
identity. Model rating ties are legitimate throughout this benchmark and
are never broken artificially.
"""

from collections import defaultdict


def average_ranks(values, items):
    """1-based average rank per item (rank 1 = highest score). Tied items
    share the mean of the rank positions their group spans."""
    ordered = sorted(items, key=lambda it: -values[it])
    ranks = {}
    i = 0
    n = len(ordered)
    while i < n:
        j = i
        while j + 1 < n and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = avg_rank
        i = j + 1
    return ranks


def tied_groups(values, items):
    """Best-first list of tie groups: [{"rank_position", "story_ids", "score"}, ...].
    story_ids within one group are sorted alphabetically for STABLE DISPLAY
    ONLY -- every member of a group shares the same rank_position and score,
    so this ordering never affects any statistic."""
    items = list(items)
    ranks = average_ranks(values, items)
    by_score = defaultdict(list)
    for it in items:
        by_score[values[it]].append(it)
    groups = []
    for score in sorted(by_score, reverse=True):
        members = sorted(by_score[score])
        groups.append({"rank_position": ranks[members[0]], "story_ids": members, "score": score})
    return groups


def pearson_correlation(xs, ys):
    n = len(xs)
    if n == 0:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def spearman_tie_aware(values_a, values_b, items):
    """SECONDARY rank statistic. Correct under ties: Pearson correlation of
    the two sides' average ranks (the standard tie-corrected formula) --
    never the tie-free shortcut formula."""
    items = list(items)
    if len(items) < 2:
        return None
    ranks_a = average_ranks(values_a, items)
    ranks_b = average_ranks(values_b, items)
    return pearson_correlation([ranks_a[it] for it in items], [ranks_b[it] for it in items])


def kendall_tau_b(values_a, values_b, items):
    """PRIMARY rank statistic. Kendall's tau-b: a pair tied on either side is
    excluded from the concordant/discordant count and folded into that
    side's tie term, rather than forced into an arbitrary order (tau-a would
    require that). Returns None if undefined (fewer than 2 items, or one
    side has every pair tied)."""
    items = list(items)
    n = len(items)
    if n < 2:
        return None
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_diff = values_a[items[i]] - values_a[items[j]]
            b_diff = values_b[items[i]] - values_b[items[j]]
            if a_diff == 0 and b_diff == 0:
                ties_a += 1
                ties_b += 1
            elif a_diff == 0:
                ties_a += 1
            elif b_diff == 0:
                ties_b += 1
            elif (a_diff > 0) == (b_diff > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    denom = ((n0 - ties_a) * (n0 - ties_b)) ** 0.5
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def human_reference_scores(ranking):
    """Turn the researcher's best-first ordinal reference list into
    {item: score} (higher = more preferred), so it can feed the same
    tie-aware functions above as a possibly-tied model score dict. The
    reference ordering itself has no ties -- it's an ordinal relation only
    (A > B > C...), not a cardinal preference intensity."""
    n = len(ranking)
    return {story_id: n - i for i, story_id in enumerate(ranking)}


def pairwise_diagnostics_from_scores(values, human_pairs):
    """For every known human judgment (winner, loser) with both stories
    present in `values`, classify using the model's raw scores: concordant
    (model score agrees with the human winner), discordant (disagrees), or
    model_tied (the model gave both stories the exact same score -- there is
    no strict model preference to compare, and this is never converted into
    a fabricated win/loss). Returns (concordant, discordant, model_tied, checked).
    """
    concordant = discordant = model_tied = 0
    for winner, loser in human_pairs:
        if winner not in values or loser not in values:
            continue
        if values[winner] == values[loser]:
            model_tied += 1
        elif values[winner] > values[loser]:
            concordant += 1
        else:
            discordant += 1
    return concordant, discordant, model_tied, concordant + discordant + model_tied


# ---------------------------------------------------------------------------
# Descriptive-only naturalistic-vs-invariance comparison. No new statistical
# model: a plain magnitude/sign comparison, used only to label an effect as
# attenuated/amplified/reversed/unchanged under text-only instructions.
# ---------------------------------------------------------------------------

def describe_attenuation(naturalistic_effect, invariance_effect):
    """Purely descriptive label comparing an invariance-regime effect to the
    same effect under naturalistic framing. Not a significance test."""
    if naturalistic_effect == 0:
        return "no_naturalistic_effect" if invariance_effect == 0 else "invariance_effect_only"
    same_sign = (naturalistic_effect > 0) == (invariance_effect > 0)
    if invariance_effect != 0 and not same_sign:
        return "reversed"
    ratio = abs(invariance_effect) / abs(naturalistic_effect)
    if ratio < 0.5:
        return "attenuated"
    if ratio > 1.5:
        return "amplified"
    return "unchanged"
