"""v3 stress family: irrelevant-evidence DOSE RESPONSE.

Design (per evaluator profile):
    66 story pairs x 6 dose levels x 8 interventions x 4 counterbalance
    cells (favoured passage = story 1 / story 2, x display position)
    = 12,672 unique cells; DOSE_REPLICATES replicates each.

Cue direction and display position are the same two counterbalance
dimensions the primary uses (assignment x position). Neither is redundant:
direction cancels each pair's underlying preference (D_pair), position
cancels A/B position bias. The zero-dose reference is the primary
experiment's matched no-context condition for the same intervention; it
is not re-collected here.

Numeric cue (frozen, one representation, fixed panel of 100):
    "In an independent reader evaluation, {n} out of 100 readers preferred
     Passage {FAV} and {100-n} out of 100 readers preferred Passage {UNFAV}."
with n in (51, 60, 70, 80, 90, 99). The intervention sentence is the
primary's frozen wording, unchanged (I4 already tells the judge that any
contextual labels were assigned at random by the experimenter).
"""

import json

from context_trials import ITEMS_FILE, load_items, story_pairs
from controllability_v3_design import INTERVENTION_IDS
from controllability_v3_trials import load_story_texts
import controllability_v3_stress_design as sd

DOSE_CUE_ID = "reader_consensus_numeric"
DOSE_REPLICATES = 5
N_PAIRS, N_DOSES, N_INTERVENTIONS, N_CELLS = 66, len(sd.DOSE_GRID), len(INTERVENTION_IDS), 4
DOSE_UNIQUE_CELLS = N_PAIRS * N_DOSES * N_INTERVENTIONS * N_CELLS   # 12,672
assert DOSE_UNIQUE_CELLS == 12672


def build_dose_trials(items=None, texts=None):
    items = items if items is not None else load_items(ITEMS_FILE)
    texts = texts if texts is not None else load_story_texts(items)
    trials = []
    for s1, s2 in story_pairs(items):
        for n in sd.DOSE_GRID:
            for iid in INTERVENTION_IDS:
                extra = {"dose": n, "dose_panel_size": sd.DOSE_PANEL_SIZE, "dose_logit": sd.dose_logit(n), "cue_family": "reader_consensus"}
                trials.extend(sd.build_stress_block(sd.FAMILY_DOSE, s1["id"], s2["id"], DOSE_CUE_ID, iid, sd.dose_id(n),
                                                    lambda fav, unfav, n=n: sd.dose_intro(n, fav, unfav), texts, extra))
    return trials


def assert_dose_manifest_well_formed(trials):
    blocks = sd.assert_stress_blocks_well_formed(trials, sd.FAMILY_DOSE)
    if len(trials) != DOSE_UNIQUE_CELLS or len(blocks) != DOSE_UNIQUE_CELLS // 4:
        raise ValueError(f"dose: expected {DOSE_UNIQUE_CELLS} cells in {DOSE_UNIQUE_CELLS // 4} blocks, found {len(trials)} in {len(blocks)}")
    from collections import Counter
    per_dose = Counter(t["dose"] for t in trials)
    if set(per_dose) != set(sd.DOSE_GRID) or len(set(per_dose.values())) != 1:
        raise ValueError(f"dose: grid not balanced: {dict(per_dose)}")
    per_dir = Counter((t["dose"], t["assignment"]) for t in trials)
    per_pos = Counter((t["dose"], t["position"]) for t in trials)
    if len(set(per_dir.values())) != 1 or len(set(per_pos.values())) != 1:
        raise ValueError("dose: cue direction / display position not balanced within every dose")
    per_int = Counter((t["dose"], t["intervention_id"]) for t in trials)
    if set(i for _, i in per_int) != set(INTERVENTION_IDS) or len(set(per_int.values())) != 1:
        raise ValueError("dose: interventions not balanced within every dose")
    for t in trials:
        if t["variant"] != sd.dose_id(t["dose"]) or t["cue_id"] != DOSE_CUE_ID:
            raise ValueError(f"dose: {t['trial_id']} variant/cue inconsistent")
        fav, unfav = sd.favoured_letters(t["story_1_id"], t["story_2_id"], t["assignment"], t["position"])
        if t["intro"] != sd.dose_intro(t["dose"], fav, unfav):
            raise ValueError(f"dose: {t['trial_id']} intro is not the frozen dose sentence")
        # direction reversal: the flipped cell of the same (pair, dose, I, position) names the other passage
    by_key = {(t["pair_id"], t["dose"], t["intervention_id"], t["position"], t["assignment"]): t for t in trials}
    for (pair, n, iid, pos, asg), t in by_key.items():
        other = by_key[(pair, n, iid, pos, "flipped" if asg == "forward" else "forward")]
        fav_t, _ = sd.favoured_letters(t["story_1_id"], t["story_2_id"], asg, pos)
        fav_o, _ = sd.favoured_letters(other["story_1_id"], other["story_2_id"], other["assignment"], pos)
        if fav_t == fav_o:
            raise ValueError(f"dose: direction reversal failed for {t['trial_id']}")
    return blocks


def design_arithmetic(replicates=DOSE_REPLICATES):
    import math
    n_pairs = math.comb(12, 2)
    cells = n_pairs * len(sd.DOSE_GRID) * len(INTERVENTION_IDS) * 2 * 2
    return {"story_pairs": n_pairs, "doses": len(sd.DOSE_GRID), "interventions": len(INTERVENTION_IDS), "cells_per_block": 4,
            "unique_cells": cells, "replicates": replicates, "judgments": cells * replicates,
            "judgments_at_10_replicates": cells * 10}


def main():
    trials = build_dose_trials()
    assert_dose_manifest_well_formed(trials)
    sd.write_manifest(trials, sd.DOSE_TRIALS_FILE)
    a = design_arithmetic()
    print(f"dose-response manifest: {len(trials)} unique cells -> {sd.DOSE_TRIALS_FILE}")
    print(f"grid: {sd.DOSE_GRID} (out of {sd.DOSE_PANEL_SIZE})  replicates: {a['replicates']}  judgments: {a['judgments']:,}  (at 10 reps: {a['judgments_at_10_replicates']:,})")


if __name__ == "__main__":
    main()
