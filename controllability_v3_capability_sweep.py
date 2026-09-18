"""v3 experimental family: PAIRED evaluator capability / reasoning sweep
(capability_sweep).

Question: as evaluator capability/reasoning changes, does selective
suppression improve, or does the judge simply change its underlying
literary preferences? Every evaluator profile sees the SAME scientific
cells (one manifest, frozen); profiles differ only in the
evaluation_observation_id suffix, so every comparison is within-cell.

Design (frozen defaults; DESIGN):
  pairs        n_pairs_per_stratum x 3 strata of blind baseline strength
               (strong / moderately ambiguous / highly ambiguous) when a
               blind baseline is available; otherwise a seeded deterministic
               subset balanced over stories, recorded as such
  interventions I0, I1, I4, I5, I7
  conditions   matched no-context (2 positions)
               ordinary context x 5 cues (4 cells each)
               strongest frozen adversarial attack per (cue, intervention)
               (4 cells each; included only when the stress attacks stage is
               frozen -- recorded in the manifest metadata)
               dose 51 / 80 / 99 (4 cells each)
  replicates   3
Cells per pair per intervention without attacks: 2 + 20 + 12 = 34;
with attacks: 54.

Ids: v3c::capability_sweep::{pair}::{cue|none}::{assignment|none}::{position}::{I}::{variant}::{ctx|noctx}
"""

import hashlib
import json
import random

import controllability_v3_stress_design as sd
from context_trials import ITEMS_FILE, load_items, story_pairs
from controllability_v3_design import CUE_ORDER, NO_CONTEXT_INTRO, build_prompt, context_intro, instruction_sentence
from controllability_v3_trials import load_contrasts, load_story_texts

FAMILY = "capability_sweep"
ID_PREFIX = "v3c"
TRIALS_FILE = "data/controllability_v3_capability_trials.jsonl"
META_FILE = "data/controllability_v3_capability_manifest_meta.json"

DESIGN = {
    "n_pairs_per_stratum": 8,
    "strata": ["strong", "moderate", "ambiguous"],
    "interventions": ["I0", "I1", "I4", "I5", "I7"],
    "cues": list(CUE_ORDER),
    "conditions": ["nocontext", "ordinary", "attack", "dose"],
    "doses": [51, 80, 99],
    "replicates": 3,
    "pair_selection_seed": 20260920,
    "profiles": ["deepseek_flash_low", "deepseek_flash_medium", "deepseek_flash_high"],
    "pair_selection": "blind-baseline strength tertiles (primary no-context I0) when available, else seeded story-balanced subset",
}


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_trial_id(pair_id, cue_id, assignment, position, intervention_id, variant, context_present):
    return "::".join([ID_PREFIX, FAMILY, pair_id, cue_id or "none", assignment or "none", position, intervention_id, variant, "ctx" if context_present else "noctx"])


def parse_observation_id(oid):
    parts = oid.split("::")
    if len(parts) != 10 or parts[0] != ID_PREFIX or parts[1] != FAMILY:
        raise ValueError(f"not a capability-sweep observation id: {oid!r}")
    return {"pair_id": parts[2], "cue_id": None if parts[3] == "none" else parts[3], "assignment": None if parts[4] == "none" else parts[4],
            "position": parts[5], "intervention_id": parts[6], "variant": parts[7], "context_present": parts[8] == "ctx", "replicate_number": int(parts[9][1:]),
            "trial_id": "::".join(parts[:9])}


def make_observation_id(trial_id, replicate):
    return f"{trial_id}::r{replicate}"


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def select_pairs(items, design, baseline_strength=None):
    """[(s1, s2, stratum)]; deterministic. With a blind baseline: tertiles of
    strength, n per stratum, chosen to spread stories (greedy by least-used
    story). Without: seeded sample of 3 x n pairs, labelled 'seeded_*'."""
    pairs = [(a["id"], b["id"]) for a, b in story_pairs(items)]
    n = design["n_pairs_per_stratum"]
    if baseline_strength:
        ordered = sorted(pairs, key=lambda p: (baseline_strength[f"{p[0]}_vs_{p[1]}"], p))
        k = len(ordered) // 3
        buckets = {"ambiguous": ordered[:k], "moderate": ordered[k:2 * k], "strong": ordered[2 * k:]}
        chosen = []
        for stratum in design["strata"]:
            use = {}
            picked = []
            for _ in range(n):
                cand = min((p for p in buckets[stratum] if p not in picked), key=lambda p: (use.get(p[0], 0) + use.get(p[1], 0), p))
                picked.append(cand)
                use[cand[0]] = use.get(cand[0], 0) + 1
                use[cand[1]] = use.get(cand[1], 0) + 1
            chosen += [(p[0], p[1], stratum) for p in picked]
        return chosen, "blind_baseline_tertiles"
    rng = random.Random(design["pair_selection_seed"])
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    picked = sorted(shuffled[:3 * n])
    return [(p[0], p[1], f"seeded_{i % 3}") for i, p in enumerate(picked)], "seeded_story_balanced_subset"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _cell(pair, stratum, cue_id, assignment, position, iid, variant, context_present, intro, texts, extra):
    s1, s2 = pair
    story_a, story_b = sd.display(s1, s2, position)
    prompt = build_prompt(intro, instruction_sentence(iid), texts[story_a], texts[story_b])
    return {"experiment_id": "context_controllability_v3_capability", "family": FAMILY, "type": "context_pairwise", "choice_mode": "forced", "response_format": "plain_ab",
            "trial_id": make_trial_id(f"{s1}_vs_{s2}", cue_id, assignment, position, iid, variant, context_present),
            "block_id": "::".join([ID_PREFIX, FAMILY, f"{s1}_vs_{s2}", cue_id or "none", iid, variant]),
            "pair_id": f"{s1}_vs_{s2}", "pair_stratum": stratum, "story_1_id": s1, "story_2_id": s2, "story_a_id": story_a, "story_b_id": story_b,
            "cue_id": cue_id, "assignment": assignment, "position": position, "intervention_id": iid, "instruction_id": iid, "context_present": context_present,
            "variant": variant, "condition": extra.get("condition"), "intro": intro, "prompt_sha256": sha256_hex(prompt), **extra}


def build_capability_trials(design, items=None, texts=None, contrasts=None, baseline_strength=None, strongest_attacks=None):
    """strongest_attacks: {(cue_id, intervention_id): attack row (template, attack_id)} or None."""
    items = items if items is not None else load_items(ITEMS_FILE)
    texts = texts if texts is not None else load_story_texts(items)
    contrasts = contrasts if contrasts is not None else load_contrasts()
    by_cue = {c["contrast_id"]: c for c in contrasts}
    chosen, basis = select_pairs(items, design, baseline_strength)
    trials = []
    for s1, s2, stratum in chosen:
        pair = (s1, s2)
        for iid in design["interventions"]:
            if "nocontext" in design["conditions"]:
                for position in sd.POSITIONS:
                    trials.append(_cell(pair, stratum, None, None, position, iid, "nocontext", False, NO_CONTEXT_INTRO, texts, {"condition": "nocontext"}))
            for cue_id in design["cues"]:
                c = by_cue[cue_id]
                for assignment in sd.ASSIGNMENTS:
                    for position in sd.POSITIONS:
                        fav, unfav = sd.favoured_letters(s1, s2, assignment, position)
                        if "ordinary" in design["conditions"]:
                            clauses = {fav: c["a_clause"], unfav: c["b_clause"]}
                            intro = context_intro(clauses["A"], clauses["B"])
                            trials.append(_cell(pair, stratum, cue_id, assignment, position, iid, "ordinary", True, intro, texts,
                                                {"condition": "ordinary", "context_a": {"value": c["a"] if fav == "A" else c["b"], "clause": clauses["A"]},
                                                 "context_b": {"value": c["a"] if fav == "B" else c["b"], "clause": clauses["B"]}}))
                        if "attack" in design["conditions"] and strongest_attacks and (cue_id, iid) in strongest_attacks:
                            atk = strongest_attacks[(cue_id, iid)]
                            trials.append(_cell(pair, stratum, cue_id, assignment, position, iid, atk["attack_id"], True, sd.fill_attack_template(atk["template"], fav, unfav), texts,
                                                {"condition": "attack", "attack_id": atk["attack_id"], "template_sha256": atk["template_sha256"]}))
            if "dose" in design["conditions"]:
                for n in design["doses"]:
                    for assignment in sd.ASSIGNMENTS:
                        for position in sd.POSITIONS:
                            fav, unfav = sd.favoured_letters(s1, s2, assignment, position)
                            trials.append(_cell(pair, stratum, "reader_consensus_numeric", assignment, position, iid, sd.dose_id(n), True, sd.dose_intro(n, fav, unfav), texts,
                                                {"condition": "dose", "dose": n, "dose_logit": sd.dose_logit(n)}))
    meta = {"pair_selection_basis": basis, "pairs": [{"story_1_id": s1, "story_2_id": s2, "stratum": st} for s1, s2, st in chosen],
            "attack_condition_included": bool(strongest_attacks), "n_cells": len(trials), "design": design}
    return trials, meta


def assert_capability_manifest_well_formed(trials, design):
    from collections import Counter
    ids = [t["trial_id"] for t in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("capability manifest has duplicate trial ids")
    for t in trials:
        p = parse_observation_id(make_observation_id(t["trial_id"], 1))
        for key in ("pair_id", "cue_id", "assignment", "position", "intervention_id", "variant", "context_present"):
            if p[key] != t[key]:
                raise ValueError(f"{t['trial_id']}: id component {key} disagrees with the row")
        if t["intervention_id"] not in design["interventions"]:
            raise ValueError(f"{t['trial_id']}: intervention not in the design")
    per_pair_int = Counter((t["pair_id"], t["intervention_id"]) for t in trials)
    if len(set(per_pair_int.values())) != 1:
        raise ValueError("capability manifest is not balanced over pair x intervention")
    for cond in ("ordinary", "dose", "attack"):
        blocks = Counter(t["block_id"] for t in trials if t["condition"] == cond)
        if any(n != 4 for n in blocks.values()):
            raise ValueError(f"{cond} blocks must have 4 counterbalance cells")
    nblocks = Counter(t["block_id"] for t in trials if t["condition"] == "nocontext")
    if any(n != 2 for n in nblocks.values()):
        raise ValueError("no-context blocks must have both positions")
    return per_pair_int


def design_arithmetic(design, attacks_included=False):
    n_pairs = design["n_pairs_per_stratum"] * len(design["strata"])
    per = 0
    if "nocontext" in design["conditions"]:
        per += 2
    if "ordinary" in design["conditions"]:
        per += 4 * len(design["cues"])
    if "attack" in design["conditions"] and attacks_included:
        per += 4 * len(design["cues"])
    if "dose" in design["conditions"]:
        per += 4 * len(design["doses"])
    cells = n_pairs * len(design["interventions"]) * per
    return {"pairs": n_pairs, "interventions": len(design["interventions"]), "cells_per_pair_intervention": per, "unique_cells": cells,
            "replicates": design["replicates"], "judgments_per_profile": cells * design["replicates"],
            "profiles": len(design["profiles"]), "judgments_all_profiles": cells * design["replicates"] * len(design["profiles"]),
            "attacks_included": attacks_included}
