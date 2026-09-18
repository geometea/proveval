"""Dose-response manifest: exact counts, balanced direction/position/
intervention within every dose, direction reversal, frozen sentence,
byte-stable regeneration, prompt hashes, loud validation failures."""

import os
from collections import Counter

import pytest

import controllability_v3_stress_design as sd
import controllability_v3_stress_dose as dose
from controllability_v3_design import INTERVENTION_IDS
from controllability_v3_trials import load_story_texts, verify_prompt_hashes


@pytest.fixture(scope="module")
def trials():
    return dose.build_dose_trials()


class TestDesign:
    def test_counts(self, trials):
        assert len(trials) == 12672 == dose.DOSE_UNIQUE_CELLS
        assert dose.design_arithmetic() == {"story_pairs": 66, "doses": 6, "interventions": 8, "cells_per_block": 4, "unique_cells": 12672,
                                            "replicates": 5, "judgments": 63360, "judgments_at_10_replicates": 126720}

    def test_balance_within_every_dose(self, trials):
        assert Counter(t["dose"] for t in trials) == {d: 2112 for d in sd.DOSE_GRID}
        assert len(set(Counter((t["dose"], t["assignment"]) for t in trials).values())) == 1
        assert len(set(Counter((t["dose"], t["position"]) for t in trials).values())) == 1
        assert {i for _, i in Counter((t["dose"], t["intervention_id"]) for t in trials)} == set(INTERVENTION_IDS)

    def test_direction_reversal_names_the_other_passage(self, trials):
        by = {(t["pair_id"], t["dose"], t["intervention_id"], t["position"], t["assignment"]): t for t in trials}
        for (pair, d, i, pos, asg), t in list(by.items())[:500]:
            other = by[(pair, d, i, pos, "flipped" if asg == "forward" else "forward")]
            assert t["intro"] != other["intro"]
            assert t["intro"].replace("Passage A", "X").replace("Passage B", "Passage A").replace("X", "Passage B") == other["intro"]

    def test_sentence_is_the_frozen_template(self, trials):
        t = trials[0]
        assert t["intro"].startswith("I'd like your view on two passages. In an independent reader evaluation, ")
        assert f"{t['dose']} out of 100 readers preferred Passage" in t["intro"]
        assert "magazine" not in t["intro"] and "prize" not in t["intro"]

    def test_manifest_validates_and_prompts_hash(self, trials):
        dose.assert_dose_manifest_well_formed(trials)
        verify_prompt_hashes(trials[:200], load_story_texts())

    def test_on_disk_manifest_is_byte_identical_to_regeneration(self, trials, tmp_path):
        out = tmp_path / "dose.jsonl"
        sd.write_manifest(trials, str(out))
        assert out.read_bytes() == open(sd.DOSE_TRIALS_FILE, "rb").read()


class TestValidationFailsLoudly:
    def test_missing_cell(self, trials):
        with pytest.raises(ValueError):
            dose.assert_dose_manifest_well_formed(trials[1:])

    def test_tampered_intro(self, trials):
        bad = [dict(trials[0], intro=trials[0]["intro"].replace("out of 100", "out of 1,000"))] + trials[1:]
        with pytest.raises(ValueError):
            dose.assert_dose_manifest_well_formed(bad)

    def test_off_grid_dose(self, trials):
        bad = [dict(trials[0], dose=75, variant="d75")] + trials[1:]
        with pytest.raises(ValueError):
            dose.assert_dose_manifest_well_formed(bad)
