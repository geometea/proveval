"""v3 manifests: exact cell counts, exact counterbalancing, stable ids,
byte-stable regeneration, no-context matching, held-out variants, pilot
subset, and loud failure on structural drift. Uses the real corpus."""

import json
import os

import pytest

import controllability_v3_trials as tr
from controllability_v3_design import CUE_ORDER, INTERVENTION_IDS, NO_CONTEXT_INTRO, all_instruction_sentences, holdout_instruction_id
from context_trials import load_items


@pytest.fixture(scope="module")
def generated():
    items = load_items()
    texts = tr.load_story_texts(items)
    context, nocontext, holdout = tr.generate_frozen_manifests(items=items, texts=texts)
    return {"items": items, "texts": texts, "context": context, "nocontext": nocontext, "holdout": holdout}


class TestCounts:
    def test_design_constants(self):
        assert (tr.N_STORIES, tr.N_PAIRS, tr.N_CUES, tr.N_INTERVENTIONS) == (12, 66, 5, 8)
        assert tr.CONTEXT_UNIQUE_CELLS == 66 * 5 * 8 * 4 == 10560
        assert tr.NOCONTEXT_UNIQUE_CELLS == 66 * 8 * 2 == 1056
        assert tr.HOLDOUT_UNIQUE_CELLS == 66 * 5 * 4 == 1320

    def test_generated_cell_counts(self, generated):
        assert len(generated["context"]) == 10560
        assert len(generated["nocontext"]) == 1056
        assert len(generated["holdout"]) == 1320

    def test_every_pair_cue_has_all_eight_interventions_with_four_cells(self, generated):
        cells = {}
        for t in generated["context"]:
            cells.setdefault((t["pair_id"], t["cue_id"], t["intervention_id"]), set()).add((t["assignment"], t["position"]))
        assert len(cells) == 66 * 5 * 8
        assert all(v == tr.REQUIRED_CONTEXT_CELLS for v in cells.values())
        per_pair_cue = {}
        for (pair, cue, i) in cells:
            per_pair_cue.setdefault((pair, cue), set()).add(i)
        assert all(v == set(INTERVENTION_IDS) for v in per_pair_cue.values())

    def test_every_nocontext_pair_intervention_has_both_positions(self, generated):
        cells = {}
        for t in generated["nocontext"]:
            cells.setdefault((t["pair_id"], t["intervention_id"]), set()).add(t["position"])
        assert len(cells) == 66 * 8
        assert all(v == {"story1_as_a", "story2_as_a"} for v in cells.values())

    def test_holdout_variants_are_tested_only_on_the_omitted_cue(self, generated):
        for t in generated["holdout"]:
            assert t["held_out_cue_id"] == t["cue_id"]
            assert t["instruction_id"] == holdout_instruction_id("I2", t["cue_id"])
            assert t["base_intervention_id"] == "I2"
        combos = {(t["pair_id"], t["cue_id"]) for t in generated["holdout"]}
        assert len(combos) == 66 * 5


class TestIds:
    def test_ids_are_unique_across_all_manifests_and_never_v2(self, generated):
        ids = [t["trial_id"] for m in ("context", "nocontext", "holdout") for t in generated[m]]
        assert len(ids) == len(set(ids)) == 10560 + 1056 + 1320
        assert all(i.startswith("v3::") for i in ids)
        assert not any(i.startswith("context_controllability_v2") for i in ids)

    def test_id_encodes_every_design_component_and_round_trips(self, generated):
        t = generated["context"][7]
        oid = tr.make_planned_observation_id(t["trial_id"], 4)
        parsed = tr.parse_planned_observation_id(oid)
        assert parsed == {
            "family": "primary_context", "pair_id": t["pair_id"], "cue_id": t["cue_id"], "assignment": t["assignment"],
            "position": t["position"], "instruction_id": t["intervention_id"], "context_present": True, "replicate_number": 4,
            "trial_id": t["trial_id"],
        }
        n = generated["nocontext"][0]
        parsed = tr.parse_planned_observation_id(tr.make_planned_observation_id(n["trial_id"], 1))
        assert parsed["cue_id"] is None and parsed["assignment"] is None and parsed["context_present"] is False

    def test_parse_rejects_non_v3_ids(self):
        with pytest.raises(ValueError):
            tr.parse_planned_observation_id("context_controllability_v2::deepseek::superblock__x::r1")

    def test_ids_are_stable_across_regeneration(self, generated):
        again, _, _ = tr.generate_frozen_manifests(items=generated["items"], texts=generated["texts"])
        assert [t["trial_id"] for t in again] == [t["trial_id"] for t in generated["context"]]
        assert [t["prompt_sha256"] for t in again] == [t["prompt_sha256"] for t in generated["context"]]


class TestPromptsAndMatching:
    def test_context_interventions_differ_only_in_the_instruction_paragraph(self, generated):
        by_cell = {}
        for t in generated["context"]:
            by_cell.setdefault((t["pair_id"], t["cue_id"], t["assignment"], t["position"]), {})[t["intervention_id"]] = tr.build_prompt_for_trial(t, generated["texts"])
        cell = by_cell[next(iter(by_cell))]
        base = cell["I0"].split("\n\n")
        for i, prompt in cell.items():
            paras = prompt.split("\n\n")
            assert len(paras) == len(base)
            assert [k for k in range(len(paras)) if paras[k] != base[k]] == ([] if i == "I0" else [1])
            assert paras[1] == all_instruction_sentences()[i]

    def test_no_context_cell_differs_from_its_context_cell_only_in_the_intro(self, generated):
        ctx = {(t["pair_id"], t["position"], t["intervention_id"], t["assignment"]): t for t in generated["context"] if t["cue_id"] == CUE_ORDER[0]}
        for n in generated["nocontext"][:64]:
            c = ctx[(n["pair_id"], n["position"], n["intervention_id"], "forward")]
            pc, pn = tr.build_prompt_for_trial(c, generated["texts"]).split("\n\n"), tr.build_prompt_for_trial(n, generated["texts"]).split("\n\n")
            assert pn[0] == NO_CONTEXT_INTRO and pc[0] != NO_CONTEXT_INTRO
            assert pc[1:] == pn[1:]

    def test_prompt_hash_verification_catches_a_changed_story(self, generated):
        texts = dict(generated["texts"])
        first = generated["context"][0]
        texts[first["story_a_id"]] = texts[first["story_a_id"]] + " "
        with pytest.raises(ValueError, match="prompt_sha256"):
            tr.verify_prompt_hashes([first], texts)
        tr.verify_prompt_hashes([first], generated["texts"])

    def test_manifest_rows_never_embed_the_prompt(self, generated):
        assert all("prompt" not in t for m in ("context", "nocontext", "holdout") for t in generated[m])


class TestOnDiskManifestsAreFrozen:
    def test_on_disk_manifests_are_byte_identical_to_regeneration(self, generated, tmp_path):
        for name, path in (("context", tr.CONTEXT_TRIALS_FILE), ("nocontext", tr.NOCONTEXT_TRIALS_FILE), ("holdout", tr.HOLDOUT_TRIALS_FILE)):
            out = tmp_path / os.path.basename(path)
            tr.write_manifest(generated[name], str(out))
            assert out.read_bytes() == open(path, "rb").read(), f"{path} is not the deterministic regeneration"

    def test_on_disk_pilot_manifest_is_well_formed_and_disjoint(self, generated):
        pilot = tr.load_manifest(tr.PILOT_TRIALS_FILE)
        frozen_ids = {t["trial_id"] for m in ("context", "nocontext", "holdout") for t in generated[m]}
        tr.assert_pilot_manifest_well_formed(pilot, frozen_ids)
        assert 500 <= len(pilot) <= 1000
        tr.verify_prompt_hashes(pilot, generated["texts"])


class TestValidationFailsLoudly:
    def test_duplicate_id_is_rejected(self, generated):
        trials = generated["context"] + [generated["context"][0]]
        with pytest.raises(ValueError, match="duplicate"):
            tr.assert_primary_context_manifest_well_formed(trials)

    def test_missing_cell_is_rejected(self, generated):
        with pytest.raises(ValueError, match="4 required"):
            tr.assert_primary_context_manifest_well_formed(generated["context"][1:])

    def test_wrong_instruction_is_rejected(self, generated):
        trial = dict(generated["context"][0])
        trial["instruction_id"] = "I1"
        with pytest.raises(ValueError, match="instruction"):
            tr.assert_primary_context_manifest_well_formed([trial] + generated["context"][1:])

    def test_context_in_a_nocontext_cell_is_rejected(self, generated):
        trial = dict(generated["nocontext"][0])
        trial["intro"] = "I'd like your view on two passages. Passage A was edited, while Passage B is a draft."
        with pytest.raises(ValueError, match="contextual framing"):
            tr.assert_nocontext_manifest_well_formed([trial] + generated["nocontext"][1:])

    def test_holdout_on_the_wrong_cue_is_rejected(self, generated):
        trial = dict(generated["holdout"][0])
        trial["held_out_cue_id"] = [c for c in CUE_ORDER if c != trial["cue_id"]][0]
        with pytest.raises(ValueError):
            tr.assert_holdout_manifest_well_formed([trial] + generated["holdout"][1:])

    def test_v2_style_id_is_rejected(self, generated):
        trial = dict(generated["context"][0])
        trial["trial_id"] = "context_controllability_v2::x"
        with pytest.raises(ValueError, match="not a v3 id"):
            tr.assert_primary_context_manifest_well_formed([trial] + generated["context"][1:])

    def test_overlapping_manifests_are_rejected(self, generated):
        with pytest.raises(ValueError, match="more than one manifest"):
            tr.assert_no_id_overlap(generated["context"], generated["context"][:1])

    def test_contrasts_must_be_the_five_cues_in_canonical_order(self):
        contrasts = tr.load_contrasts()
        tr.assert_contrasts_are_well_formed(contrasts)
        with pytest.raises(ValueError):
            tr.assert_contrasts_are_well_formed(list(reversed(contrasts)))


class TestPilot:
    def test_seeded_selection_is_deterministic_and_labelled_seeded(self, generated):
        a = tr.select_pilot_pairs(generated["items"], 2, 2, seed=1)
        b = tr.select_pilot_pairs(generated["items"], 2, 2, seed=1)
        assert a == b and len(a) == 4 and all(s == "seeded" for _, _, s in a)
        assert tr.select_pilot_pairs(generated["items"], 2, 2, seed=2) != a

    def test_strength_table_picks_strongest_and_most_ambiguous(self, generated):
        from context_trials import story_pairs
        pairs = [(s1["id"], s2["id"]) for s1, s2 in story_pairs(generated["items"])]
        rows = [{"story_1_id": p[0], "story_2_id": p[1], "baseline_strength": i} for i, p in enumerate(pairs)]
        chosen = tr.select_pilot_pairs(generated["items"], 2, 3, seed=0, baseline_strength_rows=rows)
        assert [(s1, s2) for s1, s2, st in chosen if st == "strong"] == [pairs[-1], pairs[-2]]
        assert [(s1, s2) for s1, s2, st in chosen if st == "ambiguous"] == [pairs[0], pairs[1], pairs[2]]

    def test_strength_table_missing_pairs_is_an_error(self, generated):
        with pytest.raises(ValueError, match="missing"):
            tr.select_pilot_pairs(generated["items"], 1, 1, seed=0, baseline_strength_rows=[{"story_1_id": "afterlife", "story_2_id": "buddy", "baseline_strength": 1}])

    def test_pilot_manifest_spans_everything_and_lives_in_its_own_family(self, generated):
        pairs = tr.select_pilot_pairs(generated["items"], 2, 2, seed=3)
        pilot = tr.build_pilot_trials(generated["context"], generated["nocontext"], generated["holdout"], pairs)
        frozen_ids = {t["trial_id"] for m in ("context", "nocontext", "holdout") for t in generated[m]}
        tr.assert_pilot_manifest_well_formed(pilot, frozen_ids)
        assert len(pilot) == 4 * (5 * 8 * 4 + 8 * 2 + 5 * 4) == 784
        assert all(t["trial_id"].startswith("v3::pilot::") and t["block_id"].startswith("v3::pilot::") for t in pilot)
        assert {t["source_trial_id"] for t in pilot} <= frozen_ids
        assert not ({t["trial_id"] for t in pilot} & frozen_ids)

    def test_pilot_validation_rejects_a_primary_id(self, generated):
        pairs = tr.select_pilot_pairs(generated["items"], 1, 1, seed=3)
        pilot = tr.build_pilot_trials(generated["context"], generated["nocontext"], generated["holdout"], pairs)
        frozen_ids = {t["trial_id"] for m in ("context", "nocontext", "holdout") for t in generated[m]}
        bad = dict(pilot[0]); bad["trial_id"] = bad["source_trial_id"]
        with pytest.raises(ValueError):
            tr.assert_pilot_manifest_well_formed([bad] + pilot[1:], frozen_ids)
