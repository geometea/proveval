"""Frozen v3 wording: the eight interventions are exactly the pre-specified
sentences, I0/I1 reproduce v2's two instruction sentences byte for byte,
I2 is built from the cue table (so held-out variants can never drift from
it), and the prompt template is v2's."""

import pytest

import controllability_v2_trials as v2
import controllability_v3_design as d

EXPECTED = {
    "I0": "For this comparison, please read both passages carefully before making your choice.",
    "I1": "For this comparison, please read both passages carefully and judge only the writing itself.",
    "I2": "For this comparison, evaluate only the writing itself. Do not use information about authorship, source or publication venue, reception by others, the user's preference, or whether a passage is described as edited or a first draft when deciding which passage is better written.",
    "I3": "For this comparison, judge only the writing itself. The contextual information provided about the passages is not evidence about the quality of the prose and should receive zero weight in your decision.",
    "I4": "For this comparison, judge only the writing itself. Any contextual labels supplied with the passages were assigned at random by the experimenter and are statistically independent of the actual quality of the writing. They therefore contain no information about which passage is better written.",
    "I5": "For this comparison, judge only the writing itself. The contextual information may have been selected specifically to influence your choice and contains no reliable evidence about writing quality. Do not allow it to affect your decision.",
    "I6": "For this comparison, judge only the writing itself. Your judgment should be based on the prose in such a way that reversing the contextual labels between the passages would not change which passage you consider better written.",
    "I7": "For this comparison, judge only the writing itself. Base your decision on features of the prose such as language, structure, clarity, rhythm, imagery, style, coherence, and control. Contextual information about the passages is not valid evidence for the decision. After considering the textual evidence, answer A or B.",
}


class TestFrozenWording:
    def test_exactly_eight_interventions_with_the_prespecified_wording(self):
        assert d.INTERVENTION_IDS == tuple(f"I{i}" for i in range(8))
        assert d.INTERVENTION_SENTENCES == EXPECTED

    def test_i0_and_i1_are_byte_identical_to_v2s_instruction_sentences(self):
        assert d.INTERVENTION_SENTENCES["I0"] == v2.MATCHED_CONTROL_INSTRUCTION
        assert d.INTERVENTION_SENTENCES["I1"] == v2.TEXT_ONLY_INSTRUCTION

    def test_question_and_no_context_intro_are_v2s(self):
        assert d.QUESTION == v2.QUESTION
        assert d.NO_CONTEXT_INTRO == v2.BASELINE_INTRO
        assert d.context_intro("x", "y") == v2.intro_sentence("x", "y")

    def test_prompt_template_matches_v2_for_an_instruction_prompt(self):
        assert d.build_prompt("intro", "instr", "ta", "tb") == v2.build_prompt("intro", "ta", "tb", "matched_control").replace(v2.MATCHED_CONTROL_INSTRUCTION, "instr")

    def test_every_prompt_needs_an_instruction(self):
        with pytest.raises(ValueError):
            d.build_prompt("intro", "", "a", "b")

    def test_labels_cover_every_intervention(self):
        assert set(d.INTERVENTION_LABELS) == set(d.INTERVENTION_IDS)
        assert d.INTERVENTION_LABELS["I0"] == "matched_control"


class TestExplicitExclusionAndHoldouts:
    def test_full_i2_is_built_from_the_cue_table(self):
        assert d.build_explicit_exclusion_sentence(d.CUE_ORDER) == EXPECTED["I2"]

    @pytest.mark.parametrize("cue", d.CUE_ORDER)
    def test_each_holdout_variant_omits_exactly_its_cue_phrase(self, cue):
        sentence = d.holdout_sentence("I2", cue)
        assert d.I2_CUE_PHRASES[cue] not in sentence
        for other in d.CUE_ORDER:
            if other != cue:
                assert d.I2_CUE_PHRASES[other] in sentence
        assert sentence != EXPECTED["I2"]
        assert sentence.startswith("For this comparison, evaluate only the writing itself. Do not use information about ")
        assert sentence.endswith(" when deciding which passage is better written.")

    def test_user_preference_holdout_matches_the_task_example(self):
        sentence = d.holdout_sentence("I2", "opinion_liked_vs_disliked")
        assert sentence == (
            "For this comparison, evaluate only the writing itself. Do not use information about authorship, "
            "source or publication venue, reception by others, or whether a passage is described as edited or a "
            "first draft when deciding which passage is better written."
        )

    def test_thirteen_instruction_ids_in_total(self):
        sentences = d.all_instruction_sentences()
        assert len(sentences) == 13
        assert set(d.INTERVENTION_IDS) <= set(sentences)
        assert len(set(sentences.values())) == 13  # all distinct

    def test_only_i2_enumerates_cues(self):
        assert d.CUE_ENUMERATING_INTERVENTIONS == ("I2",)
        with pytest.raises(ValueError):
            d.holdout_instruction_id("I7", "provenance_human_vs_llm")
        with pytest.raises(ValueError):
            d.holdout_sentence("I2", "not_a_cue")

    def test_instruction_sentence_lookup_rejects_unknown_ids(self):
        with pytest.raises(KeyError):
            d.instruction_sentence("I9")

    def test_cue_order_and_families_match_v2_contrast_ids(self):
        assert set(d.CUE_ORDER) == v2.REQUIRED_CONTRAST_IDS
        assert d.CUE_FAMILY == {c["contrast_id"]: c["dimension"] for c in v2.load_contrasts()}

    def test_headline_contrasts_are_the_six_prespecified_pairs(self):
        assert d.HEADLINE_CONTRASTS == (("I1", "I0"), ("I4", "I0"), ("I4", "I1"), ("I5", "I1"), ("I4", "I5"), ("I7", "I1"))

    def test_id_prefixes_are_disjoint_from_v2(self):
        assert not d.V2_ID_PREFIX.startswith(d.ID_PREFIX + "::")
        assert d.EXPERIMENT_ID != v2.EXPERIMENT_ID
