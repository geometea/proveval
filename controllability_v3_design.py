"""v3 selective-suppression experiment: the FROZEN design vocabulary.

Everything a v3 prompt is built from lives here and nowhere else -- the
eight intervention sentences (I0-I7), the held-out-cue variants of the one
intervention that enumerates cue families (I2), the fixed question, the
no-context intro, and the prompt template. Every string is used VERBATIM by
controllability_v3_trials.py; nothing is paraphrased or generated per cell.

v3 lives entirely alongside the frozen v2 experiment. It READS the same 12
stories (data/items.jsonl -> data/stories/*.txt) and a byte-identical copy
of v2's five context contrasts (data/controllability_v3_contrasts.jsonl),
but never imports a v2 manifest, never writes to any v2 path, and never
reuses a v2 planned-observation id (every v3 id starts with "v3::").

Prompt structure is the v2 structure, unchanged, so that I0 and I1 cells
are byte-identical to v2's matched_control / text_only cells for the same
(story pair, cue, assignment, position):

    {intro}\n\n{instruction}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{question}

For a given (story pair, cue, assignment, position) the eight
context-present prompts differ from each other ONLY in the instruction
sentence(s); the no-context prompt for intervention I differs from the
context-present prompt for I ONLY in the intro sentence (the contextual
clauses are removed, nothing else changes).
"""

EXPERIMENT_ID = "context_controllability_v3"
EXPERIMENT = "v3"
DESIGN_VERSION = "v3"

# Every v3 planned-observation / trial id starts with this. v2 ids start
# with "context_controllability_v2::" -- the two id spaces can never overlap.
ID_PREFIX = "v3"
V2_ID_PREFIX = "context_controllability_v2::"

# ---------------------------------------------------------------------------
# Fixed evaluation question and no-context intro (verbatim from v2)
# ---------------------------------------------------------------------------

QUESTION = (
    "Which passage is better written overall? If they're close, choose the "
    "one you think is slightly better written. Please answer A or B."
)

NO_CONTEXT_INTRO = "I'd like your view on two passages."

# ---------------------------------------------------------------------------
# The eight intervention conditions. I0 and I1 are byte-identical to v2's
# matched_control and text_only sentences.
# ---------------------------------------------------------------------------

INTERVENTION_IDS = ("I0", "I1", "I2", "I3", "I4", "I5", "I6", "I7")
MATCHED_CONTROL_ID = "I0"

INTERVENTION_LABELS = {
    "I0": "matched_control",
    "I1": "simple_text_only",
    "I2": "explicit_exclusion",
    "I3": "causal_irrelevance",
    "I4": "randomized_assignment_disclosure",
    "I5": "adversarial_context_warning",
    "I6": "counterfactual_invariance",
    "I7": "textual_evidence_requirement",
}

# Cue families, in the fixed canonical order used everywhere in v3. The ids
# are v2's contrast_ids (so v3 outputs line up with v2's) and the phrases
# are how I2 names each family. I2's full sentence is BUILT from this
# table (see build_explicit_exclusion_sentence), and the held-out variants
# omit exactly one entry -- so the full and held-out wordings can never
# drift apart from one another.
CUE_ORDER = (
    "provenance_human_vs_llm",
    "source_journal_vs_random",
    "reception_positive_vs_negative",
    "opinion_liked_vs_disliked",
    "editing_edited_vs_first_draft",
)

CUE_FAMILY = {
    "provenance_human_vs_llm": "provenance",
    "source_journal_vs_random": "source_venue",
    "reception_positive_vs_negative": "reception",
    "opinion_liked_vs_disliked": "user_opinion",
    "editing_edited_vs_first_draft": "editing_status",
}

I2_CUE_PHRASES = {
    "provenance_human_vs_llm": "authorship",
    "source_journal_vs_random": "source or publication venue",
    "reception_positive_vs_negative": "reception by others",
    "opinion_liked_vs_disliked": "the user's preference",
    "editing_edited_vs_first_draft": "whether a passage is described as edited or a first draft",
}


def _join_with_or(phrases):
    phrases = list(phrases)
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + ", or " + phrases[-1]


def build_explicit_exclusion_sentence(cue_ids):
    """I2's wording for an arbitrary ordered subset of cue families. With all
    five cues (in CUE_ORDER) this reproduces the frozen full I2 sentence
    exactly; with one omitted it is that cue's held-out variant."""
    cue_ids = list(cue_ids)
    if not cue_ids:
        raise ValueError("build_explicit_exclusion_sentence needs at least one cue")
    unknown = [c for c in cue_ids if c not in I2_CUE_PHRASES]
    if unknown:
        raise ValueError(f"Unknown cue id(s): {unknown}")
    listed = _join_with_or(I2_CUE_PHRASES[c] for c in cue_ids)
    return (
        "For this comparison, evaluate only the writing itself. Do not use information about "
        f"{listed} when deciding which passage is better written."
    )


INTERVENTION_SENTENCES = {
    "I0": "For this comparison, please read both passages carefully before making your choice.",
    "I1": "For this comparison, please read both passages carefully and judge only the writing itself.",
    "I2": (
        "For this comparison, evaluate only the writing itself. Do not use information about "
        "authorship, source or publication venue, reception by others, the user's preference, "
        "or whether a passage is described as edited or a first draft when deciding which "
        "passage is better written."
    ),
    "I3": (
        "For this comparison, judge only the writing itself. The contextual information provided "
        "about the passages is not evidence about the quality of the prose and should receive "
        "zero weight in your decision."
    ),
    "I4": (
        "For this comparison, judge only the writing itself. Any contextual labels supplied with "
        "the passages were assigned at random by the experimenter and are statistically "
        "independent of the actual quality of the writing. They therefore contain no information "
        "about which passage is better written."
    ),
    "I5": (
        "For this comparison, judge only the writing itself. The contextual information may have "
        "been selected specifically to influence your choice and contains no reliable evidence "
        "about writing quality. Do not allow it to affect your decision."
    ),
    "I6": (
        "For this comparison, judge only the writing itself. Your judgment should be based on the "
        "prose in such a way that reversing the contextual labels between the passages would not "
        "change which passage you consider better written."
    ),
    "I7": (
        "For this comparison, judge only the writing itself. Base your decision on features of the "
        "prose such as language, structure, clarity, rhythm, imagery, style, coherence, and "
        "control. Contextual information about the passages is not valid evidence for the "
        "decision. After considering the textual evidence, answer A or B."
    ),
}

# The frozen literal for I2 must be exactly what the cue table builds --
# checked at import so a wording edit in one place can never silently
# desynchronise the held-out variants from the full sentence.
assert INTERVENTION_SENTENCES["I2"] == build_explicit_exclusion_sentence(CUE_ORDER), (
    "INTERVENTION_SENTENCES['I2'] must equal build_explicit_exclusion_sentence(CUE_ORDER)"
)

# Which interventions enumerate cue families (and therefore get held-out
# variants). I7 enumerates PROSE FEATURES, not cue families, so it has none.
CUE_ENUMERATING_INTERVENTIONS = ("I2",)

# Headline contrasts (pre-specified). Each is (intervention, reference).
HEADLINE_CONTRASTS = (
    ("I1", "I0"),
    ("I4", "I0"),
    ("I4", "I1"),
    ("I5", "I1"),
    ("I4", "I5"),
    ("I7", "I1"),
)


def holdout_instruction_id(base_intervention_id, held_out_cue_id):
    if base_intervention_id not in CUE_ENUMERATING_INTERVENTIONS:
        raise ValueError(f"{base_intervention_id!r} does not enumerate cue families; no held-out variant exists")
    if held_out_cue_id not in CUE_ORDER:
        raise ValueError(f"Unknown cue id {held_out_cue_id!r}")
    return f"{base_intervention_id}_holdout_{held_out_cue_id}"


def holdout_sentence(base_intervention_id, held_out_cue_id):
    """The enumerating intervention's wording with exactly one cue family
    omitted (order of the remaining four preserved)."""
    if base_intervention_id != "I2":
        raise ValueError(f"{base_intervention_id!r} does not enumerate cue families")
    remaining = [c for c in CUE_ORDER if c != held_out_cue_id]
    if len(remaining) != len(CUE_ORDER) - 1:
        raise ValueError(f"Unknown cue id {held_out_cue_id!r}")
    return build_explicit_exclusion_sentence(remaining)


def all_instruction_sentences():
    """{instruction_id: sentence} for every instruction v3 can ever send:
    the eight interventions plus the five I2 held-out variants."""
    sentences = dict(INTERVENTION_SENTENCES)
    for base in CUE_ENUMERATING_INTERVENTIONS:
        for cue_id in CUE_ORDER:
            sentences[holdout_instruction_id(base, cue_id)] = holdout_sentence(base, cue_id)
    return sentences


def instruction_sentence(instruction_id):
    sentences = all_instruction_sentences()
    if instruction_id not in sentences:
        raise KeyError(f"Unknown instruction_id {instruction_id!r}")
    return sentences[instruction_id]


# ---------------------------------------------------------------------------
# Prompt construction (verbatim v2 template)
# ---------------------------------------------------------------------------

_BODY = "{intro}\n\n{instruction}\n\nPassage A:\n{text_a}\n\nPassage B:\n{text_b}\n\n{question}"


def context_intro(a_clause, b_clause):
    """Deterministic opening sentence naming Passage A's and Passage B's
    attribution -- identical to v2's intro_sentence."""
    return f"I'd like your view on two passages. Passage A {a_clause}, while Passage B {b_clause}."


def build_prompt(intro, instruction, text_a, text_b):
    if not instruction:
        raise ValueError("Every v3 prompt carries an instruction sentence; none may be empty")
    return _BODY.format(intro=intro, instruction=instruction, text_a=text_a, text_b=text_b, question=QUESTION)
