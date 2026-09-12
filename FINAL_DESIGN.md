# Final Design

## Status

This project has three distinct stages. Do not confuse them:

1. **Completed exploratory pilot** (Claude Sonnet 5 only). Documented in
   `PILOT_RESULTS.md`. 138 completed observations over 4 stories, provenance
   conditions only (self / AI / literary journal / etc.). Results are
   exploratory and are **never pooled** with anything below.
2. **Broader benchmark: currently being implemented.** This is what this
   document now describes. The trial-generation code, response validation,
   runner, and offline analysis all exist and are exercised with dry-runs and
   synthetic fixtures (see `CONTEXT_PACKETS.md` for what's built vs. planned).
   No benchmark API calls have been made under this design.
3. **Future API collection: not yet run.** Actually spending API budget on
   the benchmark below is a separate, deliberate step, gated on reviewing the
   generated trial manifest and choosing which contrasts/models/subsets are
   worth paying for.

This document is a working design, not a frozen one. Expect it to keep
changing as the benchmark is actually run and the results inform what's
worth testing next.

## Research question

How does task-irrelevant context affect LLM judgments in subjective
evaluations of prose, and can different model/context combinations better
match a fixed human preference ranking?

## Corpus

12 short prose texts (`gilbert`, `dunnest_smoke`, `prophet`, `santa`,
`dana_brownies`, `morrow_transport`, `eyecut_lowway`, `saturn`, `qual_panic`,
`buddy`, `charlotte_train`, `afterlife`), registered in `data/items.jsonl`.

All prose comes from this one fixed corpus. Any claimed provenance, source,
editing history, reception, or opinion attached to a text is **experimentally
manipulated metadata** and need not reflect anything true about how the text
was actually produced.

## Evaluation tasks

Two response formats, both over the same five dimensions (plot_structure,
prose_style, characterization, originality, overall_quality):

- **Single-text multidimensional ratings** -- one story, one context
  condition, a 1-5 score per dimension (`prompts.py`, `context_single`
  trials in `context_trials.py`). Every story also gets its own explicit
  no-context `neutral` baseline trial (`context_packets.neutral_condition`),
  distinct from any of the v0.1 pilot's own conditions -- this is v0.2's own
  clean single-text baseline, and the reference point every other
  single-text condition is compared against.
- **Pairwise multidimensional A/B/tie judgments** -- two different stories,
  a per-dimension choice of "A", "B", or "tie" (`context_comparisons.py`,
  `context_pairwise`/`context_prompt` trials in `context_trials.py`).

## Context families

Defined in `data/context_dimensions.jsonl`, composed into natural sentences
by `context_packets.py`:

- **provenance/authorship** -- self, someone I know, unknown human, named AI
  systems (Claude, ChatGPT, Gemini-edited), AI-assisted, generic/unnamed LLM.
- **source/venue** -- literary journal, sent by a friend, found randomly
  online.
- **social reception** -- unseen, lukewarm, negative, loved.
- **user-stated opinion** -- the prompter says they liked or disliked the
  piece. Intentionally sycophancy-adjacent: this tests whether the
  evaluator's judgment shifts toward the user's stated opinion rather than
  the text.
- **editing/writer-status signals** -- first draft vs. edited (independent
  of authorship); published/first-time/hobby writer status (dependent on
  provenance being self or a known other -- see the identity-contradiction
  guard below).
- **unrelated prompt-level context** -- a genuinely extraneous aside (for
  now: a one-line weather mention) appended to the request itself, not
  attributed to either story. Modeled separately from the story-scope
  families (`scope: "prompt"` in `data/context_dimensions.jsonl`,
  `data/context_prompt_contrasts.jsonl`, and its own builder in
  `context_trials.py`) rather than forced through the per-story A/B
  attribution abstraction, since it isn't naturally a per-story attribution
  at all.

Explicit value-vs-value contrasts (never a signal vs. nothing) live in
`data/context_contrasts.jsonl` (story-scope) and
`data/context_prompt_contrasts.jsonl` (prompt-scope).

### Forward/flipped assignment

For pairwise trials, `story_1` is always Story A and `story_2` is always
Story B; only which value of the contrast each side carries is swapped
between the **forward** and **flipped** trial. If a model's A/B/tie choice
is really about the underlying prose, the same A/B position should keep
winning across the forward/flipped pair; if it changes when only the
attribution is swapped, that's evidence of a context effect rather than a
prose-quality judgment. (See `context_contrasts.py` for why: same story
twice is detectable by the model, and one side with no context at all isn't
a valid control either -- both were tried and rejected during design.)

**Important limitation:** because every `context_pairwise` observation pits
two *different* claimed context values against each other (never the same
value on both sides), these trials are built to measure **causal context
sensitivity** -- whether swapping the attribution changes the decision --
not to produce a ranking "under" one context condition. A ranking pooled
across every contrast/forward/flipped trial is a rough diagnostic at best
(see "Rankings and human alignment" below); it is not a `context_single`
-style per-condition result and must not be read as one.

### Optional same-context pairwise family (not run)

`context_trials.build_context_pairwise_same_trials()` generates an
additional, **optional** trial family, written to a separate file
(`data/context_trials_optional_same_context.jsonl`, gitignored) rather than
into the required `data/context_trials.jsonl` manifest: both Story A and
Story B are attributed the *same* context value (e.g. "both of these were
generated by Claude"), instead of the forward/flipped contrast. If this
family were ever run, it would let a full round-robin of the corpus be
judged under one single, globally consistent context condition -- enabling
a true per-condition *pairwise* ranking, analogous to what
`analyze_context.rank_from_single_text_by_condition` already does for
single-text ratings. It deliberately skips dependent dimensions
(`writer_status`) to avoid the added pronoun-resolution complexity for a
family that is not part of any required run. **This trial family is not
run, is not part of any required or planned API budget, and no results
exist for it.**

### Named-model provenance contrasts

Provenance contrasts specifically include AI-vs-human framings AND
named-model-vs-named-model and named-model-vs-generic-LLM framings
(`provenance_claude_vs_chatgpt`, `provenance_claude_vs_llm_generic`,
`provenance_chatgpt_vs_llm_generic`). Since the evaluator in this project is
itself Claude, this is intended to let later analysis separate:

- a general AI-vs-human provenance effect,
- an effect specific to a named competitor model,
- a possible **self-family effect** (the evaluator treating "generated by
  Claude" differently from "generated by ChatGPT" or "generated by an
  unspecified LLM", independent of the general AI-vs-human effect).

No claim is made yet about which of these, if any, is present -- that's an
empirical question for the actual benchmark run.

## Human reference ranking

A fixed human preference ranking of the 12-text corpus, derived from human
pairwise judgments (`data/human_pairwise.jsonl`), not invented or estimated.
`human_ranking.py` checks the judgments for cycles/contradictions and
computes a full ranking only when the judgments collected so far uniquely
determine one (via topological sort, requiring the "next" story to be
unambiguous at every step). This is meant to let just enough adaptive
pairwise comparisons be collected to pin down a full order, rather than
requiring all C(12,2) = 66 pairs. As of this document, only 2 judgments are
known and the ranking is **not** complete or unique -- `data/human_reference.json`
does not exist yet, and nothing here claims otherwise.

## Rankings and human alignment

The question "which model + context setup best matches the fixed human
preference ranking?" needs a ranking **per context condition**, not one
ranking per model averaged across every condition the benchmark happened to
try -- averaging across conditions is exactly what would hide the effect
this benchmark exists to measure.

- **Primary analysis: per-condition single-text rankings.**
  `analyze_context.rank_from_single_text_by_condition` builds one story
  ranking for every `(model, dimension, value)` combination seen in
  `context_single` results, including the `neutral` no-context baseline as
  its own condition. For every such ranking, if `data/human_reference.json`
  exists, `analyze_context.py` computes:
  - **Spearman rank correlation** against the human reference (pure Python,
    no scipy dependency -- a few lines for two full, tie-free rankings of
    the same item set).
  - **Pairwise agreement** against the raw judgments in
    `data/human_pairwise.jsonl`, which works even before a complete
    reference ranking exists (it only needs the specific pairs already
    judged).

  Written to `results/context_analysis/human_comparison.csv`
  (`model, dimension, value, ranking_source, spearman_vs_human,
  pairwise_agreement, pairwise_agreement_n`) and
  `single_text_rankings_by_condition.csv`. This is the analysis meant to
  answer the question above -- not the diagnostics below.

- **Diagnostics only, clearly labelled as such, never the headline result:**
  - A pooled single-text ranking (`single_text_pooled_diagnostic`): mean
    `overall_quality` per story, averaged across every `context_single`
    condition. Useful as a rough sanity check; hides exactly the
    per-condition distinction the benchmark is testing for.
  - A pooled pairwise ranking (`pairwise_pooled_diagnostic`): a
    Copeland/win-rate estimate from `overall_quality` judgments, pooled
    across every contrast and both forward/flipped assignments. As noted
    above, `context_pairwise` trials don't represent any single context
    condition to begin with, so this is not, and must not be presented as,
    a ranking "under" a particular context -- it is a rough baseline at
    best. (The optional same-context pairwise family, if ever run, would be
    needed for a true per-condition pairwise ranking -- see above.)

  Both estimators are explicitly documented in `analyze_context.py` as
  simple, transparent, but not-the-only-possible ranking methods (e.g. the
  pairwise estimator ignores strength of opponent).

## Budget constraints

This is not a one-shot, all-at-once collection. The trial manifest
(`data/context_trials.jsonl`) is generated in full because generating it
costs nothing, but running it is expected to happen in deliberately chosen
slices. `run_batch.py` supports selecting a subset to actually spend API
budget on via `--type`, `--contrast`, `--condition`, `--id-prefix`,
`--limit`, and `--model`, against either trials file
(`--trials-file`)/results file (`--results-file`) pair. Which
contrasts/models/subsets are worth running is a decision made before
spending money, not something this design presumes in advance.

## Pilot data stays exploratory

The v0.1 pilot's trials, results, and analysis (`data/trials.jsonl`,
`prompts.py`, `comparisons.py`, `make_trials.py`, `run_trial.py`/
`run_batch.py` against their default paths, `analyze.py`) are untouched by
any of the above and remain independently reproducible. Pilot observations
are never pooled into the benchmark's analysis, and the benchmark's own
results live in separate files (`data/context_trials.jsonl`,
`results/context_raw.jsonl` by convention, `results/context_analysis/`) so
the two can never be accidentally mixed.
