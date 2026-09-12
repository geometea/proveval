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
  trials in `context_trials.py`).
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

## Comparison against the human reference

- **Rank correlation (Spearman)** is the main human-alignment measure, once
  a complete `data/human_reference.json` exists. Implemented in pure Python
  in `analyze_context.py` (no scipy dependency) since the formula for two
  full, tie-free rankings of the same 12 items is a few lines.
- **Pairwise agreement** against the raw judgments in
  `data/human_pairwise.jsonl` is a secondary measure, and works even before
  a complete ranking exists (it only needs the specific pairs already
  judged, not a full order).
- Two rankings are estimated per model from the benchmark data itself: a
  Copeland/win-rate ranking from pairwise `overall_quality` judgments, and a
  mean-score ranking from single-text `overall_quality` ratings. Both are
  explicitly documented as **provisional** estimators in `analyze_context.py`
  -- not the only possible ranking method, just a simple, transparent one to
  start from.

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
