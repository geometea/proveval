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
  condition, a **1.0-10.0 score per dimension, to one decimal place**
  (`data/context_tasks.jsonl`, `context_single` trials in
  `context_trials.py`) -- a distinct, separately validated schema from the
  v0.1 pilot's coarser integer 1-5 scale (`run_trial.validate_context_single_response`
  vs. `validate_single_response`; the pilot's own task/validator are
  untouched). The extra resolution is anchored, not decorative: the task
  instruction gives five verbal bands (very poor / weak / mixed-average /
  good / excellent) spanning the 1-10 range, so a score like 7.3 means
  something rather than being arbitrary precision. The point of the finer
  scale is more resolution than 5 integer bins -- **not** to eliminate ties;
  exact ties remain fully legitimate and are never broken artificially (see
  "Rankings and human alignment" below). Every story also gets its own
  explicit no-context `neutral` **baseline** trial
  (`context_packets.neutral_condition`), distinct from any of the v0.1
  pilot's own conditions. This neutral condition is a **reference point for
  measuring context sensitivity, not a ground-truth score and not assumed
  unbiased** -- see "Treatment vs. neutral baseline" below.
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

### Four-cell counterbalanced pairwise block

For a story pair {story_1, story_2} and a contrast {value a, value b},
`context_contrasts.build_contrast_block()` crosses BOTH of:

- **assignment** -- which contrast value each story is attributed
  ("forward": story_1 gets `a`, story_2 gets `b`; "flipped": the reverse),
- **position** -- which story is DISPLAYED as Story A ("story1_as_a" or
  "story2_as_a")

into 4 cells sharing one `block_id`:

```
cell 1: assignment=forward, position=story1_as_a -> A=story_1(a), B=story_2(b)
cell 2: assignment=forward, position=story2_as_a -> A=story_2(b), B=story_1(a)
cell 3: assignment=flipped, position=story1_as_a -> A=story_1(b), B=story_2(a)
cell 4: assignment=flipped, position=story2_as_a -> A=story_2(a), B=story_1(b)
```

An earlier version held position fixed (story_1 was always Story A) and
only varied assignment -- that isolates context sensitivity from story
identity, but leaves display position fully confounded with story identity,
so it can't separate a context-assignment effect from a raw display-position
effect. Crossing both factors makes them separable: cells 1/2 share one
context assignment and differ only in position; cells 3/4 share the other
assignment and also differ only in position. (See `context_contrasts.py`
for two more rejected designs: the same story shown twice is detectable by
the model, and one side with no context at all isn't a valid control
either.)

Every cell's trial carries `story_1_id`/`story_2_id` (the pair's fixed
identities, for grouping a block regardless of display position) alongside
`story_a_id`/`story_b_id` (whichever story is actually shown in which
position for that cell) and `context_a`/`context_b`, so analysis can always
recover "which story got which context" and "which story was chosen"
independent of the raw A/B letters -- see `analyze_context.choice_to_story_id`
and "Rankings and human alignment" below.

**Important limitation:** because every `context_pairwise` observation pits
two *different* claimed context values against each other (never the same
value on both sides), these trials are built to measure **causal context
sensitivity** -- whether the display-position-corrected preference changes
with context assignment -- not to produce a ranking "under" one context
condition. A ranking pooled across every contrast/cell is a rough diagnostic
at best (see "Rankings and human alignment" below); it is not a
`context_single`-style per-condition result and must not be read as one.

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

## Sampling regime

Two named sampling regimes exist (`run_trial.SAMPLING_REGIMES`), and every
v0.2 result row records which one produced it:

- **`low_variance_primary`** (default) -- pins `temperature=0` explicitly,
  the lowest-variance inference setting the API exposes for this request
  type, for the cleanest causal/context-sensitivity measurement. This is
  **not** an assumption that responses are literally deterministic --
  repeated replicates are still collected, and backend nondeterminism is
  expected to be empirically visible in them (see "Replication" below).
  This is the regime the primary benchmark analysis is run under.
- **`provider_default_secondary`** -- no temperature override; provider
  defaults. A distinct, later robustness check on whether the same effects
  persist under more naturalistic usage. Supported by the runner and result
  schema now; **not run** by anything in this repo.

`analyze_context.py` only ever analyzes one regime at a time
(`--sampling-regime`, default `low_variance_primary`) and reports how many
observations it excluded because they belonged to the other regime -- the
two regimes cannot be silently pooled by omission.

## Replication

`run_batch.py --replicates-treatment N --replicates-neutral M` (falling
back to `--replicates` for either when unset) lets the neutral no-context
baseline be sampled at a different rate than treatment conditions. By
default `M` equals `N`; setting `M` below `N` is a configuration error the
runner refuses, since the neutral baseline is reused as the reference point
for every treatment-vs-neutral delta on that story and should never be
sampled less than any individual treatment condition. There is no hardcoded
multiplier -- how much more precisely to estimate neutral is a call made at
run time, not baked into the code.

Repeated replicate observations of the exact same trial are never reduced
to a majority vote or filtered as "noise": every valid response is kept as
its own observation (see `analyze_context.collapse_attempts`, which
collapses failed *attempts*, never *replicates*). The design goal is enough
repetition to estimate quantities like P(story X preferred | condition C)
or a rating's typical spread, not to treat one API call as the model's
fixed, deterministic answer. Final replication counts are a decision to
make when choosing what to actually run, not fixed by this document.

## Human reference ranking

A fixed human preference ranking of the 12-text corpus, derived from human
pairwise judgments (`data/human_pairwise.jsonl`), not invented or estimated.
`human_ranking.py` checks the judgments for cycles/contradictions and
computes a full ranking only when the judgments collected so far uniquely
determine one (via topological sort, requiring the "next" story to be
unambiguous at every step). This was meant to let just enough adaptive
pairwise comparisons be collected to pin down a full order, rather than
requiring all C(12,2) = 66 pairs, and in the end 31 judgments were needed and
collected.

**Status: complete.** All 31 judgments in `data/human_pairwise.jsonl` are
free of direct contradictions and cycles, and uniquely determine a full
ranking. `human_ranking.py` has written `data/human_reference.json`
(`num_judgments: 31`):

```
1. santa            5. buddy            9.  dunnest_smoke
2. gilbert          6. dana_brownies    10. prophet
3. eyecut_lowway    7. morrow_transport 11. charlotte_train
4. saturn           8. afterlife        12. qual_panic
```

`analyze_context.py` picks this up automatically (see "Rankings and human
alignment" below) -- no code change was needed for this, since it was
already written to gracefully use the reference once it exists.

## Four distinct questions, not one "ranking" analysis

`analyze_context.py` deliberately keeps four questions separate rather than
collapsing them into a single ranking result:

|            | A. Effect of context (this study's main question)              | B. Resemblance to the human reference |
|------------|-------------------------------------------------------------|----------------------------------------|
| Single-text | Does context move the SAME story's score relative to its own neutral baseline? -- `analyze_treatment_vs_neutral` | Does the (possibly tied) score ordering under one condition resemble the human ordering? -- `rank_from_single_text_by_condition` + Kendall tau-b / tie-aware Spearman |
| Pairwise    | Does assigning context to a story change its probability of being preferred? -- `analyze_directional_pairwise_effects` | Do direct model pairwise choices resemble the human's direct pairwise judgments? -- `analyze_pairwise_vs_human_reference` |

### Single-text A: treatment vs. neutral baseline

For every story and treatment condition:

```
delta = treatment_rating - neutral_baseline_mean(model, story)
```

computed for all five rating dimensions, including `overall_quality`. The
neutral baseline is a **reference point**, not ground truth and not assumed
unbiased -- see "Sampling regime" and "Replication" above for how it's
estimated. Reported at three levels (never only the most-aggregated one):
individual treatment observation, story x treatment condition (averaged
across that story's replicates), and aggregated model x dimension x value
(averaged across every story). CSVs: `treatment_vs_neutral_observations.csv`,
`..._by_story_condition.csv`, `..._by_model_dimension_value.csv`.

### Single-text B: tie-aware ranking vs. the human reference

`rank_from_single_text_by_condition` builds one score dict per
`(model, dimension, value)` -- including the neutral baseline as its own
condition -- and **never** breaks a tie via story ID, filename, insertion
order, or any other arbitrary field: `tied_groups()` reports the actual tie
groups and their shared rank position. Against `data/human_reference.json`:

- **Kendall's tau-b (PRIMARY)** -- correctly excludes tied pairs from the
  concordant/discordant count on either side, rather than forcing an
  arbitrary order (which the tie-free tau-a formula would require).
- **Tie-aware Spearman correlation (SECONDARY)** -- Pearson correlation of
  each side's *average* ranks, the standard tie-corrected formula; never the
  tie-free shortcut.
- A separate **concordant / discordant / model_tied** count against the raw
  judgments in `data/human_pairwise.jsonl` -- a model tie is reported as
  `model_tied`, never silently converted into a fabricated win or loss.

Written to `results/context_analysis/human_comparison.csv` and
`single_text_rankings_by_condition.csv`. A pooled ranking across all
conditions (`single_text_pooled_diagnostic`) is also computed but is
explicitly a **diagnostic only**: it hides exactly the per-condition
distinction the benchmark exists to measure (averaging can even wash out a
real, opposite-signed effect into an apparent null -- see the offline
verification in the implementation notes).

### Pairwise A: directional context-sensitivity effect (PRIMARY)

For each `(model, contrast, story_1, story_2, category)`,
`analyze_directional_pairwise_effects` estimates, pooling over the
counterbalanced display position and over replicates:

```
P(story_1 preferred | story_1 receives contrast value "a")
  - P(story_1 preferred | story_1 receives contrast value "b")
```

The sign says which direction: positive means story_1 is favored more when
it carries value "a". This is never collapsed into a single
`changed: true/false` -- two blocks with opposite-signed effects (story_1
favored under "a" in one, under "b" in the other) both show
`changed: true` under the old raw-choice comparison, but only the
directional estimate tells them apart. CSVs:
`pairwise_directional_effects.csv` (per story pair) and
`..._by_contrast.csv` (aggregated across story pairs per contrast/category
-- read this aggregate cautiously, since it can average opposite-signed
per-pair effects back down toward zero, same as any aggregate).

A **secondary** diagnostic, `analyze_pairwise_changed_diagnostic`,
reproduces the original "did the raw A/B/tie choice change between forward
and flipped" comparison, restricted to one fixed display position so it
doesn't conflate position with context assignment. It preserves chosen
story IDs but is not the primary result.

### Pairwise B: direct comparison against human pairwise judgments

`analyze_pairwise_vs_human_reference` never builds a derived ranking for
this: for every `context_pairwise` observation whose two displayed stories
exactly match a known judgment in `data/human_pairwise.jsonl`, it checks
whether the model's `overall_quality` choice agrees (`concordant`),
disagrees (`discordant`), or was a tie (`model_tied`) -- directly, in story
identity, pair by pair.

### Diagnostics, clearly labelled, never the headline result

A pooled single-text ranking and a pooled pairwise Copeland ranking (both
tie-aware in the same way as above) are still computed as rough sanity
checks, written to `*_pooled_diagnostic.csv` files. As before, pooled
pairwise rankings additionally can't be read as a ranking "under" any one
context condition (see "Four-cell counterbalanced pairwise block" above;
the optional same-context family would be needed for that).

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
