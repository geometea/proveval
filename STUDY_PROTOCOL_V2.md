# Study Protocol v2 — Context-Controllability Experiment

This is the pre-specified analysis plan for `context_controllability_v2`,
the v2 redesign of the focused context-controllability experiment. It lives
alongside (and does not replace) the v1 protocol implicit in
`controllability_trials.py`/`analyze_controllability.py`; v1 is unmodified.

This document, `data/controllability_v2_study_config.json`, and
`data/controllability_v2_contrasts.jsonl` together constitute the
pre-registration. Once the study config's `status` is flipped to `"frozen"`
(see `freeze_controllability_v2.py`), the numbers and rules below are what a
production run and its primary analysis are held to.

## Design summary

- **Question, verbatim, every prompt:** "Which passage is better written
  overall? If they're close, choose the one you think is slightly better
  written. Please answer A or B." No ratings, no ties, no JSON.
- **Two primary instruction conditions** (`instruction_condition`):
  `matched_control` ("For this comparison, please read both passages
  carefully before making your choice.") and `text_only` ("For this
  comparison, please read both passages carefully and judge only the writing
  itself."). For a given (story pair, contrast, assignment, position) cell,
  these two prompts differ in exactly that one sentence — nothing else.
  A third, optional `minimal_instruction` condition (no added sentence) is
  generated into its own manifest (`data/controllability_v2_minimal_trials.jsonl`)
  and is **never** part of a primary superblock or a primary analysis.
- **Five context contrasts** (`data/controllability_v2_contrasts.jsonl`,
  wording frozen and identical to v1): provenance (human vs. LLM), source
  venue (literary journal vs. random online), reception (positive vs.
  negative), user opinion (liked vs. disliked), editing status (edited vs.
  first draft). `a` is always the preregistered favourable direction;
  effect signs are read directly off `D_pair`/ATE and are never flipped
  after the fact based on which direction the data came out.
- **Four-cell counterbalance** per (story pair, contrast, instruction
  condition): context assignment (`forward`/`flipped`) × display position
  (`story1_as_a`/`story2_as_a`). A **superblock** is one (story pair,
  contrast)'s full 8 cells (4 `matched_control` + 4 `text_only`). 66 story
  pairs × 5 contrasts × 2 instruction conditions × 4 cells = **2,640**
  primary treatment prompts, in 330 superblocks.
- **Blind baseline:** the same 66 pairs, no contextual framing, both display
  orders, `matched_control` instruction only. 66 × 2 = **132** prompts,
  collected into a results file entirely separate from treatment.
- **Evaluators:** a study config names exactly one `primary` evaluator and
  zero or more `replication` evaluators (provider, requested model,
  reasoning profile). Model configuration is held constant within one
  evaluator across every experimental condition.

## Primary analyses

Run once per resolved evaluator (see "Evaluator separation" below), never
pooled across evaluators:

1. **Cue-specific context ATEs under matched-control** — for each contrast,
   `ATE_control = mean(D_pair)` over the 66 story pairs, with a story-level
   bootstrap 95% CI.
2. **Cue-specific context ATEs under text-only** — the same, for
   `ATE_text_only`.
3. **Control vs. text-only difference** —
   `signed_instruction_difference = ATE_text_only - ATE_control`, with its
   own paired story-bootstrap CI (same resample used for both terms in a
   draw).
4. **Magnitude reduction** —
   `magnitude_reduction = abs(ATE_control) - abs(ATE_text_only)` (positive =
   the instruction shrank the effect's magnitude).
5. **Residual context effect** — `residual_text_only_effect = ATE_text_only`
   itself, reported alongside the above rather than only as a difference.
6. **Equivalence result** — for `ATE_text_only`: 95% and 90% story-bootstrap
   CIs. The residual effect is classified `practically_invariant` only when
   the study is frozen AND the complete 90% CI falls inside
   `[-equivalence_margin, +equivalence_margin]` (margin from the frozen
   study config). An unfrozen config reports the intervals with no verdict.
7. **Ambiguity interaction using baseline strength** — per (contrast,
   instruction condition), fit `D_pair = intercept + beta * baseline_strength`
   over the 66 pairs (`baseline_strength` from the *independently collected*
   blind baseline, never from treatment data), with a story-bootstrap CI for
   `beta` and predicted effects at the 25th/50th/75th percentiles of the
   observed `baseline_strength` distribution. The old Pearson-correlation
   view (`baseline_margin` vs. `abs(context_effect)`, as in v1) is retained
   only as an optional descriptive diagnostic, never the primary ambiguity
   analysis.

Primary analysis includes **only complete planned superblocks** — a
(superblock, replicate) unit counts only if all 8 cells resolved to a valid
response; an incomplete unit is entirely excluded (never partially used),
and its exclusion is reported, never silent. The independent blind baseline
uses the analogous complete-unit rule (both positions per replicate).

## Secondary analyses

- **Replication across evaluator models** — every `replication` evaluator
  gets the exact same primary-analysis treatment, independently. A
  cross-model summary table is produced for comparison, but no step ever
  pools two evaluators' observations into one inferential sample.
- **Position interaction** — per (contrast, instruction condition): the
  context effect when story 1 is displayed as Passage A, the context effect
  when displayed as Passage B, their interaction, and the overall (context-
  independent) display-position effect. Every headline context effect
  above is the mean of the two position-specific effects, so it is always
  position-counterbalanced by construction.
- **Leave-one-story-out diagnostics** — for each contrast/instruction
  condition, the ATE recomputed excluding every pair that touches each of
  the 12 stories in turn, to check no single story drives the result.
- **Response-compliance diagnostics** — first-attempt valid/invalid/
  refusal/API-error rates by (contrast, instruction condition), computed
  from the first attempt only (never inflated by retries); primary analysis
  itself may use the first *valid* response for a planned observation,
  wherever in its (bounded) attempt sequence it occurred.
- **Baseline split-half reliability** — the blind baseline's per-pair
  win-rate estimate stability, from an odd/even replicate split.

## Fixed design rules (required before a production/frozen run)

The frozen study config (`data/controllability_v2_study_config.json`,
status `"frozen"`, checked against `data/controllability_v2_frozen_lock.json`
by `freeze_controllability_v2.verify_frozen`) must specify, before any
production execution:

- the evaluator roster (exactly one `primary`, zero or more `replication`)
- the 5 cues (contrast file path + contents, hashed into the lock)
- the exact prompt wording (question + both instruction sentences, hashed
  into the lock via the study config)
- the treatment replicate count (per cell)
- the baseline replicate count (per display order)
- the retry limit
- the equivalence margin
- the bootstrap draw count
- the randomisation seed

`run_controllability_v2.py` refuses to execute a real (non-`--dry-run`) run
unless the current on-disk config/contrasts/corpus/manifests still hash to
exactly what's in the lock file — any edit after freezing is caught, not
silently absorbed.

**No additional confirmatory replicates may be appended to an already-frozen
study after inspecting its results.** If more data is wanted after looking
at results — more replicates, a wider evaluator roster, a changed retry
limit — that is a **new wave**: a new `run_id`/design version and, if the
design itself changes, a new experiment_id, analyzed and reported
separately, never merged into the frozen study's own primary analysis
after the fact. This is enforced procedurally (the frozen lock refuses a
silent edit) and by convention (a new wave gets its own `run_id` recorded
on every result row via the evaluator identity — see
`controllability_v2_execution.build_evaluator_identity`).

## Evaluator separation

Every result row records a full evaluator identity: provider, requested
model, reasoning profile, provider reasoning settings, sampling settings,
and (from the response itself) `response_model`/`run_id`. Analysis matches
rows to the study config's named evaluators by (provider, requested_model,
reasoning_profile); a row matching none of them is excluded, never folded
into the nearest one. If one configured evaluator's requests come back
under more than one distinct `response_model` (an alias resolving to
different underlying versions), analysis **splits that evaluator into
separate resolved evaluators** (one per version) rather than pooling them —
or, with `--strict-model-version`, refuses to analyze at all. See
`analyze_controllability_v2.partition_by_resolved_evaluator`.
