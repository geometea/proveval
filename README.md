# Proveval

Proveval studies how external contextual cues change LLM evaluations of
fixed prose, and how robust those evaluations are to contextual
perturbation — for example, being told a story was "written by an AI"
versus "written by me," found in "a literary journal" versus "somewhere
random online," or that the user "really liked" versus "wasn't a fan of"
it. It measures both **naturalistic context sensitivity** (does the cue
move an ordinary evaluation?) and **text-only invariance** (can the model
resist that same cue when explicitly instructed to judge only the prose?).

A fixed single-researcher **ordinal** preference ranking over the corpus
provides a secondary, personalized reference point: it lets us ask whether
model judgments — and the shifts context induces in them — move toward or
away from one person's preferences. It is not ground truth, not a cardinal
preference function, and not the organizing goal of this project; see
"Researcher reference ranking" in `FINAL_DESIGN.md` for exactly what it is
and isn't.

## Status

- **v0.1 pilot: complete.** See `PILOT_RESULTS.md` for what ran and what was
  found — 138 completed observations across single-story ratings and
  pairwise comparisons on 4 stories; identical-text controls all tied
  (20/20); an exploratory AI-label advantage in self-vs-AI comparisons; no
  effect in journal-vs-AI comparisons. This pipeline and its results stay
  independently reproducible and are never pooled with anything below.
- **v0.2 context benchmark: implemented, not yet run for real.** See
  `FINAL_DESIGN.md` for the current design (this supersedes that file's
  earlier, narrower provenance-only version), including a methodology pass
  that: moved single-text ratings to a 1.0-10.0 decimal scale; made rating
  ties tie-aware throughout analysis (Kendall tau-b primary, tie-aware
  Spearman secondary — never broken by story ID or insertion order); added
  an explicit treatment-vs-neutral-baseline delta analysis; made the neutral
  baseline's replicate count independently configurable; added named,
  recorded sampling regimes (a primary low-variance regime and a supported-
  but-not-run provider-default secondary); replaced the 2-cell
  forward/flipped pairwise design with a 4-cell block that counterbalances
  context assignment against display position, with a directional
  (not just boolean) effect estimate; and added a second, independent
  `evaluation_regime` factor (`naturalistic` vs. `text_only_invariance`) so
  ordinary context sensitivity and explicit-instruction invariance can be
  compared directly for the same contextual cue — see "Two questions" in
  `FINAL_DESIGN.md`. The full trial generator, response validators, batch
  runner, and offline analysis all exist and are exercised with dry-runs and
  synthetic fixtures — see `CONTEXT_PACKETS.md` for exactly what's built.
  No benchmark API calls have been made yet.
- **Researcher reference ranking (secondary benchmark): complete.** 31
  pairwise judgments are recorded in `data/human_pairwise.jsonl`, with no
  contradictions or cycles, and they uniquely determine a full **ordinal**
  ranking of all 12 corpus stories — one researcher's ordering, not a
  population preference or a cardinal utility. `human_ranking.py` has
  written `data/human_reference.json` (`num_judgments: 31`), which
  `analyze_context.py` now picks up automatically for the secondary
  reference-agreement analyses (Kendall tau-b / tie-aware Spearman /
  pairwise concordance) — see "Researcher reference ranking" in
  `FINAL_DESIGN.md` for why this is secondary and exploratory, not the
  project's organizing goal.

See `EXPERIMENT.md` for the original v0.1 design write-up.

## Repository layout

### Stories

- `data/stories/*.txt` — 12 short story texts.
- `data/items.jsonl` — the registered story manifest (`id` → `path`), all 12.
  `make_trials.py` (the v0.1 pilot pipeline) is pinned to the original 4
  pilot stories regardless of what's registered here, so expanding this file
  doesn't change the pilot's reproducibility.

### v0.1 pipeline (the completed pilot)

- `data/conditions.jsonl` — single-story provenance conditions (7: `neutral`,
  `self`, `neutral_metadata`, `professor`, `literary_journal`, `ai`,
  `friend_job`).
- `data/tasks.jsonl` — the single-story rating task (5 dimensions, each
  scored 1–5).
- `data/comparison_conditions.jsonl` — comparative provenance conditions (5:
  `neutral`, `ai_vs_journal`, `journal_vs_ai`, `self_vs_ai`, `ai_vs_self`).
- `prompts.py` — builds a single-story rating prompt from a story +
  condition + task.
- `comparisons.py` — builds pairwise comparison prompts: ordinary story
  pairs, plus identical-text controls (a story compared against itself).
- `make_trials.py` — combines both into `data/trials.jsonl` (108 trials: 28
  single, 60 comparison, 20 identical-text control; pinned to the 4 pilot
  stories on purpose — see above).
- `reparse_results.py` — re-runs response parsing/validation over an
  existing results file without making new API calls.
- `analyze.py` — collapses retries, prints a dataset inventory, runs the
  identical-text-control and provenance-swap analyses, writes CSVs to
  `results/analysis/`.

### v0.2 context benchmark (implemented; not yet run for real)

- `data/context_dimensions.jsonl` — dimension definitions: `provenance`,
  `writer_status`, `source_venue`, `editing_status`, `reception`,
  `user_opinion` (all `scope: "story"`, attributed to one story), and
  `prompt_context` (`scope: "prompt"`, an extraneous aside attached to
  neither story).
- `data/context_contrasts.jsonl` — 13 story-scope value-vs-value contrasts
  (provenance incl. named-model and self-family contrasts, source/venue,
  editing status, reception, user opinion, self-vs-known-other).
- `data/context_prompt_contrasts.jsonl` — 1 prompt-scope contrast (a minimal
  weather-mention aside), structured so more can be added without code
  changes.
- `data/context_tasks.jsonl` — v0.2's own single-text rating task: a
  1.0-10.0 decimal scale (one decimal place, e.g. 7.3) with five verbal
  anchor bands, separate from the v0.1 pilot's coarser integer 1-5 task in
  `data/tasks.jsonl`. Two variants, one per `evaluation_regime`
  (`context_rating_naturalistic` / `context_rating_text_only_invariance`),
  identical except for the evaluation instruction — see "Two questions" in
  `FINAL_DESIGN.md`.
- `context_packets.py` — loads dimensions, renders a packet into one natural
  paragraph, generates single-variable example conditions plus the explicit
  `neutral_condition()` no-context baseline.
- `context_comparisons.py` — the natural "help me choose between these two
  stories" pairwise prompt template (shared by the pairwise builders below).
  Two independent axes: `evaluation_regime` (differing only in one added
  evaluation-rule sentence for `text_only_invariance`) and `choice_mode` --
  `forced` (PRIMARY: A/B only, no tie) or `tie_allowed` (SECONDARY hedging
  diagnostic: A/B/tie), each with its own JSON example and choice
  instruction, so the prompt text itself never leaves the mode ambiguous.
- `context_contrasts.py` — `build_contrast_block()` builds the full 4-cell
  counterbalanced block from `data/context_contrasts.jsonl` for a story
  pair: context assignment (which story gets which value) crossed with
  display position (which story is shown as Story A), so the two effects
  are separable rather than confounded. `evaluation_regime` is a third,
  independent factor crossed with every block.
- `context_single_prompts.py` — drives the single-story rating pipeline
  (`prompts.py`, unmodified) with context-packet signals, using v0.2's own
  decimal task.
- `context_trials.py` — the manifest generator: combines the above into
  `data/context_trials.jsonl` (7680 trials — every family generated once per
  `evaluation_regime`: 552 `context_single` — 12 stories × (22
  single-variable conditions + 1 neutral no-context baseline) × 2 regimes —
  6864 `context_pairwise` (66 pairs × 13 contrasts × 4 cells × 2 regimes),
  264 `context_prompt`), each trial carrying explicit structured metadata
  (including `evaluation_regime`) rather than requiring trial_id parsing.
  Also generates an OPTIONAL, never-run same-context pairwise family (1254
  trials, naturalistic only, both stories sharing one claimed context value)
  to a separate gitignored file,
  `data/context_trials_optional_same_context.jsonl` — not part of the
  required manifest or any planned run. Additionally generates the
  SECONDARY `choice_mode="tie_allowed"` hedging-diagnostic family (6864
  trials, both regimes) to its own gitignored file,
  `data/context_trials_tie_allowed.jsonl` — also not part of the required
  manifest; running it is a separate decision from running the primary
  forced-choice manifest.
- `run_trial.py` / `run_batch.py` — accept `--trials-file`, `--results-file`,
  and `--model`, so the v0.1 pilot and v0.2 benchmark never share a results
  file by accident. `--sampling-regime` (`low_variance_primary` default, or
  `provider_default_secondary`) is recorded on every v0.2 result row and
  ignored/unrecorded for v0.1 trial types. `run_batch.py` also accepts
  `--replicates-treatment`/`--replicates-neutral` (independent replicate
  counts for treatment vs. the neutral baseline; neutral defaults to, and
  can never be configured below, the treatment count) and
  `--evaluation-regime` (filter to `naturalistic` and/or
  `text_only_invariance`; repeatable). `--limit` selects complete
  experimental units, never orphaning a `context_pairwise` 4-cell block (see
  `run_batch.group_trials_into_units`) — every other trial type's unit is
  one trial, unchanged. Response validation accepts the v0.2 1.0-10.0
  decimal single-text schema, the forced-choice (A/B only) and tie-allowed
  (A/B/tie) pairwise-context schemas — dispatched by each trial's
  `choice_mode`, never coercing a tie into a forced A/B answer — and the
  original v0.1 integer 1-5 and story_a/story_b/preference schemas — all
  fully backward compatible. `--dry-run` never calls the API in either
  script.
- `data/human_pairwise.jsonl` — one researcher's pairwise judgments (winner,
  loser); 31 judgments, all actually made (never invented). This is the raw
  input to the secondary, personalized reference ranking below — not a
  population survey.
- `human_ranking.py` — checks those judgments for cycles/contradictions and
  writes `data/human_reference.json` only once they uniquely determine a
  complete ranking. They now do — see Status above.
- `data/human_reference.json` — the complete, unique 12-story **ordinal**
  researcher reference ranking derived from the above (regenerate with
  `python3 human_ranking.py`; only written when the judgments are
  contradiction-free and uniquely determine a full order). An ordering only
  (A > B > C...), not a cardinal preference score, and not ground truth —
  see "Researcher reference ranking" in `FINAL_DESIGN.md`.
- `analyze_context.py` — separate from `analyze.py`: collapses retries
  (never replicates), analyzes only one `--sampling-regime` at a time, and
  is organized PRIMARY-first, SECONDARY-second (see FINAL_DESIGN.md's
  "Primary and secondary empirical questions") rather than collapsing
  everything into one ranking. **Primary**: single-text
  treatment-vs-neutral-baseline deltas, and a directional pairwise
  context-sensitivity effect in story identity (not just a boolean
  "changed") — both compared across `evaluation_regime` with a descriptive
  naturalistic-vs-invariance "attenuation" label. **Secondary**: a tie-aware
  single-text ranking's agreement with the researcher reference (Kendall
  tau-b primary statistic, tie-aware Spearman secondary statistic,
  concordant/discordant/model_tied counts — never an artificial tie-break),
  and a direct pairwise-choices-vs-researcher-judgments comparison — both
  personalized/exploratory, not evidence of an objectively "best" context.
  Every analysis is stratified by `evaluation_regime`
  (`naturalistic` / `text_only_invariance` are never pooled). Pooled
  single-text and pairwise rankings are still computed but clearly labelled
  **diagnostic only**. Writes CSVs to `results/context_analysis/`.
  `analyze_context.py` itself is a thin orchestrator (argument parsing, the
  fixed print/CSV order, and the one function genuinely spanning both
  PRIMARY effects) — the actual analysis logic lives in five focused
  modules it imports from and re-exports, each independently readable:
  `context_analysis_common.py` (shared constants), `context_analysis_io.py`
  (loading/collapsing results, sampling_regime filtering, inventory),
  `context_analysis_stats.py` (tie-aware rank statistics, the descriptive
  attenuation label), `context_analysis_single_text.py` (PRIMARY
  treatment-vs-baseline), `context_analysis_pairwise.py` (PRIMARY
  directional effect, position/interaction/heterogeneity/leave-one-out, the
  optional tie-allowed diagnostic, context_prompt), and
  `context_analysis_reference.py` (SECONDARY researcher-reference
  agreement). `import analyze_context as ac` still exposes every function
  by its original name.
- `results/context_raw.jsonl` — suggested results path for the benchmark
  (via `--results-file`); not committed, and never the same file as the
  pilot's `results/raw.jsonl`.

### Docs

- `EXPERIMENT.md` — the original v0.1 design.
- `PILOT_RESULTS.md` — what the v0.1 pilot actually found.
- `FINAL_DESIGN.md` — the current design for the v0.2 context benchmark.
- `CONTEXT_PACKETS.md` — the context-packet data model, the prompt-design
  bugs found and fixed while building it, and exactly what's wired up versus
  still planned.

## Tests

`tests/` is a persistent pytest suite (no synthetic fixtures are generated
and deleted by hand anymore -- these run every time):

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/          # 268 tests, offline, no API calls, < 5s
```

- `test_run_trial_validation.py` — forced-choice vs. tie-allowed response
  schemas; confirms the v0.1 pilot's own schemas are untouched.
- `test_context_comparisons.py` — prompt-template correctness across the
  (evaluation_regime × choice_mode) grid.
- `test_context_contrasts.py` — the 4-cell counterbalanced block builder.
- `test_run_batch.py` — block-level `--limit` selection (never orphans a
  pairwise cell), replicate-count resolution, and a regression test for a
  real bug: `sampling_regime` was missing from `run_batch.py`'s own
  already-completed/already-failed bookkeeping, so a valid result recorded
  under one sampling regime would silently block a subsequent run of the
  same trial/replicate under the *other* regime into the same results file
  (now fixed — see `result_sampling_regime`/`load_existing_results`).
- `test_analyze_context.py` — tie-aware rank statistics, `collapse_attempts`
  regime separation, the position-effect/context×position-interaction
  decomposition, per-story heterogeneity + leave-one-story-out sensitivity,
  the tie-allowed hedging diagnostic, and the single-vs-pairwise
  disagreement case.
- `test_manifest_generation.py` — integration-level checks against the real
  `data/*.jsonl` files (manifest totals to exactly 7,680, trial IDs unique,
  the tie-allowed family stays the same size but disjoint, the v0.1 pilot
  stays pinned to its original 4 stories). No files are written; the
  builder functions are called directly rather than `context_trials.main()`.
- `test_llm_provenance_trials.py` / `test_analyze_llm_provenance.py` — the
  standalone named-LLM provenance experiment: contrast/trial well-formedness
  checks (`assert_contrasts_are_well_formed`/`assert_trials_are_well_formed`
  fail loudly on malformed input), the manifest's expected sizes, a full
  manual-inspection checklist for one block (fixed prose, correct
  provenance/position swaps, one shared `block_id`), `run_batch.py`
  block-level `--limit` reuse, and — via the same synthetic-fixture style as
  `test_analyze_context.py` — the antisymmetric provenance-effect matrix and
  its separation from the display-position effect.
- `test_controllability_trials.py` / `test_analyze_controllability.py` —
  the context-controllability experiment (the current experiment): frozen
  contrast wordings, the exact naturalistic-vs-suppression prompt diff
  ("Passage", plain A/B, no JSON), `prompt_sha256` correctness, both
  manifests' expected sizes (2,640 treatment / 132 blind-baseline trials),
  `run_batch.py` block-level `--limit` reuse and `prompt_sha256`-aware
  completion checking, the `parse_plain_ab_response` parser (accepted forms,
  ambiguity/tie/refusal rejection, JSON fallback), complete-unit filtering
  (4-cell treatment / 2-cell baseline), the `suppression_magnitude` table,
  the retained position-bias/leave-one-story-out diagnostics, the
  per-contrast (never pooled) baseline-margin correlation, and first-attempt
  response compliance.

Nothing here makes an API call or reads `ANTHROPIC_API_KEY`.

## Running things

```bash
# --- v0.1 pilot (unchanged) ---
python3 prompts.py                      # preview single-story prompts
python3 comparisons.py                  # preview comparison prompts
python3 make_trials.py                  # regenerate data/trials.jsonl (108 trials)
python3 run_trial.py <trial_id>                         # one trial, real API call
python3 run_batch.py --type single --limit 5 --dry-run  # preview a batch
python3 run_batch.py --type single --limit 5            # run it for real
python3 analyze.py                      # analyze results/raw.jsonl

# --- v0.2 context benchmark ---
python3 context_packets.py              # preview single-variable example conditions
python3 context_contrasts.py            # preview the 4-cell counterbalanced story-scope prompts
python3 context_single_prompts.py       # preview single-story context prompts (1.0-10.0 decimal scale)
python3 context_trials.py               # regenerate data/context_trials.jsonl (7680 trials, both evaluation regimes)
                                         # + optional never-run same-context family (separate file)

# dry-run against the new manifest, restricted to one evaluation_regime
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_pairwise --contrast provenance_claude_vs_human \
    --evaluation-regime naturalistic --limit 10 --dry-run

# the same slice under the text-only-invariance instruction, for comparison
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_pairwise --contrast provenance_claude_vs_human \
    --evaluation-regime text_only_invariance --limit 10 --dry-run

# sample the neutral no-context baseline more precisely than each treatment condition
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_single --replicates-treatment 3 --replicates-neutral 6 --dry-run

# for real (needs ANTHROPIC_API_KEY; choose a deliberately small slice first)
# --sampling-regime defaults to low_variance_primary (temperature=0); pass
# provider_default_secondary to run the (currently unused) robustness regime
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_pairwise --contrast provenance_claude_vs_human --limit 10

python3 analyze_context.py              # analyze results/context_raw.jsonl (low_variance_primary by default;
                                         # both evaluation regimes present are stratified, never pooled)
python3 analyze_context.py --sampling-regime provider_default_secondary  # analyze the other sampling regime explicitly

# --- researcher reference ranking (secondary benchmark) ---
python3 human_ranking.py                # report + write data/human_reference.json once complete
```

All `--dry-run` invocations above are safe to run offline with no
`ANTHROPIC_API_KEY` set — nothing in this repo makes an API call unless
`--dry-run` is omitted.

## Standalone named-LLM provenance experiment (earlier/auxiliary)

An earlier, auxiliary sub-experiment (implemented and dry-run-tested, but
not the current focus — see "Context-controllability experiment" below for
that), layered entirely on top of the v0.2
`context_pairwise` machinery above: given the same underlying pair of
stories, does an evaluator's preference change depending on whether each
story is *claimed* to have been generated by Claude, ChatGPT, Gemini, or
DeepSeek? Only the claimed authorship changes — the prose is always one of
the existing 12 registered corpus stories, byte-identical across every
cell. This is **not** a comparison of actual outputs produced by those
models (no non-Anthropic API is called anywhere in this repository); it
measures the provenance-*label* sensitivity of whichever evaluator
`run_batch.py --model` runs (currently Claude), never conflating that
evaluator with the claimed-authorship labels under test.

It is additive, not a new experimental architecture: every trial it
generates is an ordinary `"type": "context_pairwise"`,
`"choice_mode": "forced"` trial, built by the same
`build_contrast_block()`/`build_context_pairwise_trials()` 4-cell
counterbalance (context assignment × display position) used everywhere
else in this README, reusing the same forced-choice prompt format and
response schema. The only thing narrowing it to this focused question is
which contrasts get generated: `data/llm_provenance_contrasts.jsonl` holds
exactly the six unordered pairs (`C(4,2)`) of the four named-LLM provenance
values, instead of the full general-benchmark `data/context_contrasts.jsonl`
set. Every trial also carries an `"experiment_id": "llm_provenance_pairwise_v1"`
field so a shared results file can still tell this manifest's rows apart
from the general benchmark's.

Because the six pairwise effects aren't assumed transitive, there is
deliberately no Bradley-Terry/Elo ranking here — a scalar ranking would
hide exactly the kind of non-transitive pattern (e.g. Claude > ChatGPT >
Gemini > Claude) this experiment exists to be able to show. The primary
output is the raw 4×4 antisymmetric matrix of directional treatment
effects (never called "win rates") for the `overall_quality` category,
alongside the same position-effect, heterogeneity, and
leave-one-story-out diagnostics used by the general benchmark.

```bash
# generate the focused manifest (naturalistic evaluation regime only, by
# default: 66 story pairs x 6 contrasts x 4 cells = 1,584 trials)
python3 llm_provenance_trials.py

# both evaluation regimes (3,168 trials)
python3 llm_provenance_trials.py --evaluation-regime naturalistic --evaluation-regime text_only_invariance

# smoke test: --limit operates on whole 4-cell blocks, so --limit 2 selects
# exactly 2 complete blocks (8 planned calls)
python3 run_batch.py --trials-file data/llm_provenance_trials.jsonl \
    --results-file results/llm_provenance/raw.jsonl \
    --type context_pairwise --limit 2 --dry-run

# for real (needs ANTHROPIC_API_KEY; choose a deliberately small slice first)
python3 run_batch.py --trials-file data/llm_provenance_trials.jsonl \
    --results-file results/llm_provenance/raw.jsonl \
    --type context_pairwise --limit 2

# analyze results/llm_provenance/raw.jsonl -- console summary plus tidy CSVs
# under results/llm_provenance/analysis/ (pairwise_effects.csv,
# per_story_pair_effects.csv, overall_quality_matrix.csv, position_effects.csv,
# per_story_context_effects.csv, leave_one_story_out.csv)
python3 analyze_llm_provenance.py
```

See `llm_provenance_trials.py` and `analyze_llm_provenance.py` for the full
rationale; `FINAL_DESIGN.md` is not rewritten around this experiment since
it introduces no new architecture, only a second, smaller contrast set and
manifest layered on the existing one.

## Context-controllability experiment (current experiment)

This is the experiment actively being developed and run in this repo. The
v0.2 broad context benchmark ("Standalone named-LLM provenance experiment"
above, and the general `context_trials.py` manifest under "Repository
layout") are earlier/auxiliary work: implemented, exercised with dry-runs
and synthetic fixtures, but not the current focus.

A focused sub-experiment, layered on the same v0.2 `context_pairwise`
architecture (4-cell assignment×position counterbalance, `evaluation_regime`,
`run_batch.py`, result persistence/retry), that asks a different question
from the broad benchmark: not just *whether* a contextual cue moves an
evaluation, but *how suppressible* each one is. For 5 isolated context
contrasts, each pitting one frozen, deterministic pair of clauses against
the other (never generated or paraphrased) — see
`data/controllability_contrasts.jsonl`:

- provenance: "was written by a human, but I don't know who" vs. "was
  generated by an LLM"
- source: "is something I found in a literary journal" vs. "is something I
  stumbled across somewhere random online"
- reception: "has been very well received by the people who've read it" vs.
  "has been very poorly received by the people who've read it"
- user opinion: "is one I really liked" vs. "is one I really didn't like"
- editing: "has already been carefully edited" vs. "is an unrevised first
  draft"

Every text pair is run under both existing evaluation regimes, with the
contextual claim *shown in both*:

- `naturalistic`: the ordinary framing (`naturalistic_effect`).
- `text_only_invariance` (the suppression condition): the exact same
  contextual claim is shown, plus exactly one added sentence — "For this
  comparison, I'm only interested in the writing itself. Please base your
  judgment on the passages as written." — asking the evaluator to judge the
  writing itself. Nothing else differs between the two regimes' prompts for
  the same cell (see `controllability_trials.SUPPRESSION_INSTRUCTION`); no
  new regime is introduced.

`suppression_magnitude = abs(naturalistic_effect) - abs(text_only_effect)`
is the primary suppression measure (positive = the instruction shrank the
effect; negative = the effect got larger under the instruction); the signed
`naturalistic_effect - text_only_effect` and the residual `text_only_effect`
itself are also kept, never discarded. Each trial tests only one context
type (no combined-context trials), and every trial is an ordinary
`"type": "context_pairwise"`, `"choice_mode": "forced"` trial with the same
4-cell counterbalance used everywhere else. Trials carry
`"experiment_id": "context_controllability_v1"` and a deterministic
`"prompt_sha256"` (so a saved result can never be mistaken for satisfying a
trial whose prompt wording has since changed).

Single plain-language outcome, not the shared 5-field JSON rubric: the
prompt asks one question — "Which passage do you prefer overall? ... Please
answer A or B." (using "Passage", never "Story") — parsed by a dedicated,
conservative pattern matcher (`response_format="plain_ab"`, see
`run_trial.parse_plain_ab_response`; no LLM is used to parse) and normalized
internally to `{"overall_quality": "A"|"B"}` so the existing pairwise
analysis is reused unmodified with `categories=["overall_quality"]`.

A separate blind baseline (`data/controllability_baseline_trials.jsonl`)
shows each of the same 66 text pairs with *no* contextual framing at all,
once in each display order (naturalistic only — there is nothing to
instruct the evaluator to ignore). `baseline_margin = 2 * abs(p_text1_wins -
0.5)` (0 = the blind evaluator was close to indifferent; 1 = maximally
decisive) is related to `abs(context_effect)`, calculated **separately for
each (model, evaluation_regime, contrast_id)** — never pooled across the 5
contrasts — via a plain Pearson correlation
(`context_analysis_stats.pearson_correlation`, reused unmodified). A
**negative** correlation means context effects are larger on pairs the
evaluator was closer to indifferent about; never a new statistical method,
and never a Bradley-Terry/Elo ranking.

Before the primary analysis, `analyze_controllability.py` also: discards any
`(block_id, model, replicate_id, sampling_regime)` unit that doesn't have
all 4 (treatment) or both (baseline) cells validly answered — reporting how
many units were complete/incomplete, never silently — and produces a
first-attempt response-compliance report (`response_compliance.csv`, by
model/evaluation_regime/contrast_id) that keeps API-call failures
distinguishable from malformed/refused model responses; this is diagnostic
only and never treats an invalid response as an A/B judgment.

```bash
# generate both manifests: the treatment trials (66 pairs x 5 contrasts x
# 4 cells x 2 regimes = 2,640 trials) and the blind baseline (66 pairs x
# 2 positions = 132 trials)
python3 controllability_trials.py

# smoke test: --limit operates on whole 4-cell blocks, so --limit 2 selects
# exactly 2 complete blocks (8 planned calls)
python3 run_batch.py --trials-file data/controllability_trials.jsonl \
    --results-file results/controllability/raw.jsonl \
    --type context_pairwise --limit 2 --dry-run

# the blind baseline: --limit 2 selects 2 pairs x 2 positions (4 planned calls)
python3 run_batch.py --trials-file data/controllability_baseline_trials.jsonl \
    --results-file results/controllability/baseline_raw.jsonl \
    --type context_pairwise --limit 2 --dry-run

# for real (needs ANTHROPIC_API_KEY; choose a deliberately small slice first)
python3 run_batch.py --trials-file data/controllability_trials.jsonl \
    --results-file results/controllability/raw.jsonl \
    --type context_pairwise --limit 2
python3 run_batch.py --trials-file data/controllability_baseline_trials.jsonl \
    --results-file results/controllability/baseline_raw.jsonl \
    --type context_pairwise --limit 2

# reparse saved raw responses offline (e.g. after a parser fix), no API call
python3 reparse_results.py --trials-file data/controllability_trials.jsonl \
    --results-file results/controllability/raw.jsonl

# analyze both results files -- console summary (the natural/text_only/
# suppression_magnitude table, position-bias and leave-one-story-out
# diagnostics, and the per-contrast baseline-margin correlation) plus tidy
# CSVs under results/controllability/analysis/ (suppression_effects.csv,
# pairwise_effects.csv, per_story_pair_effects.csv, position_effects.csv,
# leave_one_story_out.csv, baseline_margin.csv,
# context_effect_vs_baseline_margin.csv, baseline_margin_correlation.csv,
# response_compliance.csv)
python3 analyze_controllability.py
```

See `controllability_trials.py` and `analyze_controllability.py` for the
full rationale; `FINAL_DESIGN.md` is not rewritten around this experiment
for the same reason as above.
