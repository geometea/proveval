# Proveval

Proveval is an LLM-as-judge benchmark for subjective prose evaluation. It
tests whether LLM judges change their evaluation of fixed prose when given
task-irrelevant context — for example, being told a story was "written by an
AI" versus "written by me," found in "a literary journal" versus "somewhere
random online," or that the user "really liked" versus "wasn't a fan of" it.
It also asks whether some model/context combinations track a fixed human
preference ranking better than others.

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
  but-not-run provider-default secondary); and replaced the 2-cell
  forward/flipped pairwise design with a 4-cell block that counterbalances
  context assignment against display position, with a directional
  (not just boolean) effect estimate. The full trial generator, response
  validators, batch runner, and offline analysis all exist and are
  exercised with dry-runs and synthetic fixtures — see `CONTEXT_PACKETS.md`
  for exactly what's built. No benchmark API calls have been made yet.
- **Human reference ranking: complete.** 31 pairwise judgments are recorded
  in `data/human_pairwise.jsonl`, with no contradictions or cycles, and they
  uniquely determine a full ranking of all 12 corpus stories.
  `human_ranking.py` has written `data/human_reference.json`
  (`num_judgments: 31`), which `analyze_context.py` now picks up
  automatically for Spearman correlation and pairwise-agreement comparisons.

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
  `data/tasks.jsonl`.
- `context_packets.py` — loads dimensions, renders a packet into one natural
  paragraph, generates single-variable example conditions plus the explicit
  `neutral_condition()` no-context baseline.
- `context_comparisons.py` — the natural "help me choose between these two
  stories" A/B/tie prompt template (shared by the pairwise builders below).
- `context_contrasts.py` — `build_contrast_block()` builds the full 4-cell
  counterbalanced block from `data/context_contrasts.jsonl` for a story
  pair: context assignment (which story gets which value) crossed with
  display position (which story is shown as Story A), so the two effects
  are separable rather than confounded.
- `context_single_prompts.py` — drives the single-story rating pipeline
  (`prompts.py`, unmodified) with context-packet signals, using v0.2's own
  decimal task.
- `context_trials.py` — the manifest generator: combines the above into
  `data/context_trials.jsonl` (3840 trials: 276 `context_single` — 12
  stories × (22 single-variable conditions + 1 neutral no-context baseline)
  — 3432 `context_pairwise` (66 pairs × 13 contrasts × 4 cells), 132
  `context_prompt`), each trial carrying explicit structured metadata rather
  than requiring trial_id parsing. Also generates an OPTIONAL, never-run
  same-context pairwise family (1254 trials, both stories sharing one
  claimed context value) to a separate gitignored file,
  `data/context_trials_optional_same_context.jsonl` — not part of the
  required manifest or any planned run.
- `run_trial.py` / `run_batch.py` — accept `--trials-file`, `--results-file`,
  and `--model`, so the v0.1 pilot and v0.2 benchmark never share a results
  file by accident. `--sampling-regime` (`low_variance_primary` default, or
  `provider_default_secondary`) is recorded on every v0.2 result row and
  ignored/unrecorded for v0.1 trial types. `run_batch.py` also accepts
  `--replicates-treatment`/`--replicates-neutral` (independent replicate
  counts for treatment vs. the neutral baseline; neutral defaults to, and
  can never be configured below, the treatment count). Response validation
  accepts the v0.2 1.0-10.0 decimal single-text schema, the A/B/tie
  pairwise-context schema, and the original v0.1 integer 1-5 and
  story_a/story_b/preference schemas — all fully backward compatible.
  `--dry-run` never calls the API in either script.
- `data/human_pairwise.jsonl` — known human pairwise judgments (winner,
  loser); 31 judgments, all actually made (never invented).
- `human_ranking.py` — checks those judgments for cycles/contradictions and
  writes `data/human_reference.json` only once they uniquely determine a
  complete ranking. They now do — see Status above.
- `data/human_reference.json` — the complete, unique 12-story human
  preference ranking derived from the above (regenerate with
  `python3 human_ranking.py`; only written when the judgments are
  contradiction-free and uniquely determine a full order).
- `analyze_context.py` — separate from `analyze.py`: collapses retries
  (never replicates), analyzes only one `--sampling-regime` at a time, and
  keeps four questions distinct rather than collapsing them into one
  ranking (see FINAL_DESIGN.md): single-text treatment-vs-neutral-baseline
  deltas; a tie-aware single-text ranking vs. the human reference (Kendall
  tau-b primary, tie-aware Spearman secondary, concordant/discordant/
  model_tied counts — never an artificial tie-break); a directional pairwise
  context-sensitivity effect in story identity (not just a boolean
  "changed"); and a direct pairwise-choices-vs-human-judgments comparison.
  Pooled single-text and pairwise rankings are still computed but clearly
  labelled **diagnostic only**. Writes CSVs to `results/context_analysis/`.
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
python3 context_trials.py               # regenerate data/context_trials.jsonl (3840 trials)
                                         # + optional never-run same-context family (separate file)

# dry-run against the new manifest, writing to a separate results file
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_pairwise --contrast provenance_claude_vs_human --limit 10 --dry-run

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

python3 analyze_context.py              # analyze results/context_raw.jsonl (low_variance_primary by default)
python3 analyze_context.py --sampling-regime provider_default_secondary  # analyze the other regime explicitly

# --- human reference ranking ---
python3 human_ranking.py                # report + write data/human_reference.json once complete
```

All `--dry-run` invocations above are safe to run offline with no
`ANTHROPIC_API_KEY` set — nothing in this repo makes an API call unless
`--dry-run` is omitted.
