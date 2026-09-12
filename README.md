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
  earlier, narrower provenance-only version). The full trial generator,
  response validators, batch runner, and offline analysis all exist and are
  exercised with dry-runs and synthetic fixtures — see `CONTEXT_PACKETS.md`
  for exactly what's built. No benchmark API calls have been made yet.
- **Human reference ranking: partial.** 2 of the needed pairwise judgments
  are recorded in `data/human_pairwise.jsonl`; that's not enough to
  determine a full order yet, so `data/human_reference.json` does not exist.
  `human_ranking.py` will write it once enough judgments are collected.

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
- `context_packets.py` — loads dimensions, renders a packet into one natural
  paragraph, generates single-variable example conditions.
- `context_comparisons.py` — the natural "help me choose between these two
  stories" A/B/tie prompt template (shared by the pairwise builders below).
- `context_contrasts.py` — builds forward/flipped two-story trials from
  `data/context_contrasts.jsonl`: two different stories, both always
  carrying context, only the attribution swapping between forward/flipped.
- `context_single_prompts.py` — drives the existing single-story numeric
  rating pipeline (`prompts.py`, unmodified) with context-packet signals.
- `context_trials.py` — the manifest generator: combines the above into
  `data/context_trials.jsonl` (2112 trials: 264 `context_single`, 1716
  `context_pairwise`, 132 `context_prompt`), each trial carrying explicit
  structured metadata (story ids, dimension, contrast_id, assignment, etc.)
  rather than requiring trial_id parsing.
- `run_trial.py` / `run_batch.py` — now accept `--trials-file`,
  `--results-file`, and `--model`, so the v0.1 pilot and v0.2 benchmark never
  share a results file by accident. Response validation now also accepts
  the A/B/tie pairwise-context schema (5 categories, each "A"/"B"/"tie"),
  alongside the original 1-5 and story_a/story_b/preference schemas, fully
  backward compatible. `--dry-run` never calls the API in either script.
- `data/human_pairwise.jsonl` — known human pairwise judgments (winner,
  loser), seeded only with judgments actually made.
- `human_ranking.py` — checks those judgments for cycles/contradictions and
  writes `data/human_reference.json` only once they uniquely determine a
  complete ranking (not yet — see Status above).
- `analyze_context.py` — separate from `analyze.py`: collapses retries,
  joins each result to its trial's structured metadata (embedded in the
  result row, or by looking the trial back up if needed), summarizes
  forward/flipped and prompt-level context effects, computes two provisional
  rankings (pairwise Copeland/win-rate and single-text mean score), and
  compares each against the human reference (Spearman rank correlation, pure
  Python, plus pairwise agreement) when available. Writes CSVs to
  `results/context_analysis/`.
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
python3 context_contrasts.py            # preview forward/flipped story-scope prompts
python3 context_single_prompts.py       # preview single-story context prompts
python3 context_trials.py               # regenerate data/context_trials.jsonl (2112 trials)

# dry-run against the new manifest, writing to a separate results file
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_pairwise --contrast provenance_claude_vs_human --limit 10 --dry-run

# for real (needs ANTHROPIC_API_KEY; choose a deliberately small slice first)
python3 run_batch.py --trials-file data/context_trials.jsonl \
    --results-file results/context_raw.jsonl \
    --type context_pairwise --contrast provenance_claude_vs_human --limit 10

python3 analyze_context.py              # analyze results/context_raw.jsonl

# --- human reference ranking ---
python3 human_ranking.py                # report + write data/human_reference.json once complete
```

All `--dry-run` invocations above are safe to run offline with no
`ANTHROPIC_API_KEY` set — nothing in this repo makes an API call unless
`--dry-run` is omitted.
