# Proveval

Proveval tests whether LLM judges change their evaluation of fixed prose when
given irrelevant provenance or context information — for example, being told
a story was "written by an AI" versus "written by me," or that it was
"positively reviewed" versus "never shown to anyone." The goal is to find out
which contextual signals move a judge's ratings, independent of the writing
itself.

## Status

- **v0.1 pilot: complete.** See `PILOT_RESULTS.md` for what ran and what was
  found — 138 completed observations across single-story ratings and
  pairwise comparisons; identical-text controls all tied (20/20); an
  exploratory AI-label advantage in self-vs-AI comparisons; no effect in
  journal-vs-AI comparisons.
- **Final study: designed, not yet run.** See `FINAL_DESIGN.md` for the
  frozen post-pilot design — a 12-text corpus, target-anchor comparison
  units, neutral/self/other_human/ai provenance conditions, 288 primary +
  24 control observations.
- **Context-packet extension: built, not yet run.** See `CONTEXT_PACKETS.md`
  for a newer, more modular way to compose and contrast context signals
  (provenance × writer status × editing status × reception), including the
  design bugs found and fixed while building it. This only generates and
  prints prompts right now — no API calls, and it isn't wired into the
  trial/results/analysis pipeline below.

See `EXPERIMENT.md` for the original v0.1 design write-up.

## Repository layout

### Stories

- `data/stories/*.txt` — 12 short story texts.
- `data/items.jsonl` — the registered story manifest (`id` → `path`). Only 4
  stories (`gilbert`, `dunnest_smoke`, `prophet`, `santa`) are currently
  registered here and used by the pipeline below; the other 8 files exist on
  disk but aren't wired in yet.

### v0.1 pipeline (the pilot and the frozen final-study design)

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
- `make_trials.py` — combines both into `data/trials.jsonl` (currently 108
  trials: 28 single, 60 comparison, 20 identical-text control).
- `run_trial.py` — runs one trial through the Claude API
  (`python3 run_trial.py <trial_id>`; add `--dry-run` to preview without
  calling).
- `run_batch.py` — runs many trials at once, with `--type`, `--condition`,
  `--id-prefix`, `--replicates`, `--limit`, `--retry-failed`, and `--dry-run`.
  Tracks an `attempt_id` per observation and only retries observations with
  no successful attempt yet — failed attempts are never deleted.
- `reparse_results.py` — re-runs response parsing/validation over an
  existing `results/raw.jsonl` without making new API calls (for when a
  parsing bug is fixed after data has already been collected).
- `analyze.py` — collapses retries into one observation per
  `(trial_id, model, replicate_id)`, prints a dataset inventory, runs the
  identical-text-control and provenance-swap analyses, and writes tidy CSVs
  to `results/analysis/`.
- `results/raw.jsonl` — raw API responses. Not committed to the repo; local
  to wherever the batch runner was actually executed.

### Context-packet extension (exploratory, prompt generation only)

- `data/context_dimensions.jsonl` — dimension definitions (`provenance`,
  `writer_status`, `editing_status`, `reception`), each value paired with
  its natural-language phrasing.
- `data/context_contrasts.jsonl` — 8 explicit value-vs-value contrasts (e.g.
  Claude-generated vs. unknown human), each with hand-written intro-sentence
  clauses.
- `context_packets.py` — loads dimensions, renders a packet into one natural
  paragraph, and generates single-variable (empty-baseline-vs-one-signal)
  example conditions for the single-story case.
- `context_comparisons.py` — the natural "help me choose between these two
  stories" A/B/tie prompt template.
- `context_contrasts.py` — builds forward/flipped two-story comparison
  trials from `data/context_contrasts.jsonl`: two different stories, both
  always carrying context, only the attribution swapping between them.
- `context_single_prompts.py` — drives the existing single-story numeric
  rating pipeline (`prompts.py`, unmodified) with context-packet signals.

### Docs

- `EXPERIMENT.md` — the original v0.1 design.
- `PILOT_RESULTS.md` — what the v0.1 pilot actually found.
- `FINAL_DESIGN.md` — the frozen design for the next, larger study.
- `CONTEXT_PACKETS.md` — the context-packet data model, and the three
  prompt-design bugs found and fixed while building its comparison format.

## Running things

```bash
# Preview prompts without calling the API
python3 prompts.py
python3 comparisons.py
python3 context_packets.py
python3 context_contrasts.py
python3 context_single_prompts.py

# Regenerate the v0.1 trial set
python3 make_trials.py

# Run v0.1 trials for real (needs ANTHROPIC_API_KEY set)
python3 run_trial.py <trial_id>                     # one trial
python3 run_batch.py --type single --limit 5 --dry-run   # preview a batch
python3 run_batch.py --type single --limit 5             # run it for real

# Analyze whatever's in results/raw.jsonl
python3 analyze.py
```

The context-packet extension (`context_*.py`) currently only has preview
scripts — there's no batch runner or analysis wired up for it yet.
