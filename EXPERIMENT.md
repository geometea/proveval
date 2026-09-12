# Proveval v0.1 Experimental Design

> **Status:** this is the original v0.1 design doc. The pilot it describes has
> since run — see `PILOT_RESULTS.md` for what was found, `FINAL_DESIGN.md` for
> the frozen post-pilot study design, and `CONTEXT_PACKETS.md` for a separate,
> newer (not yet run) way of composing context signals. A couple of details
> below have been updated inline to match what was actually implemented.

## Research question

How sensitive are LLM evaluations of fixed prose to socially salient but
task-irrelevant contextual information?

## Single-text experiment

The exact same story is evaluated under different context conditions. Only
the context sentence changes; the story text and rating task are identical
across conditions.

### Conditions

- `neutral` — no extra context
- `self` — user claims authorship
- `friend_job` — a friend wrote it and is deciding whether to pursue writing
- `professor` — assigned by an English professor
- `literary_journal` — said to have appeared in a literary journal
- `ai` — said to be AI-generated
- `neutral_metadata` — irrelevant factual metadata with no obvious prestige
  or social valence

### Primary single-text outcomes

Five dimensions, each scored 1–5:

- `plot_structure`
- `prose_style`
- `characterization`
- `originality`
- `overall_quality`

### Primary analysis

For each story and model, compare each condition's scores against that
story's `neutral` baseline.

## Comparative experiment

Each pair of stories is shown in both A/B orders, under:

- `neutral`
- AI vs. literary journal
- literary journal vs. AI
- self (user-authored) vs. AI
- AI vs. self (user-authored)

### Primary comparative outcome

`preference`, from -2 (strongly prefer Story A) to +2 (strongly prefer
Story B).

### Primary comparative metric

How often swapping only the provenance labels (not the underlying texts)
changes or reverses the model's preference.

## Controls

- A/B order counterbalancing
- Identical underlying text across provenance conditions
- Repeated independent samples
- Structured output validation
- `neutral_metadata` condition
- Identical-text pairwise control (comparing a story against itself,
  implemented in `comparisons.py`/`make_trials.py`)

## Limitations

- Small text set
- Literary evaluation is inherently subjective
- Model judgments are stochastic
- Provenance sensitivity is not automatically equivalent to sycophancy

No results are claimed at this stage; this document describes the design
only.
