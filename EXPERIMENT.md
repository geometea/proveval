# Proveval v0.1 Experimental Design

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
  or social valence (not yet implemented in `data/conditions.jsonl`, which
  currently has `crit_group` in this slot)

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
- Later: identical-text pairwise control (comparing a story against itself)

## Limitations

- Small text set
- Literary evaluation is inherently subjective
- Model judgments are stochastic
- Provenance sensitivity is not automatically equivalent to sycophancy

No results are claimed at this stage; this document describes the design
only.
