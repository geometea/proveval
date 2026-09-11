# Pilot Results

## Dataset

- 141 raw API attempts
- 138 completed observations
- 3 failed attempts absorbed by later successful retries
- 0 unresolved failures
- 10 single-story observations
- 20 identical-text controls
- 108 ordinary comparison observations

## Identical-text controls

All 20/20 identical-text control trials produced a preference of 0. When the
two texts shown to the model were byte-identical, provenance labels alone
(neutral, AI-vs-journal, journal-vs-AI) did not cause the model to invent a
preference between them.

## Literary journal vs AI

- 12 matched label swaps
- positive 1, zero 10, negative 1
- mean shift 0.000
- estimated AI-label advantage 0.000

This is a null pilot result: swapping the AI/literary-journal labels on the
same pair of texts did not measurably move the model's preference.

## User-written vs AI

- 36 matched label swaps
- positive 13, zero 21, negative 2
- mean shift 0.500
- estimated AI-label advantage 0.250

By replicate:

- replicate 1 mean shift 0.583
- replicate 2 mean shift 0.750
- replicate 3 mean shift 0.167

Rating effects, AI-labelled minus self-labelled:

- plot_structure +0.208
- prose_style +0.097
- characterization +0.125
- originality +0.139
- overall_quality +0.153

## Interpretation

This pilot suggests Claude Sonnet 5's comparative judgments were modestly
shifted toward prose labelled as AI-generated rather than user-authored.
That shift should be read cautiously:

- This is exploratory pilot data, not a confirmatory result.
- Only four underlying stories were used.
- Repeated model calls on the same text are not independent text samples.
- The hypothesis was refined after inspecting pilot results.
- More replicates of the same four texts would not substitute for broader
  text sampling.
- The self-vs-AI comparison conflates an AI-provenance effect with a
  user-authorship effect — it cannot distinguish "labelled AI" from
  "not labelled as belonging to the specific user."
- The current evaluator instruction tells the model to base ratings on the
  text itself, which may itself affect how sensitive it is to provenance
  labels one way or another.

No claim of statistical significance is made or implied by these numbers.

## Next experiment

The next design should distinguish:

1. user-authored provenance
2. third-party-human provenance
3. AI-generated provenance

and should use a larger set of texts before drawing conclusions.
