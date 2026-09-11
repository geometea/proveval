# Proveval

Proveval is a small experiment to test whether LLM judges change their
evaluation of the same (or similar) prose when given irrelevant provenance
or user-preference information — for example, being told a passage was
"written by a Pulitzer winner" or that the user "loves flowery language."

## Current stage

This repo currently only contains toy data and prompt-building code.
There are no model API calls yet.

- `data/items.jsonl` — a few short prose samples used for testing.
- `prompts.py` — loads the samples and builds neutral evaluation prompts.
  Run it with `python prompts.py` to print the prompts to your terminal.

## Next steps (not built yet)

- Add "irrelevant info" variants of each prompt (fake provenance, fake
  user preferences).
- Send prompts to a judge model and compare scores across variants.
