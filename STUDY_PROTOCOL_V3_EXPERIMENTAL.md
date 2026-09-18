# Study Protocol v3 — Experimental families: adaptive dose, iterative red-team, capability sweep, and the design simulator

Additive to `STUDY_PROTOCOL_V3.md` (frozen primary) and
`STUDY_PROTOCOL_V3_STRESS.md` (frozen stress families). Nothing here
changes a frozen manifest, wording, id, lock, or raw-data definition; every
experimental preflight fails if the primary lock, the stress design lock,
or the v2 snapshot no longer verifies. Policies live in
`data/controllability_v3_experimental_config.json` and are frozen in
stages by `freeze_controllability_v3_experimental.py`
(`data/controllability_v3_experimental_frozen_lock.json`).

Labels: **confirmatory** = pre-specified, frozen before data;
**secondary** = pre-specified, lower priority; **exploratory** = design
support / hypothesis-generating.

## 0. Runtime planning (exploratory, methodological)

`controllability_v3_runtime.py` measures throughput from any raw JSONL
result file (completed judgments, wall and idle-corrected active time,
effective judgments/s, p50/p90/p95 latency, retry and error fractions,
tokens per successful judgment, Little's-law concurrency). Planning shows
three labelled scenarios: optimistic (empirical × 1.5, or an ASSUMED 4.0
j/s), empirical (measured, given, or an ASSUMED 2.0 j/s fallback), and
conservative = min(empirical, 1.4 j/s), the only rate ever observed (v2
recovery at concurrency 32). Nothing in the planner is a prediction.

## 1. Design / power simulator (exploratory, methodological)

`controllability_v3_design_simulator.py` draws hierarchical worlds (story
quality, pair baseline, position bias, cue effects, intervention
multipliers, intervention × cue interactions, ambiguity dependence,
no-context drift, attack recovery, dose slopes, cross-evaluator scale and
quality mixing, Bernoulli replicate noise), simulates candidate designs
(replicates 2/3/5/10, 24 vs 66 pairs, 3/4/6 dose points, dev replicates,
top-K, intervention subsets, one or two profiles) and scores them with the
real estimators (pooled CE, signed suppression, noise-corrected drift,
attack recovery, WLS dose slope and +10pp threshold, story bootstrap):
bias, RMSE, CI width, coverage, P(correct ordering), P(best intervention),
P(detect attack vulnerability), false-positive rate on null interventions,
false-negative rate, dose-slope and threshold precision. Outputs:
`design_power_summary.csv`, `design_precision_summary.csv`,
`design_cost_runtime_summary.csv`, `design_recommendation.json`, two SVG
plots. The recommendation is advisory; no design is changed by code.

## 2. Adaptive dose response — `stress_dose_adaptive` (secondary, confirmatory estimands)

**Question.** How much known-irrelevant numerical evidence is required to
shift (d10, d25) or reverse (d50) the judge's underlying preference?

**Unit of adaptation: intervention × ambiguity stratum.** A curve per
pair × intervention would rest on four binary observations per round; a
single curve per intervention hides the baseline-strength dependence the
hypothesis is about. Strata are tertiles of blind baseline strength from
the primary no-context I0 rows of the same evaluator profile (recorded once
in the state file; a single pooled stratum when no blind baseline exists,
also recorded). Every round samples every pair in the stratum at one dose ×
4 counterbalance cells × 1 replicate, so story-level inference and balance
are preserved.

**Model.** Per unit, IRLS logistic regression
logit P(context-favoured passage chosen) = α + β·x, x = logit(dose/100).
Targets: d10, d25 = doses where P(x) − P(0) = 0.10, 0.25; d50 = dose where
P = 0.5 when P(0) < 0.5. Delta-method SEs and 95% CIs in dose points; a
story bootstrap of d10 is reported in the analysis.

**Next-dose rule (deterministic, frozen).** Anchors 51, 70, 90, 99 first.
Then targets are cycled (d10, d25, d10, …) and the next dose is the
allowed dose {51, 55, …, 95, 99} nearest to the current point estimate of
that target with fewer than 3 visits (ties → lower dose): the classic
quantile-sampling design, which converges to the quantile and keeps every
adaptive observation in the transition region. A one-step
"minimise d10 variance" criterion was evaluated and rejected: it selects
the extreme doses (the threshold's variance is dominated by the slope,
which extreme leverage pins best), i.e. exactly the doses least
informative about the transition; its value is still recorded per
candidate as a diagnostic. When the fit is unidentifiable (β ≤ 0), the
least-visited allowed dose is revisited.

**Stopping (frozen; rules 2–4 only after all anchors).** (1) hard budget
12 rounds per unit; (2) d10 CI width ≤ 10 dose points; (3) after ≥ 6
rounds, slope CI upper bound < 0.05 per logit unit; (4) after ≥ 6 rounds,
the d10 CI lies entirely above 99. Global hard cap 25,344 judgments
(24 units × 12 rounds × 88). Anchors alone cost 8,448.

**Reproducibility.** Every decision is appended to
`adaptive_dose_schedule.jsonl` with the state before it, the candidate
criterion values, the chosen dose, the reason, and the data count. Resume
replays the raw observations in schedule order, re-derives each decision,
and refuses on any mismatch. Outputs: `adaptive_thresholds.csv`,
`adaptive_slopes.csv`, `adaptive_stopping_reasons.csv`,
`adaptive_decisions.csv`, `adaptive_vs_fixed_grid.csv` (same estimands and
spend from the frozen fixed grid when present).

## 3. Iterative adversarial evolution — `stress_adversarial_iterative` (secondary)

**Question.** With black-box feedback on development pairs only, how
efficiently can an attacker discover framing that defeats an intervention?

**Frozen policy** (`iterative_policy` stage): population 6 per (cue,
intervention); generation 0 = ordinary wording + valid frozen one-shot
candidates + fresh attacker candidates; 3 generations after zero; elites
2; 2 mutation children per elite + 1 recombination child; diversity rule:
normalised template must be new in the key's history; validator = the
frozen rule set of the one-shot family; selection score = dev D_pair(attack)
− dev D_pair(ordinary); final selection = best-ever top-2 per key by dev
score (ties by generation, then index). Feedback the attacker may receive:
its own dev score, the ordinary score, the previous generation's best,
cue family, intervention, generation. Never: story pairs, passages,
held-out results. `assert_prompt_clean` refuses any prompt naming a pair,
containing passage text, or mentioning held-out data.

**Break rule (frozen before data).** A key is broken at the first
generation with a member whose dev recovery ≥ 0.10 and, when the I0
ordinary reference exists, dev CE_attack ≥ 0.5 × CE_I0_ordinary_dev.
`queries_to_break` = cumulative dev judgments of that key; never-broken keys
are reported. Held-out evaluation only after the `iterative_attacks`
freeze; attacks are never revised afterwards. Outputs:
`iterative_attack_population.jsonl`, `iterative_attack_generation_summary.csv`,
`iterative_attack_learning_curve.csv`, `iterative_attack_selected.json`,
`iterative_attack_heldout_results.jsonl`, `iterative_attack_generalization.csv`.
Planned maximum: 400 attacker calls, 73,920 development judgments, 42,240
held-out judgments (all 40 keys).

## 4. Paired capability / reasoning sweep — `capability_sweep` (confirmatory contrasts)

**Question.** As evaluator capability or reasoning changes, does selective
suppression improve, or does the judge merely change its literary
preferences?

**Frozen design** (`capability_design` stage): 24 pairs (8 per strength
stratum from the blind baseline when available; currently a seeded
story-balanced subset, recorded in the manifest meta), interventions I0,
I1, I4, I5, I7, all five cues, conditions: matched no-context (2 cells),
ordinary context (4 per cue), strongest frozen one-shot attack per cue ×
intervention (4 per cue; included only once the stress attacks are
frozen), doses 51/80/99 (4 each); 3 replicates. 4,080 cells without the
attack condition (12,240 judgments per profile), 6,480 with it (19,440).
One manifest for every profile; `evaluation_observation_id =
planned_observation_id@profile_id`.

**Per profile:** blind preference, context effect, suppression, no-context
drift, attack recovery, dose slope, high-dose residual effect. **Between
profiles (paired, joint story bootstrap):** blind preference disagreement
(noise-corrected RMS, Spearman), Δ context susceptibility, Δ suppression,
Δ attack recovery, Δ dose slope. A profile is never called "better at
suppression" unless its blind disagreement is reported beside the
suppression difference. Reasoning comparisons use only profiles the
provider abstraction supports (deepseek_flash_low/medium/high).

## 5. What remains unfrozen

Adaptive observations (by design), the iterative attack population and
selection (`iterative_attacks` stage; frozen by `iterative-attack-freeze`
before any held-out call), and the capability manifest's attack condition
(regenerate + re-freeze `capability_design` once the stress attacks are
frozen). Any change to a policy before production requires re-freezing
its stage and a note here.
