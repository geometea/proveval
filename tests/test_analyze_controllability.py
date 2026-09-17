"""Tests for the standalone context-controllability experiment's focused
analyser (analyze_controllability.py).

Synthetic-fixture style, like tests/test_analyze_context.py and
tests/test_analyze_llm_provenance.py: builds observation/raw-row dicts
directly rather than running real trials/API calls. Every effect-size
calculation under test here is reused, unmodified, from
context_analysis_pairwise.py (analyze_directional_pairwise_effects,
summarize_directional_effects_by_contrast, compare_pairwise_regimes,
analyze_pairwise_cell_rates, analyze_position_and_interaction_effects) and
context_analysis_stats.py (pearson_correlation) -- what's under test is
this module's own thin composition: complete-unit filtering,
suppression_magnitude, contrast-stratified baseline-margin correlation, and
first-attempt response compliance.
"""

import pytest

from context_analysis_pairwise import (
    analyze_directional_pairwise_effects,
    analyze_pairwise_cell_rates,
    analyze_position_and_interaction_effects,
    summarize_directional_effects_by_contrast,
)

import analyze_controllability as acb
import controllability_trials as ct

CATEGORIES = [ct.PRIMARY_CATEGORY]

CONTRAST_ID = "provenance_human_vs_llm"


def make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, assignment, position, wins, n, prefix, block_id=None, replicate_id=1):
    """wins = number of replicates (out of n) where text_1 is the chosen
    text. n replicates all share replicate_id (a single "unit" for
    completeness filtering) unless the caller varies it explicitly."""
    obs = {}
    for rep in range(1, n + 1):
        text_1_chosen = rep <= wins
        if position == "story1_as_a":
            story_a_id, story_b_id = s1, s2
            choice = "A" if text_1_chosen else "B"
        else:
            story_a_id, story_b_id = s2, s1
            choice = "B" if text_1_chosen else "A"
        obs[f"{prefix}_{rep}"] = {
            "type": "context_pairwise", "model": model, "evaluation_regime": evaluation_regime,
            "choice_mode": "forced", "contrast_id": contrast_id, "story_1_id": s1, "story_2_id": s2,
            "story_a_id": story_a_id, "story_b_id": story_b_id, "assignment": assignment, "position": position,
            "replicate_id": replicate_id, "parsed_response": {"overall_quality": choice},
            "response_format": "plain_ab", "experiment_id": ct.EXPERIMENT_ID,
            "block_id": block_id or f"block__{contrast_id}__{s1}_vs_{s2}__{evaluation_regime}",
            "sampling_regime": "low_variance_primary",
        }
    return obs


def make_full_block(model, evaluation_regime, contrast_id, s1, s2, fwd_a, fwd_b, flp_a, flp_b, n=10, prefix="x", block_id=None):
    obs = {}
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "forward", "story1_as_a", fwd_a, n, f"{prefix}_fa", block_id))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "forward", "story2_as_a", fwd_b, n, f"{prefix}_fb", block_id))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "flipped", "story1_as_a", flp_a, n, f"{prefix}_pa", block_id))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "flipped", "story2_as_a", flp_b, n, f"{prefix}_pb", block_id))
    return obs


def pipeline(observations, contrasts=(CONTRAST_ID,)):
    directional_rows = analyze_directional_pairwise_effects(observations, categories=CATEGORIES)
    acb.attach_context_name(directional_rows)
    by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
    acb.attach_context_name(by_contrast_rows)
    cell_rows = analyze_pairwise_cell_rates(observations, categories=CATEGORIES)
    position_rows = analyze_position_and_interaction_effects(cell_rows)
    acb.attach_context_name(position_rows)
    return directional_rows, by_contrast_rows, position_rows


# ---------------------------------------------------------------------------
# Complete-unit filtering: 4-cell treatment blocks
# ---------------------------------------------------------------------------

def make_unit_cell(block_id, model, assignment, position, choice, replicate_id=1, regime="naturalistic"):
    return {
        "type": "context_pairwise", "model": model, "evaluation_regime": regime,
        "choice_mode": "forced", "contrast_id": CONTRAST_ID, "story_1_id": "s1", "story_2_id": "s2",
        "story_a_id": "s1" if position == "story1_as_a" else "s2",
        "story_b_id": "s2" if position == "story1_as_a" else "s1",
        "assignment": assignment, "position": position, "replicate_id": replicate_id,
        "parsed_response": {"overall_quality": choice}, "response_format": "plain_ab",
        "experiment_id": ct.EXPERIMENT_ID, "block_id": block_id, "sampling_regime": "low_variance_primary",
    }


def make_unit(block_id, model="m", replicate_id=1, cells=ct.REQUIRED_CELLS):
    return {
        f"{block_id}__{assignment}__{position}": make_unit_cell(block_id, model, assignment, position, "A", replicate_id)
        for assignment, position in cells
    }


class TestTreatmentCompleteness:
    def test_a_full_four_cell_unit_is_retained(self):
        obs = make_unit("block_1")
        complete, n_complete, n_incomplete = acb.filter_complete_treatment_units(obs)
        assert n_complete == 1
        assert n_incomplete == 0
        assert set(complete) == set(obs)

    def test_a_unit_missing_one_cell_is_entirely_excluded_and_counted_incomplete(self):
        incomplete = make_unit("block_1", cells={("forward", "story1_as_a"), ("forward", "story2_as_a"), ("flipped", "story1_as_a")})
        complete, n_complete, n_incomplete = acb.filter_complete_treatment_units(incomplete)
        assert n_complete == 0
        assert n_incomplete == 1
        assert complete == {}

    def test_raw_observations_are_never_mutated_or_deleted(self):
        incomplete = make_unit("block_1", cells={("forward", "story1_as_a"), ("forward", "story2_as_a"), ("flipped", "story1_as_a")})
        original_keys = set(incomplete)
        acb.filter_complete_treatment_units(incomplete)
        assert set(incomplete) == original_keys  # the input dict itself is untouched

    def test_two_separate_units_are_filtered_independently(self):
        complete_unit = make_unit("block_ok")
        incomplete_unit = make_unit("block_bad", cells={("forward", "story1_as_a"), ("forward", "story2_as_a")})
        combined = {**complete_unit, **incomplete_unit}

        complete, n_complete, n_incomplete = acb.filter_complete_treatment_units(combined)
        assert n_complete == 1
        assert n_incomplete == 1
        assert all(v["block_id"] == "block_ok" for v in complete.values())

    def test_different_replicate_ids_of_the_same_block_are_separate_units(self):
        """(block_id, model, replicate_id, sampling_regime) is the unit key
        -- replicate 1 being complete must not paper over replicate 2 being
        incomplete, and vice versa."""
        replicate_1 = make_unit("block_1", replicate_id=1)
        replicate_2 = make_unit("block_1", replicate_id=2, cells={("forward", "story1_as_a")})
        combined = {f"r1_{k}": v for k, v in replicate_1.items()}
        combined.update({f"r2_{k}": v for k, v in replicate_2.items()})

        complete, n_complete, n_incomplete = acb.filter_complete_treatment_units(combined)
        assert n_complete == 1
        assert n_incomplete == 1
        assert all(v["replicate_id"] == 1 for v in complete.values())


class TestBaselineCompleteness:
    def _baseline_obs(self, model, s1, s2, positions, block_id="baseline_block__s1_vs_s2"):
        obs = {}
        for position in positions:
            story_a_id, story_b_id = (s1, s2) if position == "story1_as_a" else (s2, s1)
            obs[f"{position}"] = {
                "type": "context_pairwise", "model": model, "evaluation_regime": "naturalistic",
                "choice_mode": "forced", "contrast_id": "no_context_baseline", "story_1_id": s1, "story_2_id": s2,
                "story_a_id": story_a_id, "story_b_id": story_b_id, "position": position, "replicate_id": 1,
                "parsed_response": {"overall_quality": "A"}, "response_format": "plain_ab",
                "experiment_id": ct.BASELINE_EXPERIMENT_ID, "block_id": block_id, "sampling_regime": "low_variance_primary",
            }
        return obs

    def test_both_positions_present_is_complete(self):
        obs = self._baseline_obs("m", "s1", "s2", ["story1_as_a", "story2_as_a"])
        complete, n_complete, n_incomplete = acb.filter_complete_baseline_units(obs)
        assert n_complete == 1 and n_incomplete == 0
        assert set(complete) == set(obs)

    def test_missing_one_position_is_excluded(self):
        obs = self._baseline_obs("m", "s1", "s2", ["story1_as_a"])
        complete, n_complete, n_incomplete = acb.filter_complete_baseline_units(obs)
        assert n_complete == 0 and n_incomplete == 1
        assert complete == {}


# ---------------------------------------------------------------------------
# Suppression magnitude (item 8): abs(natural) - abs(text_only), plus the
# retained signed_regime_difference and text_only_effect.
# ---------------------------------------------------------------------------

class TestSuppressionMagnitude:
    def test_positive_suppression_when_the_instruction_shrinks_the_effect(self):
        naturalistic_obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                            fwd_a=10, fwd_b=10, flp_a=0, flp_b=0, prefix="nat")
        invariance_obs = make_full_block("m", "text_only_invariance", CONTRAST_ID, "s1", "s2",
                                          fwd_a=7, fwd_b=7, flp_a=3, flp_b=3, prefix="inv")
        obs = {**naturalistic_obs, **invariance_obs}
        _, by_contrast_rows, _ = pipeline(obs)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)

        row = next(r for r in suppression_rows if r["category"] == "overall_quality")
        assert row["naturalistic_effect"] == pytest.approx(1.0)
        assert row["text_only_effect"] == pytest.approx(0.4)
        assert row["suppression_magnitude"] == pytest.approx(0.6)
        assert row["signed_regime_difference"] == pytest.approx(0.6)

    def test_negative_suppression_when_the_effect_grows_under_the_instruction(self):
        naturalistic_obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                            fwd_a=6, fwd_b=6, flp_a=4, flp_b=4, prefix="nat")
        invariance_obs = make_full_block("m", "text_only_invariance", CONTRAST_ID, "s1", "s2",
                                          fwd_a=9, fwd_b=9, flp_a=1, flp_b=1, prefix="inv")
        obs = {**naturalistic_obs, **invariance_obs}
        _, by_contrast_rows, _ = pipeline(obs)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)

        row = next(r for r in suppression_rows if r["category"] == "overall_quality")
        assert row["naturalistic_effect"] == pytest.approx(0.2)
        assert row["text_only_effect"] == pytest.approx(0.8)
        assert row["suppression_magnitude"] == pytest.approx(-0.6)  # effect got bigger, not smaller
        assert row["signed_regime_difference"] == pytest.approx(-0.6)

    def test_signed_regime_difference_and_magnitude_diverge_on_a_sign_flip(self):
        """When the effect REVERSES sign under the instruction (rather than
        just shrinking or growing), the signed difference and the magnitude
        difference must not be conflated -- they measure different things."""
        naturalistic_obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                            fwd_a=10, fwd_b=10, flp_a=0, flp_b=0, prefix="nat")
        invariance_obs = make_full_block("m", "text_only_invariance", CONTRAST_ID, "s1", "s2",
                                          fwd_a=0, fwd_b=0, flp_a=10, flp_b=10, prefix="inv")
        obs = {**naturalistic_obs, **invariance_obs}
        _, by_contrast_rows, _ = pipeline(obs)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)

        row = next(r for r in suppression_rows if r["category"] == "overall_quality")
        assert row["naturalistic_effect"] == pytest.approx(1.0)
        assert row["text_only_effect"] == pytest.approx(-1.0)
        assert row["suppression_magnitude"] == pytest.approx(0.0)  # same size, so no size change
        assert row["signed_regime_difference"] == pytest.approx(2.0)  # but the signed difference is large


# ---------------------------------------------------------------------------
# Position-bias diagnostic retained, unaffected by the suppression rework
# ---------------------------------------------------------------------------

def test_position_effect_is_still_computed_and_distinct_from_context_effect():
    obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2", fwd_a=8, fwd_b=4, flp_a=6, flp_b=2)
    directional_rows, _, position_rows = pipeline(obs)
    quality_row = next(r for r in directional_rows if r["category"] == "overall_quality")
    assert quality_row["directional_effect_a_minus_b"] == pytest.approx(0.2)
    prow = next(r for r in position_rows if r["category"] == "overall_quality")
    assert prow["position_effect_pooled"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Baseline margin (item 9): interpretation and stratified correlation
# ---------------------------------------------------------------------------

class TestBaselineMargin:
    def _baseline_obs(self, model, s1, s2, p1_wins, n, prefix):
        obs = {}
        for rep in range(1, n + 1):
            text_1_chosen = rep <= p1_wins
            obs[f"{prefix}_a_{rep}"] = {
                "type": "context_pairwise", "model": model, "story_1_id": s1, "story_2_id": s2,
                "story_a_id": s1, "story_b_id": s2, "position": "story1_as_a", "replicate_id": rep,
                "parsed_response": {"overall_quality": ("A" if text_1_chosen else "B")},
            }
            obs[f"{prefix}_b_{rep}"] = {
                "type": "context_pairwise", "model": model, "story_1_id": s1, "story_2_id": s2,
                "story_a_id": s2, "story_b_id": s1, "position": "story2_as_a", "replicate_id": rep,
                "parsed_response": {"overall_quality": ("B" if text_1_chosen else "A")},
            }
        return obs

    def test_zero_margin_means_close_balanced(self):
        obs = self._baseline_obs("m", "s1", "s2", p1_wins=5, n=10, prefix="tie")
        rows = acb.analyze_baseline_margin(obs)
        row = next(r for r in rows if r["category"] == "overall_quality")
        assert row["p_text1_wins"] == pytest.approx(0.5)
        assert row["baseline_margin"] == pytest.approx(0.0)

    def test_margin_of_one_means_maximally_decisive(self):
        obs = self._baseline_obs("m", "s1", "s2", p1_wins=10, n=10, prefix="dec")
        rows = acb.analyze_baseline_margin(obs)
        row = next(r for r in rows if r["category"] == "overall_quality")
        assert row["p_text1_wins"] == pytest.approx(1.0)
        assert row["baseline_margin"] == pytest.approx(1.0)

    def _directional_row(self, model, regime, contrast_id, s1, s2, effect):
        return {"model": model, "evaluation_regime": regime, "contrast_id": contrast_id,
                "story_1_id": s1, "story_2_id": s2, "category": "overall_quality",
                "directional_effect_a_minus_b": effect}

    def _baseline_row(self, model, s1, s2, margin):
        return {"model": model, "story_1_id": s1, "story_2_id": s2, "category": "overall_quality",
                "p_text1_wins": 0.5 + margin / 2, "n": 20, "baseline_margin": margin}

    def test_negative_correlation_means_context_moves_close_calls_more(self):
        # Large context effects on close-call pairs (low margin), small
        # effects on decisive pairs (high margin) -> negative correlation.
        pairs = [(0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0)]
        directional_rows = [
            self._directional_row("m", "naturalistic", CONTRAST_ID, f"s{i}", f"t{i}", effect)
            for i, (margin, effect) in enumerate(pairs)
        ]
        baseline_rows = [self._baseline_row("m", f"s{i}", f"t{i}", margin) for i, (margin, effect) in enumerate(pairs)]
        joined = acb.join_effect_to_baseline_margin(directional_rows, baseline_rows)
        correlation_rows = acb.correlate_effect_and_baseline_margin(joined)
        row = next(r for r in correlation_rows if r["evaluation_regime"] == "naturalistic")
        assert row["r_baseline_margin_vs_abs_context_effect"] == pytest.approx(-1.0)

    def test_correlation_is_stratified_by_contrast_never_pooled(self):
        """Two different contrasts must produce two separate correlation
        rows, each computed only over its own (<=66) story pairs -- never
        pooled into one combined correlation."""
        contrast_a_pairs = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        contrast_b_pairs = [(0.0, 0.1), (0.5, 0.1), (1.0, 0.1)]  # no relationship at all

        directional_rows = (
            [self._directional_row("m", "naturalistic", "contrast_a", f"a{i}", f"a{i}x", e) for i, (m, e) in enumerate(contrast_a_pairs)]
            + [self._directional_row("m", "naturalistic", "contrast_b", f"b{i}", f"b{i}x", e) for i, (m, e) in enumerate(contrast_b_pairs)]
        )
        baseline_rows = (
            [self._baseline_row("m", f"a{i}", f"a{i}x", m) for i, (m, e) in enumerate(contrast_a_pairs)]
            + [self._baseline_row("m", f"b{i}", f"b{i}x", m) for i, (m, e) in enumerate(contrast_b_pairs)]
        )
        joined = acb.join_effect_to_baseline_margin(directional_rows, baseline_rows)
        correlation_rows = acb.correlate_effect_and_baseline_margin(joined)

        assert len(correlation_rows) == 2
        row_a = next(r for r in correlation_rows if r["contrast_id"] == "contrast_a")
        row_b = next(r for r in correlation_rows if r["contrast_id"] == "contrast_b")
        assert row_a["n_story_pairs"] == 3
        assert row_b["n_story_pairs"] == 3
        assert row_a["r_baseline_margin_vs_abs_context_effect"] == pytest.approx(-1.0)
        assert abs(row_b["r_baseline_margin_vs_abs_context_effect"]) < 0.05  # ~no relationship, not pooled with contrast_a's -1.0


# ---------------------------------------------------------------------------
# First-attempt response compliance (item 7): diagnostic only
# ---------------------------------------------------------------------------

class TestResponseCompliance:
    def _row(self, model, evaluation_regime, contrast_id, attempt_id, valid=True, api_error=False):
        if api_error:
            return {"model": model, "attempt_id": attempt_id, "parsed_response": None,
                     "validation_error": "API call failed: timeout",
                     "trial_meta": {"experiment_id": ct.EXPERIMENT_ID, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id}}
        if valid:
            return {"model": model, "attempt_id": attempt_id, "parsed_response": {"overall_quality": "A"}, "validation_error": None,
                     "trial_meta": {"experiment_id": ct.EXPERIMENT_ID, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id}}
        return {"model": model, "attempt_id": attempt_id, "parsed_response": None, "validation_error": "ambiguous",
                 "trial_meta": {"experiment_id": ct.EXPERIMENT_ID, "evaluation_regime": evaluation_regime, "contrast_id": contrast_id}}

    def test_counts_valid_invalid_and_api_error_separately(self):
        rows = [
            self._row("m", "naturalistic", CONTRAST_ID, 1, valid=True),
            self._row("m", "naturalistic", CONTRAST_ID, 1, valid=True),
            self._row("m", "naturalistic", CONTRAST_ID, 1, valid=False),
            self._row("m", "naturalistic", CONTRAST_ID, 1, api_error=True),
        ]
        compliance = acb.first_attempt_compliance(rows, {}, {ct.EXPERIMENT_ID})
        row = next(r for r in compliance if r["evaluation_regime"] == "naturalistic" and r["contrast_id"] == CONTRAST_ID)
        assert row["n_attempted"] == 4
        assert row["n_valid_first_attempt"] == 2
        assert row["n_invalid_first_attempt"] == 1
        assert row["n_api_error_first_attempt"] == 1
        assert row["valid_rate"] == pytest.approx(0.5)

    def test_only_first_attempts_are_counted_a_retry_is_excluded(self):
        rows = [
            self._row("m", "naturalistic", CONTRAST_ID, 1, valid=False),
            self._row("m", "naturalistic", CONTRAST_ID, 2, valid=True),  # a retry, not a first attempt
        ]
        compliance = acb.first_attempt_compliance(rows, {}, {ct.EXPERIMENT_ID})
        row = next(r for r in compliance if r["contrast_id"] == CONTRAST_ID)
        assert row["n_attempted"] == 1
        assert row["n_invalid_first_attempt"] == 1

    def test_an_invalid_response_never_counts_as_an_ab_judgment(self):
        """This report never appears in, or influences, the directional
        effect calculations -- it's purely diagnostic."""
        rows = [self._row("m", "naturalistic", CONTRAST_ID, 1, valid=False)]
        compliance = acb.first_attempt_compliance(rows, {}, {ct.EXPERIMENT_ID})
        row = next(r for r in compliance if r["contrast_id"] == CONTRAST_ID)
        assert row["n_valid_first_attempt"] == 0

    def test_rows_from_a_different_experiment_are_excluded(self):
        rows = [self._row("m", "naturalistic", "some_other_contrast", 1, valid=True)]
        general_row = dict(rows[0])
        general_row["trial_meta"] = {"experiment_id": "some_other_experiment_v1", "evaluation_regime": "naturalistic", "contrast_id": "x"}
        compliance = acb.first_attempt_compliance([general_row], {}, {ct.EXPERIMENT_ID})
        assert compliance == []


# ---------------------------------------------------------------------------
# attach_context_name
# ---------------------------------------------------------------------------

def test_attach_context_name_uses_structured_display_names_not_parsed_ids():
    rows = [{"contrast_id": "editing_edited_vs_first_draft"}]
    acb.attach_context_name(rows)
    assert rows[0]["context"] == "edited vs first draft"
