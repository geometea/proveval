"""Tests for the standalone context-controllability experiment's focused
analyser (analyze_controllability.py).

Synthetic-fixture style, like tests/test_analyze_context.py and
tests/test_analyze_llm_provenance.py: builds observation dicts directly
rather than running real trials/API calls. Every effect-size calculation
under test here is reused, unmodified, from context_analysis_pairwise.py
(analyze_directional_pairwise_effects, summarize_directional_effects_by_contrast,
compare_pairwise_regimes, analyze_pairwise_cell_rates,
analyze_position_and_interaction_effects) and context_analysis_stats.py
(pearson_correlation) -- what's under test is this module's own thin
composition: the suppression_effect derived column, contrast-name
attachment, and the baseline-margin join/correlation.
"""

import pytest

from context_analysis_common import EXCERPT_RATING_FIELDS
from context_analysis_pairwise import (
    analyze_directional_pairwise_effects,
    analyze_pairwise_cell_rates,
    analyze_position_and_interaction_effects,
    summarize_directional_effects_by_contrast,
)

import analyze_controllability as acb
import controllability_trials as ct


def make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, assignment, position, wins, n, prefix):
    """wins = number of replicates (out of n) where text_1 is the chosen text.
    Mirrors tests/test_analyze_context.py's helper of the same name, using
    the excerpt rubric's rating categories instead of the full rubric's."""
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
            "replicate_id": rep, "parsed_response": {f: choice for f in EXCERPT_RATING_FIELDS},
            "rubric": "excerpt", "experiment_id": ct.EXPERIMENT_ID,
        }
    return obs


def make_full_block(model, evaluation_regime, contrast_id, s1, s2, fwd_a, fwd_b, flp_a, flp_b, n=10, prefix="x"):
    obs = {}
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "forward", "story1_as_a", fwd_a, n, f"{prefix}_fa"))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "forward", "story2_as_a", fwd_b, n, f"{prefix}_fb"))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "flipped", "story1_as_a", flp_a, n, f"{prefix}_pa"))
    obs.update(make_cell_obs(model, evaluation_regime, contrast_id, s1, s2, "flipped", "story2_as_a", flp_b, n, f"{prefix}_pb"))
    return obs


CONTRAST_ID = "provenance_human_vs_llm"


# ---------------------------------------------------------------------------
# Suppression table
# ---------------------------------------------------------------------------

class TestSuppressionTable:
    def test_naturalistic_and_text_only_effects_and_their_difference(self):
        # naturalistic: text_1 always wins when it carries value "a" -> effect = 1.0
        naturalistic_obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                            fwd_a=10, fwd_b=10, flp_a=0, flp_b=0, prefix="nat")
        # text_only_invariance: effect attenuated to 0.4 (fwd 7/10, flp 3/10)
        invariance_obs = make_full_block("m", "text_only_invariance", CONTRAST_ID, "s1", "s2",
                                          fwd_a=7, fwd_b=7, flp_a=3, flp_b=3, prefix="inv")
        obs = {**naturalistic_obs, **invariance_obs}

        directional_rows = analyze_directional_pairwise_effects(obs, categories=EXCERPT_RATING_FIELDS)
        by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)

        quality_row = next(r for r in suppression_rows if r["category"] == "overall_quality")
        assert quality_row["naturalistic_effect"] == pytest.approx(1.0)
        assert quality_row["text_only_effect"] == pytest.approx(0.4)
        assert quality_row["suppression_effect"] == pytest.approx(0.6)
        assert quality_row["context"] == "human vs LLM"

    def test_zero_suppression_when_instruction_has_no_effect(self):
        naturalistic_obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                            fwd_a=8, fwd_b=8, flp_a=2, flp_b=2, prefix="nat")
        invariance_obs = make_full_block("m", "text_only_invariance", CONTRAST_ID, "s1", "s2",
                                          fwd_a=8, fwd_b=8, flp_a=2, flp_b=2, prefix="inv")
        obs = {**naturalistic_obs, **invariance_obs}

        directional_rows = analyze_directional_pairwise_effects(obs, categories=EXCERPT_RATING_FIELDS)
        by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)

        quality_row = next(r for r in suppression_rows if r["category"] == "overall_quality")
        assert quality_row["naturalistic_effect"] == pytest.approx(0.6)
        assert quality_row["text_only_effect"] == pytest.approx(0.6)
        assert quality_row["suppression_effect"] == pytest.approx(0.0)

    def test_missing_one_regime_is_excluded_not_fabricated(self):
        """compare_pairwise_regimes (reused unmodified) skips rows missing
        one of the two regimes -- this module must never invent a
        suppression_effect from a single-regime observation."""
        naturalistic_only = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                             fwd_a=10, fwd_b=10, flp_a=0, flp_b=0)
        directional_rows = analyze_directional_pairwise_effects(naturalistic_only, categories=EXCERPT_RATING_FIELDS)
        by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)
        assert suppression_rows == []

    def test_all_five_categories_are_retained_overall_quality_is_not_the_only_one(self):
        naturalistic_obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2",
                                            fwd_a=10, fwd_b=10, flp_a=0, flp_b=0, prefix="nat")
        invariance_obs = make_full_block("m", "text_only_invariance", CONTRAST_ID, "s1", "s2",
                                          fwd_a=5, fwd_b=5, flp_a=5, flp_b=5, prefix="inv")
        obs = {**naturalistic_obs, **invariance_obs}
        directional_rows = analyze_directional_pairwise_effects(obs, categories=EXCERPT_RATING_FIELDS)
        by_contrast_rows = summarize_directional_effects_by_contrast(directional_rows)
        suppression_rows = acb.build_suppression_table(by_contrast_rows)
        assert {r["category"] for r in suppression_rows} == set(EXCERPT_RATING_FIELDS)
        assert "plot_structure" not in {r["category"] for r in suppression_rows}


# ---------------------------------------------------------------------------
# Position-bias diagnostic retained, separate from the suppression table
# ---------------------------------------------------------------------------

class TestPositionDiagnosticRetained:
    def test_position_effect_is_computed_and_distinct_from_context_effect(self):
        # Same position-confound fixture as test_analyze_context.py: forward
        # 8/10, 4/10; flipped 6/10, 2/10 -> context effect 0.2, position effect 0.4.
        obs = make_full_block("m", "naturalistic", CONTRAST_ID, "s1", "s2", fwd_a=8, fwd_b=4, flp_a=6, flp_b=2)

        directional_rows = analyze_directional_pairwise_effects(obs, categories=EXCERPT_RATING_FIELDS)
        quality_row = next(r for r in directional_rows if r["category"] == "overall_quality")
        assert quality_row["directional_effect_a_minus_b"] == pytest.approx(0.2)

        cell_rows = analyze_pairwise_cell_rates(obs, categories=EXCERPT_RATING_FIELDS)
        position_rows = analyze_position_and_interaction_effects(cell_rows)
        acb.attach_context_name(position_rows)

        prow = next(r for r in position_rows if r["category"] == "overall_quality")
        assert prow["position_effect_pooled"] == pytest.approx(0.4)
        assert prow["context"] == "human vs LLM"
        assert abs(prow["position_effect_pooled"]) > abs(quality_row["directional_effect_a_minus_b"])


# ---------------------------------------------------------------------------
# Blind baseline: margin computation
# ---------------------------------------------------------------------------

class TestBaselineMargin:
    def _baseline_obs(self, model, s1, s2, p1_wins, n, prefix):
        obs = {}
        for rep in range(1, n + 1):
            text_1_chosen = rep <= p1_wins
            obs[f"{prefix}_a_{rep}"] = {
                "type": "context_pairwise", "model": model, "evaluation_regime": "naturalistic",
                "choice_mode": "forced", "contrast_id": "no_context_baseline", "story_1_id": s1, "story_2_id": s2,
                "story_a_id": s1, "story_b_id": s2, "position": "story1_as_a", "replicate_id": rep,
                "parsed_response": {f: ("A" if text_1_chosen else "B") for f in EXCERPT_RATING_FIELDS},
                "rubric": "excerpt", "experiment_id": ct.BASELINE_EXPERIMENT_ID,
            }
            obs[f"{prefix}_b_{rep}"] = {
                "type": "context_pairwise", "model": model, "evaluation_regime": "naturalistic",
                "choice_mode": "forced", "contrast_id": "no_context_baseline", "story_1_id": s1, "story_2_id": s2,
                "story_a_id": s2, "story_b_id": s1, "position": "story2_as_a", "replicate_id": rep,
                "parsed_response": {f: ("B" if text_1_chosen else "A") for f in EXCERPT_RATING_FIELDS},
                "rubric": "excerpt", "experiment_id": ct.BASELINE_EXPERIMENT_ID,
            }
        return obs

    def test_indifferent_pair_has_zero_margin(self):
        obs = self._baseline_obs("m", "s1", "s2", p1_wins=5, n=10, prefix="tie")
        rows = acb.analyze_baseline_margin(obs, categories=EXCERPT_RATING_FIELDS)
        quality_row = next(r for r in rows if r["category"] == "overall_quality")
        assert quality_row["p_text1_wins"] == pytest.approx(0.5)
        assert quality_row["baseline_margin"] == pytest.approx(0.0)

    def test_lopsided_pair_has_large_margin(self):
        obs = self._baseline_obs("m", "s1", "s2", p1_wins=9, n=10, prefix="lop")
        rows = acb.analyze_baseline_margin(obs, categories=EXCERPT_RATING_FIELDS)
        quality_row = next(r for r in rows if r["category"] == "overall_quality")
        assert quality_row["p_text1_wins"] == pytest.approx(0.9)
        assert quality_row["baseline_margin"] == pytest.approx(0.8)

    def test_position_is_pooled_not_treated_as_a_confound(self):
        """The baseline has no context-assignment axis to counterbalance --
        position is simply pooled, exactly as documented."""
        obs = self._baseline_obs("m", "s1", "s2", p1_wins=8, n=10, prefix="pool")
        rows = acb.analyze_baseline_margin(obs, categories=EXCERPT_RATING_FIELDS)
        quality_row = next(r for r in rows if r["category"] == "overall_quality")
        assert quality_row["n"] == 20  # both positions' replicates pooled into one estimate


# ---------------------------------------------------------------------------
# Joining context effect to baseline margin, and the correlation
# ---------------------------------------------------------------------------

class TestJoinAndCorrelation:
    def _directional_row(self, model, regime, s1, s2, category, effect):
        return {
            "model": model, "evaluation_regime": regime, "contrast_id": CONTRAST_ID,
            "story_1_id": s1, "story_2_id": s2, "category": category,
            "directional_effect_a_minus_b": effect,
        }

    def _baseline_row(self, model, s1, s2, category, margin):
        return {
            "model": model, "story_1_id": s1, "story_2_id": s2, "category": category,
            "p_text1_wins": 0.5 + margin / 2, "n": 20, "baseline_margin": margin,
        }

    def test_join_pairs_matching_keys_and_computes_abs_effect(self):
        directional_rows = [self._directional_row("m", "naturalistic", "s1", "s2", "overall_quality", -0.3)]
        baseline_rows = [self._baseline_row("m", "s1", "s2", "overall_quality", 0.2)]
        joined = acb.join_effect_to_baseline_margin(directional_rows, baseline_rows)
        assert len(joined) == 1
        row = joined[0]
        assert row["context_effect"] == -0.3
        assert row["abs_context_effect"] == pytest.approx(0.3)
        assert row["baseline_margin"] == 0.2
        assert row["context"] == "human vs LLM"

    def test_join_skips_pairs_with_no_matching_baseline(self):
        directional_rows = [self._directional_row("m", "naturalistic", "s1", "s2", "overall_quality", 0.5)]
        joined = acb.join_effect_to_baseline_margin(directional_rows, baseline_rows=[])
        assert joined == []

    def test_join_skips_empty_directional_effect(self):
        row = self._directional_row("m", "naturalistic", "s1", "s2", "overall_quality", "")
        baseline_rows = [self._baseline_row("m", "s1", "s2", "overall_quality", 0.2)]
        joined = acb.join_effect_to_baseline_margin([row], baseline_rows)
        assert joined == []

    def test_correlation_is_perfect_when_effect_size_tracks_margin_exactly(self):
        # abs_context_effect == baseline_margin for every pair -> r = 1.0
        directional_rows = [
            self._directional_row("m", "naturalistic", f"s{i}", f"t{i}", "overall_quality", margin)
            for i, margin in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ]
        baseline_rows = [
            self._baseline_row("m", f"s{i}", f"t{i}", "overall_quality", margin)
            for i, margin in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ]
        joined = acb.join_effect_to_baseline_margin(directional_rows, baseline_rows)
        correlation_rows = acb.correlate_effect_and_baseline_margin(joined)
        row = next(r for r in correlation_rows if r["evaluation_regime"] == "naturalistic")
        assert row["r_baseline_margin_vs_abs_context_effect"] == pytest.approx(1.0)
        assert row["n_story_pairs"] == 6

    def test_correlation_near_zero_when_context_overrides_strong_preferences_uniformly(self):
        # abs_context_effect is constant regardless of baseline_margin -> no
        # linear relationship (variance in effect is zero) -> undefined (None),
        # reported as "" rather than fabricated.
        directional_rows = [
            self._directional_row("m", "naturalistic", f"s{i}", f"t{i}", "overall_quality", 0.5)
            for i, margin in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ]
        baseline_rows = [
            self._baseline_row("m", f"s{i}", f"t{i}", "overall_quality", margin)
            for i, margin in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ]
        joined = acb.join_effect_to_baseline_margin(directional_rows, baseline_rows)
        correlation_rows = acb.correlate_effect_and_baseline_margin(joined)
        row = next(r for r in correlation_rows if r["evaluation_regime"] == "naturalistic")
        assert row["r_baseline_margin_vs_abs_context_effect"] == ""


# ---------------------------------------------------------------------------
# attach_context_name
# ---------------------------------------------------------------------------

def test_attach_context_name_uses_structured_display_names_not_parsed_ids():
    rows = [{"contrast_id": "editing_edited_vs_first_draft"}]
    acb.attach_context_name(rows)
    assert rows[0]["context"] == "edited vs first draft"
