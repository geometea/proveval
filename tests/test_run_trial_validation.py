"""Response-validation schema tests.

Persists the "item A/B" offline verification that was previously done with
throwaway synthetic scripts and deleted afterward: the primary
context_pairwise task is forced-choice (A/B only, no silent tie coercion),
the secondary choice_mode="tie_allowed" diagnostic still allows tie, and
neither change touches the v0.1 pilot's own schemas.
"""

import json

import pytest

from run_trial import (
    is_valid_ratings,
    is_valid_context_ratings,
    parse_and_validate,
    validate_pairwise_context_response_forced,
    validate_pairwise_context_response_tie_allowed,
)

AB_RESPONSE = {
    "plot_structure": "A",
    "prose_style": "B",
    "characterization": "A",
    "originality": "B",
    "overall_quality": "B",
}

TIE_RESPONSE = {**AB_RESPONSE, "originality": "tie"}

V1_RATINGS = {"plot_structure": 3, "prose_style": 3, "characterization": 3, "originality": 3, "overall_quality": 3}
V02_RATINGS = {"plot_structure": 7.3, "prose_style": 6.0, "characterization": 8.5, "originality": 5.5, "overall_quality": 7.0}


class TestForcedChoicePrimary:
    def test_accepts_a_b_only(self):
        parsed, err = validate_pairwise_context_response_forced(AB_RESPONSE)
        assert err is None
        assert parsed == AB_RESPONSE

    def test_rejects_tie_without_coercion(self):
        parsed, err = validate_pairwise_context_response_forced(TIE_RESPONSE)
        assert parsed is None
        assert err is not None
        assert "tie" not in err.lower() or "must be exactly" in err

    def test_parse_and_validate_dispatches_to_forced_by_default(self):
        parsed, err = parse_and_validate(json.dumps(AB_RESPONSE), "context_pairwise", None)
        assert err is None
        parsed, err = parse_and_validate(json.dumps(TIE_RESPONSE), "context_pairwise", "forced")
        assert parsed is None and err is not None

    def test_context_prompt_and_pairwise_same_are_also_forced_by_default(self):
        for trial_type in ("context_prompt", "context_pairwise_same"):
            parsed, err = parse_and_validate(json.dumps(TIE_RESPONSE), trial_type, "forced")
            assert parsed is None and err is not None, trial_type

    def test_normalizes_british_spelling(self):
        british = {k.replace("characterization", "characterisation"): v for k, v in AB_RESPONSE.items()}
        parsed, err = validate_pairwise_context_response_forced(british)
        assert err is None
        assert parsed["characterization"] == AB_RESPONSE["characterization"]


class TestTieAllowedSecondary:
    def test_accepts_tie(self):
        parsed, err = validate_pairwise_context_response_tie_allowed(TIE_RESPONSE)
        assert err is None
        assert parsed == TIE_RESPONSE

    def test_still_accepts_a_b(self):
        parsed, err = validate_pairwise_context_response_tie_allowed(AB_RESPONSE)
        assert err is None

    def test_parse_and_validate_dispatches_on_choice_mode(self):
        parsed, err = parse_and_validate(json.dumps(TIE_RESPONSE), "context_pairwise", "tie_allowed")
        assert err is None
        assert parsed == TIE_RESPONSE


class TestV1SchemaUnaffected:
    """The v0.1 pilot's own schemas must be byte-for-byte unaffected by the
    v0.2 forced-choice/tie-allowed work -- they are structurally separate
    (integer ratings / integer preference), not the A/B/tie string schema."""

    def test_single_schema(self):
        assert is_valid_ratings(V1_RATINGS)
        parsed, err = parse_and_validate(json.dumps(V1_RATINGS), "single")
        assert err is None

    def test_comparison_schema_preference_is_integer_minus2_to_2(self):
        payload = {"story_a": V1_RATINGS, "story_b": V1_RATINGS, "preference": 0}
        parsed, err = parse_and_validate(json.dumps(payload), "comparison")
        assert err is None
        assert parsed["preference"] == 0

    @pytest.mark.parametrize("bad_preference", [-3, 3, "tie", 0.5])
    def test_comparison_schema_rejects_out_of_range_preference(self, bad_preference):
        payload = {"story_a": V1_RATINGS, "story_b": V1_RATINGS, "preference": bad_preference}
        parsed, err = parse_and_validate(json.dumps(payload), "comparison")
        assert parsed is None and err is not None

    def test_v02_context_single_decimal_schema_is_distinct_from_v1(self):
        assert is_valid_context_ratings(V02_RATINGS)
        assert not is_valid_ratings(V02_RATINGS)  # decimals fail the integer 1-5 schema
        parsed, err = parse_and_validate(json.dumps(V02_RATINGS), "context_single")
        assert err is None
