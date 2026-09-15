"""Prompt-template tests for context_comparisons.build_prompt across the
2x2 (evaluation_regime x choice_mode) grid."""

import json

import pytest

from context_comparisons import build_prompt, CHOICE_MODES


@pytest.mark.parametrize("evaluation_regime", ["naturalistic", "text_only_invariance"])
@pytest.mark.parametrize("choice_mode", CHOICE_MODES)
def test_prompt_contains_both_stories_and_intro(evaluation_regime, choice_mode):
    prompt = build_prompt("INTRO SENTENCE", "STORY A TEXT", "STORY B TEXT", evaluation_regime, choice_mode)
    assert "INTRO SENTENCE" in prompt
    assert "STORY A TEXT" in prompt
    assert "STORY B TEXT" in prompt


@pytest.mark.parametrize("evaluation_regime", ["naturalistic", "text_only_invariance"])
@pytest.mark.parametrize("choice_mode", CHOICE_MODES)
def test_json_example_is_actually_valid_json(evaluation_regime, choice_mode):
    """A substring check like `'"tie"' in prompt` can't catch mis-escaped
    braces around it -- parse the example for real. Regression coverage
    for a real bug: see tests/test_bug_sweep_regressions.py."""
    prompt = build_prompt("intro", "a", "b", evaluation_regime, choice_mode)
    example = prompt.split("Return only JSON:")[1].strip()
    json.loads(example)  # must not raise


def test_forced_prompt_never_mentions_tie_in_json_example():
    prompt = build_prompt("intro", "a", "b", "naturalistic", "forced")
    # The instruction text legitimately says "ties are not allowed"; what
    # must never appear is a "tie" VALUE in the JSON example, which would
    # visually suggest it's an acceptable answer.
    assert '"tie"' not in prompt


def test_tie_allowed_prompt_json_example_includes_tie():
    prompt = build_prompt("intro", "a", "b", "naturalistic", "tie_allowed")
    assert '"tie"' in prompt


def test_forced_instruction_says_must_choose_a_or_b():
    prompt = build_prompt("intro", "a", "b", "naturalistic", "forced")
    assert "must answer" in prompt.lower() or "must choose" in prompt.lower()


def test_default_choice_mode_is_forced():
    default_prompt = build_prompt("intro", "a", "b", "naturalistic")
    forced_prompt = build_prompt("intro", "a", "b", "naturalistic", "forced")
    assert default_prompt == forced_prompt


def test_text_only_invariance_adds_judge_only_prose_instruction():
    naturalistic = build_prompt("intro", "a", "b", "naturalistic", "forced")
    invariance = build_prompt("intro", "a", "b", "text_only_invariance", "forced")
    assert "Judge only the two pieces of prose" in invariance
    assert "Judge only the two pieces of prose" not in naturalistic


def test_evaluation_regime_is_required_with_no_default():
    with pytest.raises(KeyError):
        build_prompt("intro", "a", "b", "not_a_real_regime")
