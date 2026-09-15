"""Regression tests for bugs found in the 2026-09-15 bug sweep.

Each test reproduces the exact failure mode before its fix so it can never
silently come back.
"""

import json

from context_comparisons import build_prompt
from run_trial import strip_code_fence
from human_ranking import find_unknown_story_ids, topological_order


class TestPairwisePromptJsonExampleIsValidJson:
    """context_comparisons.build_prompt's JSON example must be literally
    parseable JSON, not double-braced template-escaping leftovers -- a
    model that mirrors the example's own bracing in its answer would
    otherwise produce unparseable output for every context_pairwise/
    context_prompt trial."""

    def _json_example(self, prompt):
        return prompt.split("Return only JSON:")[1].strip()

    def test_forced_prompt_json_example_parses(self):
        prompt = build_prompt("intro", "story a", "story b", "naturalistic", "forced")
        parsed = json.loads(self._json_example(prompt))
        assert parsed["plot_structure"] in ("A", "B")

    def test_tie_allowed_prompt_json_example_parses(self):
        prompt = build_prompt("intro", "story a", "story b", "naturalistic", "tie_allowed")
        parsed = json.loads(self._json_example(prompt))
        assert parsed["originality"] == "tie"

    def test_no_doubled_braces_anywhere_in_the_prompt(self):
        for choice_mode in ("forced", "tie_allowed"):
            for regime in ("naturalistic", "text_only_invariance"):
                prompt = build_prompt("intro", "a", "b", regime, choice_mode)
                assert "{{" not in prompt and "}}" not in prompt


class TestStripCodeFenceHandlesSingleLineFences:
    """A model that emits the whole fenced JSON reply on one line (no
    internal newline) must not have its content discarded."""

    def test_single_line_fence_with_language_tag(self):
        assert strip_code_fence('```json{"a": 1}```') == '{"a": 1}'

    def test_single_line_fence_without_language_tag(self):
        assert strip_code_fence('```{"a": 1}```') == '{"a": 1}'

    def test_multi_line_fence_still_works(self):
        assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_multi_line_fence_no_language_tag_still_works(self):
        assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_fence_passthrough(self):
        assert strip_code_fence('{"a": 1}') == '{"a": 1}'

    def test_trailing_whitespace_after_closing_fence(self):
        assert strip_code_fence('```json\n{"a": 1}\n```  \n') == '{"a": 1}'

    def test_result_always_parses_as_json_when_content_is_valid(self):
        for text in ['```json{"a": 1}```', '```{"a": 1}```', '```json\n{"a": 1}\n```', '{"a": 1}']:
            json.loads(strip_code_fence(text))  # must not raise


class TestHumanRankingUnknownStoryId:
    """A typo'd winner/loser id in data/human_pairwise.jsonl must produce a
    clear diagnostic, not an unhandled KeyError from topological_order."""

    def test_detects_unknown_winner(self):
        unknown = find_unknown_story_ids([("promopt", "gilbert")], ["gilbert", "prophet"])
        assert unknown == ["promopt"]

    def test_detects_unknown_loser(self):
        unknown = find_unknown_story_ids([("gilbert", "promopt")], ["gilbert", "prophet"])
        assert unknown == ["promopt"]

    def test_no_false_positives_on_clean_data(self):
        unknown = find_unknown_story_ids([("gilbert", "prophet"), ("prophet", "santa")], ["gilbert", "prophet", "santa"])
        assert unknown == []

    def test_topological_order_would_have_raised_keyerror_on_unknown_id(self):
        """Documents why the check is needed: calling topological_order
        directly with an id absent from `nodes` still raises -- the fix is
        to check with find_unknown_story_ids before ever reaching this call,
        which is what human_ranking.main() now does."""
        import pytest

        with pytest.raises(KeyError):
            topological_order(["gilbert", "prophet"], [("gilbert", "promopt")])
