"""Regression tests for the 2026-09-15 bug sweep of the v0.1 pilot files
(context_packets.py, prompts.py, comparisons.py, make_trials.py).
"""

import os
import tempfile

import pytest

from context_packets import load_dimensions, render_packet, validate_packet
from prompts import load_story as load_story_single
from comparisons import load_story as load_story_comparison
from make_trials import pilot_items, PILOT_STORY_IDS


@pytest.fixture(scope="module")
def dimensions():
    return load_dimensions()


class TestContextPacketScopeValidation:
    """A prompt-scope dimension (prompt_context) mixed into a packet must be
    rejected up front, not silently dropped by render_packet's
    DIMENSION_ORDER-only loop (which only walks story-scope dimensions)."""

    def test_story_scope_packet_still_renders(self, dimensions):
        text = render_packet(dimensions, {"provenance": "self"})
        assert text  # non-empty; the legitimate case is unaffected

    def test_prompt_scope_dimension_is_rejected(self, dimensions):
        with pytest.raises(ValueError):
            validate_packet(dimensions, {"prompt_context": "weather_mention"})

    def test_mixed_story_and_prompt_scope_packet_is_rejected(self, dimensions):
        """Before the fix, this validated cleanly and render_packet silently
        dropped the prompt_context content -- a validated-looking packet
        with content missing from its rendered text."""
        with pytest.raises(ValueError):
            render_packet(dimensions, {"provenance": "self", "prompt_context": "weather_mention"})


class TestStoryTextDelimiterCollision:
    """build_prompt/build_comparison_prompt both wrap story text in a
    literal \"\"\" delimiter; a story containing that sequence must be
    rejected at load time rather than silently producing a confusing
    prompt where the story appears to end early."""

    def _write_story(self, text):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return path

    def test_prompts_load_story_rejects_delimiter_collision(self):
        path = self._write_story('some text with """ inside it')
        try:
            with pytest.raises(ValueError):
                load_story_single(path)
        finally:
            os.unlink(path)

    def test_comparisons_load_story_rejects_delimiter_collision(self):
        path = self._write_story('some text with """ inside it')
        try:
            with pytest.raises(ValueError):
                load_story_comparison(path)
        finally:
            os.unlink(path)

    def test_normal_story_text_loads_fine(self):
        path = self._write_story("An ordinary story with no special characters.")
        try:
            assert load_story_single(path) == "An ordinary story with no special characters."
            assert load_story_comparison(path) == "An ordinary story with no special characters."
        finally:
            os.unlink(path)


class TestPilotItemsCompleteness:
    """pilot_items() must never silently return fewer than the 4 pinned
    pilot stories -- that would shrink the reproduced trial set (28/60/20)
    with no error, contradicting make_trials.py's own stated purpose."""

    def test_all_four_pilot_stories_found_in_real_corpus(self):
        import prompts

        items = prompts.load_items(prompts.ITEMS_FILE)
        found = pilot_items(items)
        assert len(found) == 4
        assert {i["id"] for i in found} == set(PILOT_STORY_IDS)

    def test_raises_when_a_pilot_story_is_missing(self):
        with pytest.raises(ValueError):
            pilot_items([{"id": "gilbert"}])  # missing the other 3

    def test_raises_when_no_pilot_stories_present(self):
        with pytest.raises(ValueError):
            pilot_items([{"id": "some_other_story"}])


class TestMakeTrialsWriteOrder:
    """A duplicate trial_id must abort before data/trials.jsonl is written,
    never after -- a print-only check that still exits 0 leaves a corrupted
    file on disk for downstream consumers."""

    def test_uniqueness_check_happens_before_write(self, tmp_path, monkeypatch):
        import make_trials

        trials_path = tmp_path / "trials.jsonl"
        monkeypatch.setattr(make_trials, "TRIALS_FILE", str(trials_path))
        monkeypatch.setattr(
            make_trials,
            "build_single_trials",
            lambda: [{"trial_id": "dup", "type": "single"}, {"trial_id": "dup", "type": "single"}],
        )
        monkeypatch.setattr(make_trials, "build_comparison_trials", lambda: [])
        monkeypatch.setattr(make_trials, "build_comparison_control_trials", lambda: [])

        with pytest.raises(ValueError):
            make_trials.main()

        assert not trials_path.exists(), "trials.jsonl must not be written when trial_ids collide"
