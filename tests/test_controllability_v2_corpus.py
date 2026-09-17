"""Tests for controllability_v2_corpus.py: frozen story hashes, no student
identity, and no corpus metadata ever leaking into a model-facing prompt.
"""

import json

import controllability_v2_corpus as corpus
import controllability_v2_trials as t
from context_trials import load_items


def test_corpus_metadata_has_exactly_12_frozen_story_hashes():
    metadata = corpus.build_corpus_metadata()
    assert len(metadata["stories"]) == 12
    assert len(metadata["item_ids"]) == 12
    for story in metadata["stories"]:
        assert len(story["sha256"]) == 64
        assert all(c in "0123456789abcdef" for c in story["sha256"])
        assert story["word_count"] > 0


def test_story_hashes_match_the_actual_files():
    metadata = corpus.build_corpus_metadata()
    items_by_id = {item["id"]: item for item in load_items()}
    for story in metadata["stories"]:
        text = corpus.load_story_text(items_by_id[story["id"]]["path"])
        assert story["sha256"] == corpus.sha256_hex(text)


def test_corpus_hash_is_deterministic():
    a = corpus.build_corpus_metadata()
    b = corpus.build_corpus_metadata()
    assert a["corpus_hash"] == b["corpus_hash"]


def test_no_student_identity_in_corpus_metadata():
    metadata = corpus.build_corpus_metadata()
    serialized = json.dumps(metadata).lower()
    # the anonymised author_group is a fixed opaque label, never a name
    assert metadata["author_group"] == corpus.AUTHOR_GROUP
    assert "name" not in metadata
    assert "author_name" not in metadata
    assert "student" not in serialized


def test_corpus_metadata_never_appears_in_any_generated_prompt():
    """corpus_hash / author_group / individual story sha256s must never leak
    into a model-facing prompt -- corpus metadata is bookkeeping ABOUT the
    stories, never content shown to a model."""
    metadata = corpus.build_corpus_metadata()
    forbidden = [metadata["corpus_hash"], metadata["author_group"]] + [s["sha256"] for s in metadata["stories"]]

    items = load_items()
    contrasts = t.load_contrasts()
    treatment = t.build_all_treatment_trials(contrasts, items, t.PRIMARY_INSTRUCTION_CONDITIONS)
    baseline = t.build_baseline_trials(items)

    for trial in treatment[:40] + baseline[:20]:
        for value in forbidden:
            assert value not in trial["prompt"]
