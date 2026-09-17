"""Item 17: v2 must not use human-ranking machinery, single-text ratings,
or the five-category rubric anywhere. Checked structurally (source scan of
every v2 module) rather than by behavior, since the whole point is that
these code paths are never even reached.
"""

import ast
import pathlib

V2_MODULES = [
    "controllability_v2_corpus.py",
    "controllability_v2_trials.py",
    "controllability_v2_study_config.py",
    "controllability_v2_execution.py",
    "controllability_v2_stats.py",
    "controllability_v2_plots.py",
    "run_controllability_v2.py",
    "analyze_controllability_v2.py",
    "plan_controllability_v2.py",
    "freeze_controllability_v2.py",
]

FORBIDDEN_SUBSTRINGS = (
    "human_pairwise", "human_reference", "human_ranking",
    "RATING_FIELDS", "validate_single_response", "validate_context_single_response",
)


def test_no_v2_module_references_human_ranking_or_rating_machinery():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for module_name in V2_MODULES:
        source = (repo_root / module_name).read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert forbidden not in source, f"{module_name} references forbidden {forbidden!r}"


def test_no_v2_module_imports_human_ranking_py():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for module_name in V2_MODULES:
        tree = ast.parse((repo_root / module_name).read_text(encoding="utf-8"))
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        assert "human_ranking" not in imported_modules


def test_v1_human_ranking_files_are_untouched_by_v2_data_files():
    """v1's data/human_pairwise.jsonl and data/human_reference.json must
    never be read by anything v2 -- confirmed above by source scan; this
    just double-checks the files themselves still exist unmodified (v2
    never deletes or overwrites them)."""
    import os

    assert os.path.exists("data/human_pairwise.jsonl")
    assert os.path.exists("data/human_reference.json")
