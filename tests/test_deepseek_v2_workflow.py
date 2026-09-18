"""Tests for .github/workflows/deepseek-v2.yml: the Wave 2 recovery mode is
additive and configured for source run 35355521243, and the existing
scheduled/production trigger is untouched. Pure YAML-structure assertions --
no GitHub Actions run happens here."""

import yaml

WORKFLOW_PATH = ".github/workflows/deepseek-v2.yml"


def _load_workflow():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _steps():
    return _load_workflow()["jobs"]["experiment"]["steps"]


class TestExistingTriggerUntouched:
    def test_schedule_cron_and_timezone_are_unchanged(self):
        workflow = _load_workflow()
        schedule = workflow[True]["schedule"]  # YAML parses bare "on" as True
        assert schedule == [{"cron": "7 3 18 9 *", "timezone": "America/Los_Angeles"}]

    def test_mode_choices_include_recovery_alongside_the_originals(self):
        workflow = _load_workflow()
        mode_input = workflow[True]["workflow_dispatch"]["inputs"]["mode"]
        assert mode_input["options"] == ["preflight", "production", "recovery"]
        assert mode_input["default"] == "preflight"

    def test_production_steps_still_run_on_schedule_or_mode_production(self):
        for step in _steps():
            if step["name"] in ("Run baseline", "Run treatment", "Analyze completed experiment"):
                assert "recovery" not in step["if"]
                assert "schedule" in step["if"] or "production" in step["if"]


class TestRecoveryMode:
    def test_actions_read_permission_is_present_for_cross_run_download(self):
        workflow = _load_workflow()
        assert workflow["permissions"]["actions"] == "read"
        assert workflow["permissions"]["contents"] == "read"

    def test_download_step_targets_the_exact_source_run(self):
        steps = _steps()
        download_steps = [s for s in steps if s.get("uses", "").startswith("actions/download-artifact")]
        assert len(download_steps) == 1
        step = download_steps[0]
        assert step["with"]["run-id"] == 35355521243
        assert step["with"]["name"] == "proveval-v2-results-35355521243"
        assert "recovery" in step["if"]

    def test_every_recovery_step_is_gated_on_recovery_mode_only(self):
        steps = _steps()
        recovery_step_names = {
            "Restore previous recovery state", "Download Wave 1 artifact (run 35355521243)",
            "Recovery preflight", "Run recovery (unresolved Wave 1 observations only)",
            "Run new no-context/text_only baseline condition",
            "Run validation sample (500 non-primary duplicate observations)",
            "Merge Wave 1 + recovery results", "Analyze the merged Wave 2 dataset",
            "Save resumable recovery state", "Upload Wave 2 recovery results",
        }
        by_name = {s["name"]: s for s in steps}
        assert recovery_step_names <= set(by_name)
        for name in recovery_step_names:
            assert "inputs.mode == 'recovery'" in by_name[name]["if"]

    def test_recovery_never_reruns_the_full_original_experiment(self):
        """The recovery job must call recovery-run/baseline-text-only/
        validation-run/merge -- never the plain treatment/baseline
        subcommands that would rerun everything from scratch."""
        steps = _steps()
        recovery_run_texts = " ".join(
            s.get("run", "") for s in steps
            if isinstance(s.get("if"), str) and "recovery" in s["if"]
        )
        assert "recovery-run" in recovery_run_texts
        assert "baseline-text-only" in recovery_run_texts
        assert "validation-run" in recovery_run_texts
        assert "run_controllability_v2.py merge" in recovery_run_texts
        assert " treatment --production" not in recovery_run_texts
        assert " baseline --production" not in recovery_run_texts

    def test_recovery_uploads_a_new_artifact_distinct_from_the_production_one(self):
        steps = _steps()
        upload_steps = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")]
        names = {s["with"]["name"] for s in upload_steps}
        assert "proveval-v2-results-${{ github.run_id }}" in names
        assert "proveval-v2-wave2-recovery-results-${{ github.run_id }}" in names
