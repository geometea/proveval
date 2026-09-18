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

    def test_mode_choices_include_recovery_and_recovery_preflight_alongside_the_originals(self):
        workflow = _load_workflow()
        mode_input = workflow[True]["workflow_dispatch"]["inputs"]["mode"]
        assert mode_input["options"] == ["preflight", "production", "recovery-preflight", "recovery"]
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

    def test_every_paid_call_recovery_step_is_gated_on_the_exact_recovery_mode_string(self):
        """These steps make (or gate) real API calls, or act on their
        results -- they must fire for mode: recovery only, and GitHub
        Actions' == is exact string equality, so 'recovery-preflight' can
        never satisfy inputs.mode == 'recovery'."""
        steps = _steps()
        paid_call_step_names = {
            "Restore previous recovery state", "Run recovery (unresolved Wave 1 observations only)",
            "Run new no-context/text_only baseline condition",
            "Run validation sample (500 non-primary duplicate observations)",
            "Validation diagnostics report",
            "Merge Wave 1 + recovery results", "Analyze the merged Wave 2 dataset",
            "Save resumable recovery state", "Upload Wave 2 recovery results",
        }
        by_name = {s["name"]: s for s in steps}
        assert paid_call_step_names <= set(by_name)
        for name in paid_call_step_names:
            condition = by_name[name]["if"]
            assert "inputs.mode == 'recovery'" in condition
            assert "recovery-preflight" not in condition

    def test_download_and_recovery_preflight_steps_fire_for_both_recovery_modes(self):
        """The download + recovery-preflight steps are the ONLY two steps
        mode: recovery-preflight ever reaches -- they must fire for either
        mode: recovery or mode: recovery-preflight."""
        steps = _steps()
        by_name = {s["name"]: s for s in steps}
        for name in ("Download Wave 1 artifact (run 35355521243)", "Recovery preflight"):
            condition = by_name[name]["if"]
            assert "inputs.mode == 'recovery'" in condition
            assert "inputs.mode == 'recovery-preflight'" in condition

    def test_recovery_preflight_mode_cannot_reach_any_api_running_step(self):
        """Acceptance test for the standalone recovery-preflight mode: no
        step that ever dispatches a model API call (recovery-run,
        baseline-text-only, validation-run) can fire when
        inputs.mode == 'recovery-preflight' -- each is gated on the exact
        string 'recovery', which is never equal to 'recovery-preflight'."""
        steps = _steps()
        # "py baseline-text-only" (the actual subcommand invocation) is
        # deliberately more specific than "baseline-text-only" alone, which
        # would also match the unrelated --baseline-text-only-results-file
        # CLI flag on the analysis step.
        api_calling_run_texts = {
            "recovery-run": None, "py baseline-text-only": None, "validation-run": None,
        }
        for step in steps:
            run_text = step.get("run", "")
            condition = step.get("if", "")
            for keyword in api_calling_run_texts:
                if keyword in run_text:
                    api_calling_run_texts[keyword] = condition
        for keyword, condition in api_calling_run_texts.items():
            assert condition is not None, f"no step found containing {keyword!r}"
            assert condition == "${{ inputs.mode == 'recovery' }}", (
                f"step containing {keyword!r} has condition {condition!r} -- "
                "it must be gated on exactly mode == 'recovery', excluding 'recovery-preflight'"
            )

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

    def test_analyze_step_reads_baseline_text_only_from_wave2_recovery_not_production(self):
        by_name = {s["name"]: s for s in _steps()}
        run_text = by_name["Analyze the merged Wave 2 dataset"]["run"]
        assert "results/controllability_v2/wave2_recovery/baseline_text_only_raw.jsonl" in run_text
        assert "results/controllability_v2/production/baseline_text_only_raw.jsonl" not in run_text

    def test_recovery_cache_and_upload_cover_baseline_text_only_via_wave2_recovery_dir(self):
        by_name = {s["name"]: s for s in _steps()}
        for name in ("Restore previous recovery state", "Save resumable recovery state"):
            assert by_name[name]["with"]["path"] == "results/controllability_v2/wave2_recovery"
        upload_path = by_name["Upload Wave 2 recovery results"]["with"]["path"]
        assert "results/controllability_v2/wave2_recovery/" in upload_path

    def test_validation_report_step_is_wired_into_the_recovery_pipeline(self):
        by_name = {s["name"]: s for s in _steps()}
        assert "Validation diagnostics report" in by_name
        run_text = by_name["Validation diagnostics report"]["run"]
        assert "validation-report" in run_text


class TestWave1SourceDirectoryResolution:
    """Regression coverage for the artifact-layout bug: actions/upload-artifact@v4
    strips the LEAST COMMON ANCESTOR of every path given to one upload. The
    "Upload results" step uploads both results/controllability_v2/production/
    and results/controllability_v2/analysis/, whose common ancestor is
    results/controllability_v2/ -- so the downloaded artifact's real layout
    is production/... and analysis/..., never results/controllability_v2/production/....
    Every Wave 1 consumer must read from ONE resolved directory rather than
    each hardcoding a (previously wrong) path."""

    def test_upload_results_step_implies_production_and_analysis_are_siblings_after_stripping(self):
        """Confirms the actual upload configuration this fix is based on:
        both paths share results/controllability_v2/ as their only common
        ancestor, so that prefix is what upload-artifact strips."""
        by_name = {s["name"]: s for s in _steps()}
        upload_path = by_name["Upload results"]["with"]["path"]
        assert "results/controllability_v2/production/" in upload_path
        assert "results/controllability_v2/analysis/" in upload_path

    def test_diagnostic_step_immediately_follows_download_and_makes_no_api_calls(self):
        steps = _steps()
        names = [s["name"] for s in steps]
        download_index = names.index("Download Wave 1 artifact (run 35355521243)")
        assert names[download_index + 1] == "Show downloaded Wave 1 artifact contents"
        diagnostic_step = steps[download_index + 1]
        assert "uses" not in diagnostic_step  # a plain shell step, not an action
        assert "python3" not in diagnostic_step["run"]
        assert "find wave1_artifact" in diagnostic_step["run"]
        assert "inputs.mode == 'recovery'" in diagnostic_step["if"]
        assert "inputs.mode == 'recovery-preflight'" in diagnostic_step["if"]

    def test_resolver_step_derives_production_as_the_source_directory_and_validates_both_files(self):
        by_name = {s["name"]: s for s in _steps()}
        resolver = by_name["Resolve Wave 1 source directory"]
        run_text = resolver["run"]
        assert 'WAVE1_RESULTS_DIR="wave1_artifact/production"' in run_text
        assert "treatment_raw.jsonl" in run_text
        assert "baseline_raw.jsonl" in run_text
        assert "GITHUB_ENV" in run_text
        assert "exit 1" in run_text  # fails loudly rather than silently continuing
        assert "python3" not in run_text  # pure bash, no API calls possible
        assert "inputs.mode == 'recovery'" in resolver["if"]
        assert "inputs.mode == 'recovery-preflight'" in resolver["if"]

    def test_resolver_runs_before_recovery_preflight(self):
        names = [s["name"] for s in _steps()]
        assert names.index("Resolve Wave 1 source directory") < names.index("Recovery preflight")

    def test_recovery_preflight_uses_the_resolved_directory_not_a_hardcoded_path(self):
        by_name = {s["name"]: s for s in _steps()}
        run_text = by_name["Recovery preflight"]["run"]
        assert "--source-results \"$WAVE1_RESULTS_DIR\"" in run_text
        assert "results/controllability_v2/production" not in run_text

    def test_every_wave1_consumer_uses_the_same_resolved_directory_variable(self):
        """recovery-run, validation-run, and merge must all read
        $WAVE1_RESULTS_DIR -- the SAME variable recovery-preflight uses --
        never a separately hardcoded (and therefore driftable) path."""
        by_name = {s["name"]: s for s in _steps()}
        consumer_steps = {
            "Recovery preflight",
            "Run recovery (unresolved Wave 1 observations only)",
            "Run validation sample (500 non-primary duplicate observations)",
            "Merge Wave 1 + recovery results",
        }
        for name in consumer_steps:
            run_text = by_name[name]["run"]
            assert "--source-results \"$WAVE1_RESULTS_DIR\"" in run_text, f"{name!r} does not use the resolved directory"
            assert "results/controllability_v2/production" not in run_text, f"{name!r} still hardcodes the old wrong path"

    def test_validation_report_does_not_need_or_use_the_wave1_directory(self):
        """validation-report only ever reads validation_raw.jsonl -- it
        never consumes Wave 1 directly, so it correctly has no
        --source-results argument at all."""
        by_name = {s["name"]: s for s in _steps()}
        run_text = by_name["Validation diagnostics report"]["run"]
        assert "--source-results" not in run_text
        assert "WAVE1_RESULTS_DIR" not in run_text
