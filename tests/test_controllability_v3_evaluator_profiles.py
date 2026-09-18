"""Evaluator profiles: validated against model_providers, independent of
the scientific manifest, deterministic evaluation ids, per-profile paths."""

import pytest

import model_providers
import controllability_v3_evaluator_profiles as ep
from controllability_v3_execution import attach_observation_ids, plan_execution_order
from controllability_v3_study_config import load_study_config


class TestProfiles:
    def test_every_profile_validates_against_model_providers(self):
        for pid in ep.profile_ids():
            assert ep.validate_profile(pid) == (True, None), pid
            p = ep.get_profile(pid)
            assert p["provider"] in model_providers.PROVIDERS
            assert p["reasoning_profile"] in model_providers.REASONING_PROFILES_LOGICAL
            assert model_providers.resolve_reasoning_settings(p["provider"], p["reasoning_profile"])

    def test_unknown_or_invalid_profiles_fail(self, monkeypatch):
        ok, reason = ep.validate_profile("nope")
        assert not ok and "Unknown" in reason
        monkeypatch.setitem(ep.PROFILES, "bad_tokens", {**ep.PROFILES["deepseek_flash_low"], "max_output_tokens": 512})
        ok, reason = ep.validate_profile("bad_tokens")
        assert not ok and "safe window" in reason
        monkeypatch.setitem(ep.PROFILES, "bad_reasoning", {**ep.PROFILES["deepseek_flash_low"], "reasoning_profile": "ultra"})
        assert not ep.validate_profile("bad_reasoning")[0]
        monkeypatch.setitem(ep.PROFILES, "bad_provider", {**ep.PROFILES["deepseek_flash_low"], "provider": "mistral"})
        assert not ep.validate_profile("bad_provider")[0]

    def test_primary_profile_matches_the_frozen_study_config(self):
        assert ep.profile_matches_study_config(ep.PRIMARY_PROFILE_ID, load_study_config())
        assert not ep.profile_matches_study_config("deepseek_flash_medium", load_study_config())

    def test_evaluation_id_round_trip_and_paths(self):
        eid = ep.make_evaluation_observation_id("v3::primary_context::x::r1", "deepseek_flash_high")
        assert eid == "v3::primary_context::x::r1@deepseek_flash_high"
        assert ep.split_evaluation_observation_id(eid) == ("v3::primary_context::x::r1", "deepseek_flash_high")
        with pytest.raises(ValueError):
            ep.split_evaluation_observation_id("no-at-sign")
        assert ep.results_path_for_profile("results/x/production/primary_raw.jsonl", ep.PRIMARY_PROFILE_ID) == "results/x/production/primary_raw.jsonl"
        assert ep.results_path_for_profile("results/x/production/primary_raw.jsonl", "deepseek_flash_high") == "results/x/production/profiles/deepseek_flash_high/primary_raw.jsonl"

    def test_profile_from_row_handles_new_legacy_and_unknown_rows(self):
        assert ep.profile_id_from_row({"evaluator_profile_id": "gemini_3_8_flash_low"}) == "gemini_3_8_flash_low"
        legacy = {"evaluator": {"provider": "deepseek", "requested_model": "deepseek-flash", "reasoning_profile": "low"}}
        assert ep.profile_id_from_row(legacy) == ep.PRIMARY_PROFILE_ID
        assert ep.profile_id_from_row({"provider": "deepseek", "model": "deepseek-flash", "reasoning_effort": "high", "max_output_tokens": 16384}) == "deepseek_flash_high"
        assert ep.profile_id_from_row({}) == ep.PRIMARY_PROFILE_ID
        assert ep.profile_id_from_row({"provider": "deepseek", "model": "deepseek-v9", "reasoning_effort": "low", "max_output_tokens": 4096}).startswith("unknown__")


class TestProfileIndependence:
    def test_same_manifest_two_profiles_share_planned_ids_but_not_evaluation_ids(self):
        trials = [{"trial_id": f"v3::primary_nocontext::a_vs_b::none::none::{pos}::I0::noctx", "block_id": "b", "family": "primary_nocontext"} for pos in ("story1_as_a", "story2_as_a")]
        a = attach_observation_ids(plan_execution_order(trials, 2, 5), "deepseek_flash_low")
        b = attach_observation_ids(plan_execution_order(trials, 2, 5), "deepseek_flash_high")
        assert [e["planned_observation_id"] for e in a] == [e["planned_observation_id"] for e in b]
        assert [e["execution_order_index"] for e in a] == [e["execution_order_index"] for e in b]
        assert {e["evaluation_observation_id"] for e in a}.isdisjoint({e["evaluation_observation_id"] for e in b})
        assert all(e["evaluation_observation_id"].endswith("@deepseek_flash_high") for e in b)
