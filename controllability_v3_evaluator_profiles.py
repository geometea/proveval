"""Evaluator profiles: the model-side identity of a v3 run, kept entirely
separate from the (model-independent) scientific manifests.

A scientific manifest row + replicate gives a planned_observation_id. An
evaluator profile gives WHO judged it. Their combination is the
evaluation_observation_id, which is what resume/dedupe keys on, so the
same frozen manifest can be executed under several profiles without ever
being regenerated:

    evaluation_observation_id = f"{planned_observation_id}@{profile_id}"

Every profile field maps onto something model_providers.py actually
supports (PROVIDERS, DEFAULT_MODELS, REASONING_PROFILES_LOGICAL). Nothing
here invents a provider setting: validate_profile() resolves the provider's
native reasoning settings through model_providers.resolve_reasoning_settings
and refuses anything that module does not know.
"""

import os

import model_providers

# The frozen v3 primary evaluator (must equal the study config's
# primary_evaluator + max_output_tokens -- checked by tests and preflight).
PRIMARY_PROFILE_ID = "deepseek_flash_low"

# Safe output-token window: never below the v2 recovery-era ceiling (512
# truncated hidden reasoning), never absurdly large.
MIN_SAFE_MAX_OUTPUT_TOKENS = 4096
MAX_SAFE_MAX_OUTPUT_TOKENS = 32768

PROFILES = {
    "deepseek_flash_low": {"provider": "deepseek", "model": "deepseek-flash", "reasoning_profile": "low", "max_output_tokens": 4096,
                           "description": "v3 primary evaluator (identical to v2's recovery-era production configuration)"},
    "deepseek_flash_medium": {"provider": "deepseek", "model": "deepseek-flash", "reasoning_profile": "medium", "max_output_tokens": 8192,
                              "description": "same judge, more reasoning (capability-dependence sweep)"},
    "deepseek_flash_high": {"provider": "deepseek", "model": "deepseek-flash", "reasoning_profile": "high", "max_output_tokens": 16384,
                            "description": "same judge, high reasoning (capability-dependence sweep)"},
    "anthropic_claude_sonnet_5_low": {"provider": "anthropic", "model": "claude-sonnet-5", "reasoning_profile": "low", "max_output_tokens": 4096,
                                      "description": "replication evaluator (v2 study config roster)"},
    "openai_gpt_5_6_low": {"provider": "openai", "model": "gpt-5.6", "reasoning_profile": "low", "max_output_tokens": 4096,
                           "description": "replication evaluator (v2 study config roster)"},
    "gemini_3_8_flash_low": {"provider": "gemini", "model": "gemini-3.8-flash", "reasoning_profile": "low", "max_output_tokens": 4096,
                             "description": "replication evaluator (v2 study config roster)"},
    # Attacker profiles (adversarial red-team generation stage only).
    "attacker_deepseek_flash_high": {"provider": "deepseek", "model": "deepseek-flash", "reasoning_profile": "high", "max_output_tokens": 16384,
                                     "description": "default attacker for stress-adversarial-generate; never used as a judge"},
}


def profile_ids():
    return sorted(PROFILES)


def get_profile(profile_id):
    if profile_id not in PROFILES:
        raise KeyError(f"Unknown evaluator profile {profile_id!r}. Known: {profile_ids()}")
    return {"profile_id": profile_id, **PROFILES[profile_id]}


def validate_profile(profile_id):
    """Returns (ok, reason). Checks every field against model_providers."""
    try:
        profile = get_profile(profile_id)
    except KeyError as e:
        return False, str(e)
    if profile["provider"] not in model_providers.PROVIDERS:
        return False, f"provider {profile['provider']!r} is not supported by model_providers ({list(model_providers.PROVIDERS)})"
    if not isinstance(profile["model"], str) or not profile["model"]:
        return False, "model must be a non-empty string"
    if profile["reasoning_profile"] not in model_providers.REASONING_PROFILES_LOGICAL:
        return False, f"reasoning_profile {profile['reasoning_profile']!r} is not one of {list(model_providers.REASONING_PROFILES_LOGICAL)}"
    try:
        settings = model_providers.resolve_reasoning_settings(profile["provider"], profile["reasoning_profile"])
    except ValueError as e:
        return False, str(e)
    if not settings:
        return False, "provider resolved an empty reasoning setting (would silently use the provider default)"
    tokens = profile["max_output_tokens"]
    if not isinstance(tokens, int) or not (MIN_SAFE_MAX_OUTPUT_TOKENS <= tokens <= MAX_SAFE_MAX_OUTPUT_TOKENS):
        return False, f"max_output_tokens {tokens!r} outside the safe window [{MIN_SAFE_MAX_OUTPUT_TOKENS}, {MAX_SAFE_MAX_OUTPUT_TOKENS}]"
    return True, None


def profile_matches_study_config(profile_id, study_config):
    """True if the profile is exactly the study config's primary evaluator
    (provider/model/reasoning) at the config's max_output_tokens."""
    p, e = get_profile(profile_id), study_config["primary_evaluator"]
    return (p["provider"], p["model"], p["reasoning_profile"], p["max_output_tokens"]) == (
        e["provider"], e["requested_model"], e["reasoning_profile"], study_config["max_output_tokens"])


def profile_id_from_row(row):
    """The profile a result row was judged under. Rows written before
    profiles existed carry no evaluator_profile_id; they are matched to a
    known profile by (provider, model, reasoning, max_output_tokens), else
    labelled by those fields."""
    if row.get("evaluator_profile_id"):
        return row["evaluator_profile_id"]
    ev = row.get("evaluator") or {}
    key = (row.get("provider") or ev.get("provider"), row.get("model") or ev.get("requested_model"),
           row.get("reasoning_effort") or ev.get("reasoning_profile"), row.get("max_output_tokens") or ev.get("max_output_tokens"))
    if all(k is None for k in key):
        return PRIMARY_PROFILE_ID  # a row with no evaluator information predates profiles: the frozen primary
    judges = [(pid, p) for pid, p in PROFILES.items() if not pid.startswith("attacker_")]
    for pid, p in judges:
        if (p["provider"], p["model"], p["reasoning_profile"], p["max_output_tokens"]) == key:
            return pid
    if key[3] is None:  # a row that never recorded its token ceiling: match on provider/model/reasoning alone
        for pid, p in judges:
            if (p["provider"], p["model"], p["reasoning_profile"]) == key[:3]:
                return pid
    return "unknown__" + "__".join(str(k) for k in key)


def make_evaluation_observation_id(planned_observation_id, profile_id):
    return f"{planned_observation_id}@{profile_id}"


def split_evaluation_observation_id(evaluation_observation_id):
    planned, sep, profile = evaluation_observation_id.rpartition("@")
    if not sep:
        raise ValueError(f"Not an evaluation_observation_id: {evaluation_observation_id!r}")
    return planned, profile


def results_path_for_profile(base_path, profile_id):
    """The primary profile keeps the historical path; any other profile
    gets its own sub-directory so profiles never share a raw file."""
    if profile_id == PRIMARY_PROFILE_ID:
        return base_path
    directory, name = os.path.split(base_path)
    return os.path.join(directory, "profiles", profile_id, name)
