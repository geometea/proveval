"""Tests for model_providers.py: direct-call normalization for every
provider, reasoning-profile configuration, and the pure (network-free)
batch-request-payload builder.

No real API calls: every provider's SDK is faked by injecting a module into
sys.modules before the (lazily-imported) provider function runs. anthropic
happens to be installed in this environment but is still faked here, same
as the others, so these tests never depend on what's actually installed.
"""

import sys
import types

import pytest

import model_providers as mp


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    for env_var in mp.API_KEY_ENV_VARS.values():
        monkeypatch.setenv(env_var, "fake-key")


# ---------------------------------------------------------------------------
# Reasoning profiles: table-driven, one clear config, provider-neutral CLI
# ---------------------------------------------------------------------------

class TestReasoningProfiles:
    @pytest.mark.parametrize("provider", mp.PROVIDERS)
    def test_every_provider_defines_all_three_logical_profiles(self, provider):
        for profile in mp.REASONING_PROFILES_LOGICAL:
            settings = mp.resolve_reasoning_settings(provider, profile)
            assert isinstance(settings, dict)

    def test_low_is_the_default_profile(self):
        assert mp.DEFAULT_REASONING_PROFILE == "low"

    def test_anthropic_uses_adaptive_thinking_and_effort_not_budget_tokens(self):
        """Claude Sonnet 5: budget_tokens is removed (400) and thinking is
        always adaptive -- an empty {} is never treated as "low reasoning",
        since depth is controlled by output_config.effort instead."""
        for profile in ("low", "medium", "high"):
            settings = mp.resolve_reasoning_settings("anthropic", profile)
            assert settings["thinking"] == {"type": "adaptive"}
            assert settings["output_config"] == {"effort": profile}
            assert "budget_tokens" not in str(settings)

    def test_anthropic_settings_are_never_empty(self):
        for profile in mp.REASONING_PROFILES_LOGICAL:
            assert mp.resolve_reasoning_settings("anthropic", profile) != {}

    def test_openai_maps_profile_to_reasoning_effort(self):
        assert mp.resolve_reasoning_settings("openai", "low") == {"reasoning": {"effort": "low"}}
        assert mp.resolve_reasoning_settings("openai", "medium") == {"reasoning": {"effort": "medium"}}
        assert mp.resolve_reasoning_settings("openai", "high") == {"reasoning": {"effort": "high"}}

    def test_gemini_maps_profile_to_nested_thinking_config(self):
        """thinking_level must be nested under thinking_config, matching the
        real generation_config shape -- never a bare top-level thinking_level."""
        assert mp.resolve_reasoning_settings("gemini", "high") == {"thinking_config": {"thinking_level": "high"}}
        assert mp.resolve_reasoning_settings("gemini", "low") == {"thinking_config": {"thinking_level": "low"}}

    def test_deepseek_reasoning_effort_is_explicitly_set_for_every_profile(self):
        """DeepSeek must not rely on its provider default -- reasoning_effort
        is sent explicitly and does vary by profile (unlike a provider with
        no real lever, DeepSeek's OpenAI-compatible API does expose one)."""
        assert mp.resolve_reasoning_settings("deepseek", "low") == {"reasoning_effort": "low"}
        assert mp.resolve_reasoning_settings("deepseek", "medium") == {"reasoning_effort": "medium"}
        assert mp.resolve_reasoning_settings("deepseek", "high") == {"reasoning_effort": "high"}

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            mp.resolve_reasoning_settings("not_a_real_provider", "low")

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            mp.resolve_reasoning_settings("anthropic", "extreme")

    def test_default_max_output_tokens_is_512_for_every_provider(self):
        """The controllability experiment's visible answer is still just
        A/B -- the 512 cap exists so hidden/adaptive reasoning can't consume
        the whole output budget before any provider emits the letter."""
        for provider in mp.PROVIDERS:
            assert mp.default_max_output_tokens(provider) == 512


# ---------------------------------------------------------------------------
# Anthropic direct call
# ---------------------------------------------------------------------------

def _fake_anthropic_module(response_text="A", model_version="claude-sonnet-5-20250929", stop_reason="end_turn"):
    block = types.SimpleNamespace(type="text", text=response_text)
    usage = types.SimpleNamespace(input_tokens=42, output_tokens=3)
    response = types.SimpleNamespace(content=[block], model=model_version, id="msg_abc", stop_reason=stop_reason, usage=usage)

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return response

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    return types.SimpleNamespace(Anthropic=FakeClient), captured


class TestAnthropicDirectCall:
    def test_normalizes_response_shape(self, monkeypatch):
        fake_module, captured = _fake_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)

        result = mp.call_model("anthropic", "claude-sonnet-5", "hello", max_output_tokens=64, reasoning_profile="low")

        assert result["response_text"] == "A"
        assert result["provider"] == "anthropic"
        assert result["requested_model"] == "claude-sonnet-5"
        assert result["response_model"] == "claude-sonnet-5-20250929"
        assert result["reasoning_profile"] == "low"
        assert result["provider_reasoning_settings"] == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
        assert result["input_tokens"] == 42 and result["output_tokens"] == 3

    def test_high_reasoning_profile_sends_thinking_but_not_the_prompt_text(self, monkeypatch):
        fake_module, captured = _fake_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)

        mp.call_model("anthropic", "claude-sonnet-5", "the experimental prompt", max_output_tokens=64, reasoning_profile="high")
        kwargs = captured["kwargs"]
        assert kwargs["thinking"]["type"] == "adaptive"
        assert kwargs["output_config"]["effort"] == "high"
        assert kwargs["messages"] == [{"role": "user", "content": "the experimental prompt"}]

    def test_reasoning_profile_never_alters_the_prompt(self, monkeypatch):
        fake_module, captured = _fake_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)

        for profile in mp.REASONING_PROFILES_LOGICAL:
            mp.call_model("anthropic", "claude-sonnet-5", "fixed prompt text", max_output_tokens=64, reasoning_profile=profile)
            assert captured["kwargs"]["messages"][0]["content"] == "fixed prompt text"

    def test_no_reasoning_tokens_reported_for_anthropic(self, monkeypatch):
        fake_module, _ = _fake_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        result = mp.call_model("anthropic", "claude-sonnet-5", "hello", reasoning_profile="low")
        assert result["reasoning_tokens"] is None

    def test_no_chain_of_thought_content_anywhere_in_the_result(self, monkeypatch):
        fake_module, _ = _fake_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        result = mp.call_model("anthropic", "claude-sonnet-5", "hello", reasoning_profile="high")
        serialized = str(result)
        assert "thinking" not in serialized.lower() or result["provider_reasoning_settings"].get("thinking") is not None
        # the only place "thinking" may legitimately appear is the recorded
        # request setting, never as returned content
        assert "raw" in result and "thought" not in str(result["raw"]).lower()

    def test_sampling_params_are_applied(self, monkeypatch):
        fake_module, captured = _fake_anthropic_module()
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        mp.call_model("anthropic", "claude-sonnet-5", "hello", reasoning_profile="low", sampling_params={"temperature": 0})
        assert captured["kwargs"]["temperature"] == 0


# ---------------------------------------------------------------------------
# OpenAI direct call (Responses API)
# ---------------------------------------------------------------------------

def _fake_openai_module(output_text="B", reasoning_tokens=5):
    details = types.SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = types.SimpleNamespace(input_tokens=20, output_tokens=4, output_tokens_details=details)
    response = types.SimpleNamespace(output_text=output_text, model="gpt-5.6-2025", id="resp_abc", status="completed", usage=usage)

    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return response

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    return types.SimpleNamespace(OpenAI=FakeClient), captured


class TestOpenAIDirectCall:
    def test_normalizes_response_shape(self, monkeypatch):
        fake_module, captured = _fake_openai_module()
        monkeypatch.setitem(sys.modules, "openai", fake_module)

        result = mp.call_model("openai", "gpt-5.6", "hello", max_output_tokens=64, reasoning_profile="low")
        assert result["response_text"] == "B"
        assert result["provider"] == "openai"
        assert result["response_model"] == "gpt-5.6-2025"
        assert result["reasoning_tokens"] == 5

    def test_reasoning_effort_is_sent_and_prompt_is_unaffected(self, monkeypatch):
        fake_module, captured = _fake_openai_module()
        monkeypatch.setitem(sys.modules, "openai", fake_module)
        mp.call_model("openai", "gpt-5.6", "fixed prompt", reasoning_profile="medium")
        assert captured["kwargs"]["reasoning"] == {"effort": "medium"}
        assert captured["kwargs"]["input"] == "fixed prompt"

    def test_reasoning_tokens_recorded_when_available(self, monkeypatch):
        fake_module, _ = _fake_openai_module(reasoning_tokens=17)
        monkeypatch.setitem(sys.modules, "openai", fake_module)
        result = mp.call_model("openai", "gpt-5.6", "hello", reasoning_profile="low")
        assert result["reasoning_tokens"] == 17


# ---------------------------------------------------------------------------
# Gemini direct call
# ---------------------------------------------------------------------------

def _fake_gemini_module(text="A", thoughts_tokens=8):
    class FakeFinish:
        finish_reason = "STOP"

    class FakeUsage:
        prompt_token_count = 30
        candidates_token_count = 2
        thoughts_token_count = thoughts_tokens

    class FakeResponse:
        text = None  # set below

    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            resp = FakeResponse()
            resp.text = text
            resp.model_version = "gemini-3.8-flash-002"
            resp.usage_metadata = FakeUsage()
            resp.candidates = [FakeFinish()]
            return resp

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    fake_genai = types.SimpleNamespace(Client=FakeClient)

    class FakeThinkingConfig:
        def __init__(self, thinking_level=None):
            self.thinking_level = thinking_level

    class FakeGenerateContentConfig:
        def __init__(self, max_output_tokens=None, thinking_config=None):
            self.max_output_tokens = max_output_tokens
            self.thinking_config = thinking_config

    fake_types = types.SimpleNamespace(ThinkingConfig=FakeThinkingConfig, GenerateContentConfig=FakeGenerateContentConfig)
    fake_genai.types = fake_types  # so "from google.genai import types" resolves via plain attribute access
    fake_genai_pkg = types.SimpleNamespace(genai=fake_genai)
    return fake_genai_pkg, fake_genai, fake_types, captured


class TestGeminiDirectCall:
    def test_normalizes_response_shape(self, monkeypatch):
        fake_google_pkg, fake_genai, fake_types, captured = _fake_gemini_module()
        monkeypatch.setitem(sys.modules, "google", fake_google_pkg)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

        result = mp.call_model("gemini", "gemini-3.8-flash", "hello", max_output_tokens=512, reasoning_profile="low")
        assert result["response_text"] == "A"
        assert result["provider"] == "gemini"
        assert result["response_model"] == "gemini-3.8-flash-002"
        assert result["reasoning_tokens"] == 8
        assert captured["config"]["thinking_config"]["thinking_level"] == "low"

    def test_prompt_passed_through_unchanged_across_profiles(self, monkeypatch):
        fake_google_pkg, fake_genai, fake_types, captured = _fake_gemini_module()
        monkeypatch.setitem(sys.modules, "google", fake_google_pkg)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
        for profile in mp.REASONING_PROFILES_LOGICAL:
            mp.call_model("gemini", "gemini-3.8-flash", "fixed prompt", reasoning_profile=profile)
            assert captured["contents"] == [{"role": "user", "parts": [{"text": "fixed prompt"}]}]


# ---------------------------------------------------------------------------
# DeepSeek direct call (OpenAI-compatible client)
# ---------------------------------------------------------------------------

def _fake_deepseek_module(content="A", reasoning_content="secret chain of thought -- must never be read"):
    class FakeMessage:
        def __init__(self):
            self.content = content
            self.reasoning_content = reasoning_content  # must never be extracted

    class FakeChoice:
        def __init__(self):
            self.message = FakeMessage()
            self.finish_reason = "stop"

    class FakeUsage:
        prompt_tokens = 15
        completion_tokens = 1

    class FakeResponse:
        choices = [FakeChoice()]
        model = "deepseek-v4-pro"
        id = "chatcmpl_abc"
        usage = FakeUsage()

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = FakeChat()

    return types.SimpleNamespace(OpenAI=FakeClient), captured


class TestDeepSeekDirectCall:
    def test_normalizes_response_shape_and_uses_the_openai_compatible_base_url(self, monkeypatch):
        fake_module, captured = _fake_deepseek_module()
        monkeypatch.setitem(sys.modules, "openai", fake_module)

        result = mp.call_model("deepseek", "deepseek-v4-pro", "hello", max_output_tokens=64, reasoning_profile="low")
        assert result["response_text"] == "A"
        assert result["provider"] == "deepseek"
        assert captured["base_url"] == "https://api.deepseek.com"

    def test_reasoning_content_is_never_extracted_into_the_result(self, monkeypatch):
        """The response's reasoning_content field (if the model returns one)
        must never be read, stored, or leaked anywhere in the normalized
        result -- only the ordinary visible `content` field."""
        fake_module, _ = _fake_deepseek_module(content="B", reasoning_content="TOP SECRET REASONING")
        monkeypatch.setitem(sys.modules, "openai", fake_module)
        result = mp.call_model("deepseek", "deepseek-v4-pro", "hello", reasoning_profile="low")
        assert result["response_text"] == "B"
        assert "TOP SECRET REASONING" not in str(result)


# ---------------------------------------------------------------------------
# Pure, network-free batch request payload construction
# ---------------------------------------------------------------------------

class TestBuildBatchRequestPayload:
    @pytest.mark.parametrize("provider", mp.BATCH_SUPPORTED_PROVIDERS)
    def test_never_imports_any_sdk(self, provider, monkeypatch):
        """This must work with no provider package installed at all --
        remove anthropic/openai/google from sys.modules for the duration."""
        for name in list(sys.modules):
            if name.split(".")[0] in ("anthropic", "openai", "google"):
                monkeypatch.delitem(sys.modules, name, raising=False)
        payload = mp.build_batch_request_payload(provider, "some-model", "req-1", "a prompt", 64, {})
        assert isinstance(payload, dict)

    def test_anthropic_payload_shape(self):
        payload = mp.build_batch_request_payload("anthropic", "claude-sonnet-5", "req-1", "hi", 64, {"thinking": {"type": "enabled", "budget_tokens": 1024}})
        assert payload["custom_id"] == "req-1"
        assert payload["params"]["model"] == "claude-sonnet-5"
        assert payload["params"]["thinking"]["type"] == "enabled"

    def test_openai_payload_shape(self):
        payload = mp.build_batch_request_payload("openai", "gpt-5.6", "req-1", "hi", 64, {"reasoning": {"effort": "low"}})
        assert payload["custom_id"] == "req-1"
        assert payload["url"] == "/v1/responses"
        assert payload["body"]["input"] == "hi"

    def test_gemini_payload_shape(self):
        payload = mp.build_batch_request_payload("gemini", "gemini-3.8-flash", "req-1", "hi", 512, {"thinking_config": {"thinking_level": "low"}})
        assert payload["key"] == "req-1"
        assert payload["request"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
        assert payload["request"]["generation_config"]["thinking_config"]["thinking_level"] == "low"
        assert payload["request"]["generation_config"]["max_output_tokens"] == 512

    def test_gemini_batch_and_direct_settings_match(self):
        """Direct and batch Gemini calls must share the same request-building
        code (see _gemini_contents / _gemini_generation_config) so they
        cannot drift apart."""
        settings = mp.resolve_reasoning_settings("gemini", "high")
        payload = mp.build_batch_request_payload("gemini", "gemini-3.8-flash", "req-1", "hi", 512, settings)
        assert payload["request"]["contents"] == mp._gemini_contents("hi")
        assert payload["request"]["generation_config"] == mp._gemini_generation_config(512, settings)

    def test_deepseek_is_not_batch_supported(self):
        with pytest.raises(ValueError, match="does not support native batch"):
            mp.build_batch_request_payload("deepseek", "deepseek-v4-pro", "req-1", "hi", 64, {})

    def test_unique_request_ids_across_a_batch(self):
        payloads = [mp.build_batch_request_payload("anthropic", "m", f"req-{i}", "p", 64, {}) for i in range(5)]
        assert len({p["custom_id"] for p in payloads}) == 5


# ---------------------------------------------------------------------------
# Short deterministic batch request ids (provider custom_id / key)
# ---------------------------------------------------------------------------

class TestMakeShortRequestId:
    def test_within_length_and_charset_limit(self):
        request_id = mp.make_short_request_id(
            "some::very::long::trial::id::with::lots::of::components::baked::in",
            "rep-3", "a" * 64, "anthropic", "claude-sonnet-5", "high",
        )
        assert len(request_id) <= mp.SHORT_REQUEST_ID_MAX_LEN
        assert all(c.isalnum() or c in "_-" for c in request_id)

    def test_deterministic(self):
        args = ("trial-1", "rep-1", "abc123", "openai", "gpt-5.6", "low")
        assert mp.make_short_request_id(*args) == mp.make_short_request_id(*args)

    def test_unique_across_replicates_and_trials(self):
        base = ("trial-1", "rep-1", "abc123", "openai", "gpt-5.6", "low")
        ids = {
            mp.make_short_request_id(*base),
            mp.make_short_request_id("trial-2", "rep-1", "abc123", "openai", "gpt-5.6", "low"),
            mp.make_short_request_id("trial-1", "rep-2", "abc123", "openai", "gpt-5.6", "low"),
        }
        assert len(ids) == 3

    def test_changes_when_evaluator_identity_changes(self):
        """A different provider/model/reasoning_profile/prompt must yield a
        different id, since run_batch.is_completed keys on all of these."""
        base = mp.make_short_request_id("trial-1", "rep-1", "abc123", "anthropic", "claude-sonnet-5", "low")
        assert base != mp.make_short_request_id("trial-1", "rep-1", "abc123", "anthropic", "claude-sonnet-5", "high")
        assert base != mp.make_short_request_id("trial-1", "rep-1", "abc123", "openai", "claude-sonnet-5", "low")
        assert base != mp.make_short_request_id("trial-1", "rep-1", "def456", "anthropic", "claude-sonnet-5", "low")
