"""Provider-neutral model-calling abstraction: one direct-call interface
(call_model) plus native-batch request/status/collect primitives for each
supported provider. All provider-specific request/response handling lives
here -- run_batch.py, run_trial.py, and batch_run.py never construct or
parse a provider-native payload directly.

Supported providers (--provider on the CLI; arbitrary provider-supported
model strings are always accepted, these are just the defaults):

    anthropic  -> claude-sonnet-5   (ANTHROPIC_API_KEY)
    openai     -> gpt-5.6           (OPENAI_API_KEY)
    gemini     -> gemini-3.8-flash  (GEMINI_API_KEY)
    deepseek   -> deepseek-flash    (DEEPSEEK_API_KEY)

Every SDK is imported lazily, inside the function that actually needs it
(matching the pre-existing run_trial.call_claude convention), so:
  - --dry-run and unit tests never require openai/google-genai to be
    installed (see build_batch_request_payload, which needs no SDK at all);
  - tests mock a provider's SDK by injecting a fake module into
    sys.modules, never by calling a real API.

Reasoning: each provider gets a FIXED reasoning configuration per logical
profile ("low", the default, "medium", or "high" -- see REASONING_PROFILES).
The profile is kept identical across every experimental condition for a
given evaluator (naturalistic vs. suppression, every contrast, every
replicate) and is recorded on every result row alongside the provider's
literal native settings (provider_reasoning_settings) -- the two are NOT
pretended to be equivalent across providers; each provider's docstring
below says exactly what its profile does and doesn't control. Chain-of-
thought / extended-thinking CONTENT is never requested, exposed, parsed, or
stored anywhere in this module -- only token-count usage (reasoning_tokens),
when a provider reports it, is kept.
"""

import os

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.6",
    "gemini": "gemini-3.8-flash",
    "deepseek": "deepseek-flash",
}

API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

PROVIDERS = tuple(DEFAULT_MODELS)

REASONING_PROFILES_LOGICAL = ("low", "medium", "high")
DEFAULT_REASONING_PROFILE = "low"

# Output-token budget for this experiment (controllability's plain A/B
# task): 512 for every provider. The visible answer is still just "A"/"B" --
# the larger cap exists so a provider's hidden/adaptive reasoning doesn't
# consume the entire output budget before the model gets to answer (this
# was previously 64 for three of the four providers, which is exactly the
# failure mode: adaptive thinking can use up a 64-token cap before ever
# emitting the visible letter).
DEFAULT_MAX_OUTPUT_TOKENS = {
    "anthropic": 512,
    "openai": 512,
    "gemini": 512,
    "deepseek": 512,
}

# ---------------------------------------------------------------------------
# Reasoning profile -> provider-native settings. One table, not scattered
# through runner code. Each provider's "low" is its own low/default-low
# configuration -- never an empty {} standing in for "low reasoning" (an
# empty settings dict is not itself a valid low-reasoning request on any of
# these APIs; it either falls back to that provider's own default, which is
# not necessarily "low", or -- for Claude Sonnet 5 specifically -- adaptive
# thinking runs regardless of what `thinking` is set to, so the request
# must still set output_config.effort explicitly to get a "low" response).
# ---------------------------------------------------------------------------

REASONING_PROFILES = {
    # Claude Sonnet 5: thinking is ALWAYS adaptive (there is no off switch
    # analogous to Opus 4.7/4.8's "omit thinking" behavior, and
    # budget_tokens is REMOVED -- a 400 error -- on Sonnet 5). Depth is
    # controlled by output_config.effort instead. Sonnet 5 also rejects
    # temperature/top_p/top_k entirely (see run_trial.resolve_sampling_params
    # / the provider-default sampling regime used for this experiment) --
    # unrelated to reasoning, but a consequence of the same adaptive-thinking
    # architecture.
    "anthropic": {
        "low": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
        "medium": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
        "high": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
    },
    # GPT-5.6 Responses API: reasoning.effort is the model's own documented
    # lever, mapped one-to-one onto our low/medium/high.
    "openai": {
        "low": {"reasoning": {"effort": "low"}},
        "medium": {"reasoning": {"effort": "medium"}},
        "high": {"reasoning": {"effort": "high"}},
    },
    # Gemini 3.8 Flash: generation_config.thinking_config.thinking_level is
    # the model's own documented lever -- nested under thinking_config, not
    # a bare top-level thinking_level (see _gemini_generation_config, used
    # by both the direct call and the batch payload builder so they can't
    # drift apart).
    "gemini": {
        "low": {"thinking_config": {"thinking_level": "low"}},
        "medium": {"thinking_config": {"thinking_level": "medium"}},
        "high": {"thinking_config": {"thinking_level": "high"}},
    },
    # DeepSeek Flash: explicitly set reasoning_effort on every request
    # (sent via the OpenAI-compatible client's extra_body, since it isn't a
    # parameter name the openai-python client itself defines) rather than
    # relying on whatever DeepSeek's own default happens to be.
    "deepseek": {
        "low": {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "medium"},
        "high": {"reasoning_effort": "high"},
    },
}


def get_api_key(provider):
    env_var = API_KEY_ENV_VARS[provider]
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"{env_var} is not set (required for provider={provider!r})")
    return key


def resolve_reasoning_settings(provider, reasoning_profile):
    if provider not in REASONING_PROFILES:
        raise ValueError(f"Unknown provider: {provider!r}. Supported: {list(PROVIDERS)}")
    if reasoning_profile not in REASONING_PROFILES_LOGICAL:
        raise ValueError(f"Unknown reasoning_profile: {reasoning_profile!r}. Supported: {list(REASONING_PROFILES_LOGICAL)}")
    return dict(REASONING_PROFILES[provider][reasoning_profile])


def default_max_output_tokens(provider):
    return DEFAULT_MAX_OUTPUT_TOKENS[provider]


def normalize_response(provider, requested_model, response_text, response_model=None, request_id=None,
                        input_tokens=None, output_tokens=None, reasoning_tokens=None, stop_reason=None,
                        reasoning_profile=DEFAULT_REASONING_PROFILE, provider_reasoning_settings=None, raw=None,
                        system_fingerprint=None, prompt_cache_hit_tokens=None, prompt_cache_miss_tokens=None):
    """The one normalized shape every provider's direct-call and
    batch-collect path returns. `raw` is small, provider-native debug
    metadata (ids, stop reasons, token breakdowns) for troubleshooting --
    never chain-of-thought/reasoning content.

    system_fingerprint/prompt_cache_hit_tokens/prompt_cache_miss_tokens are
    DeepSeek-specific usage fields (see _call_deepseek); every other
    provider leaves them at their None default rather than fabricating a
    value. `prompt_tokens`/`completion_tokens` are literal aliases of
    input_tokens/output_tokens -- some callers (and DeepSeek's own raw
    usage shape) name them that way; both names always agree.
    """
    return {
        "response_text": response_text,
        "provider": provider,
        "requested_model": requested_model,
        "response_model": response_model,
        "request_id": request_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
        "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
        "system_fingerprint": system_fingerprint,
        "stop_reason": stop_reason,
        "reasoning_profile": reasoning_profile,
        "provider_reasoning_settings": provider_reasoning_settings or {},
        "raw": raw or {},
    }


# ---------------------------------------------------------------------------
# Direct (synchronous, one-request) execution
# ---------------------------------------------------------------------------

def call_model(provider, model, prompt, max_output_tokens=None, reasoning_profile=DEFAULT_REASONING_PROFILE, sampling_params=None):
    """Provider-neutral direct call. Returns a normalize_response() dict.
    max_output_tokens defaults to DEFAULT_MAX_OUTPUT_TOKENS[provider] when
    omitted. Changing reasoning_profile never alters `prompt`.

    sampling_params (e.g. {"temperature": 0}, see run_trial.SAMPLING_REGIMES)
    is the pre-existing v0.2 sampling-regime mechanism, orthogonal to
    reasoning_profile -- it's applied for Anthropic (which is what every
    existing sampling-regime run uses) and otherwise accepted-and-ignored,
    since no other provider's low-variance/temperature-equivalent mapping
    was specified here.
    """
    if max_output_tokens is None:
        max_output_tokens = default_max_output_tokens(provider)
    settings = resolve_reasoning_settings(provider, reasoning_profile)
    try:
        call_fn = _DIRECT_CALL_FUNCTIONS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider: {provider!r}. Supported: {list(PROVIDERS)}")
    return call_fn(model, prompt, max_output_tokens, reasoning_profile, settings, sampling_params)


def _call_anthropic(model, prompt, max_output_tokens, reasoning_profile, settings, sampling_params=None):
    import anthropic  # lazy: keeps --dry-run/tests independent of the SDK being installed

    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    create_kwargs = dict(model=model, max_tokens=max_output_tokens, messages=[{"role": "user", "content": prompt}])
    create_kwargs.update(settings)
    if sampling_params:
        create_kwargs.update(sampling_params)
    response = client.messages.create(**create_kwargs)

    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        raise ValueError(f"No text blocks in response. Block types were: {[b.type for b in response.content]}")

    return normalize_response(
        provider="anthropic", requested_model=model, response_text="\n".join(text_blocks),
        response_model=getattr(response, "model", None), request_id=getattr(response, "id", None),
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
        reasoning_tokens=None, stop_reason=response.stop_reason,
        reasoning_profile=reasoning_profile, provider_reasoning_settings=settings,
        raw={"id": getattr(response, "id", None), "stop_reason": response.stop_reason},
    )


def _call_openai(model, prompt, max_output_tokens, reasoning_profile, settings, sampling_params=None):
    import openai  # lazy

    client = openai.OpenAI(api_key=get_api_key("openai"))
    create_kwargs = dict(model=model, input=prompt, max_output_tokens=max_output_tokens)
    create_kwargs.update(settings)
    response = client.responses.create(**create_kwargs)

    usage = getattr(response, "usage", None)
    reasoning_tokens = None
    if usage is not None:
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None

    return normalize_response(
        provider="openai", requested_model=model, response_text=getattr(response, "output_text", "") or "",
        response_model=getattr(response, "model", None), request_id=getattr(response, "id", None),
        input_tokens=getattr(usage, "input_tokens", None) if usage is not None else None,
        output_tokens=getattr(usage, "output_tokens", None) if usage is not None else None,
        reasoning_tokens=reasoning_tokens, stop_reason=getattr(response, "status", None),
        reasoning_profile=reasoning_profile, provider_reasoning_settings=settings,
        raw={"id": getattr(response, "id", None), "status": getattr(response, "status", None)},
    )


def _gemini_contents(prompt):
    """The structured `contents` shape a native GenerateContentRequest
    needs (a list of Content objects, each with a role and parts) -- shared
    by the direct call and the batch payload builder so they cannot drift
    apart. A bare string `contents=prompt` is Python-SDK-only convenience
    sugar for the direct call; the native batch JSON format requires the
    real structured shape, so this is used everywhere, not just in batch.
    """
    return [{"role": "user", "parts": [{"text": prompt}]}]


def _gemini_generation_config(max_output_tokens, settings):
    """generation_config, with thinking_config nested correctly (never a
    bare top-level thinking_level) -- shared by the direct call and the
    batch payload builder. `settings` is one of REASONING_PROFILES["gemini"]'s
    values, e.g. {"thinking_config": {"thinking_level": "low"}}."""
    config = {"max_output_tokens": max_output_tokens}
    if "thinking_config" in settings:
        config["thinking_config"] = dict(settings["thinking_config"])
    return config


def _call_gemini(model, prompt, max_output_tokens, reasoning_profile, settings, sampling_params=None):
    from google import genai  # lazy

    client = genai.Client(api_key=get_api_key("gemini"))
    # The google-genai SDK accepts plain dicts for `contents`/`config` (it
    # validates them the same way as its typed Content/GenerateContentConfig
    # classes), which is exactly what lets this share _gemini_contents/
    # _gemini_generation_config with the batch payload builder below instead
    # of maintaining two request-building implementations.
    response = client.models.generate_content(
        model=model, contents=_gemini_contents(prompt), config=_gemini_generation_config(max_output_tokens, settings)
    )

    usage = getattr(response, "usage_metadata", None)
    reasoning_tokens = getattr(usage, "thoughts_token_count", None) if usage is not None else None

    return normalize_response(
        provider="gemini", requested_model=model, response_text=getattr(response, "text", "") or "",
        response_model=getattr(response, "model_version", None), request_id=None,
        input_tokens=getattr(usage, "prompt_token_count", None) if usage is not None else None,
        output_tokens=getattr(usage, "candidates_token_count", None) if usage is not None else None,
        reasoning_tokens=reasoning_tokens,
        stop_reason=(response.candidates[0].finish_reason if getattr(response, "candidates", None) else None),
        reasoning_profile=reasoning_profile, provider_reasoning_settings=settings,
        raw={"model_version": getattr(response, "model_version", None)},
    )


def _call_deepseek(model, prompt, max_output_tokens, reasoning_profile, settings, sampling_params=None):
    """DeepSeek is served through the OpenAI-compatible chat.completions
    endpoint. `reasoning_content` (if the model returns any) is DELIBERATELY
    never read here -- only the ordinary visible `content` field is ever
    extracted; chain-of-thought is never requested, exposed, parsed, or
    stored."""
    import openai  # lazy; DeepSeek uses the OpenAI-compatible client/interface

    client = openai.OpenAI(api_key=get_api_key("deepseek"), base_url="https://api.deepseek.com")
    # `settings` (e.g. {"reasoning_effort": "low"}) is sent via extra_body:
    # it isn't a parameter the openai-python client's typed signature
    # defines, and extra_body is the documented way to pass a provider
    # -specific field straight through into the raw request body.
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_output_tokens,
        extra_body=dict(settings) if settings else None,
    )
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    reasoning_tokens = None
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None

    return normalize_response(
        provider="deepseek", requested_model=model, response_text=choice.message.content or "",
        response_model=getattr(response, "model", None), request_id=getattr(response, "id", None),
        input_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
        reasoning_tokens=reasoning_tokens, stop_reason=choice.finish_reason,
        reasoning_profile=reasoning_profile, provider_reasoning_settings=settings,
        system_fingerprint=getattr(response, "system_fingerprint", None),
        prompt_cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", None) if usage is not None else None,
        prompt_cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", None) if usage is not None else None,
        raw={"id": getattr(response, "id", None), "finish_reason": choice.finish_reason},
    )


_DIRECT_CALL_FUNCTIONS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
    "deepseek": _call_deepseek,
}


# ---------------------------------------------------------------------------
# Native batch: submit / status / collect. Every function here is pure
# request/response translation -- trial selection, block-aware --limit,
# and replicate handling all live in batch_run.py (reused from run_batch.py),
# never duplicated here.
#
# `requests` is always a list of {"request_key": str, "prompt": str}, where
# request_key is caller-assigned and stable per (trial_id, replicate_id) --
# batch_run.py is responsible for that mapping. Results are always matched
# back by request_key/custom_id, NEVER by output position/order.
# ---------------------------------------------------------------------------

BATCH_SUPPORTED_PROVIDERS = ("anthropic", "openai", "gemini")

# All four batch APIs (Anthropic custom_id, OpenAI custom_id, Gemini request
# key) cap request identifiers at 64 characters and/or restrict the
# character set. A raw "trial_id::replicate_id" string can exceed 64 chars
# once contrast/pair/regime/rubric are baked into trial_id (see
# controllability_trials.py's block_id scheme), so batch requests use a
# short deterministic hash instead -- see make_short_request_id.
SHORT_REQUEST_ID_MAX_LEN = 64


def make_short_request_id(trial_id, replicate_id, prompt_sha256, provider, requested_model, reasoning_profile):
    """A short (<=64 char), safe-character ("r_" + 32 hex chars = 34 chars),
    deterministic request id for one (trial_id, replicate_id) under one
    (provider, requested_model, reasoning_profile, prompt) identity.
    Deterministic: the same inputs always produce the same id, so a
    resubmission of the exact same request reuses the exact same id.
    Unique within a batch: trial_id/replicate_id alone already uniquely
    identify one request in any single submission, and folding in
    prompt_sha256/provider/requested_model/reasoning_profile means an id
    also changes if any of that identity changes (see
    run_batch.is_completed, which requires all of these to match before a
    saved result counts as satisfying a trial). The caller (batch_run.py)
    keeps the local mapping from this id back to the full trial metadata --
    this function never needs to be reversed.
    """
    import hashlib

    basis = "\x1f".join(
        str(part) for part in (trial_id, replicate_id, prompt_sha256, provider, requested_model, reasoning_profile)
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]
    return f"r_{digest}"


def build_batch_request_payload(provider, model, request_key, prompt, max_output_tokens, reasoning_settings):
    """Pure, network-free construction of one provider-native batch request
    item, for --dry-run/testing: builds the exact payload submit_batch()
    would send, without importing any SDK or making any call."""
    if provider == "anthropic":
        return {
            "custom_id": request_key,
            "params": {
                "model": model, "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": prompt}], **reasoning_settings,
            },
        }
    if provider == "openai":
        return {
            "custom_id": request_key, "method": "POST", "url": "/v1/responses",
            "body": {"model": model, "input": prompt, "max_output_tokens": max_output_tokens, **reasoning_settings},
        }
    if provider == "gemini":
        return {
            "key": request_key,
            "request": {
                "model": model,
                "contents": _gemini_contents(prompt),
                "generation_config": _gemini_generation_config(max_output_tokens, reasoning_settings),
            },
        }
    raise ValueError(f"{provider!r} does not support native batch submission -- supported: {BATCH_SUPPORTED_PROVIDERS}")


def submit_batch(provider, model, requests, reasoning_profile=DEFAULT_REASONING_PROFILE, max_output_tokens=None):
    """Submit a real native batch job. Returns {"provider_batch_id": str, "raw": {...}}."""
    if max_output_tokens is None:
        max_output_tokens = default_max_output_tokens(provider)
    settings = resolve_reasoning_settings(provider, reasoning_profile)
    try:
        submit_fn = _BATCH_SUBMIT_FUNCTIONS[provider]
    except KeyError:
        raise ValueError(f"{provider!r} does not support native batch submission -- supported: {BATCH_SUPPORTED_PROVIDERS}")
    return submit_fn(model, requests, max_output_tokens, settings)


def batch_status(provider, provider_batch_id):
    """Returns {"status": str, "raw": {...}}. `status` is normalized to one
    of "in_progress", "completed", "failed", "canceled" where the provider's
    own status maps cleanly; otherwise the provider's raw status string."""
    try:
        status_fn = _BATCH_STATUS_FUNCTIONS[provider]
    except KeyError:
        raise ValueError(f"{provider!r} does not support native batch submission -- supported: {BATCH_SUPPORTED_PROVIDERS}")
    return status_fn(provider_batch_id)


def collect_batch(provider, provider_batch_id, request_keys):
    """Returns {request_key: normalize_response()_dict_or_{"error": str}}
    for every key in request_keys that the provider has a result for.
    Matched strictly by request_key/custom_id -- batch result order is
    never relied on. A request_key with no matching result yet is simply
    omitted (the caller decides what "still pending" means)."""
    try:
        collect_fn = _BATCH_COLLECT_FUNCTIONS[provider]
    except KeyError:
        raise ValueError(f"{provider!r} does not support native batch submission -- supported: {BATCH_SUPPORTED_PROVIDERS}")
    return collect_fn(provider_batch_id, set(request_keys))


# --- Anthropic: Message Batches API ----------------------------------------

def _submit_batch_anthropic(model, requests, max_output_tokens, settings):
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    payload = [build_batch_request_payload("anthropic", model, r["request_key"], r["prompt"], max_output_tokens, settings)
               for r in requests]
    batch = client.messages.batches.create(requests=payload)
    return {"provider_batch_id": batch.id, "raw": {"processing_status": getattr(batch, "processing_status", None)}}


def _status_batch_anthropic(provider_batch_id):
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    batch = client.messages.batches.retrieve(provider_batch_id)
    processing_status = getattr(batch, "processing_status", None)
    status = {"ended": "completed", "in_progress": "in_progress", "canceling": "canceled"}.get(processing_status, processing_status)
    counts = getattr(batch, "request_counts", None)
    return {"status": status, "raw": {"processing_status": processing_status, "request_counts": vars(counts) if counts else None}}


def _collect_batch_anthropic(provider_batch_id, request_keys):
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    results = {}
    # client.messages.batches.results() streams one entry per custom_id;
    # never assume the first/only content block is the visible text, and
    # never rely on stream order to match a request back to its key.
    for entry in client.messages.batches.results(provider_batch_id):
        custom_id = entry.custom_id
        if custom_id not in request_keys:
            continue
        result_type = getattr(entry.result, "type", None)
        if result_type == "succeeded":
            message = entry.result.message
            text_blocks = [b.text for b in message.content if getattr(b, "type", None) == "text"]
            results[custom_id] = normalize_response(
                provider="anthropic", requested_model=message.model, response_text="\n".join(text_blocks),
                response_model=message.model, request_id=getattr(message, "id", None),
                input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens,
                stop_reason=message.stop_reason, raw={"batch_result_type": result_type},
            )
        else:
            error = getattr(entry.result, "error", None)
            results[custom_id] = {"error": f"Anthropic batch request failed ({result_type}): {error}"}
    return results


# --- OpenAI: Batch API over the Responses endpoint --------------------------

def _submit_batch_openai(model, requests, max_output_tokens, settings):
    import io
    import json as _json

    import openai

    client = openai.OpenAI(api_key=get_api_key("openai"))
    lines = [
        _json.dumps(build_batch_request_payload("openai", model, r["request_key"], r["prompt"], max_output_tokens, settings))
        for r in requests
    ]
    buffer = io.BytesIO("\n".join(lines).encode("utf-8"))
    buffer.name = "batch_input.jsonl"
    input_file = client.files.create(file=buffer, purpose="batch")
    batch = client.batches.create(input_file_id=input_file.id, endpoint="/v1/responses", completion_window="24h")
    return {"provider_batch_id": batch.id, "raw": {"status": batch.status, "input_file_id": input_file.id}}


def _status_batch_openai(provider_batch_id):
    import openai

    client = openai.OpenAI(api_key=get_api_key("openai"))
    batch = client.batches.retrieve(provider_batch_id)
    status = {"completed": "completed", "failed": "failed", "cancelled": "canceled",
              "in_progress": "in_progress", "validating": "in_progress", "finalizing": "in_progress"}.get(batch.status, batch.status)
    return {"status": status, "raw": {"status": batch.status, "request_counts": getattr(batch, "request_counts", None)}}


def _collect_batch_openai(provider_batch_id, request_keys):
    import json as _json

    import openai

    client = openai.OpenAI(api_key=get_api_key("openai"))
    batch = client.batches.retrieve(provider_batch_id)
    results = {}

    def _read_lines(file_id):
        if not file_id:
            return []
        content = client.files.content(file_id)
        return [line for line in content.text.splitlines() if line.strip()]

    for line in _read_lines(getattr(batch, "output_file_id", None)):
        row = _json.loads(line)
        custom_id = row.get("custom_id")
        if custom_id not in request_keys:
            continue
        response = row.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code") == 200:
            output_text = "".join(
                part.get("text", "")
                for item in body.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            usage = body.get("usage") or {}
            results[custom_id] = normalize_response(
                provider="openai", requested_model=body.get("model"), response_text=output_text,
                response_model=body.get("model"), request_id=body.get("id"),
                input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                stop_reason=body.get("status"), raw={"status_code": response.get("status_code")},
            )
        else:
            results[custom_id] = {"error": f"OpenAI batch request failed: {response}"}

    for line in _read_lines(getattr(batch, "error_file_id", None)):
        row = _json.loads(line)
        custom_id = row.get("custom_id")
        if custom_id in request_keys and custom_id not in results:
            results[custom_id] = {"error": f"OpenAI batch request errored: {row.get('error')}"}

    return results


# --- Gemini: native Batch API, file-based input -----------------------------

def _submit_batch_gemini(model, requests, max_output_tokens, settings):
    import io

    from google import genai

    client = genai.Client(api_key=get_api_key("gemini"))
    lines = [
        __import__("json").dumps(build_batch_request_payload("gemini", model, r["request_key"], r["prompt"], max_output_tokens, settings))
        for r in requests
    ]
    buffer = io.BytesIO("\n".join(lines).encode("utf-8"))
    uploaded = client.files.upload(file=buffer, config={"mime_type": "jsonl", "display_name": "controllability_batch_input"})
    job = client.batches.create(model=model, src=uploaded.name)
    return {"provider_batch_id": job.name, "raw": {"state": getattr(job, "state", None)}}


def _status_batch_gemini(provider_batch_id):
    from google import genai

    client = genai.Client(api_key=get_api_key("gemini"))
    job = client.batches.get(name=provider_batch_id)
    state = getattr(job, "state", None)
    status = {"JOB_STATE_SUCCEEDED": "completed", "JOB_STATE_FAILED": "failed",
              "JOB_STATE_CANCELLED": "canceled", "JOB_STATE_RUNNING": "in_progress",
              "JOB_STATE_PENDING": "in_progress"}.get(str(state), str(state))
    return {"status": status, "raw": {"state": str(state)}}


def _collect_batch_gemini(provider_batch_id, request_keys):
    import json as _json

    from google import genai

    client = genai.Client(api_key=get_api_key("gemini"))
    job = client.batches.get(name=provider_batch_id)
    results = {}
    dest = getattr(job, "dest", None)
    output_file = getattr(dest, "file_name", None) if dest is not None else None
    if not output_file:
        return results

    raw_bytes = client.files.download(file=output_file)
    for line in raw_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        # Matched by the same stable "key" every request carried in (see
        # build_batch_request_payload) -- never by line/output position.
        key = row.get("key")
        if key not in request_keys:
            continue
        if "response" in row:
            response = row["response"]
            candidates = response.get("candidates") or [{}]
            parts = (candidates[0].get("content") or {}).get("parts") or []
            output_text = "".join(p.get("text", "") for p in parts)
            usage = response.get("usageMetadata") or {}
            results[key] = normalize_response(
                provider="gemini", requested_model=response.get("modelVersion"), response_text=output_text,
                response_model=response.get("modelVersion"), request_id=None,
                input_tokens=usage.get("promptTokenCount"), output_tokens=usage.get("candidatesTokenCount"),
                reasoning_tokens=usage.get("thoughtsTokenCount"),
                stop_reason=(candidates[0].get("finishReason") if candidates else None), raw={},
            )
        else:
            results[key] = {"error": f"Gemini batch request failed: {row.get('error')}"}
    return results


_BATCH_SUBMIT_FUNCTIONS = {
    "anthropic": _submit_batch_anthropic,
    "openai": _submit_batch_openai,
    "gemini": _submit_batch_gemini,
}
_BATCH_STATUS_FUNCTIONS = {
    "anthropic": _status_batch_anthropic,
    "openai": _status_batch_openai,
    "gemini": _status_batch_gemini,
}
_BATCH_COLLECT_FUNCTIONS = {
    "anthropic": _collect_batch_anthropic,
    "openai": _collect_batch_openai,
    "gemini": _collect_batch_gemini,
}
