"""Per-run cost accounting for v2 executions (item 8).

Pricing lives in ONE small explicit structure (PRICING_USD_PER_MILLION_TOKENS)
so it can be updated without touching any analysis logic. Every dollar
figure this module produces is an ESTIMATE derived from the provider's own
reported usage tokens and a configured per-million-token rate -- never a
real invoice. compute_cost_summary() always sets "cost_is_estimated": True
and never silently invents a rate: an unpriced (provider, requested_model)
pair reports None costs rather than guessing.

DeepSeek's cache-hit/cache-miss prompt-token split (see model_providers.
_call_deepseek and controllability_v2_execution.run_one_observation's
attempt records) is used when present; when a run's usage data never
reports the split (e.g. an older result, or the field genuinely absent),
this falls back to pricing every prompt token at the (higher, conservative)
cache-miss rate rather than assuming free cache hits.
"""

# Illustrative/placeholder rates -- DeepSeek Flash is not a real, currently
# priced model, so these numbers are NOT sourced from a real price sheet.
# Update this table with actual published per-million-token rates before
# using estimated_*_cost_usd for any real budget decision.
PRICING_USD_PER_MILLION_TOKENS = {
    "deepseek": {
        "deepseek-flash": {"cache_miss_input": 0.27, "cache_hit_input": 0.07, "output": 1.10},
    },
}


def compute_cost_summary(rows, provider, requested_model, pricing=None):
    """rows: result rows (the v2 row schema -- each has "attempts" and
    "parsing_status") for ONE evaluator/(provider, requested_model) pair.
    Sums usage across EVERY attempt (a retried, ultimately-failed
    observation still spent tokens on each attempt), and counts
    "successful_planned_observations" as rows with parsing_status ==
    "resolved" (at most one per planned observation, regardless of how many
    attempts it took)."""
    pricing = pricing if pricing is not None else PRICING_USD_PER_MILLION_TOKENS

    total_calls_attempted = 0
    successful_observations = 0
    prompt_tokens = completion_tokens = reasoning_tokens = 0
    cache_hit_tokens = cache_miss_tokens = 0
    have_cache_breakdown = False

    for row in rows:
        if row.get("parsing_status") == "resolved":
            successful_observations += 1
        for attempt in row.get("attempts") or []:
            total_calls_attempted += 1
            prompt_tokens += attempt.get("input_tokens") or 0
            completion_tokens += attempt.get("output_tokens") or 0
            reasoning_tokens += attempt.get("reasoning_tokens") or 0
            hit, miss = attempt.get("prompt_cache_hit_tokens"), attempt.get("prompt_cache_miss_tokens")
            if hit is not None or miss is not None:
                have_cache_breakdown = True
                cache_hit_tokens += hit or 0
                cache_miss_tokens += miss or 0

    rates = (pricing.get(provider) or {}).get(requested_model)
    if rates is None:
        estimated_input_cost = estimated_output_cost = estimated_total_cost = None
        pricing_note = f"no pricing configured for provider={provider!r} requested_model={requested_model!r}"
    else:
        if have_cache_breakdown:
            estimated_input_cost = (cache_hit_tokens * rates["cache_hit_input"] + cache_miss_tokens * rates["cache_miss_input"]) / 1_000_000
        else:
            # No cache-hit/miss split reported anywhere in these rows --
            # conservatively price every prompt token at the (higher)
            # cache-miss rate rather than assuming any hits.
            estimated_input_cost = (prompt_tokens * rates["cache_miss_input"]) / 1_000_000
        estimated_output_cost = (completion_tokens * rates["output"]) / 1_000_000
        estimated_total_cost = estimated_input_cost + estimated_output_cost
        pricing_note = "estimated from configured placeholder rates -- update PRICING_USD_PER_MILLION_TOKENS with real published pricing before using for budget decisions"

    return {
        "provider": provider,
        "requested_model": requested_model,
        "total_calls_attempted": total_calls_attempted,
        "successful_planned_observations": successful_observations,
        "prompt_tokens": prompt_tokens,
        "cache_hit_prompt_tokens": cache_hit_tokens if have_cache_breakdown else None,
        "cache_miss_prompt_tokens": cache_miss_tokens if have_cache_breakdown else None,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_input_cost_usd": estimated_input_cost,
        "estimated_output_cost_usd": estimated_output_cost,
        "estimated_total_cost_usd": estimated_total_cost,
        "cost_is_estimated": True,
        "pricing_note": pricing_note,
    }
