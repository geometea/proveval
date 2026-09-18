"""Empirical throughput / runtime planning for v3 runs.

The only throughput ever OBSERVED for this pipeline is v2's Wave 2 recovery:
roughly 1.4 judgments/s at concurrency 32 for difficult reasoning calls
(OBSERVED_V2_RECOVERY_JPS). No v3 throughput has been measured. Anything
else is an assumption and is labelled as one everywhere it is printed.

summarize_run() derives throughput and latency statistics from any v3 raw
JSONL result file (rows carry a write timestamp, the resolving attempt's
latency, per-attempt timestamps/latencies, token counts). Wall-clock time
is the span of row timestamps with idle gaps removed (a gap longer than
IDLE_GAP_FACTOR x the median latency is treated as a pause between
invocations, not as running time), so a resumed run does not inflate its
elapsed time. Effective concurrency is estimated by Little's law
(sum of latencies / active time) when the run did not record it.

plan_scenarios() turns a judgment count into three labelled scenarios:
  optimistic   : empirical x OPTIMISTIC_FACTOR (or the ASSUMED optimistic rate)
  empirical    : measured from --throughput-from-results, or --throughput-jps
  conservative : min(empirical, OBSERVED_V2_RECOVERY_JPS) -- never faster
                 than the one rate ever observed
None of this is precise; every scenario carries its basis in the output.
"""

import json
import os
import statistics
from datetime import datetime

OBSERVED_V2_RECOVERY_JPS = 1.4          # observed: v2 Wave 2 recovery, concurrency 32, reasoning calls
OBSERVED_V2_RECOVERY_CONCURRENCY = 32
ASSUMED_OPTIMISTIC_JPS = 4.0            # assumption only (no v3 measurement exists)
ASSUMED_EMPIRICAL_FALLBACK_JPS = 2.0    # assumption only, used when neither a file nor a rate is given
OPTIMISTIC_FACTOR = 1.5
IDLE_GAP_FACTOR = 10.0
MIN_IDLE_GAP_SECONDS = 120.0


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def active_time(timestamps, median_latency):
    """Sum of intervals between consecutive row timestamps, excluding idle
    gaps (> max(IDLE_GAP_FACTOR x median latency, MIN_IDLE_GAP_SECONDS)).
    Returns (active_seconds, wall_seconds, n_gaps_removed)."""
    ts = sorted(t for t in timestamps if t is not None)
    if len(ts) < 2:
        return 0.0, 0.0, 0
    gap_limit = max(IDLE_GAP_FACTOR * (median_latency or 0.0), MIN_IDLE_GAP_SECONDS)
    active, gaps = 0.0, 0
    for a, b in zip(ts, ts[1:]):
        d = b - a
        if d > gap_limit:
            gaps += 1
        else:
            active += d
    return active, ts[-1] - ts[0], gaps


def summarize_run(rows, concurrency=None):
    """Throughput/latency/error/token summary of a list of raw result rows."""
    valid = [r for r in rows if r.get("parsing_status") == "resolved"]
    latencies = sorted(r["latency_seconds"] for r in valid if r.get("latency_seconds") is not None)
    attempt_latencies = sorted(a["latency_seconds"] for r in rows for a in (r.get("attempts") or []) if a.get("latency_seconds") is not None)
    median_latency = statistics.median(latencies) if latencies else None
    active, wall, gaps = active_time([_parse_ts(r.get("timestamp")) for r in rows], median_latency)
    n_attempts = sum(len(r.get("attempts") or []) for r in rows)
    first_valid = sum(1 for r in rows if r.get("first_attempt_status") == "valid")
    summary = {
        "n_rows": len(rows),
        "completed_judgments": len(valid),
        "unresolved_rows": len(rows) - len(valid),
        "elapsed_wall_seconds": wall,
        "active_seconds": active,
        "idle_gaps_removed": gaps,
        "effective_judgments_per_second": (len(valid) / active) if active > 0 else None,
        "wall_clock_judgments_per_second": (len(valid) / wall) if wall > 0 else None,
        "median_latency_seconds": median_latency,
        "p50_latency_seconds": percentile(latencies, 50),
        "p90_latency_seconds": percentile(latencies, 90),
        "p95_latency_seconds": percentile(latencies, 95),
        "p50_attempt_latency_seconds": percentile(attempt_latencies, 50),
        "total_attempts": n_attempts,
        "attempts_per_row": (n_attempts / len(rows)) if rows else None,
        "first_attempt_error_fraction": (1 - first_valid / len(rows)) if rows else None,
        "retry_fraction": (sum(1 for r in rows if (r.get("total_attempts") or 1) > 1) / len(rows)) if rows else None,
        "unresolved_fraction": ((len(rows) - len(valid)) / len(rows)) if rows else None,
        "recorded_concurrency": concurrency,
        "estimated_concurrency_littles_law": (sum(attempt_latencies) / active) if active > 0 and attempt_latencies else None,
    }
    for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens"):
        vals = [r[key] for r in valid if r.get(key) is not None]
        summary[f"mean_{key}_per_successful_judgment"] = (sum(vals) / len(vals)) if vals else None
    return summary


def summarize_results_file(path, concurrency=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return summarize_run(load_rows(path), concurrency)


def plan_scenarios(n_judgments, empirical_jps=None, empirical_basis=None):
    """Three labelled runtime scenarios for n_judgments. empirical_jps may
    come from a results file (basis 'measured from <path>') or a CLI rate
    (basis 'given on the command line'); None means 'assumed'."""
    if empirical_jps is None:
        empirical, basis = ASSUMED_EMPIRICAL_FALLBACK_JPS, f"ASSUMED fallback {ASSUMED_EMPIRICAL_FALLBACK_JPS} judgments/s (no measurement supplied)"
        optimistic, opt_basis = ASSUMED_OPTIMISTIC_JPS, f"ASSUMED optimistic {ASSUMED_OPTIMISTIC_JPS} judgments/s"
    else:
        empirical, basis = empirical_jps, empirical_basis or "given"
        optimistic, opt_basis = empirical_jps * OPTIMISTIC_FACTOR, f"empirical x {OPTIMISTIC_FACTOR}"
    conservative = min(empirical, OBSERVED_V2_RECOVERY_JPS)
    scenarios = {
        "optimistic": {"judgments_per_second": optimistic, "basis": opt_basis},
        "empirical": {"judgments_per_second": empirical, "basis": basis},
        "conservative": {"judgments_per_second": conservative,
                         "basis": f"min(empirical, OBSERVED v2 recovery {OBSERVED_V2_RECOVERY_JPS} judgments/s at concurrency {OBSERVED_V2_RECOVERY_CONCURRENCY})"},
    }
    for s in scenarios.values():
        s["hours"] = n_judgments / s["judgments_per_second"] / 3600 if s["judgments_per_second"] > 0 else None
    return {"n_judgments": n_judgments, "scenarios": scenarios, "note": "Runtime scenarios are planning ranges, not predictions."}


def resolve_throughput(throughput_jps=None, throughput_from_results=None, concurrency=None):
    """(empirical_jps or None, basis string, summary or None)."""
    if throughput_from_results:
        summary = summarize_results_file(throughput_from_results, concurrency)
        jps = summary["effective_judgments_per_second"]
        if jps:
            return jps, f"measured from {throughput_from_results} ({summary['completed_judgments']} judgments over {summary['active_seconds']:.0f}s active)", summary
        return None, f"{throughput_from_results} has too little data to measure throughput", summary
    if throughput_jps:
        return throughput_jps, "given on the command line (--throughput-jps)", None
    return None, "no measurement supplied", None


def format_scenarios(plan, label=""):
    lines = [f"runtime scenarios{(' for ' + label) if label else ''}: {plan['n_judgments']:,} judgments"]
    for name in ("optimistic", "empirical", "conservative"):
        s = plan["scenarios"][name]
        lines.append(f"  {name:12s} {s['judgments_per_second']:.2f} j/s -> {s['hours']:.1f} h   [{s['basis']}]")
    lines.append(f"  {plan['note']}")
    return "\n".join(lines)


def suggested_time_budget_minutes(n_judgments, jps=OBSERVED_V2_RECOVERY_JPS, safety_margin=0.8, job_cap_minutes=300):
    """Minutes to allow for n_judgments at `jps` with a safety margin,
    capped by the Actions job budget (resume covers the rest)."""
    needed = n_judgments / jps / 60 / safety_margin
    return min(job_cap_minutes, needed)
