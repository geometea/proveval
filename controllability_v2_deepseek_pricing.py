"""DeepSeek off-peak pricing guard.

DeepSeek's current peak pricing windows (Monday-Friday, UTC):
    01:00:00 <= t < 04:00:00   (peak)
    06:00:00 <= t < 10:00:00   (peak)
Everything else -- including all of Saturday and Sunday -- is off-peak.

This module is pure scheduling logic (is_deepseek_off_peak/
next_off_peak_boundary) plus one small coordination primitive
(DeepSeekPricingGate) used by run_controllability_v2.py to pause new
DeepSeek dispatch during a peak window without busy-looping and without
interrupting requests already in flight. It never makes a network call and
never touches any other provider.
"""

import threading
import time
from datetime import datetime, time as time_of_day, timedelta, timezone

# Half-open [start, end) peak windows, UTC, Monday-Friday only.
PEAK_WINDOWS_UTC = (
    (time_of_day(1, 0), time_of_day(4, 0)),
    (time_of_day(6, 0), time_of_day(10, 0)),
)

SAFETY_BUFFER_SECONDS = 60


def _require_utc(dt):
    """Timezone-aware input is required; normalises to UTC."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("dt_utc must be timezone-aware (got a naive datetime)")
    return dt.astimezone(timezone.utc)


def is_deepseek_off_peak(dt_utc: datetime) -> bool:
    """True if `dt_utc` (any timezone-aware datetime; normalised to UTC
    internally) falls outside DeepSeek's Monday-Friday peak windows.
    Saturday and Sunday are always off-peak."""
    dt_utc = _require_utc(dt_utc)
    if dt_utc.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    t = dt_utc.timetz().replace(tzinfo=None)
    for start, end in PEAK_WINDOWS_UTC:
        if start <= t < end:
            return False
    return True


def next_off_peak_boundary(dt_utc: datetime) -> datetime:
    """The next UTC instant at which off-peak begins. If `dt_utc` is
    already off-peak, returns it unchanged. Only meaningful within a peak
    window (which, by construction, is always Monday-Friday and always
    ends later the same UTC day -- no weekend-skipping is ever needed)."""
    dt_utc = _require_utc(dt_utc)
    if is_deepseek_off_peak(dt_utc):
        return dt_utc
    t = dt_utc.timetz().replace(tzinfo=None)
    for start, end in PEAK_WINDOWS_UTC:
        if start <= t < end:
            return dt_utc.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    raise AssertionError(f"{dt_utc!r} was reported peak but matches no configured peak window")


def format_utc_z(dt_utc: datetime) -> str:
    """ISO-8601 UTC, seconds precision, "Z" suffix -- e.g. 2026-09-18T06:00:03Z."""
    return _require_utc(dt_utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DeepSeekPricingGate:
    """Shared across every worker in the concurrent pool. Before each real
    DeepSeek dispatch, wait_until_dispatch_allowed() blocks (without
    busy-looping -- one computed-duration sleep per check) until the
    current time is off-peak. Only the first thread to observe a
    peak-window transition prints the pause message; only the first thread
    to observe the return to off-peak prints the resume message -- every
    other thread blocked on the same transition prints nothing, so
    concurrency never spams the log.

    allow_peak=True makes every call a no-op (the --allow-peak-pricing
    override); this gate is only ever constructed/consulted for the
    DeepSeek provider -- other providers never see it.
    """

    def __init__(self, allow_peak=False, now_fn=None, sleep_fn=None):
        self.allow_peak = allow_peak
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._sleep_fn = sleep_fn or time.sleep
        self._lock = threading.Lock()
        self._paused_until = None  # None => not currently paused

    def wait_until_dispatch_allowed(self):
        if self.allow_peak:
            return
        while True:
            now = self._now_fn()
            if is_deepseek_off_peak(now):
                self._maybe_announce_resume(now)
                return
            wake_at = self._maybe_announce_pause(now)
            sleep_seconds = max(0.0, (wake_at - self._now_fn()).total_seconds())
            self._sleep_fn(sleep_seconds)
            # loop back and re-check -- defensive against clock drift /
            # scheduling jitter, never assumes the single sleep was exact

    def _maybe_announce_pause(self, now):
        with self._lock:
            if self._paused_until is None:
                boundary = next_off_peak_boundary(now)
                self._paused_until = boundary + timedelta(seconds=SAFETY_BUFFER_SECONDS)
                print(f"DeepSeek peak pricing active at {format_utc_z(now)}.")
                print(f"Pausing new API dispatch until {format_utc_z(self._paused_until)}.")
            return self._paused_until

    def _maybe_announce_resume(self, now):
        with self._lock:
            if self._paused_until is not None:
                self._paused_until = None
                print(f"DeepSeek dispatch resumed at {format_utc_z(now)}.")
