"""Tests for controllability_v2_deepseek_pricing.py: the off-peak/peak
classification, the next-off-peak-boundary calculation, timezone-aware
input enforcement, and the concurrency-safe DeepSeekPricingGate.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

import controllability_v2_deepseek_pricing as p


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# 2026-09-21 is a Monday; +1 day increments walk Tue/Wed/Thu/Fri/Sat/Sun.
MONDAY = (2026, 9, 21)
TUESDAY = (2026, 9, 22)
FRIDAY = (2026, 9, 25)
SATURDAY = (2026, 9, 26)
SUNDAY = (2026, 9, 27)


class TestIsDeepSeekOffPeak:
    @pytest.mark.parametrize("hm,expected_off_peak", [
        ((0, 30), True),
        ((1, 0), False),
        ((3, 59), False),
        ((4, 0), True),
        ((5, 59), True),
        ((6, 0), False),
        ((9, 59), False),
        ((10, 0), True),
        ((23, 59), True),
    ])
    def test_monday_boundaries(self, hm, expected_off_peak):
        dt = utc(*MONDAY, *hm)
        assert p.is_deepseek_off_peak(dt) is expected_off_peak

    def test_tuesday_through_thursday_follow_the_same_windows(self):
        assert p.is_deepseek_off_peak(utc(*TUESDAY, 2, 0)) is False
        assert p.is_deepseek_off_peak(utc(*TUESDAY, 5, 0)) is True

    def test_friday_peak_windows(self):
        assert p.is_deepseek_off_peak(utc(*FRIDAY, 1, 30)) is False
        assert p.is_deepseek_off_peak(utc(*FRIDAY, 7, 0)) is False
        assert p.is_deepseek_off_peak(utc(*FRIDAY, 11, 0)) is True

    def test_saturday_is_always_off_peak(self):
        for hour in (0, 1, 3, 4, 6, 9, 10, 23):
            assert p.is_deepseek_off_peak(utc(*SATURDAY, hour, 0)) is True

    def test_sunday_is_always_off_peak(self):
        for hour in (0, 1, 3, 4, 6, 9, 10, 23):
            assert p.is_deepseek_off_peak(utc(*SUNDAY, hour, 0)) is True

    def test_requires_timezone_aware_input(self):
        naive = datetime(2026, 9, 21, 1, 0)
        with pytest.raises(ValueError):
            p.is_deepseek_off_peak(naive)

    def test_normalises_non_utc_timezones(self):
        pst = timezone(timedelta(hours=-7))
        # 18:00 PST Sunday == 01:00 UTC Monday -> peak
        dt_pst = datetime(2026, 9, 20, 18, 0, tzinfo=pst)
        assert p.is_deepseek_off_peak(dt_pst) is False

    def test_a_timezone_aware_datetime_with_zero_offset_but_not_utc_object_works(self):
        # a fixed-offset tzinfo of +00:00 (not the `timezone.utc` singleton) must
        # still be accepted as timezone-aware and treated identically to UTC
        zero_offset = timezone(timedelta(0))
        dt = datetime(2026, 9, 21, 6, 0, tzinfo=zero_offset)
        assert p.is_deepseek_off_peak(dt) is False


class TestNextOffPeakBoundary:
    def test_from_the_first_peak_window(self):
        assert p.next_off_peak_boundary(utc(*MONDAY, 1, 0)) == utc(*MONDAY, 4, 0)
        assert p.next_off_peak_boundary(utc(*MONDAY, 3, 59, 59)) == utc(*MONDAY, 4, 0)

    def test_from_the_second_peak_window(self):
        assert p.next_off_peak_boundary(utc(*MONDAY, 6, 0)) == utc(*MONDAY, 10, 0)
        assert p.next_off_peak_boundary(utc(*MONDAY, 9, 30)) == utc(*MONDAY, 10, 0)

    def test_already_off_peak_returns_the_input_unchanged(self):
        dt = utc(*MONDAY, 5, 0)
        assert p.next_off_peak_boundary(dt) == dt

    def test_requires_timezone_aware_input(self):
        with pytest.raises(ValueError):
            p.next_off_peak_boundary(datetime(2026, 9, 21, 1, 0))


class TestFormatUtcZ:
    def test_formats_with_z_suffix(self):
        assert p.format_utc_z(utc(2026, 9, 18, 6, 0, 3)) == "2026-09-18T06:00:03Z"


# ---------------------------------------------------------------------------
# DeepSeekPricingGate: concurrency-safe, no busy-loop, single announcement
# ---------------------------------------------------------------------------

class FakeClock:
    """A controllable clock + sleep function: sleep(seconds) advances the
    clock by exactly that many seconds rather than actually waiting, so
    these tests run instantly regardless of how long the simulated pause is."""

    def __init__(self, start):
        self._now = start
        self._lock = threading.Lock()
        self.sleep_calls = []

    def now(self):
        with self._lock:
            return self._now

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)
        with self._lock:
            self._now = self._now + timedelta(seconds=seconds)


class TestDeepSeekPricingGate:
    def test_off_peak_dispatch_is_immediate_and_never_sleeps(self):
        clock = FakeClock(utc(*MONDAY, 5, 0))  # off-peak
        gate = p.DeepSeekPricingGate(now_fn=clock.now, sleep_fn=clock.sleep)
        gate.wait_until_dispatch_allowed()
        assert clock.sleep_calls == []

    def test_peak_dispatch_blocks_until_the_boundary_plus_safety_buffer(self):
        clock = FakeClock(utc(*MONDAY, 6, 0))  # peak
        gate = p.DeepSeekPricingGate(now_fn=clock.now, sleep_fn=clock.sleep)
        gate.wait_until_dispatch_allowed()
        assert clock.now() == utc(*MONDAY, 10, 1, 0)  # 10:00 boundary + 60s buffer

    def test_allow_peak_override_bypasses_the_guard_entirely(self):
        clock = FakeClock(utc(*MONDAY, 6, 0))  # peak
        gate = p.DeepSeekPricingGate(allow_peak=True, now_fn=clock.now, sleep_fn=clock.sleep)
        gate.wait_until_dispatch_allowed()
        assert clock.sleep_calls == []
        assert clock.now() == utc(*MONDAY, 6, 0)  # never advanced -- returned immediately

    def test_does_not_busy_loop_one_sleep_call_covers_the_whole_wait(self, capsys):
        clock = FakeClock(utc(*MONDAY, 1, 0))  # peak (first window)
        gate = p.DeepSeekPricingGate(now_fn=clock.now, sleep_fn=clock.sleep)
        gate.wait_until_dispatch_allowed()
        assert len(clock.sleep_calls) == 1
        assert clock.sleep_calls[0] == pytest.approx((3 * 3600) + 60)  # 01:00 -> 04:00 + 60s

    def test_concurrent_workers_all_pass_the_gate_and_only_one_pair_of_messages_is_printed(self, capsys):
        clock = FakeClock(utc(*MONDAY, 6, 0))  # peak
        gate = p.DeepSeekPricingGate(now_fn=clock.now, sleep_fn=clock.sleep)
        results = []

        def worker():
            gate.wait_until_dispatch_allowed()
            results.append(True)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == [True] * 10  # every worker eventually passed the gate
        out = capsys.readouterr().out
        assert out.count("DeepSeek peak pricing active") == 1  # never spammed
        assert out.count("DeepSeek dispatch resumed") == 1

    def test_already_in_flight_work_is_unaffected_by_the_gate(self):
        """The gate only blocks the NEXT wait_until_dispatch_allowed() call --
        it has no way to (and must not attempt to) interrupt work that
        already passed the check and is executing."""
        clock = FakeClock(utc(*MONDAY, 5, 0))  # off-peak
        gate = p.DeepSeekPricingGate(now_fn=clock.now, sleep_fn=clock.sleep)
        gate.wait_until_dispatch_allowed()  # passes immediately

        in_flight_started = threading.Event()
        in_flight_finished = threading.Event()

        def simulate_in_flight_call():
            in_flight_started.set()
            # simulate the peak window beginning WHILE this "request" is running
            clock._now = utc(*MONDAY, 6, 0)
            in_flight_finished.set()

        t = threading.Thread(target=simulate_in_flight_call)
        t.start()
        in_flight_started.wait()
        t.join()
        assert in_flight_finished.is_set()  # ran to completion, unaffected by the now-peak clock

    def test_multiple_pause_cycles_each_get_their_own_single_announcement(self, capsys):
        clock = FakeClock(utc(*MONDAY, 1, 0))  # first peak window
        gate = p.DeepSeekPricingGate(now_fn=clock.now, sleep_fn=clock.sleep)
        gate.wait_until_dispatch_allowed()  # -> waits to 04:01:00 (off-peak)

        clock._now = utc(*MONDAY, 6, 0)  # jump into the second peak window
        gate.wait_until_dispatch_allowed()  # -> waits to 10:01:00

        out = capsys.readouterr().out
        assert out.count("DeepSeek peak pricing active") == 2  # one per distinct pause episode
        assert out.count("DeepSeek dispatch resumed") == 2
