"""`sources/_http.py`: the pacing and backoff both metered clients share.

The module exists because `pubmed.py` and `ctgov.py` had drifting copies of the same
three pieces. These tests pin the behaviour that drift would break: a `Retry-After` that
outranks the exponential ladder, a ceiling nothing may exceed, and jitter that is actually
jitter rather than a constant.
"""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from deep_life_sci.sources._http import (
    RETRY_STATUSES,
    Throttle,
    backoff_delay,
    chunks,
    retry_after,
)


def _response(**headers: str) -> httpx.Response:
    return httpx.Response(429, headers=headers)


class TestRetryAfter:
    def test_reads_delta_seconds(self):
        assert retry_after(_response(**{"Retry-After": "12"})) == 12.0

    def test_tolerates_surrounding_whitespace(self):
        assert retry_after(_response(**{"Retry-After": "  4.5 "})) == 4.5

    def test_absent_header_is_none(self):
        assert retry_after(_response()) is None

    def test_http_date_form_is_none(self):
        """The legal date form is deliberately not parsed — see the docstring."""
        assert retry_after(_response(**{"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None

    def test_negative_clamps_to_zero(self):
        assert retry_after(_response(**{"Retry-After": "-5"})) == 0.0


class TestBackoffDelay:
    def test_retry_after_wins_over_the_ladder(self):
        resp = _response(**{"Retry-After": "7"})
        assert backoff_delay(1, base=1.0, maximum=30.0, resp=resp) == 7.0

    def test_retry_after_is_still_capped_by_maximum(self):
        """A server asking for an hour must not park a fan-out for an hour."""
        resp = _response(**{"Retry-After": "3600"})
        assert backoff_delay(1, base=1.0, maximum=30.0, resp=resp) == 30.0

    def test_response_without_the_header_falls_through_to_the_ladder(self):
        delay = backoff_delay(1, base=4.0, maximum=30.0, resp=_response())
        assert 2.0 <= delay <= 4.0

    @pytest.mark.parametrize("attempt", [1, 2, 3, 4])
    def test_stays_between_half_the_ceiling_and_the_ceiling(self, attempt: int):
        ceiling = min(1.0 * 2 ** (attempt - 1), 30.0)
        for _ in range(50):
            delay = backoff_delay(attempt, base=1.0, maximum=30.0)
            assert ceiling / 2 <= delay <= ceiling

    def test_never_exceeds_the_maximum_however_high_the_attempt(self):
        for _ in range(50):
            assert backoff_delay(20, base=1.0, maximum=30.0) <= 30.0

    def test_is_jittered_rather_than_constant(self):
        """Concurrent callers must not back off in lockstep and collide again."""
        delays = {backoff_delay(3, base=1.0, maximum=30.0) for _ in range(50)}
        assert len(delays) > 1


class TestChunks:
    def test_splits_into_request_sized_pieces(self):
        assert list(chunks(list("abcde"), 2)) == [["a", "b"], ["c", "d"], ["e"]]

    def test_exact_multiple_yields_no_empty_tail(self):
        assert list(chunks(list("abcd"), 2)) == [["a", "b"], ["c", "d"]]

    def test_empty_input_yields_nothing(self):
        assert list(chunks([], 10)) == []

    def test_size_above_the_list_yields_one_piece(self):
        assert list(chunks(["a"], 200)) == [["a"]]


class TestThrottle:
    @pytest.fixture(autouse=True)
    def clock(self, monkeypatch):
        from deep_life_sci.sources import _http

        now = [100.0]
        real_sleep = asyncio.sleep

        async def sleep(delay):
            # Yield while the throttle holds its lock: concurrent waiters must queue.
            await real_sleep(0)
            now[0] += delay

        self.sleep = AsyncMock(side_effect=sleep)
        self.now = lambda: now[0]
        monkeypatch.setattr(_http, "time", SimpleNamespace(monotonic=self.now))
        monkeypatch.setattr(_http, "asyncio", SimpleNamespace(Lock=asyncio.Lock, sleep=self.sleep))

    async def test_serialises_concurrent_callers_to_the_interval(self):
        """The pacing is process-wide, so a gather must not let calls overlap."""
        throttle = Throttle(lambda: 0.02)
        stamps: list[float] = []

        async def call() -> None:
            await throttle.wait()
            stamps.append(self.now())

        await asyncio.gather(*(call() for _ in range(4)))

        gaps = [b - a for a, b in itertools.pairwise(stamps)]
        assert all(gap >= 0.015 for gap in gaps), gaps

    async def test_interval_is_read_per_call_not_captured(self):
        """NCBI's interval depends on an env var read after import — see the docstring."""
        interval = [0.0]
        throttle = Throttle(lambda: interval[0])

        await throttle.wait()
        interval[0] = 0.02
        started = self.now()
        await throttle.wait()

        assert self.now() - started >= 0.015

    async def test_does_not_wait_when_the_interval_has_already_passed(self):
        throttle = Throttle(lambda: 0.0)
        started = self.now()
        for _ in range(5):
            await throttle.wait()
        assert self.now() == started
        self.sleep.assert_not_awaited()


def test_retry_statuses_are_the_rate_limiter_and_the_gateway():
    """Anything else is the caller's problem and must surface rather than be retried."""
    assert frozenset({429, 500, 502, 503, 504}) == RETRY_STATUSES
    assert 400 not in RETRY_STATUSES
    assert 404 not in RETRY_STATUSES
    assert 414 not in RETRY_STATUSES
