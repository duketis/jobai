"""Tests for the per-host politeness rate limiter + its fetcher wrapper.

Deterministic: an injected monotonic clock + fake sleep means no
wall-clock waits and exact assertions on the spacing the limiter
enforces. The point of the limiter is that request *rate* to a host
stays human no matter how many slugs/pages pile on — that's what
trips LinkedIn's fingerprint throttle, not total volume.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from jobai.fetcher.base import Response
from jobai.fetcher.ratelimit import HostRateLimiter, RateLimitedFetcher


class _Clock:
    """Mutable fake monotonic clock advanced only by the fake sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0
        self.now += seconds


def _limiter(clock: _Clock, *, min_interval: float = 2.0, jitter: float = 0.0) -> HostRateLimiter:
    return HostRateLimiter(
        min_interval=min_interval,
        jitter=jitter,
        clock=clock.time,
        sleep=clock.sleep,
        rand=lambda: 0.0,
    )


async def test_first_acquire_for_host_does_not_wait() -> None:
    clock = _Clock()
    limiter = _limiter(clock)
    await limiter.acquire("linkedin.com")
    assert clock.now == 0.0


async def test_second_acquire_same_host_waits_min_interval() -> None:
    clock = _Clock()
    limiter = _limiter(clock, min_interval=2.0)
    await limiter.acquire("linkedin.com")
    await limiter.acquire("linkedin.com")
    assert clock.now == pytest.approx(2.0)


async def test_distinct_hosts_are_independent() -> None:
    clock = _Clock()
    limiter = _limiter(clock, min_interval=5.0)
    await limiter.acquire("linkedin.com")
    await limiter.acquire("seek.com.au")  # different host → no wait
    assert clock.now == 0.0


async def test_elapsed_real_gap_is_not_double_charged() -> None:
    """If enough time already passed between calls, the next acquire
    doesn't sleep — we only ever wait for the *remaining* interval."""
    clock = _Clock()
    limiter = _limiter(clock, min_interval=2.0)
    await limiter.acquire("x")
    clock.now += 5.0  # caller spent 5s doing other work
    await limiter.acquire("x")
    assert clock.now == pytest.approx(5.0)  # no extra sleep


async def test_jitter_extends_the_interval() -> None:
    clock = _Clock()
    limiter = HostRateLimiter(
        min_interval=2.0,
        jitter=0.5,
        clock=clock.time,
        sleep=clock.sleep,
        rand=lambda: 1.0,  # max jitter
    )
    await limiter.acquire("h")
    await limiter.acquire("h")
    assert clock.now == pytest.approx(3.0)  # 2.0 * (1 + 0.5*1.0)


async def test_concurrent_acquires_same_host_serialize() -> None:
    """Two coroutines hitting the same host must not both see a free
    slot — they drain one min_interval apart, not simultaneously."""
    clock = _Clock()
    limiter = _limiter(clock, min_interval=2.0)
    await asyncio.gather(limiter.acquire("h"), limiter.acquire("h"), limiter.acquire("h"))
    # 1st: 0, 2nd: +2, 3rd: +2 → total 4.0 of enforced spacing.
    assert clock.now == pytest.approx(4.0)


def test_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="min_interval"):
        HostRateLimiter(min_interval=0.0)


# ---------------------------------------------------------------------------
# RateLimitedFetcher wrapper
# ---------------------------------------------------------------------------


class _SpyFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - matches Fetcher Protocol
        wait_for_selector: str | None = None,
        wait_until: str = "networkidle",
    ) -> Response:
        self.calls.append(url)
        return Response(
            url=url,
            status_code=200,
            headers={},
            body=b"ok",
            fetched_at=datetime.now(tz=UTC),
        )

    async def aclose(self) -> None:
        self.closed = True


async def test_wrapper_gates_each_fetch_on_the_host() -> None:
    clock = _Clock()
    inner = _SpyFetcher()
    fetcher = RateLimitedFetcher(inner, limiter=_limiter(clock, min_interval=3.0))

    await fetcher.fetch("https://www.linkedin.com/a")
    await fetcher.fetch("https://www.linkedin.com/b")

    assert inner.calls == ["https://www.linkedin.com/a", "https://www.linkedin.com/b"]
    assert clock.now == pytest.approx(3.0)  # second call paced behind the first


async def test_wrapper_passes_kwargs_through_and_returns_response() -> None:
    clock = _Clock()
    inner = _SpyFetcher()
    fetcher = RateLimitedFetcher(inner, limiter=_limiter(clock))
    resp = await fetcher.fetch("https://x.test/y", wait_until="domcontentloaded")
    assert resp.status_code == 200
    assert inner.calls == ["https://x.test/y"]


async def test_wrapper_handles_urls_without_a_host() -> None:
    clock = _Clock()
    fetcher = RateLimitedFetcher(_SpyFetcher(), limiter=_limiter(clock))
    # No hostname → bucketed under "" but must not crash.
    resp = await fetcher.fetch("not-a-url")
    assert resp.status_code == 200


async def test_wrapper_aclose_and_context_manager_delegate() -> None:
    inner = _SpyFetcher()
    clock = _Clock()
    async with RateLimitedFetcher(inner, limiter=_limiter(clock)) as fetcher:
        await fetcher.fetch("https://h.test/")
    assert inner.closed is True
