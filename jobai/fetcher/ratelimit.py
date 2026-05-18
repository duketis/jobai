"""Per-host politeness rate limiter + a Fetcher decorator for it.

LinkedIn (and the other fingerprinting boards we hit on the stealth
tier) throttle on request *rate*, not total volume. Before this there
was no pacing anywhere: a slug walk fired ~100 page requests
back-to-back and every slug triggered on the hour, so adding slugs
meant a burstier pattern, not just more data.

:class:`HostRateLimiter` enforces a minimum gap between successive
requests *to the same registrable host*, with jitter so the cadence
isn't a robotic metronome. :class:`RateLimitedFetcher` wraps any
:class:`Fetcher` and gates every ``fetch`` through the limiter.

A single process-wide limiter (:func:`get_global_limiter`) is shared
by every tier-3 fetcher built via :func:`jobai.fetcher.dispatch.build_fetcher`,
so the rate cap holds across *all* concurrent slug scrapes — not
per-fetcher. That's what makes "add as many slugs as you like, it
just drains slowly and safely" true: slug count stops mattering;
only the global per-host rate does.

The clock / sleep / rand are injectable so the spacing is asserted
deterministically without wall-clock waits.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlparse

from jobai.fetcher.base import Fetcher, Response, WaitUntil

#: Default minimum seconds between two requests to the same host on
#: the stealth tier. ~2.5s + jitter ≈ a human clicking through a
#: results list, two orders of magnitude under LinkedIn's guest
#: fingerprint throttle. Slower than the old unbounded burst by
#: design — completeness over speed (the user's explicit call).
DEFAULT_MIN_INTERVAL_SECONDS = 2.5

#: Fraction of ``min_interval`` added as uniform random noise so the
#: spacing isn't a detectable fixed period.
DEFAULT_JITTER = 0.4


class HostRateLimiter:
    """Enforce a min interval between requests to the same host."""

    def __init__(
        self,
        *,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        jitter: float = DEFAULT_JITTER,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        """Wrap a per-host token gate.

        Args:
            min_interval: seconds every same-host request waits behind
                the previous one. Must be > 0.
            jitter: fraction of ``min_interval`` added as uniform
                noise (``interval * (1 + jitter*rand())``).
            clock: monotonic time source (injected in tests).
            sleep: async sleep (injected in tests).
            rand: 0..1 source for the jitter (injected in tests).
        """
        if min_interval <= 0:
            msg = f"min_interval must be > 0, got {min_interval}"
            raise ValueError(msg)
        self._min_interval = min_interval
        self._jitter = jitter
        self._clock = clock
        self._sleep = sleep
        self._rand = rand
        self._next_allowed: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, host: str) -> asyncio.Lock:
        lock = self._locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[host] = lock
        return lock

    async def acquire(self, host: str) -> None:
        """Block until a request to ``host`` is allowed, then reserve
        the next slot. Concurrent callers for the same host serialize
        through a per-host lock so two coroutines can't both slip
        through one free slot."""
        async with self._lock_for(host):
            now = self._clock()
            allowed = self._next_allowed.get(host, now)
            wait = allowed - now
            if wait > 0:
                await self._sleep(wait)
                now = allowed
            base = max(now, allowed)
            self._next_allowed[host] = base + self._min_interval * (
                1.0 + self._jitter * self._rand()
            )


class RateLimitedFetcher:
    """Fetcher decorator that paces every request via a shared limiter."""

    def __init__(self, inner: Fetcher, *, limiter: HostRateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - delegates to inner fetcher
        wait_for_selector: str | None = None,
        wait_until: WaitUntil = "networkidle",
    ) -> Response:
        await self._limiter.acquire(urlparse(url).hostname or "")
        return await self._inner.fetch(
            url,
            method=method,
            headers=headers,
            json=json,
            data=data,
            timeout=timeout,
            wait_for_selector=wait_for_selector,
            wait_until=wait_until,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


_GLOBAL_LIMITER: HostRateLimiter | None = None


def get_global_limiter() -> HostRateLimiter:
    """Return the process-wide limiter shared by every stealth fetcher.

    Lazily created so importing this module has no side effects and
    tests can use their own instances. Sharing one instance is the
    whole point: the rate cap is global per host, independent of how
    many slugs/fetchers run concurrently.
    """
    global _GLOBAL_LIMITER  # noqa: PLW0603 - module-singleton accessor
    if _GLOBAL_LIMITER is None:
        _GLOBAL_LIMITER = HostRateLimiter()
    return _GLOBAL_LIMITER


__all__ = [
    "DEFAULT_JITTER",
    "DEFAULT_MIN_INTERVAL_SECONDS",
    "HostRateLimiter",
    "RateLimitedFetcher",
    "get_global_limiter",
]
