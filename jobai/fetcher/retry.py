"""Retrying fetcher: exponential backoff + ``Retry-After`` honour.

Wraps any :class:`Fetcher` so callers get automatic recovery from
transient failures without coding a loop into every source. Retries
happen on:

* network-level exceptions (``httpx.ConnectError``,
  ``httpx.TimeoutException``, etc.) — anything raised before a status
  code lands;
* HTTP 429 with the ``Retry-After`` header honoured (seconds or
  HTTP-date);
* HTTP 5xx (transient server errors).

Genuine client errors (4xx other than 429) and 2xx pass through
unchanged. The retry budget is bounded by ``max_attempts`` so a
permanently failing source can't hang the whole scrape cycle.

Why a wrapper rather than a tenacity decorator on each source: the
fetcher is the *only* place we touch the network, so wrapping it is
a single point of correctness. Per-source retry logic would drift
out of sync the moment a new source skipped it.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx

from jobai.fetcher.base import Fetcher, Response, WaitUntil

_log = logging.getLogger(__name__)

#: Network-level exception classes we treat as transient. Subclassing
#: ``httpx.TransportError`` covers connect/read/write/protocol issues
#: in a single check; ``TimeoutException`` is its own hierarchy.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    httpx.TimeoutException,
)

#: Status codes that warrant a retry. 429 is rate-limited; 502/503/504
#: are upstream proxy / gateway transients; 500 covers the long tail
#: of server-side flakes.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class RetryingFetcher:
    """Fetcher decorator with exponential backoff."""

    def __init__(
        self,
        inner: Fetcher,
        *,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        jitter: float = 0.25,
        sleep: Any = asyncio.sleep,
    ) -> None:
        """Wrap ``inner`` with retry semantics.

        Args:
            inner: any :class:`Fetcher` implementation.
            max_attempts: total attempts (initial + retries). Must be
                at least 1.
            backoff_base: seconds for the first retry. Doubles each
                further attempt up to ``backoff_max``.
            backoff_max: ceiling for the per-retry sleep.
            jitter: fraction of the computed sleep added as random
                noise (``sleep * (1 + random.uniform(0, jitter))``).
                Spreads concurrent retrants apart so they don't pile
                up in a thundering herd.
            sleep: async sleep function. Tests inject a fake to avoid
                wall-clock waits.
        """
        if max_attempts < 1:
            msg = f"max_attempts must be >= 1, got {max_attempts}"
            raise ValueError(msg)
        self._inner = inner
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._jitter = jitter
        self._sleep = sleep

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109  - delegates to inner fetcher
        wait_for_selector: str | None = None,
        wait_until: WaitUntil = "networkidle",
    ) -> Response:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._inner.fetch(
                    url,
                    method=method,
                    headers=headers,
                    json=json,
                    data=data,
                    timeout=timeout,
                    wait_for_selector=wait_for_selector,
                    wait_until=wait_until,
                )
            except _TRANSIENT_EXCEPTIONS as exc:
                if attempt >= self._max_attempts:
                    raise
                delay = self._compute_delay(attempt, retry_after=None)
                _log.info(
                    "fetch_transient_failure",
                    extra={
                        "url": url,
                        "attempt": attempt,
                        "max_attempts": self._max_attempts,
                        "delay": delay,
                        "error_class": type(exc).__name__,
                    },
                )
                await self._sleep(delay)
                continue

            if response.status_code in _RETRYABLE_STATUSES and attempt < self._max_attempts:
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                delay = self._compute_delay(attempt, retry_after=retry_after)
                _log.info(
                    "fetch_retryable_status",
                    extra={
                        "url": url,
                        "attempt": attempt,
                        "max_attempts": self._max_attempts,
                        "status": response.status_code,
                        "delay": delay,
                    },
                )
                await self._sleep(delay)
                continue

            return response

        # The loop only exits this way if every attempt failed with a
        # retryable status (we'd have raised on transient exceptions).
        # ``last_exc`` is None here because exceptions are re-raised
        # immediately on the final attempt; we just return the last
        # response the inner fetcher produced.
        # Defensive: keep a sane fallback so type-checkers stay happy.
        msg = "RetryingFetcher exited the retry loop unexpectedly"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    # Forwards to BrowserFetcher.run_in_page, which drives real Playwright
    # and is itself excluded. Integration-only.
    async def run_in_page(  # pragma: no cover
        self, *args: Any, **kwargs: Any
    ) -> Response:
        """Forward to the inner fetcher's ``run_in_page`` (if browser-tier).

        Browser workflows go through Playwright, which already has
        its own waits and exception handling — re-running the script
        on transient errors would double up. So we forward without
        retry semantics.
        """
        method = self._inner.run_in_page  # type: ignore[attr-defined]
        return await method(*args, **kwargs)  # type: ignore[no-any-return]

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

    def _compute_delay(self, attempt: int, *, retry_after: float | None) -> float:
        """Return the sleep before the next attempt (seconds).

        ``Retry-After`` always wins when present — the server is
        explicitly telling us how long to wait. Otherwise, exponential
        backoff with jitter.
        """
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self._backoff_max)
        base = self._backoff_base * (2 ** (attempt - 1))
        bounded = min(base, self._backoff_max)
        jitter_amount = bounded * random.uniform(0, self._jitter)  # noqa: S311  - non-crypto
        return float(bounded + jitter_amount)


def _parse_retry_after(header_value: str | None) -> float | None:
    """Parse a ``Retry-After`` header in seconds or HTTP-date form.

    Returns ``None`` for missing or unparseable values so the caller
    falls back to exponential backoff.
    """
    if not header_value:
        return None
    stripped = header_value.strip()
    if stripped.isdigit():
        return float(stripped)
    try:
        target = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    now = datetime.now(tz=UTC)
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    delta = (target - now).total_seconds()
    return max(delta, 0.0)
