"""Tier-escalating fetcher: HTTP first, browser on block signals.

Many sources work over plain HTTP until they don't. A single 403 or a
Cloudflare interstitial means the rest of the cycle should run through
a real browser; staying on HTTP just burns the request budget against
a wall.

:class:`EscalatingFetcher` wraps a primary (typically tier-1
:class:`HttpFetcher`) and a fallback factory (typically tier-2
:class:`BrowserFetcher`). On a fetch, it tries the primary; if the
response looks blocked (403 / 429 / Cloudflare interstitial), it
constructs the fallback once and re-issues the request through it.
After a single escalation the fetcher stays on the fallback for the
rest of its life — re-probing the primary every fetch would waste
trips and leak the same signal again.

Why a factory rather than a pre-built fallback: building a browser is
expensive (Playwright start + Chromium launch). The factory pattern
lets callers compose the fallback eagerly only for sources known to
need it, while keeping the cheap HTTP-only path free of any browser
overhead when no escalation happens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Self

from jobai.fetcher.base import Fetcher, Response

#: HTTP status codes that signal we should escalate. 401/404 are NOT
#: in here — those mean "the source genuinely doesn't have this", not
#: "we're being blocked".
_BLOCK_STATUS_CODES = frozenset({403, 429})

#: Cloudflare / Akamai / similar interstitial fingerprints. Lowercased
#: substrings searched in the first slice of the body.
_BLOCK_BODY_SIGNALS = (
    "just a moment",
    "checking your browser",
    "cloudflare ray id",
    "cf-chl-bypass",
    "attention required",
)

#: Search at most this many bytes of the body for block signals.
#: Cloudflare's challenge HTML is small; longer scans waste cycles on
#: legitimate large pages.
_BLOCK_BODY_SCAN_BYTES = 8_192


class EscalatingFetcher:
    """Tier-2 escalator: primary fetcher, browser fallback on blocks."""

    def __init__(
        self,
        *,
        primary: Fetcher,
        fallback_factory: Callable[[], Fetcher],
    ) -> None:
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: Fetcher | None = None
        self._escalated = False

    @property
    def escalated(self) -> bool:
        """True once a block has triggered escalation."""
        return self._escalated

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109  - delegates to wrapped fetcher
    ) -> Response:
        if self._escalated:
            return await self._call_fallback(
                url,
                method=method,
                headers=headers,
                json=json,
                timeout=timeout,
            )

        response = await self._primary.fetch(
            url,
            method=method,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        if not _looks_blocked(response):
            return response

        self._escalated = True
        return await self._call_fallback(
            url,
            method=method,
            headers=headers,
            json=json,
            timeout=timeout,
        )

    async def _call_fallback(
        self,
        url: str,
        *,
        method: str,
        headers: Mapping[str, str] | None,
        json: Any,
        timeout: float | None,  # noqa: ASYNC109  - delegates to wrapped fetcher
    ) -> Response:
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return await self._fallback.fetch(
            url,
            method=method,
            headers=headers,
            json=json,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._primary.aclose()
        if self._fallback is not None:
            await self._fallback.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _looks_blocked(response: Response) -> bool:
    """Heuristically decide if a response is a block / challenge."""
    if response.status_code in _BLOCK_STATUS_CODES:
        return True
    if response.status_code != 200:
        return False
    body_slice = (
        response.body[:_BLOCK_BODY_SCAN_BYTES]
        .decode(
            "utf-8",
            errors="replace",
        )
        .lower()
    )
    return any(signal in body_slice for signal in _BLOCK_BODY_SIGNALS)
