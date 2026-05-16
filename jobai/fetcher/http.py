"""Tier-1 HTTP fetcher backed by :mod:`httpx`.

Suitable for any source whose listing data is reachable over plain HTTP
without JavaScript execution: ATS aggregators (Greenhouse, Lever,
Ashby, Workable, SmartRecruiters), public job-board feeds, the HN
Algolia endpoint, etc.

Uses HTTP/2 transport for connection multiplexing, a configurable
User-Agent, and automatic redirect following. Sources should construct
one fetcher per scrape cycle (the underlying connection pool is reused
for the duration) and close it in a ``finally`` or ``async with`` block.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import httpx

from jobai import __version__
from jobai.fetcher.base import Response, WaitUntil

#: Default UA — versioned via ``jobai.__version__`` so a single bump
#: propagates here, the browser fetcher, and the stealth fetcher.
_DEFAULT_USER_AGENT = f"jobai/{__version__} (+https://github.com/duketis/jobai)"


class HttpFetcher:
    """Tier-1 fetcher: plain HTTP with HTTP/2 + connection pooling."""

    def __init__(
        self,
        *,
        user_agent: str = _DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        follow_redirects: bool = True,
    ) -> None:
        self._client = httpx.AsyncClient(
            http2=True,
            follow_redirects=follow_redirects,
            timeout=timeout,
            headers={"User-Agent": user_agent},
        )

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
        wait_for_selector: str | None = None,
        wait_until: WaitUntil = "networkidle",
    ) -> Response:
        # ``wait_for_selector`` / ``wait_until`` are part of the Fetcher
        # Protocol so browser-tier sources can request rendering, but
        # plain HTTP has nothing to wait for or navigate. Accept and
        # ignore both.
        del wait_for_selector, wait_until
        kwargs: dict[str, Any] = {}
        if headers is not None:
            kwargs["headers"] = dict(headers)
        if json is not None:
            kwargs["json"] = json
        if data is not None:
            kwargs["data"] = dict(data)
        if timeout is not None:
            kwargs["timeout"] = timeout

        httpx_response = await self._client.request(method, url, **kwargs)

        return Response(
            url=str(httpx_response.url),
            status_code=httpx_response.status_code,
            headers=dict(httpx_response.headers),
            body=httpx_response.content,
            fetched_at=datetime.now(tz=UTC),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
