"""Tier-2 browser fetcher backed by Playwright (Chromium).

For sources whose listings or job-detail pages are JavaScript-rendered
and therefore invisible to a plain HTTP fetch — Workable's careers
page, Seek, most modern enterprise SPA listings.

The fetcher owns a long-lived browser instance and creates a fresh
context per ``fetch`` call. Per-fetch contexts isolate cookies and
storage between sources without paying the multi-second
:meth:`Playwright.start` + :meth:`Browser.launch` cost on every
request.

Browser plumbing lives in :class:`PlaywrightDriver`. The
:class:`BrowserFetcher` orchestrates argument validation,
:class:`Response` translation, and lifecycle. Splitting them keeps
unit tests small — tests inject a fake driver and assert the
fetcher's contract without ever launching Chromium.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Protocol, Self

from playwright.async_api import Browser, Playwright, async_playwright

from jobai.fetcher.base import Response

#: A Chrome-on-macOS UA suffixed with ``jobai/<version>`` so we keep a
#: realistic fingerprint without misrepresenting ourselves.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 jobai/0.0.1"
)


class _Driver(Protocol):
    """The minimal surface :class:`BrowserFetcher` needs from a browser.

    Defined as a Protocol so tests can inject a fake without subclassing
    or touching Playwright at all.
    """

    async def fetch_page(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        timeout_ms: float,
    ) -> Response: ...

    async def close(self) -> None: ...


class PlaywrightDriver:
    """Default driver: Chromium launched lazily, one context per fetch.

    Lazy launch keeps the fetcher cheap to construct (no Playwright
    process started until the first fetch). A single :class:`Browser`
    is reused across requests; each call gets a fresh
    :class:`BrowserContext` so cookies / localStorage do not leak
    between sources.
    """

    def __init__(
        self,
        *,
        user_agent: str = _DEFAULT_USER_AGENT,
        headless: bool = True,
        playwright_factory: Any = async_playwright,
    ) -> None:
        self._user_agent = user_agent
        self._headless = headless
        self._factory = playwright_factory
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_browser(self) -> Browser:
        async with self._init_lock:
            if self._browser is None:
                self._playwright = await self._factory().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self._headless,
                )
            return self._browser

    async def fetch_page(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        timeout_ms: float,
    ) -> Response:
        browser = await self._ensure_browser()
        context = await browser.new_context(user_agent=self._user_agent)
        try:
            page = await context.new_page()
            if headers:
                await page.set_extra_http_headers(dict(headers))
            response = await page.goto(url, timeout=timeout_ms)
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            html = await page.content()

            if response is None:
                # Same-document navigations and a few edge cases yield a
                # null response in Playwright. Surface 0 so the caller
                # can treat it as a soft failure rather than crashing.
                return Response(
                    url=url,
                    status_code=0,
                    headers={},
                    body=html.encode("utf-8"),
                    fetched_at=datetime.now(tz=UTC),
                )

            return Response(
                url=response.url,
                status_code=response.status,
                headers=dict(await response.all_headers()),
                body=html.encode("utf-8"),
                fetched_at=datetime.now(tz=UTC),
            )
        finally:
            await context.close()

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


class BrowserFetcher:
    """Tier-2 fetcher implementing the :class:`Fetcher` Protocol."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        driver: _Driver | None = None,
        user_agent: str = _DEFAULT_USER_AGENT,
        headless: bool = True,
    ) -> None:
        self._timeout = timeout
        self._driver: _Driver = driver or PlaywrightDriver(
            user_agent=user_agent,
            headless=headless,
        )

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109  - delegates to playwright
    ) -> Response:
        if method != "GET":
            msg = (
                f"BrowserFetcher only supports GET; got {method!r}. "
                "Use HttpFetcher for non-GET endpoints."
            )
            raise ValueError(msg)
        if json is not None:
            msg = "BrowserFetcher does not support `json` payloads."
            raise ValueError(msg)

        timeout_ms = (timeout if timeout is not None else self._timeout) * 1000
        return await self._driver.fetch_page(
            url,
            headers=headers,
            timeout_ms=timeout_ms,
        )

    async def aclose(self) -> None:
        await self._driver.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
