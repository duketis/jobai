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
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Protocol, Self

from playwright.async_api import Browser, Page, Playwright, async_playwright

from jobai.fetcher.base import Response

#: Sources that need full Playwright control (form fill, click,
#: multi-step navigation) pass a callable to
#: :meth:`BrowserFetcher.run_in_page`. The callable receives a
#: :class:`Page` already navigated to ``url`` and is expected to
#: drive any extra interactions; on return the fetcher snapshots
#: ``page.content()`` and packages it as a :class:`Response`.
PageScript = Callable[["Page"], Awaitable[None]]

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
        wait_for_selector: str | None = None,
    ) -> Response: ...

    async def run_in_page(
        self,
        url: str,
        *,
        timeout_ms: float,
        page_script: PageScript,
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
        wait_for_selector: str | None = None,
    ) -> Response:
        browser = await self._ensure_browser()
        context = await browser.new_context(user_agent=self._user_agent)
        try:
            page = await context.new_page()
            if headers:
                await page.set_extra_http_headers(dict(headers))
            response = await page.goto(url, timeout=timeout_ms)
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            if wait_for_selector is not None:
                # Wait for the SPA to populate the requested selector
                # (Next.js / Salesforce Lightning / React lazy-load
                # results after first paint). Soft-fail: a timeout
                # here yields the partially-rendered DOM rather than
                # raising — better than zero data on the first cycle.
                with contextlib.suppress(Exception):
                    await page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
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

    async def run_in_page(
        self,
        url: str,
        *,
        timeout_ms: float,
        page_script: PageScript,
    ) -> Response:
        """Navigate to ``url`` and hand the Page to ``page_script``.

        Used by sources that need full Playwright control — form
        fills, click chains, multi-step navigation — that the
        single-shot :meth:`fetch_page` can't express. The script
        runs against a clean per-call context (same isolation as
        :meth:`fetch_page`); the fetcher snapshots ``page.content()``
        when the script returns.
        """
        browser = await self._ensure_browser()
        context = await browser.new_context(user_agent=self._user_agent)
        try:
            page = await context.new_page()
            response = await page.goto(url, timeout=timeout_ms)
            await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            await page_script(page)
            html = await page.content()
            status = response.status if response is not None else 0
            response_url = response.url if response is not None else url
            return Response(
                url=response_url,
                status_code=status,
                headers={},
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
        wait_for_selector: str | None = None,
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
            wait_for_selector=wait_for_selector,
        )

    async def run_in_page(
        self,
        url: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        page_script: PageScript,
    ) -> Response:
        """Escape hatch: run a Playwright Page-aware script.

        Sources that need form fills or multi-step navigation drive
        the page directly via this method. Passes through to the
        underlying driver; HTTP-tier wrappers don't expose this.
        """
        timeout_ms = (timeout if timeout is not None else self._timeout) * 1000
        return await self._driver.run_in_page(url, timeout_ms=timeout_ms, page_script=page_script)

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
