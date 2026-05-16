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

from jobai import __version__
from jobai.fetcher.base import Response, WaitUntil

#: Sources that need full Playwright control (form fill, click,
#: multi-step navigation) pass a callable to
#: :meth:`BrowserFetcher.run_in_page`. The callable receives a
#: :class:`Page` already navigated to ``url`` and is expected to
#: drive any extra interactions; on return the fetcher snapshots
#: ``page.content()`` and packages it as a :class:`Response`.
PageScript = Callable[["Page"], Awaitable[None]]

#: A Chrome-on-macOS UA suffixed with ``jobai/<version>`` so we keep a
#: realistic fingerprint without misrepresenting ourselves. The version
#: is interpolated from ``jobai.__version__`` so a single bump
#: propagates everywhere.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    f"Chrome/120.0.0.0 Safari/537.36 jobai/{__version__}"
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
        wait_until: WaitUntil = "networkidle",
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
    is reused across requests.

    Two context modes:

    * **Per-fetch (default)** — each call gets a fresh
      :class:`BrowserContext` so cookies / localStorage do not leak
      between sources. Right for the common case.
    * **Session-persistent** (``persistent_session=True``) — one
      context lives for the lifetime of the driver. All fetches
      share cookies, localStorage, and (importantly) the TLS
      handshake state Cloudflare ties its ``cf_clearance`` cookie
      to. Solves CF once at the start of a scrape run instead of
      hitting the challenge interstitial on every request. Use for
      CF-protected sources only - sharing state across unrelated
      sources risks cookie pollution.
    """

    def __init__(
        self,
        *,
        user_agent: str = _DEFAULT_USER_AGENT,
        headless: bool = True,
        playwright_factory: Any = async_playwright,
        persistent_session: bool = False,
    ) -> None:
        self._user_agent = user_agent
        self._headless = headless
        self._factory = playwright_factory
        self._persistent_session = persistent_session
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._persistent_context: Any = None  # BrowserContext when persistent_session
        self._init_lock = asyncio.Lock()

    async def _ensure_browser(self) -> Browser:  # pragma: no cover - drives real Playwright
        # Spawning Chromium via Playwright cannot be exercised under
        # unit tests; the integration soak (docker compose up -d) covers
        # the boot path on every release.
        async with self._init_lock:
            if self._browser is None:
                self._playwright = await self._factory().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self._headless,
                )
            return self._browser

    async def _get_context(self) -> Any:  # pragma: no cover - drives real Playwright
        """Return a context for the next fetch.

        In per-fetch mode (default) this is a fresh context the caller
        is responsible for closing. In session-persistent mode it's
        the long-lived shared context that lives until ``close()``;
        the caller MUST NOT close it.
        """
        browser = await self._ensure_browser()
        if not self._persistent_session:
            return await browser.new_context(user_agent=self._user_agent)
        # Persistent: lazily construct once, reuse on every call.
        if self._persistent_context is None:
            async with self._init_lock:
                if self._persistent_context is None:
                    self._persistent_context = await browser.new_context(
                        user_agent=self._user_agent,
                        viewport={"width": 1280, "height": 800},
                    )
        return self._persistent_context

    async def fetch_page(  # pragma: no cover - drives real Playwright
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        timeout_ms: float,
        wait_for_selector: str | None = None,
        wait_until: WaitUntil = "networkidle",
    ) -> Response:
        # Integration-only -- exercises Chromium nav, networkidle wait,
        # wait_for_selector, and the page.content() snapshot. The fake
        # driver fixture in tests/unit/fetcher exercises the surrounding
        # BrowserFetcher wiring without hitting real Chromium.
        context = await self._get_context()
        # In persistent mode the context is shared - never close it
        # here; close() takes care of it on driver shutdown.
        close_context = not self._persistent_session
        try:
            page = await context.new_page()
            if headers:
                await page.set_extra_http_headers(dict(headers))
            # ``wait_until='networkidle'`` BLOCKS goto until the network
            # is idle for 500ms, which forces SPA initial XHRs to
            # complete BEFORE we proceed. Doing it as a separate
            # ``wait_for_load_state('networkidle')`` after a default
            # goto returns immediately when the empty shell DOM
            # reaches idle - missing the SPA's data fetch entirely.
            # Also gives Cloudflare's challenge JS time to solve and
            # redirect on protected sources. Callers whose SPA never
            # reaches network idle (Seek job-detail pages poll
            # forever) pass wait_until='domcontentloaded' and rely on
            # wait_for_selector below to gate on the real content.
            response = await page.goto(url, timeout=timeout_ms, wait_until=wait_until)
            if wait_for_selector is not None:
                # Wait for the SPA to populate the requested selector
                # (Next.js / Salesforce Lightning / React lazy-load
                # results after first paint). Soft-fail: a timeout
                # here yields the partially-rendered DOM rather than
                # raising — better than zero data on the first cycle.
                with contextlib.suppress(Exception):
                    await page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
            # Post-load grace period: some SPAs (NSW iworkfor's Angular
            # app, ad-core04 backed) authenticate-then-fetch-then-render
            # after networkidle reports settled. Wait an extra beat so
            # we snapshot the fully-rendered DOM, not a transient state.
            # 2s is short enough not to hurt fast sources but covers
            # the "render right after the auth XHR returns" window.
            await page.wait_for_timeout(2000)
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
            if close_context:
                await context.close()
            else:
                # Persistent mode: leave context alive but free the
                # page so we don't leak tabs across fetches.
                with contextlib.suppress(Exception):
                    await page.close()

    async def run_in_page(  # pragma: no cover - drives real Playwright
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
        context = await self._get_context()
        close_context = not self._persistent_session
        try:
            page = await context.new_page()
            response = await page.goto(url, timeout=timeout_ms)
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
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
            if close_context:
                await context.close()
            else:
                with contextlib.suppress(Exception):
                    await page.close()

    async def close(self) -> None:  # pragma: no cover - tears down real Playwright
        if self._persistent_context is not None:
            with contextlib.suppress(Exception):
                await self._persistent_context.close()
            self._persistent_context = None
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
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109  - delegates to playwright
        wait_for_selector: str | None = None,
        wait_until: WaitUntil = "networkidle",
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
        if data is not None:
            msg = "BrowserFetcher does not support `data` (form) payloads."
            raise ValueError(msg)

        timeout_ms = (timeout if timeout is not None else self._timeout) * 1000
        return await self._driver.fetch_page(
            url,
            headers=headers,
            timeout_ms=timeout_ms,
            wait_for_selector=wait_for_selector,
            wait_until=wait_until,
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
