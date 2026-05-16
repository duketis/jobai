"""Test helpers for sources that drive the browser tier via run_in_page.

State-government sources (SA / WA / VIC) need a fetcher with both a
``fetch()`` and a ``run_in_page()`` method. Production wires them
through :class:`jobai.fetcher.browser.BrowserFetcher`; tests stub
them with :class:`FakeBrowserFetcher` below — canned response, no
Playwright, no Chromium.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from jobai.fetcher.base import Response


class FakeBrowserFetcher:
    """Browser-tier fetcher stub that returns a single canned response.

    ``run_in_page`` accepts and discards the ``page_script`` callable —
    the script's job is to drive a real Playwright Page, but in unit
    tests we already know what HTML we want to feed the parser, so
    bypassing the script is simpler than scripting a fake Page.
    """

    def __init__(self, response: Response) -> None:
        self._response = response
        self.calls: list[str] = []

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
        wait_until: str = "networkidle",
    ) -> Response:
        del method, headers, json, data, timeout, wait_for_selector
        self.calls.append(url)
        return self._response

    async def run_in_page(
        self,
        url: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        page_script: Any = None,
    ) -> Response:
        del timeout, page_script
        self.calls.append(url)
        return self._response

    async def aclose(self) -> None:
        return None


def html_response(html: str, status_code: int = 200) -> Response:
    """Build a :class:`Response` carrying ``html`` as the body."""
    return Response(
        url="https://example.test/page",
        status_code=status_code,
        headers={},
        body=html.encode("utf-8"),
        fetched_at=datetime.now(tz=UTC),
    )
