"""Tests for the shared Seek job-detail fetch + parse helper.

The fixture mirrors Seek's documented ``data-automation`` contract
(stable across their Next.js redeploys). The fake fetcher records the
navigation strategy so we lock in the verified-against-live combo:
``wait_until='domcontentloaded'`` + the JD-container selector wait.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from jobai.fetcher.base import Response
from jobai.sources.seek_detail import (
    SEEK_JD_SELECTOR,
    fetch_seek_jd_text,
    parse_seek_description,
)

_JD_HTML = (
    "<html><body>"
    '<div data-automation="jobAdDetails"><p>Build the thing.</p>'
    "<ul><li>Python</li><li>FastAPI</li></ul></div>"
    "</body></html>"
)


class _RecordingFetcher:
    """Returns one canned response and records the fetch kwargs."""

    def __init__(self, response: Response) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109  - matches Fetcher Protocol
        wait_for_selector: str | None = None,
        wait_until: str = "networkidle",
    ) -> Response:
        self.calls.append(
            {
                "url": url,
                "wait_for_selector": wait_for_selector,
                "wait_until": wait_until,
            },
        )
        return self._response

    async def aclose(self) -> None:
        return None


def _resp(status: int, body: bytes) -> Response:
    return Response(
        url="https://www.seek.com.au/job/1",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


def test_parse_extracts_jd_text() -> None:
    text = parse_seek_description(_JD_HTML)
    assert text is not None
    assert "Build the thing." in text
    assert "FastAPI" in text


def test_parse_returns_none_when_container_absent() -> None:
    assert parse_seek_description("<html><body>nope</body></html>") is None


def test_parse_returns_none_when_container_empty() -> None:
    assert parse_seek_description('<div data-automation="jobAdDetails"></div>') is None


async def test_fetch_happy_path_uses_domcontentloaded_and_selector() -> None:
    fetcher = _RecordingFetcher(_resp(200, _JD_HTML.encode()))
    text = await fetch_seek_jd_text("https://www.seek.com.au/job/91797185", fetcher)
    assert text is not None
    assert "Build the thing." in text
    assert fetcher.calls[0]["wait_until"] == "domcontentloaded"
    assert fetcher.calls[0]["wait_for_selector"] == SEEK_JD_SELECTOR


async def test_fetch_returns_none_on_non_ok_response() -> None:
    fetcher = _RecordingFetcher(_resp(403, b"blocked"))
    assert await fetch_seek_jd_text("https://www.seek.com.au/job/1", fetcher) is None


async def test_fetch_returns_none_when_body_unparsable() -> None:
    fetcher = _RecordingFetcher(_resp(200, b"<html><body>challenge</body></html>"))
    assert await fetch_seek_jd_text("https://www.seek.com.au/job/1", fetcher) is None
