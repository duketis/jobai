"""Tests for the shared LinkedIn job-detail fetch + parse helper.

The guest ``jobPosting/<id>`` fragment is publicly fetchable with no
auth wall (verified live 2026-05-18: HTTP 200, full description),
unlike the ``/jobs/view/<slug>-<id>`` page which carries auth-wall
markup. ``guest_jd_url`` rewrites any LinkedIn job URL onto that
fragment so the tailor resolver pulls a clean JD on jobai's stealth
tier instead of handing the hostile page to the siblings.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from jobai.fetcher.base import Response
from jobai.sources.linkedin_detail import (
    fetch_linkedin_jd_text,
    guest_jd_url,
    parse_linkedin_description,
)

_JD_HTML = (
    "<html><body>"
    '<div class="description__text"><section class="show-more-less-html">'
    '<div class="show-more-less-html__markup"><p>Build the platform.</p>'
    "<ul><li>Python</li><li>asyncio</li></ul></div></section></div>"
    "</body></html>"
)


class _RecordingFetcher:
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
        timeout: float | None = None,  # noqa: ASYNC109 - matches Fetcher Protocol
        wait_for_selector: str | None = None,
        wait_until: str = "networkidle",
    ) -> Response:
        self.calls.append({"url": url})
        return self._response

    async def aclose(self) -> None:
        return None


def _resp(status: int, body: bytes) -> Response:
    return Response(
        url="https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/1",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


# -- guest_jd_url -----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://au.linkedin.com/jobs/view/software-engineer-at-evp-4413516164",
        "https://au.linkedin.com/jobs/view/software-engineer-at-evp-4413516164?refId=x",
        "https://www.linkedin.com/jobs/view/4413516164",
        "https://www.linkedin.com/jobs/search?currentJobId=4413516164&keywords=x",
    ],
)
def test_guest_jd_url_extracts_id_from_every_shape(url: str) -> None:
    assert guest_jd_url(url) == "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/4413516164"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.seek.com.au/job/12345678",
        "https://example.com/jobs/view/4413516164",
        "https://au.linkedin.com/jobs/view/no-numeric-id-here",
        "not a url",
    ],
)
def test_guest_jd_url_returns_none_for_non_linkedin_or_idless(url: str) -> None:
    assert guest_jd_url(url) is None


# -- parse_linkedin_description --------------------------------------------


def test_parse_extracts_jd_text() -> None:
    text = parse_linkedin_description(_JD_HTML)
    assert text is not None
    assert "Build the platform." in text
    assert "asyncio" in text


def test_parse_returns_none_when_container_absent() -> None:
    assert parse_linkedin_description("<html><body>nope</body></html>") is None


def test_parse_falls_back_to_inner_markup_wrapper() -> None:
    """Some guest fragments omit the outer description__text and only
    carry the inner show-more-less-html__markup wrapper."""
    html = '<html><body><div class="show-more-less-html__markup">Markup body</div></body></html>'
    assert parse_linkedin_description(html) == "Markup body"


def test_parse_returns_none_when_container_empty() -> None:
    assert parse_linkedin_description('<div class="description__text"></div>') is None


# -- fetch_linkedin_jd_text -------------------------------------------------


async def test_fetch_happy_path_hits_guest_fragment() -> None:
    fetcher = _RecordingFetcher(_resp(200, _JD_HTML.encode()))
    text = await fetch_linkedin_jd_text(
        "https://au.linkedin.com/jobs/view/x-4413516164?trk=public",
        fetcher,
    )
    assert text is not None
    assert "Build the platform." in text
    assert (
        fetcher.calls[0]["url"]
        == "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/4413516164"
    )


async def test_fetch_returns_none_for_non_linkedin_url() -> None:
    fetcher = _RecordingFetcher(_resp(200, _JD_HTML.encode()))
    assert await fetch_linkedin_jd_text("https://www.seek.com.au/job/1", fetcher) is None
    assert fetcher.calls == []  # never fetched — not a LinkedIn URL


async def test_fetch_returns_none_on_non_ok_response() -> None:
    fetcher = _RecordingFetcher(_resp(429, b"rate limited"))
    assert (
        await fetch_linkedin_jd_text("https://www.linkedin.com/jobs/view/4413516164", fetcher)
        is None
    )


async def test_fetch_returns_none_when_body_unparsable() -> None:
    fetcher = _RecordingFetcher(_resp(200, b"<html><body>wall</body></html>"))
    assert (
        await fetch_linkedin_jd_text("https://www.linkedin.com/jobs/view/4413516164", fetcher)
        is None
    )
