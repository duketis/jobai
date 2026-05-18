"""Tests for the multi-platform Tailor-from-URL JD resolver.

Locks in: every gated board we scrape (Seek / LinkedIn / Indeed) is
resolved on jobai's stealth tier via its description recipe; every
non-gated host defers to the sibling (returns None); and every
failure mode degrades to None instead of raising.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from jobai.fetcher.base import Response
from jobai.tailor import jd_resolution

_SEEK_HTML = '<html><body><div data-automation="jobAdDetails">Seek JD body.</div></body></html>'
_LI_HTML = '<html><body><div class="description__text">LinkedIn JD body.</div></body></html>'
_INDEED_HTML = '<html><body><div id="jobDescriptionText">Indeed JD body.</div></body></html>'


class _FakeFetcher:
    def __init__(self, response: Response | None, *, raises: bool = False) -> None:
        self._response = response
        self._raises = raises
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
        self.calls.append(
            {"url": url, "wait_for_selector": wait_for_selector, "wait_until": wait_until},
        )
        if self._raises:
            msg = "boom"
            raise RuntimeError(msg)
        assert self._response is not None
        return self._response

    async def aclose(self) -> None:
        return None


def _resp(status: int, body: bytes) -> Response:
    return Response(
        url="https://x/y",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


def _patch_fetcher(monkeypatch: pytest.MonkeyPatch, fetcher: _FakeFetcher) -> None:
    monkeypatch.setattr(jd_resolution, "build_fetcher", lambda *, tier: fetcher)


async def test_non_gated_host_defers_to_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _FakeFetcher(_resp(200, b"x"))
    _patch_fetcher(monkeypatch, fetcher)
    assert await jd_resolution.resolve_jd_text("https://boards.greenhouse.io/x/jobs/1") is None
    assert fetcher.calls == []  # never even built a fetcher path


async def test_seek_url_resolved_via_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _FakeFetcher(_resp(200, _SEEK_HTML.encode()))
    _patch_fetcher(monkeypatch, fetcher)
    text = await jd_resolution.resolve_jd_text("https://www.seek.com.au/job/12345678")
    assert text == "Seek JD body."
    # Seek's recipe drives the verified domcontentloaded + selector combo.
    assert fetcher.calls[0]["wait_until"] == "domcontentloaded"


async def test_linkedin_url_rewritten_to_guest_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _FakeFetcher(_resp(200, _LI_HTML.encode()))
    _patch_fetcher(monkeypatch, fetcher)
    text = await jd_resolution.resolve_jd_text(
        "https://au.linkedin.com/jobs/view/software-engineer-at-x-4413516164?refId=z",
    )
    assert text == "LinkedIn JD body."
    assert (
        fetcher.calls[0]["url"]
        == "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/4413516164"
    )


async def test_indeed_url_resolved_via_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _FakeFetcher(_resp(200, _INDEED_HTML.encode()))
    _patch_fetcher(monkeypatch, fetcher)
    text = await jd_resolution.resolve_jd_text("https://au.indeed.com/viewjob?jk=abc123")
    assert text == "Indeed JD body."


async def test_non_2xx_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetcher(monkeypatch, _FakeFetcher(_resp(403, b"blocked")))
    assert await jd_resolution.resolve_jd_text("https://www.seek.com.au/job/9") is None


async def test_unparsable_body_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetcher(monkeypatch, _FakeFetcher(_resp(200, b"<html>wall</html>")))
    assert (
        await jd_resolution.resolve_jd_text("https://www.linkedin.com/jobs/view/4413516164") is None
    )


async def test_fetch_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetcher(monkeypatch, _FakeFetcher(None, raises=True))
    assert await jd_resolution.resolve_jd_text("https://www.seek.com.au/job/9") is None
