"""Tests for the Eightfold (Microsoft Careers) job-detail resolver.

`apply.careers.microsoft.com/careers/job/<id>` is an Eightfold ATS
SPA — the raw page is a ~640KB JS shell with no parseable JD or
company, which is why tailoring it produced generic, blind output.
The job body lives at the Eightfold JSON API
`/api/apply/v2/jobs/<id>` (verified live 2026-05-18: HTTP 200,
``job_description`` ~6.2KB of HTML). ``eightfold_jd_url`` rewrites
onto that endpoint; the parser strips the HTML to text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from jobai.fetcher.base import Response
from jobai.sources.eightfold_detail import (
    eightfold_jd_url,
    fetch_eightfold_jd_text,
    parse_eightfold_description,
)

_JOB_JSON = json.dumps(
    {
        "id": 1970393556621959,
        "name": "Software Engineer",
        "location": "Australia, Multiple Locations",
        "job_description": (
            "<b>Overview</b><br>Azure is growing faster than ever."
            "<ul><li>Build distributed systems</li><li>Python, C#</li></ul>"
            "<b>Qualifications</b><br>5+ years."
        ),
    }
)


class _RecordingFetcher:
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
        timeout: float | None = None,  # noqa: ASYNC109 - matches Fetcher Protocol
        wait_for_selector: str | None = None,
        wait_until: str = "networkidle",
    ) -> Response:
        self.calls.append(url)
        return self._response

    async def aclose(self) -> None:
        return None


def _resp(status: int, body: bytes) -> Response:
    return Response(
        url="https://apply.careers.microsoft.com/api/apply/v2/jobs/1",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://apply.careers.microsoft.com/careers/job/1970393556621959",
        "https://apply.careers.microsoft.com/careers/job/1970393556621959?utm_source=linkedin&src=LinkedIn",
    ],
)
def test_eightfold_jd_url_rewrites_to_the_json_api(url: str) -> None:
    assert eightfold_jd_url(url) == (
        "https://apply.careers.microsoft.com/api/apply/v2/jobs/1970393556621959"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.seek.com.au/job/12345678",
        "https://apply.careers.microsoft.com/careers/search",  # no job id
        "https://careers.microsoft.com/v2/global/en/home.html",
        "not a url",
    ],
)
def test_eightfold_jd_url_returns_none_when_not_a_job_url(url: str) -> None:
    assert eightfold_jd_url(url) is None


def test_parse_strips_html_to_text() -> None:
    text = parse_eightfold_description(_JOB_JSON)
    assert text is not None
    assert "Overview" in text
    assert "Build distributed systems" in text
    assert "<b>" not in text  # HTML stripped


def test_parse_returns_none_on_missing_or_blank_description() -> None:
    assert parse_eightfold_description(json.dumps({"name": "x"})) is None
    assert parse_eightfold_description(json.dumps({"job_description": "  "})) is None


def test_parse_returns_none_on_non_json() -> None:
    assert parse_eightfold_description("<html>not json</html>") is None


def test_parse_returns_none_when_json_is_not_an_object() -> None:
    # Valid JSON but a list / scalar, not the expected job object.
    assert parse_eightfold_description("[]") is None
    assert parse_eightfold_description("42") is None


async def test_fetch_happy_path_hits_api_and_returns_text() -> None:
    fetcher = _RecordingFetcher(_resp(200, _JOB_JSON.encode()))
    text = await fetch_eightfold_jd_text(
        "https://apply.careers.microsoft.com/careers/job/1970393556621959?src=LinkedIn",
        fetcher,
    )
    assert text is not None
    assert "Azure is growing faster than ever." in text
    assert fetcher.calls == [
        "https://apply.careers.microsoft.com/api/apply/v2/jobs/1970393556621959"
    ]


async def test_fetch_returns_none_for_non_eightfold_url() -> None:
    fetcher = _RecordingFetcher(_resp(200, _JOB_JSON.encode()))
    assert await fetch_eightfold_jd_text("https://www.seek.com.au/job/1", fetcher) is None
    assert fetcher.calls == []


async def test_fetch_returns_none_on_non_ok() -> None:
    fetcher = _RecordingFetcher(_resp(404, b"{}"))
    assert (
        await fetch_eightfold_jd_text("https://apply.careers.microsoft.com/careers/job/1", fetcher)
        is None
    )
