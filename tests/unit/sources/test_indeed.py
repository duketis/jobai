"""Tests for the Indeed (au.indeed.com) source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.indeed import IndeedFetchError, IndeedSource, build_query

_FIXTURE = (Path(__file__).parent / "fixtures" / "indeed_python_melbourne.html").read_text(
    encoding="utf-8"
)
_QUERY = "q=python&l=Melbourne&fromage=7"
_URL = f"https://au.indeed.com/jobs?{_QUERY}"


def test_indeed_source_name_includes_query() -> None:
    source = IndeedSource(account=_QUERY)
    assert source.name == f"indeed:{_QUERY}"


def test_build_query_url_encodes_inputs() -> None:
    assert build_query(keywords="python", location="Melbourne", recency_days=7) == _QUERY
    assert "Melbourne+VIC" in build_query(
        keywords="python", location="Melbourne VIC"
    ) or "Melbourne%20VIC" in build_query(keywords="python", location="Melbourne VIC")


async def test_discover_yields_one_job_per_result() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    assert {j.source_external_id for j in jobs} == {"abc123", "def456", "ghi789"}


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    senior = next(j for j in jobs if j.source_external_id == "abc123")
    assert senior.title == "Senior Python Engineer"
    assert senior.company == "Atlassian"
    assert senior.apply_url == "https://au.indeed.com/viewjob?jk=abc123"
    assert senior.location_country == "Australia"
    assert senior.salary_min == 140_000
    assert senior.salary_max == 170_000
    assert senior.salary_currency == "AUD"
    assert senior.remote_type == "hybrid"


async def test_discover_handles_remote_and_missing_salary() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["def456"].remote_type == "remote"
    assert by_id["def456"].salary_min == 120_000
    assert by_id["def456"].salary_max is None
    assert by_id["ghi789"].salary_min is None
    assert by_id["ghi789"].salary_max is None


async def test_discover_returns_empty_on_missing_island() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text="<html><body></body></html>"))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]
    assert jobs == []


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(403))
        async with HttpFetcher() as fetcher:
            with pytest.raises(IndeedFetchError) as excinfo:
                async for _ in IndeedSource(account=_QUERY).discover(fetcher):
                    pass
    assert excinfo.value.status_code == 403
