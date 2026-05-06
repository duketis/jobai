"""Tests for the Indeed (au.indeed.com) source.

Driven against a real fixture cut from a live au.indeed.com search
result page (mosaic-provider-jobcards data island, slimmed to 3
representative entries).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.indeed import IndeedFetchError, IndeedSource, build_query

_FIXTURE = (Path(__file__).parent / "fixtures" / "indeed_python_au.html").read_text(
    encoding="utf-8"
)
_QUERY = "q=python&l=Australia&fromage=7"
_URL = f"https://au.indeed.com/jobs?{_QUERY}"


def test_indeed_source_name_includes_query() -> None:
    source = IndeedSource(account=_QUERY)
    assert source.name == f"indeed:{_QUERY}"


def test_build_query_url_encodes_inputs() -> None:
    assert build_query(keywords="python", location="Australia", recency_days=7) == _QUERY


async def test_discover_yields_one_job_per_result() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    assert len(jobs) == 3
    # Real ids from the captured fixture
    assert {j.source_external_id for j in jobs} == {
        "e3eba48ce44f8a99",
        "2903c36943e71c58",
        "31d32114b8222d25",
    }


async def test_discover_canonicalises_apply_url() -> None:
    """``viewjob?jk={id}`` only — no tracking params."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    for j in jobs:
        assert j.apply_url == f"https://au.indeed.com/viewjob?jk={j.source_external_id}"
        assert "advn=" not in j.apply_url
        assert "tk=" not in j.apply_url


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in IndeedSource(account=_QUERY).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    first = by_id["e3eba48ce44f8a99"]
    assert "Python" in first.title
    assert first.company  # whatever Indeed surfaced
    assert first.location_country == "Australia"


async def test_discover_returns_empty_when_data_island_missing() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(
            return_value=httpx.Response(200, text="<html><body><p>nope</p></body></html>")
        )
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
