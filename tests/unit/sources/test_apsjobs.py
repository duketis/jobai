"""Tests for the APS Jobs source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.apsjobs import APSJobsFetchError, APSJobsSource

_FIXTURE = (Path(__file__).parent / "fixtures" / "apsjobs_software.atom").read_text(
    encoding="utf-8"
)
_QUERY = "Keywords=software"
_URL = f"https://www.apsjobs.gov.au/s/search.atom?{_QUERY}"
_DEFAULT_URL = "https://www.apsjobs.gov.au/s/search.atom"


def test_apsjobs_source_name_includes_query() -> None:
    source = APSJobsSource(account=_QUERY)
    assert source.name == f"apsjobs:{_QUERY}"


async def test_discover_yields_one_job_per_entry() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account=_QUERY).discover(fetcher)]

    assert {j.source_external_id for j in jobs} == {"3000123", "3000124", "3000125"}


async def test_discover_empty_account_uses_default_feed() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_DEFAULT_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]
    assert len(jobs) == 3


async def test_discover_extracts_agency_from_summary() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account=_QUERY).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert "Department of Finance" in by_id["3000123"].company
    assert by_id["3000124"].company == "Australian Bureau of Statistics"
    assert "Department of Defence" in by_id["3000125"].company


async def test_discover_extracts_location_from_summary() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account=_QUERY).discover(fetcher)]
    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["3000123"].location_raw == "Canberra ACT"
    assert by_id["3000123"].location_city == "Canberra ACT"
    assert by_id["3000123"].location_country == "Australia"


async def test_discover_extracts_salary_when_present() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account=_QUERY).discover(fetcher)]
    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["3000123"].salary_min == 120_000
    assert by_id["3000123"].salary_max == 140_000
    assert by_id["3000123"].salary_currency == "AUD"
    # No salary in entry 124
    assert by_id["3000124"].salary_min is None


async def test_discover_uses_published_or_updated_for_posted_at() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account=_QUERY).discover(fetcher)]
    by_id = {j.source_external_id: j for j in jobs}
    # Updated wins over published
    assert by_id["3000123"].posted_at == "2026-05-04T09:00:00Z"


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(500))
        async with HttpFetcher() as fetcher:
            with pytest.raises(APSJobsFetchError) as excinfo:
                async for _ in APSJobsSource(account=_QUERY).discover(fetcher):
                    pass
    assert excinfo.value.status_code == 500
