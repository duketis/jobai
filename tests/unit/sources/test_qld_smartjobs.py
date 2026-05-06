"""Tests for the QLD smartjobs.qld.gov.au source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.qld_smartjobs import (
    QLDSmartJobsFetchError,
    QLDSmartJobsSource,
)

_FIXTURE = (Path(__file__).parent / "fixtures" / "qld_smartjobs.html").read_text(encoding="utf-8")
_URL = "https://smartjobs.qld.gov.au/jobtools/jncustomsearch.searchResults?in_organid=14904"


def test_source_uses_default_orgid_when_account_blank() -> None:
    source = QLDSmartJobsSource(account="")
    assert source.account == "14904"


def test_source_name_includes_orgid() -> None:
    source = QLDSmartJobsSource(account="14904")
    assert source.name == "qld_smartjobs:14904"


async def test_discover_yields_one_job_per_li() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in QLDSmartJobsSource(account="14904").discover(fetcher)]

    # Real ids from fixture (extracted from /jobs/QLD-{id}-{year} hrefs)
    assert len(jobs) == 3
    assert all(j.source_external_id.startswith("QLD-") for j in jobs)


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in QLDSmartJobsSource(account="14904").discover(fetcher)]

    teacher = next(
        (j for j in jobs if j.source_external_id == "QLD-684271"),
        None,
    )
    assert teacher is not None
    assert "Teacher" in teacher.title
    assert teacher.company  # Non-empty agency name
    assert teacher.apply_url.startswith("https://smartjobs.qld.gov.au/jobs/QLD-684271")
    assert teacher.location_country == "Australia"
    assert teacher.salary_min == 61_570
    assert teacher.salary_max == 98_481
    assert teacher.salary_currency == "AUD"
    assert teacher.employment_type and "Fixed" in teacher.employment_type


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(503))
        async with HttpFetcher() as fetcher:
            with pytest.raises(QLDSmartJobsFetchError) as excinfo:
                async for _ in QLDSmartJobsSource(account="14904").discover(fetcher):
                    pass
    assert excinfo.value.status_code == 503


async def test_discover_returns_empty_on_no_cards() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(
            return_value=httpx.Response(200, text="<html><body><p>nothing</p></body></html>")
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in QLDSmartJobsSource(account="14904").discover(fetcher)]
    assert jobs == []
