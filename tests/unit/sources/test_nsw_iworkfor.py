"""Tests for the NSW iworkfor.nsw.gov.au source.

Driven against a real fixture cut from a live iworkfor.nsw.gov.au
search-results page (3 representative ``article.search-job-card``
entries).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.nsw_iworkfor import NSWIWorkForFetchError, NSWIWorkForSource

_FIXTURE = (Path(__file__).parent / "fixtures" / "nsw_iworkfor.html").read_text(encoding="utf-8")
_SLUG = (
    "jobs/all-keywords/all-agencies/all-organisations-entities/"
    "all-categories/all-locations/all-worktypes"
)
_URL = f"https://iworkfor.nsw.gov.au/{_SLUG}"


def test_source_name_includes_slug() -> None:
    source = NSWIWorkForSource(account=_SLUG)
    assert source.name == f"nsw_iworkfor:{_SLUG}"


async def test_discover_yields_one_job_per_card() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="iworkfor.nsw.gov.au").mock(
            side_effect=lambda req: (
                httpx.Response(200, text=_FIXTURE)
                if "page=" not in str(req.url)
                else httpx.Response(200, text="<html><body><main></main></body></html>")
            ),
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in NSWIWorkForSource(account=_SLUG).discover(fetcher)]

    assert {j.source_external_id for j in jobs} == {"576439", "575973", "576521"}


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="iworkfor.nsw.gov.au").mock(
            side_effect=lambda req: (
                httpx.Response(200, text=_FIXTURE)
                if "page=" not in str(req.url)
                else httpx.Response(200, text="<html><body><main></main></body></html>")
            ),
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in NSWIWorkForSource(account=_SLUG).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    deputy = by_id["576439"]
    assert deputy.title == "Deputy Commissioner"
    assert deputy.company == "NSW Police Force"
    assert deputy.apply_url.startswith("https://iworkfor.nsw.gov.au/job/")
    assert deputy.location_country == "Australia"
    # Deputy Commissioner has a salary range in the fixture
    assert deputy.salary_min == 373_951
    assert deputy.salary_max == 527_050
    assert deputy.salary_currency == "AUD"


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(503))
        async with HttpFetcher() as fetcher:
            with pytest.raises(NSWIWorkForFetchError) as excinfo:
                async for _ in NSWIWorkForSource(account=_SLUG).discover(fetcher):
                    pass
    assert excinfo.value.status_code == 503
    assert excinfo.value.slug == _SLUG


async def test_discover_returns_empty_on_page_with_no_cards() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(
            return_value=httpx.Response(200, text="<html><body><main></main></body></html>")
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in NSWIWorkForSource(account=_SLUG).discover(fetcher)]
    assert jobs == []


async def test_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        NSWIWorkForSource(account=_SLUG, max_pages=0)
