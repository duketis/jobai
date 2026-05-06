"""Tests for the Seek source.

Drives :class:`SeekSource` against a captured Next.js HTML fixture so
the parser is exercised end-to-end. Real shape calibration happens
when the browser fetcher actually hits seek.com.au; these tests pin
the parser behaviour against a known-good fixture in the meantime.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.seek import SeekFetchError, SeekSource

_FIXTURE = (Path(__file__).parent / "fixtures" / "seek_python_melbourne.html").read_text(
    encoding="utf-8"
)
_SLUG = "python-jobs/in-Melbourne-VIC"
_URL = f"https://www.seek.com.au/{_SLUG}"


def test_seek_source_name_includes_slug() -> None:
    source = SeekSource(account=_SLUG)
    assert source.name == f"seek:{_SLUG}"
    assert source.kind == "seek"


async def test_discover_yields_one_job_per_result() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))

        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    assert len(jobs) == 3
    assert {j.source_external_id for j in jobs} == {"70123456", "70234567", "70345678"}


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    senior = next(j for j in jobs if j.source_external_id == "70123456")
    assert senior.title == "Senior Python Engineer"
    assert senior.company == "Atlassian"
    assert senior.apply_url == "https://www.seek.com.au/job/70123456"
    assert senior.location_city == "Melbourne"
    assert senior.location_country == "Australia"
    assert senior.remote_type == "hybrid"
    assert senior.employment_type == "Full Time"
    assert senior.salary_min == 140_000
    assert senior.salary_max == 170_000
    assert senior.salary_currency == "AUD"
    assert senior.posted_at == "2026-05-01T09:00:00.000Z"


async def test_discover_infers_remote_type_from_work_arrangements() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["70123456"].remote_type == "hybrid"
    assert by_id["70234567"].remote_type == "remote"
    assert by_id["70345678"].remote_type == "onsite"


async def test_discover_handles_missing_salary_gracefully() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    junior = next(j for j in jobs if j.source_external_id == "70345678")
    assert junior.salary_min is None
    assert junior.salary_max is None
    assert junior.salary_currency is None


async def test_derives_apply_url_from_id_when_missing() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    canva = next(j for j in jobs if j.source_external_id == "70234567")
    # The fixture leaves `url` off this entry — the source must
    # synthesise a URL from the id.
    assert canva.apply_url == "https://www.seek.com.au/job/70234567"


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(503))
        async with HttpFetcher() as fetcher:
            source = SeekSource(account=_SLUG)
            with pytest.raises(SeekFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.status_code == 503
    assert excinfo.value.slug == _SLUG


async def test_discover_returns_empty_on_missing_data_island() -> None:
    """A page without ``__NEXT_DATA__`` is treated as zero results, not error."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(
            return_value=httpx.Response(200, text="<html><body>No island here</body></html>"),
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    assert jobs == []


async def test_discover_returns_empty_on_malformed_json() -> None:
    with respx.mock(assert_all_called=False) as router:
        bad = '<script id="__NEXT_DATA__" type="application/json">{not valid json}</script>'
        router.get(_URL).mock(return_value=httpx.Response(200, text=bad))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    assert jobs == []


async def test_discover_skips_results_missing_required_fields() -> None:
    """Entries without id/title are skipped, not raising."""
    bad_fixture = (
        "<html><body>"
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"searchResults":{"results":['
        '{"id": null, "title": "no id"},'
        '{"id": 1, "title": ""},'
        '{"id": 2, "title": "ok", "advertiser": {"description": "Co"}, '
        '"locations": [{"label":"Sydney","country":"Australia","city":"Sydney"}]}'
        "]}}}}"
        "</script></body></html>"
    )
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=bad_fixture))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    assert len(jobs) == 1
    assert jobs[0].source_external_id == "2"
