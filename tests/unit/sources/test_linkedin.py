"""Tests for the LinkedIn (guest mode) source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.linkedin import (
    LinkedInFetchError,
    LinkedInSource,
    build_query,
)

_FIXTURE = (Path(__file__).parent / "fixtures" / "linkedin_python_au.html").read_text(
    encoding="utf-8"
)
_QUERY = "keywords=python&location=Australia"
_URL = f"https://www.linkedin.com/jobs/search?{_QUERY}"


def test_linkedin_source_name_includes_query() -> None:
    source = LinkedInSource(account=_QUERY)
    assert source.name == f"linkedin:{_QUERY}"


def test_build_query_url_encodes_inputs() -> None:
    assert build_query(keywords="python", location="Australia") == _QUERY
    # urlencode uses '+' for spaces in form-encoded queries.
    assert "senior+python" in build_query(keywords="senior python", location="Australia")


async def test_discover_yields_one_job_per_card() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]

    assert {j.source_external_id for j in jobs} == {
        "3784567890",
        "3784567891",
        "3784567892",
    }


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]

    senior = next(j for j in jobs if j.source_external_id == "3784567890")
    assert senior.title == "Senior Python Engineer"
    assert senior.company == "Atlassian"
    assert senior.apply_url.startswith("https://www.linkedin.com/jobs/view/")
    assert senior.location_country == "Australia"
    assert senior.posted_at == "2026-04-25"


async def test_discover_extracts_remote_and_hybrid_from_location_text() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["3784567891"].remote_type == "remote"
    assert by_id["3784567892"].remote_type == "hybrid"


async def test_discover_skips_cards_missing_id_or_title() -> None:
    bad = (
        "<html><body><ul>"
        '<li><div class="base-card">'
        '<a class="base-card__full-link" href="/jobs/view/no-id">x</a>'
        '<h3 class="base-search-card__title">No id</h3>'
        "</div></li>"
        '<li><div class="base-card" data-entity-urn="urn:li:jobPosting:1">'
        '<a class="base-card__full-link" href="/jobs/view/x-1"></a>'
        '<h3 class="base-search-card__title"></h3>'
        "</div></li>"
        "</ul></body></html>"
    )
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=bad))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]
    assert jobs == []


async def test_discover_raises_on_non_2xx() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(429))
        async with HttpFetcher() as fetcher:
            with pytest.raises(LinkedInFetchError) as excinfo:
                async for _ in LinkedInSource(account=_QUERY).discover(fetcher):
                    pass
    assert excinfo.value.status_code == 429
