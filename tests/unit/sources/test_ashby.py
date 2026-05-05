"""Tests for the Ashby source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.ashby import AshbyFetchError, AshbySource

_FIXTURE = (Path(__file__).parent / "fixtures" / "ashby_linear.json").read_text(encoding="utf-8")


def test_ashby_source_name_includes_account() -> None:
    source = AshbySource(account="linear")
    assert source.name == "ashby:linear"
    assert source.kind == "ashby"


async def test_discover_yields_one_job_per_listed_entry() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/linear").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = AshbySource(account="linear")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert len(jobs) == 5
    assert all(j.source_external_id for j in jobs)
    assert all(j.title for j in jobs)


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/linear").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = AshbySource(account="linear")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    first = jobs[0]
    assert first.company == "linear"
    assert first.apply_url.startswith("http")
    assert first.description_html is not None
    assert first.description_text is not None


async def test_discover_normalises_workplace_type() -> None:
    payload = {
        "jobs": [
            {
                "id": "a",
                "title": "Remote role",
                "isListed": True,
                "workplaceType": "Remote",
                "applyUrl": "https://example.com/a",
                "descriptionHtml": "",
                "descriptionPlain": "",
            },
            {
                "id": "b",
                "title": "Hybrid role",
                "isListed": True,
                "workplaceType": "Hybrid",
                "applyUrl": "https://example.com/b",
                "descriptionHtml": "",
                "descriptionPlain": "",
            },
            {
                "id": "c",
                "title": "Onsite role",
                "isListed": True,
                "workplaceType": "OnSite",
                "applyUrl": "https://example.com/c",
                "descriptionHtml": "",
                "descriptionPlain": "",
            },
        ],
        "apiVersion": "1",
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = AshbySource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["a"].remote_type == "remote"
    assert by_id["b"].remote_type == "hybrid"
    assert by_id["c"].remote_type == "onsite"


async def test_discover_normalises_employment_type() -> None:
    payload = {
        "jobs": [
            {
                "id": "a",
                "title": "FullTime role",
                "isListed": True,
                "employmentType": "FullTime",
                "applyUrl": "https://example.com/a",
            },
            {
                "id": "b",
                "title": "Contract role",
                "isListed": True,
                "employmentType": "Contract",
                "applyUrl": "https://example.com/b",
            },
        ]
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = AshbySource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["a"].employment_type == "full-time"
    assert by_id["b"].employment_type == "contract"


async def test_discover_skips_unlisted_jobs() -> None:
    """isListed=False marks a draft / hidden job; we must not surface it."""
    payload = {
        "jobs": [
            {
                "id": "shown",
                "title": "Published",
                "isListed": True,
                "applyUrl": "https://example.com/shown",
            },
            {
                "id": "hidden",
                "title": "Draft",
                "isListed": False,
                "applyUrl": "https://example.com/hidden",
            },
        ]
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = AshbySource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert [j.source_external_id for j in jobs] == ["shown"]


async def test_discover_extracts_country_from_postal_address() -> None:
    payload = {
        "jobs": [
            {
                "id": "a",
                "title": "Role",
                "isListed": True,
                "applyUrl": "https://example.com/a",
                "address": {"postalAddress": {"addressCountry": "Australia"}},
            }
        ]
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = AshbySource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs[0].location_country == "Australia"


async def test_discover_pulls_compensation_when_present() -> None:
    payload = {
        "jobs": [
            {
                "id": "a",
                "title": "Role",
                "isListed": True,
                "applyUrl": "https://example.com/a",
                "compensation": {
                    "summaryComponents": [
                        {
                            "compensationType": "Salary",
                            "minValue": 150000,
                            "maxValue": 220000,
                            "currencyCode": "USD",
                        }
                    ]
                },
            }
        ]
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = AshbySource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs[0].salary_min == 150000
    assert jobs[0].salary_max == 220000
    assert jobs[0].salary_currency == "USD"


async def test_discover_raises_on_non_2xx_status() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/missing").mock(
            return_value=httpx.Response(401, text="Unauthorized"),
        )

        source = AshbySource(account="missing")
        async with HttpFetcher() as fetcher:
            with pytest.raises(AshbyFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.status_code == 401


async def test_discover_handles_empty_jobs() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.ashbyhq.com/posting-api/job-board/empty").mock(
            return_value=httpx.Response(200, json={"jobs": [], "apiVersion": "1"}),
        )

        source = AshbySource(account="empty")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs == []
