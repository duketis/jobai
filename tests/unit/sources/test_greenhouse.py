"""Tests for the Greenhouse source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.greenhouse import GreenhouseFetchError, GreenhouseSource

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ATLASSIAN_FIXTURE = (_FIXTURE_DIR / "greenhouse_atlassian.json").read_text(encoding="utf-8")


def test_greenhouse_source_name_includes_board() -> None:
    source = GreenhouseSource(account="atlassian")
    assert source.name == "greenhouse:atlassian"
    assert source.kind == "greenhouse"
    assert source.account == "atlassian"


async def test_discover_yields_one_job_per_listing_entry() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/atlassian/jobs").mock(
            return_value=httpx.Response(200, text=_ATLASSIAN_FIXTURE),
        )

        source = GreenhouseSource(account="atlassian")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert len(jobs) == 3


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/atlassian/jobs").mock(
            return_value=httpx.Response(200, text=_ATLASSIAN_FIXTURE),
        )

        source = GreenhouseSource(account="atlassian")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    first = jobs[0]
    assert first.source_external_id == "12345"
    assert first.title == "Senior Software Engineer, Backend"
    assert first.company == "atlassian"
    assert first.apply_url == "https://www.atlassian.com/company/careers/details/12345"
    assert first.location_raw == "Sydney, Australia"
    assert first.description_html is not None
    assert first.posted_at == "2026-04-15T12:00:00-04:00"


async def test_discover_infers_remote_type_from_location() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/atlassian/jobs").mock(
            return_value=httpx.Response(200, text=_ATLASSIAN_FIXTURE),
        )

        source = GreenhouseSource(account="atlassian")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["12345"].remote_type is None  # plain "Sydney, Australia"
    assert by_id["12346"].remote_type == "remote"  # "Remote, Australia"
    assert by_id["12347"].remote_type == "hybrid"  # "Melbourne, Australia (Hybrid)"


async def test_discover_preserves_raw_payload() -> None:
    """The raw_data field must contain the original Greenhouse dict so we can
    re-extract fields later without re-scraping."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/atlassian/jobs").mock(
            return_value=httpx.Response(200, text=_ATLASSIAN_FIXTURE),
        )

        source = GreenhouseSource(account="atlassian")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs[0].raw_data["requisition_id"] == "R-1001"
    assert jobs[0].raw_data["departments"][0]["name"] == "Engineering"


async def test_discover_raises_on_non_2xx_status() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/missing-co/jobs").mock(
            return_value=httpx.Response(404, text="not found"),
        )

        source = GreenhouseSource(account="missing-co")
        async with HttpFetcher() as fetcher:
            with pytest.raises(GreenhouseFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.status_code == 404
    assert excinfo.value.board == "missing-co"


async def test_discover_handles_empty_jobs_array() -> None:
    """A board with no jobs should yield zero items, not raise."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/empty-co/jobs").mock(
            return_value=httpx.Response(200, json={"jobs": [], "meta": {"total": 0}}),
        )

        source = GreenhouseSource(account="empty-co")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs == []


async def test_discover_handles_missing_optional_location() -> None:
    """Some jobs ship without a location field; we must not crash."""
    payload = {
        "jobs": [
            {
                "id": 999,
                "title": "Engineer",
                "absolute_url": "https://example.com/999",
                "content": "<p>job</p>",
                "updated_at": "2026-05-01T00:00:00Z",
            }
        ]
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://boards-api.greenhouse.io/v1/boards/x/jobs").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = GreenhouseSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert len(jobs) == 1
    assert jobs[0].location_raw is None
    assert jobs[0].remote_type is None
