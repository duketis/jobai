"""Tests for the Workable source."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.workable import WorkableFetchError, WorkableSource

_FIXTURE = (Path(__file__).parent / "fixtures" / "workable_huggingface.json").read_text(
    encoding="utf-8"
)


def test_workable_source_name_includes_account() -> None:
    source = WorkableSource(account="huggingface")
    assert source.name == "workable:huggingface"
    assert source.kind == "workable"


async def test_discover_yields_one_job_per_listing_entry() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/huggingface").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = WorkableSource(account="huggingface")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert len(jobs) == 7
    assert all(j.source_external_id for j in jobs)


async def test_discover_uses_board_name_as_company() -> None:
    """Workable's widget gives us the human company name; prefer that over the
    URL slug."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/huggingface").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = WorkableSource(account="huggingface")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs[0].company == "Hugging Face"


async def test_discover_marks_telecommuting_jobs_remote() -> None:
    payload = {
        "name": "Co",
        "jobs": [
            {
                "shortcode": "ABC",
                "title": "Remote role",
                "telecommuting": True,
                "application_url": "https://example.com/apply/ABC",
            },
            {
                "shortcode": "DEF",
                "title": "Onsite role",
                "telecommuting": False,
                "application_url": "https://example.com/apply/DEF",
            },
            {
                "shortcode": "GHI",
                "title": "Unknown",
                "application_url": "https://example.com/apply/GHI",
            },
        ],
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = WorkableSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["ABC"].remote_type == "remote"
    assert by_id["DEF"].remote_type == "onsite"
    assert by_id["GHI"].remote_type is None  # absent stays None


async def test_discover_normalises_employment_type() -> None:
    payload = {
        "name": "Co",
        "jobs": [
            {
                "shortcode": "A",
                "title": "FT",
                "employment_type": "Full-time",
                "application_url": "https://example.com/A",
            },
            {
                "shortcode": "B",
                "title": "PT",
                "employment_type": "Part-time",
                "application_url": "https://example.com/B",
            },
            {
                "shortcode": "C",
                "title": "Contract",
                "employment_type": "Contract",
                "application_url": "https://example.com/C",
            },
        ],
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/x").mock(
            return_value=httpx.Response(200, json=payload),
        )

        source = WorkableSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["A"].employment_type == "full-time"
    assert by_id["B"].employment_type == "part-time"
    assert by_id["C"].employment_type == "contract"


async def test_discover_composes_location_from_city_state_country() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/huggingface").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = WorkableSource(account="huggingface")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    # First HF fixture job: Paris, Île-de-France, France
    assert jobs[0].location_raw is not None
    assert "Paris" in jobs[0].location_raw
    assert "France" in jobs[0].location_raw


async def test_discover_description_is_none_pending_browser_tier() -> None:
    """Workable's listing API does not expose the description. This is a
    documented gap until Phase 5 adds the browser-tier fetcher to render
    the per-job SPA page.
    """
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/huggingface").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = WorkableSource(account="huggingface")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    for job in jobs:
        assert job.description_html is None
        assert job.description_text is None


async def test_discover_preserves_raw_payload() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/huggingface").mock(
            return_value=httpx.Response(200, text=_FIXTURE),
        )

        source = WorkableSource(account="huggingface")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert "shortcode" in jobs[0].raw_data
    assert "locations" in jobs[0].raw_data


async def test_discover_raises_on_non_2xx_status() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/missing").mock(
            return_value=httpx.Response(404, text="Not Found"),
        )

        source = WorkableSource(account="missing")
        async with HttpFetcher() as fetcher:
            with pytest.raises(WorkableFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.status_code == 404


async def test_discover_handles_empty_jobs() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://apply.workable.com/api/v1/widget/accounts/empty").mock(
            return_value=httpx.Response(200, json={"name": "Co", "jobs": []}),
        )

        source = WorkableSource(account="empty")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs == []
