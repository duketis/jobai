"""Tests for the SmartRecruiters source."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.smartrecruiters import SmartRecruitersFetchError, SmartRecruitersSource

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_LISTING_FIXTURE = (_FIXTURE_DIR / "smartrecruiters_visa_listing.json").read_text(encoding="utf-8")
_DETAIL_FIXTURE = (_FIXTURE_DIR / "smartrecruiters_visa_detail.json").read_text(encoding="utf-8")


def _mock_visa_board(router: respx.Router) -> None:
    """Mock the listing page and reuse the captured Visa detail for every UUID."""
    listing = json.loads(_LISTING_FIXTURE)
    # Patch totalFound so the listing iterator stops after the captured page.
    listing["totalFound"] = len(listing["content"])

    router.get("https://api.smartrecruiters.com/v1/companies/visa/postings").mock(
        return_value=httpx.Response(200, json=listing)
    )

    for entry in listing["content"]:
        url = f"https://api.smartrecruiters.com/v1/companies/visa/postings/{entry['uuid']}"
        router.get(url).mock(return_value=httpx.Response(200, text=_DETAIL_FIXTURE))


def test_source_name_includes_account() -> None:
    source = SmartRecruitersSource(account="visa")
    assert source.name == "smartrecruiters:visa"
    assert source.kind == "smartrecruiters"


async def test_discover_yields_one_job_per_listing_entry() -> None:
    with respx.mock(assert_all_called=False) as router:
        _mock_visa_board(router)

        source = SmartRecruitersSource(account="visa")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    listing = json.loads(_LISTING_FIXTURE)
    assert len(jobs) == len(listing["content"])


async def test_discover_attaches_description_from_detail_endpoint() -> None:
    with respx.mock(assert_all_called=False) as router:
        _mock_visa_board(router)

        source = SmartRecruitersSource(account="visa")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    for job in jobs:
        assert job.description_html is not None
        assert "Job Description" in job.description_html


async def test_discover_uses_company_name_from_listing() -> None:
    with respx.mock(assert_all_called=False) as router:
        _mock_visa_board(router)

        source = SmartRecruitersSource(account="visa")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert all(j.company == "Visa" for j in jobs)


async def test_discover_paginates_listing() -> None:
    """Two listing pages with totalFound=4 should fetch both pages."""
    page_one = {
        "offset": 0,
        "limit": 2,
        "totalFound": 4,
        "content": [
            {
                "uuid": "u-1",
                "name": "Role One",
                "applyUrl": "https://example.com/1",
                "company": {"name": "Co"},
                "location": {"fullLocation": "Sydney"},
            },
            {
                "uuid": "u-2",
                "name": "Role Two",
                "applyUrl": "https://example.com/2",
                "company": {"name": "Co"},
                "location": {"fullLocation": "Sydney"},
            },
        ],
    }
    page_two = {
        "offset": 2,
        "limit": 2,
        "totalFound": 4,
        "content": [
            {
                "uuid": "u-3",
                "name": "Role Three",
                "applyUrl": "https://example.com/3",
                "company": {"name": "Co"},
                "location": {"fullLocation": "Sydney"},
            },
            {
                "uuid": "u-4",
                "name": "Role Four",
                "applyUrl": "https://example.com/4",
                "company": {"name": "Co"},
                "location": {"fullLocation": "Sydney"},
            },
        ],
    }
    detail_template = {"jobAd": {"sections": {}}, "applyUrl": ""}

    with respx.mock(assert_all_called=False) as router:
        # Both pages live on the same path; respx returns them in sequence.
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            side_effect=[
                httpx.Response(200, json=page_one),
                httpx.Response(200, json=page_two),
            ]
        )
        for uuid in ("u-1", "u-2", "u-3", "u-4"):
            router.get(f"https://api.smartrecruiters.com/v1/companies/x/postings/{uuid}").mock(
                return_value=httpx.Response(200, json=detail_template)
            )

        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert [j.source_external_id for j in jobs] == ["u-1", "u-2", "u-3", "u-4"]


async def test_discover_remote_type_from_structured_flags() -> None:
    listing = {
        "offset": 0,
        "limit": 100,
        "totalFound": 3,
        "content": [
            {
                "uuid": "r",
                "name": "Remote",
                "company": {"name": "X"},
                "location": {"remote": True, "hybrid": False, "fullLocation": "Anywhere"},
            },
            {
                "uuid": "h",
                "name": "Hybrid",
                "company": {"name": "X"},
                "location": {"remote": False, "hybrid": True, "fullLocation": "City"},
            },
            {
                "uuid": "o",
                "name": "Onsite",
                "company": {"name": "X"},
                "location": {"remote": False, "hybrid": False, "fullLocation": "City"},
            },
        ],
    }
    detail_template = {"jobAd": {"sections": {}}, "applyUrl": ""}
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            return_value=httpx.Response(200, json=listing),
        )
        for uuid in ("r", "h", "o"):
            router.get(f"https://api.smartrecruiters.com/v1/companies/x/postings/{uuid}").mock(
                return_value=httpx.Response(200, json=detail_template)
            )

        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    assert by_id["r"].remote_type == "remote"
    assert by_id["h"].remote_type == "hybrid"
    assert by_id["o"].remote_type == "onsite"


async def test_discover_raises_on_listing_error() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/missing/postings").mock(
            return_value=httpx.Response(404, text="not found"),
        )

        source = SmartRecruitersSource(account="missing")
        async with HttpFetcher() as fetcher:
            with pytest.raises(SmartRecruitersFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.stage == "listing"
    assert excinfo.value.status_code == 404


async def test_discover_raises_on_detail_error() -> None:
    """A failure on detail fetch should bubble up so the runner records the
    failure on the scrape_runs row."""
    listing = {
        "offset": 0,
        "limit": 100,
        "totalFound": 1,
        "content": [
            {
                "uuid": "u-1",
                "name": "Role",
                "company": {"name": "X"},
                "location": {"fullLocation": "X"},
            }
        ],
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            return_value=httpx.Response(200, json=listing),
        )
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings/u-1").mock(
            return_value=httpx.Response(500, text="server error")
        )

        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            with pytest.raises(SmartRecruitersFetchError) as excinfo:
                async for _ in source.discover(fetcher):
                    pass

    assert excinfo.value.stage == "detail"
    assert excinfo.value.status_code == 500


async def test_discover_handles_empty_board() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/empty/postings").mock(
            return_value=httpx.Response(
                200,
                json={"offset": 0, "limit": 100, "totalFound": 0, "content": []},
            ),
        )

        source = SmartRecruitersSource(account="empty")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]

    assert jobs == []


async def test_discover_skips_listing_entries_without_uuid_or_id() -> None:
    """A listing entry missing both ``uuid`` and ``id`` is unusable;
    the loop continues to the next without crashing."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "offset": 0,
                    "limit": 100,
                    "totalFound": 1,
                    "content": [{"title": "missing-uuid"}],
                },
            ),
        )
        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]
    assert jobs == []


async def test_discover_stops_when_content_is_not_a_list() -> None:
    """A 200 whose ``content`` key is non-list ends the pagination
    rather than crashing on iteration."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            return_value=httpx.Response(
                200,
                json={"offset": 0, "limit": 100, "totalFound": 0, "content": "broken"},
            ),
        )
        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]
    assert jobs == []


async def test_discover_skips_non_dict_content_entries() -> None:
    """Stray non-dict values in ``content`` skip without crashing."""
    listing = {
        "offset": 0,
        "limit": 100,
        "totalFound": 1,
        "content": ["not-a-dict", {"uuid": "u-1", "name": "Engineer"}],
    }
    detail = {
        "id": "u-1",
        "name": "Engineer",
        "company": {"name": "Co"},
        "location": {},
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            return_value=httpx.Response(200, json=listing),
        )
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings/u-1").mock(
            return_value=httpx.Response(200, json=detail),
        )
        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]
    assert len(jobs) == 1


async def test_discover_stops_when_total_found_is_not_int() -> None:
    """If totalFound isn't an int, the loop bails after the first page."""
    listing = {
        "offset": 0,
        "limit": 100,
        "totalFound": "not-an-int",
        "content": [{"uuid": "u-1", "name": "Engineer"}],
    }
    detail = {
        "id": "u-1",
        "name": "Engineer",
        "company": {"name": "Co"},
        "location": {},
    }
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings").mock(
            return_value=httpx.Response(200, json=listing),
        )
        router.get("https://api.smartrecruiters.com/v1/companies/x/postings/u-1").mock(
            return_value=httpx.Response(200, json=detail),
        )
        source = SmartRecruitersSource(account="x")
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in source.discover(fetcher)]
    assert len(jobs) == 1


async def test_discover_raises_when_listing_payload_is_not_dict() -> None:
    """A 200 with a list at the top instead of an envelope raises."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.smartrecruiters.com/v1/companies/wrong/postings").mock(
            return_value=httpx.Response(200, json=["not-an-envelope"]),
        )
        source = SmartRecruitersSource(account="wrong")
        async with HttpFetcher() as fetcher:
            with pytest.raises(SmartRecruitersFetchError):
                async for _ in source.discover(fetcher):
                    pass


def test_extract_label_returns_none_for_non_dict_or_non_string_label() -> None:
    from jobai.sources.smartrecruiters import _extract_label  # noqa: PLC0415

    assert _extract_label(None) is None
    assert _extract_label("not-a-dict") is None
    assert _extract_label({"label": 42}) is None
    assert _extract_label({"label": "Full Time"}) == "full-time"


def test_format_location_falls_back_to_component_join_or_none() -> None:
    from jobai.sources.smartrecruiters import _format_location  # noqa: PLC0415

    assert _format_location(None) is None
    assert _format_location("not-a-dict") is None
    # No fullLocation but city/region/country present.
    assert (
        _format_location({"city": "Sydney", "region": "NSW", "country": "AU"})
        == "Sydney, NSW, AU"
    )
    # Entirely empty dict -> None.
    assert _format_location({}) is None


def test_extract_location_field_handles_missing_and_non_string() -> None:
    from jobai.sources.smartrecruiters import _extract_location_field  # noqa: PLC0415

    assert _extract_location_field({"location": "not-dict"}, "city") is None
    assert _extract_location_field({"location": {"city": 42}}, "city") is None
    assert _extract_location_field({"location": {"city": ""}}, "city") is None
    assert _extract_location_field({"location": {"city": "Sydney"}}, "city") == "Sydney"


def test_remote_type_from_location_handles_every_combination() -> None:
    from jobai.sources.smartrecruiters import _remote_type_from_location  # noqa: PLC0415

    assert _remote_type_from_location(None) is None
    assert _remote_type_from_location({"remote": True}) == "remote"
    assert _remote_type_from_location({"hybrid": True}) == "hybrid"
    assert _remote_type_from_location({"remote": False, "hybrid": False}) == "onsite"
    assert _remote_type_from_location({}) is None


def test_compose_description_html_falls_through_on_missing_blocks() -> None:
    from jobai.sources.smartrecruiters import _compose_description_html  # noqa: PLC0415

    # No jobAd block at all -> None.
    assert _compose_description_html({}) is None
    # jobAd present but sections missing -> None.
    assert _compose_description_html({"jobAd": {}}) is None
    # sections present but every key missing or wrong shape -> None.
    assert _compose_description_html({"jobAd": {"sections": {"x": "not-a-dict"}}}) is None
    # Section with only a title (no text) -> just <h2>.
    out = _compose_description_html(
        {
            "jobAd": {
                "sections": {
                    "companyDescription": {"title": "About", "text": ""},
                    "jobDescription": {"title": "", "text": "Body."},
                },
            },
        },
    )
    assert out is not None
    assert "<h2>About</h2>" in out
    assert "Body." in out
