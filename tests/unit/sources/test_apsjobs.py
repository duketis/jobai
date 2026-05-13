"""Tests for the APS Jobs (Salesforce Lightning) source."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher
from jobai.sources.apsjobs import APSJobsFetchError, APSJobsSource

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_BOOTSTRAP_HTML = (_FIXTURE_DIR / "apsjobs_bootstrap.html").read_text(encoding="utf-8")
_PAGE1 = (_FIXTURE_DIR / "apsjobs_aura_page1.json").read_text(encoding="utf-8")
_PAGE2 = (_FIXTURE_DIR / "apsjobs_aura_page2.json").read_text(encoding="utf-8")
_EMPTY = (_FIXTURE_DIR / "apsjobs_aura_empty.json").read_text(encoding="utf-8")

_BOOTSTRAP_URL = "https://www.apsjobs.gov.au/s/job-search"
_AURA_URL = "https://www.apsjobs.gov.au/s/sfsites/aura"


def _aura_route(router: respx.MockRouter, *bodies: str) -> respx.Route:
    """Mock the Aura POST and return successive bodies on each call."""
    route = router.post(url__regex=rf"{_AURA_URL}.*")
    route.side_effect = [httpx.Response(200, text=b) for b in bodies]
    return route


def test_apsjobs_source_name_includes_query() -> None:
    source = APSJobsSource(account="software")
    assert source.name == "apsjobs:software"


def test_apsjobs_source_name_omits_query_when_empty() -> None:
    source = APSJobsSource(account="")
    assert source.name == "apsjobs"


async def test_discover_paginates_until_empty() -> None:
    """Bootstrap then walk pages until ``jobListings`` comes back empty."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        _aura_route(router, _PAGE1, _PAGE2, _EMPTY)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]

    assert {j.source_external_id for j in jobs} == {
        "a05OY00000ORdYvYAL",
        "a05OY00000ORVhpYAH",
        "a05OY00000ORTT7YAP",
        "a05OY00000ORTRVYA5",
    }


async def test_discover_dedupes_repeated_listings() -> None:
    """If the API returns the same id twice across pages we yield it once."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        # Same listings on every page until empty terminates the walk.
        _aura_route(router, _PAGE1, _PAGE1, _EMPTY)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]
    assert len(jobs) == 3


async def test_discover_extracts_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        _aura_route(router, _PAGE1, _EMPTY)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    first = by_id["a05OY00000ORdYvYAL"]
    assert first.title == "Program Director"
    assert first.company == "Australian Securities and Investments Commission"
    assert first.apply_url == ("https://www.apsjobs.gov.au/s/job-details?jobId=a05OY00000ORdYvYAL")
    assert first.location_country == "Australia"
    assert first.location_city == "Adelaide SA"
    assert first.posted_at == "2026-05-06"
    assert first.description_html is not None
    assert "ASIC" in first.description_html


async def test_discover_extracts_salary_when_present() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        _aura_route(router, _PAGE1, _EMPTY)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]
    by_id = {j.source_external_id: j for j in jobs}
    salaried = by_id["a05OY00000ORTT7YAP"]
    assert salaried.salary_min == 125_820
    assert salaried.salary_max == 137_127
    assert salaried.salary_currency == "AUD"
    # Listings without salary collapse to None / no currency.
    no_salary = by_id["a05OY00000ORdYvYAL"]
    assert no_salary.salary_min is None
    assert no_salary.salary_currency is None


async def test_discover_normalises_office_arrangement_to_remote_type() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        _aura_route(router, _PAGE1, _EMPTY)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]
    by_id = {j.source_external_id: j for j in jobs}
    # First listing has officeArrangement = "Hybrid"
    assert by_id["a05OY00000ORdYvYAL"].remote_type == "hybrid"
    # Sport Integrity Australia listing is "Flexible" (remote-friendly)
    assert by_id["a05OY00000ORTT7YAP"].remote_type == "remote"


async def test_discover_passes_search_string_through_filter() -> None:
    """A non-empty ``account`` populates the ``filter.searchString`` slot."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        aura_route = _aura_route(router, _EMPTY)
        async with HttpFetcher() as fetcher:
            _ = [j async for j in APSJobsSource(account="python").discover(fetcher)]

        # Decode the form-encoded body and inspect the inner filter
        request = aura_route.calls.last.request
        form = parse_qs(request.content.decode("utf-8"))
        message = json.loads(form["message"][0])
        action_params = message["actions"][0]["params"]
        assert action_params["classname"] == "aps_jobSearchController"
        assert action_params["method"] == "retrieveJobListings"
        inner_filter = json.loads(action_params["params"]["filter"])
        assert inner_filter["searchString"] == "python"
        assert inner_filter["offset"] == 0


async def test_discover_paginates_using_returned_new_offset() -> None:
    """Each page request sends ``offset`` set from the previous ``newOffset``."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        aura_route = _aura_route(router, _PAGE1, _PAGE2, _EMPTY)
        async with HttpFetcher() as fetcher:
            _ = [j async for j in APSJobsSource(account="").discover(fetcher)]

        offsets = []
        for call in aura_route.calls:
            form = parse_qs(call.request.content.decode("utf-8"))
            inner_filter = json.loads(
                json.loads(form["message"][0])["actions"][0]["params"]["params"]["filter"],
            )
            offsets.append(inner_filter["offset"])
    # First page asks offset=0; page 1 returned newOffset=3, page 2 newOffset=4.
    assert offsets == [0, 3, 4]


async def test_discover_raises_when_bootstrap_fails() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(return_value=httpx.Response(500))
        async with HttpFetcher() as fetcher:
            with pytest.raises(APSJobsFetchError) as excinfo:
                async for _ in APSJobsSource(account="").discover(fetcher):
                    pass
    assert excinfo.value.status_code == 500
    assert excinfo.value.stage == "bootstrap"


async def test_discover_raises_when_aura_call_fails() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        router.post(url__regex=rf"{_AURA_URL}.*").mock(
            return_value=httpx.Response(503),
        )
        async with HttpFetcher() as fetcher:
            with pytest.raises(APSJobsFetchError) as excinfo:
                async for _ in APSJobsSource(account="").discover(fetcher):
                    pass
    assert excinfo.value.status_code == 503
    assert excinfo.value.stage.startswith("page-")


async def test_discover_extracts_aura_context_from_link_header() -> None:
    """Salesforce's small first-render HTML omits the inline JS aura
    config but always includes both tokens in the URL-encoded ``Link``
    response header. We must parse them from there too.
    """
    link_header = (
        "</s/sfsites/auraFW/javascript/HEADER_FWUID_999/aura_prod.js>;"
        "rel=preload;as=script;nopush, "
        "</s/sfsites/l/%7B%22mode%22%3A%22PROD%22%2C%22fwuid%22%3A%22"
        "HEADER_FWUID_999"
        "%22%2C%22loaded%22%3A%7B%22APPLICATION%40markup%3A%2F%2F"
        "siteforce%3AcommunityApp%22%3A%22HEADER_APP_TOKEN_888%22%7D%7D"
        "/resources.js>;rel=preload;as=script;nopush"
    )
    body_without_aura = "<html><body>maintenance</body></html>"
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(
                200,
                text=body_without_aura,
                headers={"Link": link_header},
            ),
        )
        aura_route = _aura_route(router, _EMPTY)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]

        request = aura_route.calls.last.request
        form = parse_qs(request.content.decode("utf-8"))
        aura_context = json.loads(form["aura.context"][0])

    assert jobs == []
    assert aura_context["fwuid"] == "HEADER_FWUID_999"
    assert (
        aura_context["loaded"]["APPLICATION@markup://siteforce:communityApp"]
        == "HEADER_APP_TOKEN_888"
    )


async def test_discover_raises_when_bootstrap_missing_aura_context() -> None:
    """A bootstrap HTML lacking fwuid surfaces a clear error, not a regex crash."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text="<html>maintenance</html>"),
        )
        async with HttpFetcher() as fetcher:
            with pytest.raises(APSJobsFetchError) as excinfo:
                async for _ in APSJobsSource(account="").discover(fetcher):
                    pass
    assert excinfo.value.stage == "bootstrap-parse"


def test_decode_aura_response_raises_on_non_success_state() -> None:
    """When the inner action state isn't SUCCESS, _decode_aura_response
    surfaces an APSJobsFetchError with stage='apex'."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from jobai.fetcher.base import Response  # noqa: PLC0415
    from jobai.sources.apsjobs import (  # noqa: PLC0415
        APSJobsFetchError,
        _decode_aura_response,
    )

    body = json.dumps(
        {
            "actions": [
                {
                    "state": "ERROR",
                    "error": [{"message": "boom"}],
                },
            ],
        },
    )
    response = Response(
        url="https://apsjobs.example/aura",
        status_code=200,
        headers={},
        body=body.encode("utf-8"),
        fetched_at=datetime.now(tz=UTC),
    )
    with pytest.raises(APSJobsFetchError) as excinfo:
        _decode_aura_response(response)
    assert excinfo.value.stage == "apex"


def test_parse_listing_returns_none_when_missing_id_or_title() -> None:
    from jobai.sources.apsjobs import _parse_listing  # noqa: PLC0415

    assert _parse_listing({}) is None
    assert _parse_listing({"jobId": "1"}) is None
    assert _parse_listing({"jobName": "Engineer"}) is None
    assert _parse_listing({"jobId": "1", "jobName": "Engineer"}) is not None


def test_join_html_returns_none_when_every_part_is_empty() -> None:
    from jobai.sources.apsjobs import _join_html  # noqa: PLC0415

    assert _join_html(None, None, "") is None
    assert _join_html("a", None, "b") == "a\nb"


def test_first_segment_returns_none_for_empty_or_comma_only_input() -> None:
    from jobai.sources.apsjobs import _first_segment  # noqa: PLC0415

    assert _first_segment(None) is None
    assert _first_segment("") is None
    assert _first_segment("Canberra, ACT") == "Canberra"


def test_to_int_returns_none_on_unparseable_input() -> None:
    from jobai.sources.apsjobs import _to_int  # noqa: PLC0415

    assert _to_int(None) is None
    assert _to_int("not-a-number") is None
    assert _to_int(125_820.0) == 125_820


def test_normalise_arrangement_handles_empty_and_unknown_strings() -> None:
    from jobai.sources.apsjobs import _normalise_arrangement  # noqa: PLC0415

    assert _normalise_arrangement(None) is None
    assert _normalise_arrangement("") is None
    # An unknown token alone returns None.
    assert _normalise_arrangement("Telepathic") is None
    assert _normalise_arrangement("Hybrid") == "hybrid"
    assert _normalise_arrangement("Flexible") == "remote"
    assert _normalise_arrangement("On Site") == "onsite"


async def test_discover_exhausts_page_cap_when_every_page_returns_new_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard _MAX_PAGES ceiling exists so a runaway pagination loop
    can't spin forever. Shrink it to 2 and feed two pages of unique
    listings so the for-loop completes all iterations (exit branch)."""
    from jobai.sources import apsjobs as apsjobs_mod  # noqa: PLC0415

    monkeypatch.setattr(apsjobs_mod, "_MAX_PAGES", 2)

    def _page(job_id: str, new_offset: int) -> dict[str, object]:
        return {
            "actions": [
                {
                    "state": "SUCCESS",
                    "returnValue": {
                        "returnValue": {
                            "jobListingCount": 1,
                            "newOffset": new_offset,
                            "jobListings": [
                                {
                                    "jobId": job_id,
                                    "jobName": f"Engineer {job_id}",
                                    "jobLocation": "Canberra",
                                },
                            ],
                        },
                    },
                },
            ],
        }

    page = _page("unique-1", new_offset=1)
    page2 = _page("unique-2", new_offset=2)
    with respx.mock(assert_all_called=False) as router:
        router.get(_BOOTSTRAP_URL).mock(
            return_value=httpx.Response(200, text=_BOOTSTRAP_HTML),
        )
        _aura_route(router, json.dumps(page), json.dumps(page2))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in APSJobsSource(account="").discover(fetcher)]
    # Both pages yielded distinct jobs; loop hit the _MAX_PAGES ceiling.
    assert {j.source_external_id for j in jobs} == {"unique-1", "unique-2"}
