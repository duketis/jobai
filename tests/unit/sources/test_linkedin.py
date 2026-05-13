"""Tests for the LinkedIn (guest mode) source.

Driven against a real fixture cut from a live LinkedIn guest search
results page (3 representative ``base-card`` entries).
"""

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

# Real ids from the captured fixture
_FIXTURE_IDS = {"4137058028", "4409734067", "4410548782"}


def _only_first_page(request: httpx.Request) -> httpx.Response:
    """Serve the captured fixture for page 0; empty for pages > 0.

    LinkedinSource paginates via ``&start=N``; default tests only
    care about page-0 behaviour, so we short-circuit further pages
    to terminate the walk after one round.
    """
    if "start=" in str(request.url):
        return httpx.Response(200, text="<html><body></body></html>")
    return httpx.Response(200, text=_FIXTURE)


def test_linkedin_source_name_includes_query() -> None:
    source = LinkedInSource(account=_QUERY)
    assert source.name == f"linkedin:{_QUERY}"


def test_build_query_url_encodes_inputs() -> None:
    assert build_query(keywords="python", location="Australia") == _QUERY
    # urlencode uses '+' for spaces in form-encoded queries.
    assert "senior+python" in build_query(keywords="senior python", location="Australia")


async def test_discover_yields_one_job_per_card() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="www.linkedin.com", path="/jobs/search").mock(side_effect=_only_first_page)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]

    assert {j.source_external_id for j in jobs} == _FIXTURE_IDS


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="www.linkedin.com", path="/jobs/search").mock(side_effect=_only_first_page)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    one = by_id["4137058028"]
    assert one.title  # whatever LinkedIn surfaced; non-empty
    assert one.company  # non-empty
    # LinkedIn returns regional hosts (au.linkedin.com for AU searches).
    assert ".linkedin.com/jobs/view/" in one.apply_url
    assert one.location_country == "Australia"


async def test_discover_canonicalises_apply_url() -> None:
    """Tracking params (``refId``, ``trackingId``, ``position``) stripped."""
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="www.linkedin.com", path="/jobs/search").mock(side_effect=_only_first_page)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY).discover(fetcher)]

    for j in jobs:
        assert "?" not in j.apply_url, f"query params not stripped: {j.apply_url}"
        assert "#" not in j.apply_url


async def test_discover_skips_cards_missing_id_or_title() -> None:
    """Synthetic check that the parser is defensive on degenerate input."""
    bad = (
        "<html><body><ul>"
        '<li><div class="base-card">'  # no urn, no title
        '<a class="base-card__full-link" href="/jobs/view/no-id">x</a>'
        "</div></li>"
        '<li><div class="base-card" data-entity-urn="urn:li:jobPosting:1">'
        '<a class="base-card__full-link" href="/jobs/view/x-1"></a>'
        '<h3 class="base-search-card__title"></h3>'  # empty title
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


async def test_discover_walks_multiple_pages_and_dedups() -> None:
    """LinkedIn paginates via ``&start=N`` (offset, 25 per page)."""

    def page_for(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        if start == 0:
            return httpx.Response(200, text=_FIXTURE)
        if start == 25:
            # One new card; rest are dups of page-0 ids
            return httpx.Response(
                200,
                text=(
                    "<html><body><ul>"
                    '<li><div class="base-card" data-entity-urn="urn:li:jobPosting:9999999999">'
                    '<a class="base-card__full-link" href="/jobs/view/new-9999999999">x</a>'
                    '<h3 class="base-search-card__title">New Role</h3>'
                    '<h4 class="base-search-card__subtitle">Other Co</h4>'
                    '<span class="job-search-card__location">Sydney, Australia</span>'
                    "</div></li>"
                    "</ul></body></html>"
                ),
            )
        # Pages from start=50 onward serve nothing → walk terminates
        return httpx.Response(200, text="<html><body></body></html>")

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="www.linkedin.com", path="/jobs/search").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY, max_pages=4).discover(fetcher)]

    ids = {j.source_external_id for j in jobs}
    assert ids == _FIXTURE_IDS | {"9999999999"}


async def test_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        LinkedInSource(account=_QUERY, max_pages=0)


async def test_discover_stops_silently_on_mid_walk_failure() -> None:
    """A non-2xx after page 0 ends the walk; everything yielded so far survives."""

    calls = {"n": 0}

    def page_for(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text=_FIXTURE)
        return httpx.Response(500, text="server-side")

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="www.linkedin.com", path="/jobs/search").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY, max_pages=5).discover(fetcher)]
    # Page 0's fixture jobs are preserved despite the mid-walk failure.
    assert len(jobs) > 0


async def test_discover_walks_to_max_pages_when_every_page_has_new_cards() -> None:
    """Force the for loop in discover() to exhaust all max_pages by
    giving every page a unique URN."""

    def page_for(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        body = (
            "<html><body><ul>"
            f'<li><div class="base-card" data-entity-urn="urn:li:jobPosting:{start}">'
            f'<a class="base-card__full-link" href="/jobs/view/{start}">x</a>'
            f'<h3 class="base-search-card__title">R {start}</h3>'
            '<h4 class="base-search-card__subtitle">Co</h4>'
            '<span class="job-search-card__location">Sydney</span>'
            "</div></li></ul></body></html>"
        )
        return httpx.Response(200, text=body)

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="www.linkedin.com", path="/jobs/search").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in LinkedInSource(account=_QUERY, max_pages=2).discover(fetcher)]
    assert len(jobs) == 2


def test_extract_job_id_falls_back_to_href_when_urn_missing() -> None:
    """A card without a valid urn but with a viewable href returns the
    id pulled from the URL path. Covers the 182->184 + 187 branches."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.linkedin import _extract_job_id  # noqa: PLC0415

    card = HTMLParser(
        '<div class="base-card" data-entity-urn="not-a-urn">'
        '<a class="base-card__full-link" href="/jobs/view/role-77777">x</a>'
        "</div>"
    ).css_first("div")
    assert card is not None
    assert _extract_job_id(card) == "77777"


def test_extract_job_id_returns_none_when_both_urn_and_href_missing() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.linkedin import _extract_job_id  # noqa: PLC0415

    card = HTMLParser('<div class="base-card"></div>').css_first("div")
    assert card is not None
    assert _extract_job_id(card) is None


def test_text_helper_handles_missing_node_and_blank_text() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.linkedin import _text  # noqa: PLC0415

    card = HTMLParser("<div><span></span></div>").css_first("div")
    assert card is not None
    assert _text(card, ".missing") is None
    assert _text(card, "span") is None
    card = HTMLParser("<div><span>hi</span></div>").css_first("div")
    assert card is not None
    assert _text(card, "span") == "hi"


def test_href_helper_returns_none_for_missing_node() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.linkedin import _href  # noqa: PLC0415

    card = HTMLParser("<div></div>").css_first("div")
    assert card is not None
    assert _href(card, "a") is None


def test_attr_helper_returns_none_for_missing_node() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.linkedin import _attr  # noqa: PLC0415

    card = HTMLParser("<div></div>").css_first("div")
    assert card is not None
    assert _attr(card, "a", "href") is None


def test_country_from_recognises_au_us_uk_and_falls_through() -> None:
    from jobai.sources.linkedin import _country_from  # noqa: PLC0415

    assert _country_from(None) is None
    assert _country_from("") is None
    assert _country_from("Sydney, Australia") == "Australia"
    assert _country_from("San Francisco, United States") == "United States"
    assert _country_from("Greater London Area, UK") == "United Kingdom"
    # Unknown country -> None.
    assert _country_from("Tokyo, Japan") is None


def test_city_from_handles_empty_input() -> None:
    from jobai.sources.linkedin import _city_from  # noqa: PLC0415

    assert _city_from(None) is None
    assert _city_from("") is None
    assert _city_from(",   ") is None
    assert _city_from("Sydney, AU") == "Sydney"


def test_remote_from_returns_none_for_empty_or_no_keyword() -> None:
    from jobai.sources.linkedin import _remote_from  # noqa: PLC0415

    assert _remote_from(None) is None
    assert _remote_from("") is None
    assert _remote_from("Sydney NSW") is None
    assert _remote_from("Remote, AU") == "remote"
    assert _remote_from("Hybrid - Sydney") == "hybrid"
