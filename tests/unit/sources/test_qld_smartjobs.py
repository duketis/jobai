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
_URL_PAGE_1 = (
    "https://smartjobs.qld.gov.au/jobtools/jncustomsearch.searchResults?in_organid=14904&in_pg=0"
)
# Pagination beyond page 1 returns the empty results template; the
# walker stops on the first zero-yield page.
_EMPTY_RESPONSE = httpx.Response(
    200,
    text="<html><body><p>no more results</p></body></html>",
)


def test_source_uses_default_orgid_when_account_blank() -> None:
    source = QLDSmartJobsSource(account="")
    assert source.account == "14904"


def test_source_name_includes_orgid() -> None:
    source = QLDSmartJobsSource(account="14904")
    assert source.name == "qld_smartjobs:14904"


def _qld_paged_router(router: respx.Router) -> None:
    """Wire the QLD smartjobs host so page 1 returns the fixture and
    page 2+ returns the empty-results body. The walker stops on the
    first zero-yield page."""
    router.get(host="smartjobs.qld.gov.au").mock(
        side_effect=lambda req: (
            httpx.Response(200, text=_FIXTURE) if "in_pg=0" in str(req.url) else _EMPTY_RESPONSE
        ),
    )


async def test_discover_yields_one_job_per_li() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        _qld_paged_router(router)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in QLDSmartJobsSource(account="14904").discover(fetcher)]

    # Real ids from fixture (extracted from /jobs/QLD-{id}-{year} hrefs)
    assert len(jobs) == 3
    assert all(j.source_external_id.startswith("QLD-") for j in jobs)


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        _qld_paged_router(router)
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
        router.get(_URL_PAGE_1).mock(return_value=httpx.Response(503))
        async with HttpFetcher() as fetcher:
            with pytest.raises(QLDSmartJobsFetchError) as excinfo:
                async for _ in QLDSmartJobsSource(account="14904").discover(fetcher):
                    pass
    assert excinfo.value.status_code == 503


async def test_discover_returns_empty_on_no_cards() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="smartjobs.qld.gov.au").mock(
            return_value=httpx.Response(200, text="<html><body><p>nothing</p></body></html>")
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in QLDSmartJobsSource(account="14904").discover(fetcher)]
    assert jobs == []


async def test_discover_walks_pages_and_dedups() -> None:
    """Pagination is wired: page 1 yields fixture cards, page 2
    re-serves the same fixture (all dupes), so the walker exits
    without duplicating the per-cycle output."""
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        # Both pages return the SAME fixture; the walker should detect
        # zero new IDs on page 2 and exit cleanly.
        router.get(host="smartjobs.qld.gov.au").mock(
            return_value=httpx.Response(200, text=_FIXTURE)
        )
        async with HttpFetcher() as fetcher:
            jobs = [
                j async for j in QLDSmartJobsSource(account="14904", max_pages=5).discover(fetcher)
            ]
    assert len(jobs) == 3  # 3 unique from fixture, page 2's repeats deduped


async def test_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        QLDSmartJobsSource(account="14904", max_pages=0)


async def test_discover_stops_silently_on_mid_walk_failure() -> None:
    """A non-2xx on page 2+ ends the walk; everything yielded on
    earlier pages survives."""

    calls = {"n": 0}

    def page_for(request: httpx.Request) -> httpx.Response:
        del request
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text=_FIXTURE)
        return httpx.Response(500, text="server-side error")

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="smartjobs.qld.gov.au").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [
                j async for j in QLDSmartJobsSource(account="14904", max_pages=5).discover(fetcher)
            ]
    # Page 1's three cards are preserved despite page 2 erroring.
    assert len(jobs) == 3


async def test_discover_walks_to_max_pages_when_every_page_has_new_cards() -> None:
    """A scenario where every page returns unique cards must exhaust
    the for-loop's range (loop-exit branch)."""

    def page_for(request: httpx.Request) -> httpx.Response:
        # Each request returns one unique card. Job ID regex requires
        # /jobs/QLD-<digits>, so embed the offset (which is numeric).
        offset = int(request.url.params.get("in_pg", "0"))
        # Use 100000+offset to avoid collisions with values like '0' that
        # might otherwise stay constant across all pages.
        job_id = 100000 + offset
        body = (
            f"<html><body><ul><li>"
            f'<a href="/jobs/QLD-{job_id}/"><span class="result-title">'
            f"<strong>Engineer {offset}</strong></span></a>"
            f"</li></ul></body></html>"
        )
        return httpx.Response(200, text=body)

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.get(host="smartjobs.qld.gov.au").mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [
                j async for j in QLDSmartJobsSource(account="14904", max_pages=3).discover(fetcher)
            ]
    assert len({j.source_external_id for j in jobs}) == 3


def test_parse_card_returns_none_on_missing_pieces() -> None:
    """The card parser bails on each missing-piece short-circuit
    (anchor absent, blank title, job-id regex no-match)."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.qld_smartjobs import _parse_card  # noqa: PLC0415

    # No QLD anchor at all.
    li = HTMLParser('<li><a href="/somewhere/else">x</a></li>').css_first("li")
    assert li is not None
    assert _parse_card(li) is None

    # Anchor matches selector but has no strong (title None).
    li = HTMLParser('<li><a href="/jobs/QLD-NEAR/"></a></li>').css_first("li")
    assert li is not None
    assert _parse_card(li) is None

    # Job-id regex doesn't match the path.
    li = HTMLParser('<li><a href="/jobs/QLD-/"><strong>Title</strong></a></li>').css_first("li")
    assert li is not None
    assert _parse_card(li) is None


def test_parse_card_returns_none_when_title_text_missing() -> None:
    """An anchor that satisfies the CSS selector AND the job-id regex
    but has no inner strong text leaves title=None and the parser
    bails (line 142)."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.qld_smartjobs import _parse_card  # noqa: PLC0415

    li = HTMLParser('<li><a href="/jobs/QLD-12345/"></a></li>').css_first("li")
    assert li is not None
    assert _parse_card(li) is None


def test_split_title_company_handles_title_with_no_trailing_company() -> None:
    """When the anchor text is exactly the title (no company suffix),
    company is None and the trailing-strip branches don't fire."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.qld_smartjobs import _split_title_company  # noqa: PLC0415

    anchor = HTMLParser(
        '<a><span class="result-title"><strong>Engineer</strong></span></a>'
    ).css_first("a")
    assert anchor is not None
    title, company = _split_title_company(anchor)
    assert title == "Engineer"
    assert company is None


def test_split_title_company_when_full_text_does_not_start_with_title() -> None:
    """Defensive: if anchor outer-text doesn't begin with the inner-strong
    title (e.g. surrounding text gets pulled out by stripping), the
    company-extraction branch silently skips. Covers 187 False branch."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.qld_smartjobs import _split_title_company  # noqa: PLC0415

    # The anchor's full stripped text 'Prefix Engineer' doesn't start
    # with the inner-strong title 'Engineer'; the conditional drops out.
    anchor = HTMLParser("<a>Prefix <strong>Engineer</strong></a>").css_first("a")
    assert anchor is not None
    title, company = _split_title_company(anchor)
    assert title == "Engineer"
    assert company is None


def test_text_helper_returns_none_for_missing_or_blank_node() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.qld_smartjobs import _text  # noqa: PLC0415

    card = HTMLParser("<div><span></span></div>").css_first("div")
    assert card is not None
    # Missing selector -> None.
    assert _text(card, ".missing") is None
    # Selector matches but text is empty -> None.
    assert _text(card, "span") is None


def test_parse_salary_handles_missing_and_unparseable_values() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.qld_smartjobs import _parse_salary  # noqa: PLC0415

    # No salary span -> all None.
    card = HTMLParser("<div></div>").css_first("div")
    assert card is not None
    assert _parse_salary(card) == (None, None, None)

    # Salary span present but the inline script has no sal1/sal2 -> all None.
    card = HTMLParser('<div><span class="salary"><script>nothing</script></span></div>').css_first(
        "div"
    )
    assert card is not None
    assert _parse_salary(card) == (None, None, None)


def test_to_int_drops_empty_and_non_digit_tokens() -> None:
    from jobai.sources.qld_smartjobs import _to_int  # noqa: PLC0415

    assert _to_int(None) is None
    assert _to_int("") is None
    assert _to_int("not-a-number") is None
    assert _to_int("85,000") == 85_000
