"""Tests for the Seek source.

Drives :class:`SeekSource` against a minimised fixture cut from a
real Seek search-results page (``data-automation`` selectors are
Seek's documented contract for testing/automation tools, so the
fixture's selector shape mirrors production).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.seek import (
    SeekFetchError,
    SeekSource,
    _href_from_overlay,
    _infer_remote_type,
    _parse_card,
    _parse_salary,
    _strip_query_anchors,
    _to_int,
)

_FIXTURE = (Path(__file__).parent / "fixtures" / "seek_python_au.html").read_text(encoding="utf-8")
_SLUG = "python-jobs/in-All-Australia"
_URL = f"https://www.seek.com.au/{_SLUG}"


def test_seek_source_name_includes_slug() -> None:
    source = SeekSource(account=_SLUG)
    assert source.name == f"seek:{_SLUG}"
    assert source.kind == "seek"


async def test_discover_yields_one_job_per_card() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    assert len(jobs) == 3
    assert {j.source_external_id for j in jobs} == {
        "91899557",
        "91749277",
        "91818594",
    }


async def test_discover_maps_core_fields() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    senior = by_id["91899557"]
    assert senior.title == "Software Engineer"
    assert senior.company == "GTurbo"
    assert senior.apply_url.startswith("https://www.seek.com.au/job/91899557")
    # Tracking params/anchors stripped so dedup keys are stable
    assert "ref=" not in senior.apply_url
    assert "#sol=" not in senior.apply_url
    assert senior.location_country == "Australia"
    assert senior.salary_min == 110_000
    assert senior.salary_max == 130_000
    assert senior.salary_currency == "AUD"


async def test_discover_handles_missing_salary() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    by_id = {j.source_external_id: j for j in jobs}
    no_salary = by_id["91749277"]
    assert no_salary.salary_min is None
    assert no_salary.salary_max is None
    assert no_salary.salary_currency is None


async def test_discover_picks_up_employment_type_from_card_text() -> None:
    """Seek emits 'This is a Full time job' inside each card."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    # Every card in the fixture has "Full time" — verify at least one.
    types = {j.employment_type for j in jobs}
    assert any(t and "full" in t.lower() for t in types)


async def test_discover_includes_classification_in_extra_tags() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(return_value=httpx.Response(200, text=_FIXTURE))
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    # Software Engineer roles in the fixture all sit under
    # Information & Communication Technology -> Engineering - Software.
    senior = next(j for j in jobs if j.source_external_id == "91899557")
    tags_text = " | ".join(senior.extra_tags).lower()
    assert "information" in tags_text or "engineering" in tags_text


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


async def test_discover_returns_empty_on_page_with_no_cards() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(
            return_value=httpx.Response(200, text="<html><body><p>nothing here</p></body></html>"),
        )
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG).discover(fetcher)]

    assert jobs == []


async def test_discover_walks_multiple_pages_and_dedups() -> None:
    """Pages 2..N are walked; jobs already seen on page 1 are skipped.

    Mirrors Seek's real behaviour where the tail of the result set
    sometimes pads with already-shown listings — without dedup we'd
    over-count and over-write the canonical row.
    """

    def page_for(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(200, text=_FIXTURE)
        if page == 2:
            # One new card + one already-seen card from page 1
            return httpx.Response(
                200,
                text=(
                    "<html><body>"
                    '<article data-automation="normalJob" data-job-id="91899557">'
                    '<a data-automation="jobTitle" href="/job/91899557">Software Engineer</a>'
                    '<a data-automation="jobCompany">GTurbo</a>'
                    "</article>"
                    '<article data-automation="normalJob" data-job-id="99999999">'
                    '<a data-automation="jobTitle" href="/job/99999999">New Engineer</a>'
                    '<a data-automation="jobCompany">Other Co</a>'
                    "</article>"
                    "</body></html>"
                ),
            )
        # Page 3 onward: only already-seen cards → walk terminates
        return httpx.Response(
            200,
            text=(
                "<html><body>"
                '<article data-automation="normalJob" data-job-id="91899557">'
                '<a data-automation="jobTitle" href="/job/91899557">Software Engineer</a>'
                '<a data-automation="jobCompany">GTurbo</a>'
                "</article>"
                "</body></html>"
            ),
        )

    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG, max_pages=5).discover(fetcher)]

    ids = {j.source_external_id for j in jobs}
    assert ids == {"91899557", "91749277", "91818594", "99999999"}


async def test_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        SeekSource(account=_SLUG, max_pages=0)


async def test_discover_terminates_silently_on_later_page_failure() -> None:
    """A non-2xx after page 1 ends the walk without raising — we keep
    everything yielded so far."""

    def respond(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(200, text=_FIXTURE)
        return httpx.Response(503)

    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(side_effect=respond)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG, max_pages=3).discover(fetcher)]
    # Page 1 still yielded its three cards.
    assert len(jobs) == 3


# ---------------------------------------------------------------------------
# Helper-function coverage
# ---------------------------------------------------------------------------


def test_strip_query_anchors_drops_query_and_anchor() -> None:
    """Apply URL strips ``?ref=...`` and ``#sol=...`` to keep dedup keys
    stable across runs that capture different referrer params."""
    assert _strip_query_anchors("/job/91899557?ref=search&type=standard#sol=abc") == "/job/91899557"


def test_strip_query_anchors_returns_input_when_path_empty() -> None:
    """An input that urlparse can't extract a path from is passed through
    untouched (Seek emits these for tracking-only links)."""
    assert _strip_query_anchors("?only=query") == "?only=query"


def test_href_from_overlay_returns_none_when_no_overlay() -> None:
    card = HTMLParser('<article data-automation="normalJob"></article>').css_first("article")
    assert card is not None
    assert _href_from_overlay(card) is None


def test_href_from_overlay_picks_up_alt_link_data_automation() -> None:
    card = HTMLParser(
        '<article data-automation="normalJob">'
        '<a data-automation="job-list-view-job-link" href="/job/abc">x</a>'
        "</article>",
    ).css_first("article")
    assert card is not None
    assert _href_from_overlay(card) == "/job/abc"


def test_parse_card_returns_none_when_no_job_id() -> None:
    """Cards without a ``data-job-id`` attribute are not yieldable."""
    card = HTMLParser(
        '<article data-automation="normalJob">'
        '<a data-automation="jobTitle" href="/job/x">Engineer</a>'
        "</article>",
    ).css_first("article")
    assert card is not None
    assert _parse_card(card) is None


def test_parse_card_returns_none_when_no_apply_path() -> None:
    """Card with id+title but no link is unusable downstream."""
    card = HTMLParser(
        '<article data-automation="normalJob" data-job-id="42">'
        '<a data-automation="jobTitle">Engineer</a>'
        "</article>",
    ).css_first("article")
    assert card is not None
    assert _parse_card(card) is None


def test_parse_card_returns_none_when_no_title_node() -> None:
    """Cards with a job id but no ``data-automation='jobTitle'`` element
    are wire-format mismatches; skip rather than crash."""
    card = HTMLParser(
        '<article data-automation="normalJob" data-job-id="42">'
        '<span>no title element</span>'
        "</article>",
    ).css_first("article")
    assert card is not None
    assert _parse_card(card) is None


def test_parse_card_returns_none_when_title_text_is_empty() -> None:
    """A title anchor that exists but is blank (e.g. icon-only) is unusable."""
    card = HTMLParser(
        '<article data-automation="normalJob" data-job-id="42">'
        '<a data-automation="jobTitle" href="/job/x"></a>'
        "</article>",
    ).css_first("article")
    assert card is not None
    assert _parse_card(card) is None


def test_infer_remote_type_picks_up_remote_and_hybrid() -> None:
    assert _infer_remote_type("Remote, Australia") == "remote"
    assert _infer_remote_type("Hybrid - Sydney") == "hybrid"
    assert _infer_remote_type("Sydney NSW") is None
    assert _infer_remote_type(None) is None


def test_parse_salary_handles_single_value_and_unparseable() -> None:
    assert _parse_salary("$95,000 per year") == (95_000, None, "AUD")
    assert _parse_salary("Negotiable") == (None, None, None)


def test_to_int_upscales_thousands_shorthand() -> None:
    """Seek sometimes lists short-form salaries (``$200``) meaning 200k."""
    assert _to_int("250") == 250_000
    assert _to_int("not a number") is None
    assert _to_int("85,000") == 85_000


def test_employment_type_skips_paragraphs_not_starting_with_this_is_a() -> None:
    """``_employment_type_from_card`` loops every <p>; paragraphs that
    aren't the employment-type line must be skipped (loop continuation
    branch) without returning prematurely."""
    from jobai.sources.seek import _employment_type_from_card  # noqa: PLC0415

    card = HTMLParser(
        "<article>"
        "<p>Posted 2 days ago</p>"
        "<p>This is a Full time job</p>"
        "</article>"
    ).css_first("article")
    assert card is not None
    assert _employment_type_from_card(card) == "Full time"


def test_employment_type_returns_none_when_no_paragraph_matches() -> None:
    """All <p> nodes inspected, none match -- returns None (loop completes)."""
    from jobai.sources.seek import _employment_type_from_card  # noqa: PLC0415

    card = HTMLParser("<article><p>x</p><p>y</p></article>").css_first("article")
    assert card is not None
    assert _employment_type_from_card(card) is None


def test_parse_salary_range_with_unparseable_endpoints_falls_through() -> None:
    """The range regex captures ``[\\d,]+`` so a stripped-empty input
    (e.g. lone commas) matches structurally but parses to None on
    _to_int. Both range AND single fallbacks must drop through to
    (None, None, None) -- covers 252->255 and 258->261 branches."""
    from jobai.sources.seek import _parse_salary  # noqa: PLC0415

    # ', to ,' matches the range pattern with both groups = ','; _to_int(',')
    # strips the comma, gets '', returns None. Both inner ifs are False.
    assert _parse_salary("$, to $,") == (None, None, None)


async def test_discover_walks_to_max_pages_when_every_page_has_new_cards() -> None:
    """Force the for-loop in discover() to complete all max_pages iterations
    by giving every page a unique card. Covers the loop-exit branch."""

    def page_for(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        return httpx.Response(
            200,
            text=(
                "<html><body>"
                f'<article data-automation="normalJob" data-job-id="p{page}">'
                f'<a data-automation="jobTitle" href="/job/p{page}">Engineer {page}</a>'
                '<a data-automation="jobCompany">Co</a>'
                "</article>"
                "</body></html>"
            ),
        )

    with respx.mock(assert_all_called=False) as router:
        router.get(_URL).mock(side_effect=page_for)
        async with HttpFetcher() as fetcher:
            jobs = [j async for j in SeekSource(account=_SLUG, max_pages=3).discover(fetcher)]
    assert {j.source_external_id for j in jobs} == {"p1", "p2", "p3"}
