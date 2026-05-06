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

from jobai.fetcher.http import HttpFetcher
from jobai.sources.seek import SeekFetchError, SeekSource

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
