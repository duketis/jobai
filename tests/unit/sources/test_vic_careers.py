"""Tests for the VIC jobs.careers.vic.gov.au source.

VICCareersSource requires a BrowserFetcher (form-fill workflow).
The tests parse a captured results-table fixture by invoking
``_parse_row`` directly — exercising end-to-end via TestClient
would mean booting a real Chromium, which is what the integration
test (post-tag) covers separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.vic_careers import (
    VICCareersFetchError,
    VICCareersSource,
    _parse_row,
    _parse_salary,
    _walk_all_pages,
)
from tests.unit.sources._browser_fakes import FakeBrowserFetcher, html_response

_FIXTURE = (Path(__file__).parent / "fixtures" / "vic_careers.html").read_text(encoding="utf-8")


def test_source_name_includes_orgid() -> None:
    source = VICCareersSource(account="14123")
    assert source.name == "vic_careers:14123"


def test_source_uses_default_orgid_when_blank() -> None:
    source = VICCareersSource(account="")
    assert source.account == "14123"


def test_parse_row_extracts_id_and_title() -> None:
    tree = HTMLParser(_FIXTURE)
    rows = tree.css("tr.odd, tr.even")
    parsed = [_parse_row(r) for r in rows]
    by_id = {p.source_external_id: p for p in parsed if p is not None}
    assert "226530447" in by_id
    assert "226473543" in by_id
    assert "226530398" in by_id
    senior = by_id["226530447"]
    assert "Plumbing" in senior.title
    assert senior.apply_url.startswith("https://jobs.careers.vic.gov.au/jobs/VG-")


def test_parse_row_maps_agency_and_location() -> None:
    tree = HTMLParser(_FIXTURE)
    parsed = [_parse_row(r) for r in tree.css("tr.odd, tr.even")]
    senior = next(p for p in parsed if p and p.source_external_id == "226530447")
    assert senior.company  # whatever VIC put in column 4
    assert senior.location_raw is not None
    assert senior.location_country == "Australia"


def test_parse_row_returns_none_on_short_row() -> None:
    bad = '<table><tr class="odd"><td>only one cell</td></tr></table>'
    row = HTMLParser(bad).css_first("tr.odd")
    assert row is not None
    assert _parse_row(row) is None


def test_parse_salary_handles_see_advertisement() -> None:
    assert _parse_salary("See Advertisement") == (None, None, None)
    assert _parse_salary(None) == (None, None, None)
    assert _parse_salary("") == (None, None, None)


def test_parse_salary_extracts_range() -> None:
    assert _parse_salary("$70,000 - $90,000") == (70_000, 90_000, "AUD")


async def test_discover_rejects_non_browser_fetcher() -> None:
    """The form-fill workflow needs run_in_page; HTTP-only fails fast."""
    async with HttpFetcher() as fetcher:
        with pytest.raises(TypeError, match="run_in_page"):
            async for _ in VICCareersSource(account="14123").discover(fetcher):
                pass


async def test_discover_yields_jobs_through_run_in_page() -> None:
    fetcher = FakeBrowserFetcher(html_response(_FIXTURE))
    jobs = [j async for j in VICCareersSource(account="14123").discover(fetcher)]
    assert jobs
    assert fetcher.calls == [
        "https://jobs.careers.vic.gov.au/jobtools/jncustomsearch.jobsearch?in_organid=14123",
    ]


async def test_discover_raises_on_non_2xx() -> None:
    fetcher = FakeBrowserFetcher(html_response("<html/>", status_code=502))
    with pytest.raises(VICCareersFetchError) as excinfo:
        async for _ in VICCareersSource().discover(fetcher):
            pass
    assert excinfo.value.status_code == 502


async def test_walk_all_pages_returns_when_button_missing() -> None:
    """If the Search button isn't on the page, the walker returns
    early via the bare ``except`` after page.click — the caller still
    gets the partially-rendered DOM. Same contract as the previous
    _submit_search_form smoke; just on the new walker entrypoint."""

    class _NoButtonPage:
        async def click(self, *_args: object, **_kwargs: object) -> None:
            msg = "no button"
            raise RuntimeError(msg)

        async def wait_for_selector(
            self, *_args: object, **_kwargs: object
        ) -> None:  # pragma: no cover - never reached
            return

    script = _walk_all_pages(max_pages=3)
    await script(_NoButtonPage())  # type: ignore[arg-type]


def test_vic_careers_max_pages_validation() -> None:
    """max_pages must be >= 1; zero raises ValueError."""
    with pytest.raises(ValueError, match="max_pages"):
        VICCareersSource(max_pages=0)


def test_vic_careers_discover_requires_run_in_page() -> None:
    """A fetcher without run_in_page (eg HttpFetcher) raises TypeError."""
    import asyncio  # noqa: PLC0415

    class _HttpOnly:
        pass

    async def _runner() -> None:
        async for _ in VICCareersSource().discover(_HttpOnly()):  # type: ignore[arg-type]
            pass

    with pytest.raises(TypeError):
        asyncio.run(_runner())


def test_vic_parse_row_returns_none_for_missing_id_or_title() -> None:
    """Cells missing the in_select checkbox / title anchor / href fall
    through to None."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.vic_careers import _parse_row  # noqa: PLC0415

    # 7 cells but no in_select checkbox.
    row = HTMLParser(
        "<table><tr class='odd'>" + "".join("<td></td>" for _ in range(7)) + "</tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None

    # 7 cells with checkbox but no title anchor.
    row = HTMLParser(
        "<table><tr class='odd'>"
        '<td><input name="in_select" value="42"/></td>' + "<td></td>" * 6 + "</tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None

    # Title anchor exists but with no href.
    row = HTMLParser(
        "<table><tr class='odd'>"
        '<td><input name="in_select" value="42"/></td>'
        "<td><a>Engineer</a></td>" + "<td></td>" * 5 + "</tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None

    # Title anchor with empty title text but a real href -> blank title
    # triggers the ``if not title or not apply_path`` False branch.
    row = HTMLParser(
        "<table><tr class='odd'>"
        '<td><input name="in_select" value="42"/></td>'
        '<td><a href="/jobs/x"></a></td>' + "<td></td>" * 5 + "</tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None


def test_vic_cell_text_returns_none_when_idx_out_of_range() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.vic_careers import _cell_text  # noqa: PLC0415

    cells = list(HTMLParser("<table><tr><td>x</td></tr></table>").css("td"))
    assert _cell_text(cells, 0) == "x"
    assert _cell_text(cells, 99) is None


def test_vic_first_segment_handles_empty_input() -> None:
    from jobai.sources.vic_careers import _first_segment  # noqa: PLC0415

    assert _first_segment(None) is None
    assert _first_segment("") is None
    assert _first_segment(",") is None
    assert _first_segment("Melbourne, VIC") == "Melbourne"


def test_vic_parse_salary_skips_advertisement_and_partial_ranges() -> None:
    from jobai.sources.vic_careers import _parse_salary  # noqa: PLC0415

    assert _parse_salary(None) == (None, None, None)
    assert _parse_salary("Salary commensurate with advertisement") == (None, None, None)
    assert _parse_salary("no salary info here") == (None, None, None)
    # Range with unparseable endpoints (commas only) -> None.
    assert _parse_salary("$, - $,") == (None, None, None)


def test_vic_to_int_returns_none_and_upscales_shorthand() -> None:
    from jobai.sources.vic_careers import _to_int  # noqa: PLC0415

    assert _to_int("not-a-number") is None
    assert _to_int("85") == 85_000
    assert _to_int("85,000") == 85_000


async def test_vic_discover_dedups_repeated_rows() -> None:
    """The seen_ids set guards against duplicate row HTML being emitted
    twice (the JS-driven walker theoretically could). Exercises the
    ``continue`` branch at line 120."""
    from jobai.fetcher.base import Response  # noqa: PLC0415

    class _DupBrowserFetcher:
        async def aclose(self) -> None:
            return None

        async def run_in_page(self, *_args: object, **_kwargs: object) -> Response:
            from datetime import UTC, datetime  # noqa: PLC0415

            # Two rows with the SAME in_select value.
            html = (
                "<html><body><table>"
                "<tr class='odd'>"
                '<td><input name="in_select" value="DUPE-1"/></td>'
                '<td><a href="/jobs/dupe-1">Engineer</a></td>' + "<td></td>" * 5 + "</tr>"
                "<tr class='even'>"
                '<td><input name="in_select" value="DUPE-1"/></td>'
                '<td><a href="/jobs/dupe-1">Engineer</a></td>' + "<td></td>" * 5 + "</tr>"
                "</table></body></html>"
            )
            return Response(
                url="https://x",
                status_code=200,
                headers={},
                body=html.encode("utf-8"),
                fetched_at=datetime.now(tz=UTC),
            )

    jobs = []
    async for job in VICCareersSource().discover(_DupBrowserFetcher()):  # type: ignore[arg-type]
        jobs.append(job)
    assert len(jobs) == 1
