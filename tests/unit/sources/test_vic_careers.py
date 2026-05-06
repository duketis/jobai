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
    _submit_search_form,
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


async def test_submit_search_form_returns_when_button_missing() -> None:
    """If the Search button isn't on the page, ``_submit_search_form``
    returns early so the caller still gets the rendered DOM."""

    class _NoButtonPage:
        async def click(self, *_args: object, **_kwargs: object) -> None:
            msg = "no button"
            raise RuntimeError(msg)

        async def wait_for_selector(
            self, *_args: object, **_kwargs: object
        ) -> None:  # pragma: no cover - never reached
            return

    await _submit_search_form(_NoButtonPage())  # type: ignore[arg-type]
