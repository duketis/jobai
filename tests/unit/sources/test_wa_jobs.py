"""Tests for the WA search.jobs.wa.gov.au source."""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.wa_jobs import WAJobsSource, _parse_row

_FIXTURE = (Path(__file__).parent / "fixtures" / "wa_jobs.html").read_text(encoding="utf-8")


def test_source_name_includes_default_path() -> None:
    source = WAJobsSource(account="")
    assert source.account == "page.php?pageID=215"


def test_parse_row_extracts_advert_id() -> None:
    tree = HTMLParser(_FIXTURE)
    rows = tree.css("tr.oddrow, tr.evenrow")
    parsed = [_parse_row(r) for r in rows]
    by_id = {p.source_external_id: p for p in parsed if p}
    # AdvertID values from the captured fixture
    assert "408414" in by_id
    role = by_id["408414"]
    assert "Coordinator" in role.title
    assert role.apply_url.startswith("https://search.jobs.wa.gov.au/page.php")
    assert role.location_country == "Australia"


def test_parse_row_returns_none_when_no_advert_id() -> None:
    bad = (
        '<tr class="oddrow"><td data-fieldname="Job title">'
        '<a href="page.php?nope=1">Title</a></td></tr>'
    )
    row = HTMLParser(f"<table>{bad}</table>").css_first("tr.oddrow")
    assert row is not None
    assert _parse_row(row) is None


async def test_discover_rejects_non_browser_fetcher() -> None:
    async with HttpFetcher() as fetcher:
        with pytest.raises(TypeError, match="run_in_page"):
            async for _ in WAJobsSource().discover(fetcher):
                pass
