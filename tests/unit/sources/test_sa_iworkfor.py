"""Tests for the SA iworkfor.sa.gov.au source."""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.sa_iworkfor import SAIWorkForSource, _parse_row

_FIXTURE = (Path(__file__).parent / "fixtures" / "sa_iworkfor.html").read_text(encoding="utf-8")


def test_source_name_includes_default_path() -> None:
    source = SAIWorkForSource(account="")
    assert source.account == "jb/list/all"
    assert source.name == "sa_iworkfor:jb/list/all"


def test_parse_row_extracts_id_title_agency() -> None:
    tree = HTMLParser(_FIXTURE)
    rows = tree.css("tr.oddrow, tr.evenrow")
    parsed = [_parse_row(r) for r in rows]
    by_id = {p.source_external_id: p for p in parsed if p}
    assert "607195" in by_id
    correctional = by_id["607195"]
    assert correctional.title == "Correctional Officer"
    assert "Correctional Services" in correctional.company
    assert correctional.apply_url.startswith("https://iworkfor.sa.gov.au/jb/page/")
    assert correctional.location_country == "Australia"


def test_parse_row_returns_none_when_title_missing() -> None:
    bad = '<tr class="oddrow"><td data-fieldname="Reference No">123</td></tr>'
    row = HTMLParser(f"<table>{bad}</table>").css_first("tr.oddrow")
    assert row is not None
    assert _parse_row(row) is None


async def test_discover_rejects_non_browser_fetcher() -> None:
    async with HttpFetcher() as fetcher:
        with pytest.raises(TypeError, match="run_in_page"):
            async for _ in SAIWorkForSource().discover(fetcher):
                pass
