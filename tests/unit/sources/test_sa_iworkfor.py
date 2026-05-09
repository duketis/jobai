"""Tests for the SA iworkfor.sa.gov.au source."""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.sa_iworkfor import (
    SAIWorkForFetchError,
    SAIWorkForSource,
    _parse_row,
    _walk_all_pages,
)
from tests.unit.sources._browser_fakes import FakeBrowserFetcher, html_response

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


async def test_discover_yields_jobs_through_run_in_page() -> None:
    fetcher = FakeBrowserFetcher(html_response(_FIXTURE))
    jobs = [j async for j in SAIWorkForSource().discover(fetcher)]
    assert jobs, "fixture should yield at least one job"
    # Same row appearing twice in the table is yielded only once.
    assert len({j.source_external_id for j in jobs}) == len(jobs)
    assert fetcher.calls == [
        "https://iworkfor.sa.gov.au/jb/list/all",
    ]


async def test_discover_raises_on_non_2xx() -> None:
    fetcher = FakeBrowserFetcher(html_response("<html/>", status_code=503))
    with pytest.raises(SAIWorkForFetchError) as excinfo:
        async for _ in SAIWorkForSource().discover(fetcher):
            pass
    assert excinfo.value.status_code == 503


async def test_walk_all_pages_swallows_missing_button_and_selector() -> None:
    """The walker is best-effort — if the search button or result
    selector isn't present (e.g. site UI redesign) the script must
    return rather than raise, so the caller still gets the partially-
    rendered DOM. Same as the previous _submit_search_form contract,
    just on the new pagination-walker entrypoint."""

    class _DeadPage:
        async def click(self, *_args: object, **_kwargs: object) -> None:
            msg = "no such selector"
            raise RuntimeError(msg)

        async def wait_for_selector(self, *_args: object, **_kwargs: object) -> None:
            msg = "timed out"
            raise RuntimeError(msg)

        async def eval_on_selector_all(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> list[str]:
            return []

    # Should not raise.
    script = _walk_all_pages(max_pages=3)
    await script(_DeadPage())  # type: ignore[arg-type]
