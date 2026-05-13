"""Tests for the WA search.jobs.wa.gov.au source."""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from jobai.fetcher.http import HttpFetcher
from jobai.sources.wa_jobs import (
    WAJobsFetchError,
    WAJobsSource,
    _parse_row,
    _walk_all_pages,
)
from tests.unit.sources._browser_fakes import FakeBrowserFetcher, html_response

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


async def test_discover_yields_jobs_through_run_in_page() -> None:
    fetcher = FakeBrowserFetcher(html_response(_FIXTURE))
    jobs = [j async for j in WAJobsSource().discover(fetcher)]
    assert jobs
    assert len({j.source_external_id for j in jobs}) == len(jobs)
    assert fetcher.calls == [
        "https://search.jobs.wa.gov.au/page.php?pageID=215",
    ]


async def test_discover_raises_on_non_2xx() -> None:
    fetcher = FakeBrowserFetcher(html_response("<html/>", status_code=500))
    with pytest.raises(WAJobsFetchError) as excinfo:
        async for _ in WAJobsSource().discover(fetcher):
            pass
    assert excinfo.value.status_code == 500


async def test_walk_all_pages_swallows_missing_button_and_selector() -> None:
    """Same best-effort contract as the previous _submit_search_form
    smoke - walker must not raise on a dead page (UI redesign)."""

    class _DeadPage:
        async def click(self, *_args: object, **_kwargs: object) -> None:
            msg = "no button"
            raise RuntimeError(msg)

        async def wait_for_selector(self, *_args: object, **_kwargs: object) -> None:
            msg = "timed out"
            raise RuntimeError(msg)

        async def eval_on_selector_all(self, *_args: object, **_kwargs: object) -> list[str]:
            return []

    script = _walk_all_pages(max_pages=3)
    await script(_DeadPage())  # type: ignore[arg-type]


def test_wa_jobs_max_pages_validation() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        WAJobsSource(max_pages=0)


def test_wa_jobs_discover_requires_run_in_page() -> None:
    import asyncio  # noqa: PLC0415

    class _HttpOnly:
        pass

    async def _runner() -> None:
        async for _ in WAJobsSource().discover(_HttpOnly()):  # type: ignore[arg-type]
            pass

    with pytest.raises(TypeError):
        asyncio.run(_runner())


def test_wa_parse_row_returns_none_for_missing_pieces() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.wa_jobs import _parse_row  # noqa: PLC0415

    # No Job title cell + an unlabeled <td> so the data-fieldname-skip
    # branch fires.
    row = HTMLParser("<table><tr class='oddrow'><td>no-fieldname</td></tr></table>").css_first("tr")
    assert row is not None
    assert _parse_row(row) is None

    # Title cell present but anchor missing.
    row = HTMLParser(
        "<table><tr class='oddrow'><td data-fieldname=\"Job title\">Engineer</td></tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None

    # Anchor present but empty text / blank href.
    row = HTMLParser(
        "<table><tr class='oddrow'><td data-fieldname=\"Job title\"><a></a></td></tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None

    # Anchor href doesn't match the advert-id regex.
    row = HTMLParser(
        "<table><tr class='oddrow'>"
        '<td data-fieldname="Job title"><a href="/no-advert-id/here">Engineer</a></td>'
        "</tr></table>"
    ).css_first("tr")
    assert row is not None
    assert _parse_row(row) is None


def test_wa_cell_text_returns_none_for_missing_or_blank_cell() -> None:
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.wa_jobs import _cell_text  # noqa: PLC0415

    # Empty dict -> None.
    assert _cell_text({}, "Agency") is None

    # Cell present but text empty -> None.
    cell = HTMLParser("<table><tr><td></td></tr></table>").css_first("td")
    assert cell is not None
    assert _cell_text({"Agency": cell}, "Agency") is None


def test_wa_cell_text_strips_mobile_field_prefix() -> None:
    """WA's mobile responsive views prefix each cell text with
    ``"FieldName :"``; the helper strips it."""
    from selectolax.parser import HTMLParser  # noqa: PLC0415

    from jobai.sources.wa_jobs import _cell_text  # noqa: PLC0415

    cell = HTMLParser("<table><tr><td>Agency : Department of Health</td></tr></table>").css_first(
        "td"
    )
    assert cell is not None
    assert _cell_text({"Agency": cell}, "Agency") == "Department of Health"

    # When the prefix isn't there, the text passes through untouched.
    cell = HTMLParser("<table><tr><td>Department of Health</td></tr></table>").css_first("td")
    assert cell is not None
    assert _cell_text({"Agency": cell}, "Agency") == "Department of Health"


async def test_wa_jobs_discover_dedups_repeated_rows() -> None:
    """Two rows with the same advert id collapse to one job."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from jobai.fetcher.base import Response  # noqa: PLC0415

    class _DupFetcher:
        async def aclose(self) -> None:
            return None

        async def run_in_page(self, *_args: object, **_kwargs: object) -> Response:
            cell = (
                '<td data-fieldname="Job title">'
                '<a href="/jobs/advert?AdvertID=1234">Engineer</a></td>'
            )
            html = (
                "<html><body><table>"
                f"<tr class='oddrow'>{cell}</tr>"
                f"<tr class='evenrow'>{cell}</tr>"
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
    async for job in WAJobsSource().discover(_DupFetcher()):  # type: ignore[arg-type]
        jobs.append(job)
    assert len(jobs) == 1
