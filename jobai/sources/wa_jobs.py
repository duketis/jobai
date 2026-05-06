"""WA Government — search.jobs.wa.gov.au source.

WA's central jobs portal mirrors SA's structure: a search form
(``input[value="Search"]`` is the submit button) renders a
``table.Report`` of result rows after submission. Cells are keyed
by ``data-fieldname`` so we look up by name. Each row's title cell
contains an anchor whose ``AdvertID`` query param is the stable
job identifier.

Driven via :meth:`BrowserFetcher.run_in_page` for the form-fill
workflow; default tier 2.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator
from urllib.parse import urljoin

from playwright.async_api import Page
from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://search.jobs.wa.gov.au"
_DEFAULT_PATH = "/page.php?pageID=215"

#: Pull the AdvertID off the per-job ``page.php?...&AdvertID=12345`` link.
_ADVERT_ID_RE = re.compile(r"AdvertID=(\d+)")


class WAJobsFetchError(RuntimeError):
    """Raised when search.jobs.wa.gov.au returns no usable results."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"wa_jobs:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class WAJobsSource(BaseSource):
    """One search.jobs.wa.gov.au search-results page."""

    kind = "wa_jobs"

    def __init__(self, account: str = "") -> None:
        super().__init__(account or _DEFAULT_PATH.lstrip("/"))

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        run_in_page = getattr(fetcher, "run_in_page", None)
        if run_in_page is None:
            msg = f"WAJobsSource requires a fetcher with run_in_page; got {type(fetcher).__name__}"
            raise TypeError(msg)
        url = f"{_BASE_URL}/{self.account.lstrip('/')}"
        response = await run_in_page(url, page_script=_submit_search_form)
        if not response.is_ok:
            raise WAJobsFetchError(self.account, response.status_code)

        tree = HTMLParser(response.text)
        seen: set[str] = set()
        for row in tree.css("tr.oddrow, tr.evenrow"):
            job = _parse_row(row)
            if job is None or job.source_external_id in seen:
                continue
            seen.add(job.source_external_id)
            yield job


async def _submit_search_form(page: Page) -> None:
    """Click WA's ``input[value="Search"]`` to render the results table."""
    with contextlib.suppress(Exception):
        await page.click('input[value="Search"]', timeout=15_000)
    with contextlib.suppress(Exception):
        await page.wait_for_selector(
            "table.Report tr.oddrow, table.Report tr.evenrow", timeout=20_000
        )


def _parse_row(row: Node) -> NormalizedJob | None:
    """Extract a job from one ``<tr.oddrow|.evenrow>`` row."""
    cells: dict[str, Node] = {}
    for cell in row.css("td"):
        fieldname = cell.attributes.get("data-fieldname")
        if fieldname:
            cells[fieldname.strip()] = cell

    title_cell = cells.get("Job title") or cells.get("Job Title")
    if title_cell is None:
        return None
    anchor = title_cell.css_first("a")
    if anchor is None:
        return None

    title = anchor.text(strip=True)
    apply_path = anchor.attributes.get("href")
    if not title or not apply_path:
        return None

    match = _ADVERT_ID_RE.search(apply_path)
    if match is None:
        return None
    job_id = match.group(1)

    posted_at = _cell_text(cells, "Posting date") or _cell_text(cells, "Posting Date")
    agency = _cell_text(cells, "Agency") or "Western Australian Government"
    branch = _cell_text(cells, "Branch")
    closing = _cell_text(cells, "Closing date") or _cell_text(cells, "Closing Date")

    return NormalizedJob(
        source_external_id=job_id,
        title=title,
        company=agency,
        apply_url=urljoin(_BASE_URL + "/", apply_path),
        raw_data={
            "title": title,
            "agency": agency,
            "branch": branch,
            "posted_at": posted_at,
            "closing_at": closing,
            "advert_id": job_id,
        },
        location_country="Australia",
        posted_at=posted_at,
        extra_tags=tuple(t for t in (branch,) if t),
    )


def _cell_text(cells: dict[str, Node], key: str) -> str | None:
    cell = cells.get(key)
    if cell is None:
        return None
    text = cell.text(strip=True)
    if not text:
        return None
    # WA prefixes each cell text with "FieldName : " in mobile views;
    # strip the leading label noise so the stored value is clean.
    prefix = f"{key} :"
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    return text or None


__all__: list[str] = ["WAJobsFetchError", "WAJobsSource"]
