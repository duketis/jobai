"""SA Government — iworkfor.sa.gov.au source.

SA's iworkfor portal is a search-form site: results only render
after the user clicks the search button. We drive that workflow
with :meth:`BrowserFetcher.run_in_page` — load the page, click the
``#brsSearchBtn`` button, then snapshot the resulting table.

Per-row structure (stable across the deployment we tested against):

* Wrapper: ``table.Report``
* Each result is a ``tr.oddrow`` / ``tr.evenrow`` with cells keyed
  by ``data-fieldname`` (Job Title, Reference No, Posting Date,
  Agency)
* The job title cell contains an ``<a href="/jb/page/{token}">``
  with the role name; the ``Reference No`` cell holds the stable
  numeric id we use as ``source_external_id``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from urllib.parse import urljoin

from playwright.async_api import Page
from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://iworkfor.sa.gov.au"
_DEFAULT_PATH = "/jb/list/all"


class SAIWorkForFetchError(RuntimeError):
    """Raised when iworkfor.sa.gov.au returns no usable results."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"sa_iworkfor:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class SAIWorkForSource(BaseSource):
    """One iworkfor.sa.gov.au search-results page.

    ``account`` is the URL path fragment after ``iworkfor.sa.gov.au``;
    default ``"jb/list/all"`` is the unfiltered listing. Pass other
    SA-internal filter URLs to seed sub-feeds.
    """

    kind = "sa_iworkfor"

    def __init__(self, account: str = "") -> None:
        super().__init__(account or _DEFAULT_PATH.lstrip("/"))

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        run_in_page = getattr(fetcher, "run_in_page", None)
        if run_in_page is None:
            msg = (
                "SAIWorkForSource requires a fetcher with run_in_page; got "
                f"{type(fetcher).__name__}"
            )
            raise TypeError(msg)
        url = f"{_BASE_URL}/{self.account.lstrip('/')}"
        response = await run_in_page(url, page_script=_submit_search_form)
        if not response.is_ok:
            raise SAIWorkForFetchError(self.account, response.status_code)

        tree = HTMLParser(response.text)
        seen: set[str] = set()
        for row in tree.css("tr.oddrow, tr.evenrow"):
            job = _parse_row(row)
            if job is None or job.source_external_id in seen:
                continue
            seen.add(job.source_external_id)
            yield job


async def _submit_search_form(page: Page) -> None:
    """Click SA's ``#brsSearchBtn`` button to trigger the search."""
    with contextlib.suppress(Exception):
        await page.click("#brsSearchBtn", timeout=15_000)
    with contextlib.suppress(Exception):
        await page.wait_for_selector(
            "table.Report tr.oddrow, table.Report tr.evenrow", timeout=20_000
        )


def _parse_row(row: Node) -> NormalizedJob | None:
    """Extract a job from one ``<tr.oddrow|.evenrow>`` row.

    SA emits each cell with ``data-fieldname="..."`` so we look up
    by name rather than positional index — robust against cell
    re-ordering that happens between filter combinations.
    """
    cells: dict[str, Node] = {}
    for cell in row.css("td"):
        fieldname = cell.attributes.get("data-fieldname")
        if fieldname:
            cells[fieldname.strip()] = cell
    title_cell = cells.get("Job Title")
    if title_cell is None:
        return None
    anchor = title_cell.css_first("a")
    if anchor is None:
        return None
    title = anchor.text(strip=True)
    apply_path = anchor.attributes.get("href")
    if not title or not apply_path:
        return None

    ref_cell = cells.get("Reference No")
    job_id = ref_cell.text(strip=True) if ref_cell is not None else None
    if not job_id:
        # Fall back to the encoded /jb/page/ token so we still dedup.
        job_id = apply_path.rsplit("/", 1)[-1]
    if not job_id:
        return None

    posted_cell = cells.get("Posting Date")
    agency_cell = cells.get("Agency")
    posted_at = posted_cell.text(strip=True) if posted_cell is not None else None
    agency = (
        agency_cell.text(strip=True) if agency_cell is not None else "South Australian Government"
    )

    return NormalizedJob(
        source_external_id=str(job_id),
        title=title,
        company=agency or "South Australian Government",
        apply_url=urljoin(_BASE_URL, apply_path),
        raw_data={
            "title": title,
            "agency": agency,
            "reference_no": job_id,
            "posted_at": posted_at,
        },
        location_country="Australia",
        posted_at=posted_at,
    )


__all__: list[str] = ["SAIWorkForFetchError", "SAIWorkForSource"]
