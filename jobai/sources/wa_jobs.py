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
#: Results per page on WA's iworkfor portal. The ``Next >`` link
#: jumps the offset by 10 per click via ``jnext_prev(N)``.
_PAGE_SIZE = 10
#: Hard cap on pagination hops. Walker early-exits on a zero-yield
#: page; cap is just a guard against runaway loops on a UI change.
_DEFAULT_MAX_PAGES = 200

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

    def __init__(
        self,
        account: str = "",
        *,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        super().__init__(account or _DEFAULT_PATH.lstrip("/"))
        if max_pages < 1:
            msg = f"max_pages must be >= 1, got {max_pages}"
            raise ValueError(msg)
        self._max_pages = max_pages

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        run_in_page = getattr(fetcher, "run_in_page", None)
        if run_in_page is None:
            msg = f"WAJobsSource requires a fetcher with run_in_page; got {type(fetcher).__name__}"
            raise TypeError(msg)
        url = f"{_BASE_URL}/{self.account.lstrip('/')}"
        page_script = _walk_all_pages(self._max_pages)
        response = await run_in_page(url, page_script=page_script)
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


def _walk_all_pages(max_pages: int) -> PageScript:
    """Build a Playwright page-script that walks every paginated page.

    Same pattern as SA iworkfor (the two portals share the iworkfor
    SaaS platform): click Search to get page 1, then loop calling
    ``jnext_prev(N)`` for each subsequent offset, accumulating row
    HTML into the current DOM so the final snapshot contains every
    row across all pages.
    """

    async def script(page: Page) -> None:
        with contextlib.suppress(Exception):
            await page.click('input[value="Search"]', timeout=15_000)
        with contextlib.suppress(Exception):
            await page.wait_for_selector(
                "table.Report tr.oddrow, table.Report tr.evenrow",
                timeout=20_000,
            )

        # Walk every page, accumulating in Python, append once at end
        # so the in-DOM accumulation doesn't pollute the seen-ids check.
        all_chunks: list[str] = []
        seen_ids: set[str] = set()
        for hop in range(max_pages):
            current_ids = await page.eval_on_selector_all(
                'tr.oddrow a[href*="AdvertID="], tr.evenrow a[href*="AdvertID="]',
                "xs => xs.map(x => x.href.match(/AdvertID=(\\d+)/)?.[1] || '')",
            )
            new_ids = {str(x) for x in current_ids if x and x not in seen_ids}
            if not new_ids:
                break
            seen_ids |= new_ids

            chunk = await page.eval_on_selector_all(
                "tr.oddrow, tr.evenrow",
                "xs => xs.map(x => x.outerHTML).join('')",
            )
            if chunk:
                all_chunks.append(chunk)

            if hop + 1 < max_pages:
                offset = (hop + 1) * _PAGE_SIZE
                # ``jnext_prev()`` triggers a full form submission +
                # navigation. Wrap in expect_navigation so the next
                # eval doesn't fire while the context is destroyed.
                try:
                    async with page.expect_navigation(
                        wait_until="domcontentloaded",
                        timeout=20_000,
                    ):
                        await page.evaluate(f"jnext_prev({offset})")
                except Exception:  # noqa: BLE001 - end of pagination
                    break
                with contextlib.suppress(Exception):
                    await page.wait_for_selector(
                        "table.Report tr.oddrow, table.Report tr.evenrow",
                        timeout=20_000,
                    )

        if all_chunks:
            with contextlib.suppress(Exception):
                await _append_rows_to_first_table(page, "".join(all_chunks))

    return script


async def _append_rows_to_first_table(page: Page, row_html: str) -> None:
    """Append captured ``<tr>`` HTML to the current page's results tbody."""
    if not row_html:
        return
    await page.evaluate(
        "rowHtml => {"
        "  const tbody = document.querySelector('tr.oddrow, tr.evenrow')?.parentElement;"
        "  if (tbody) tbody.insertAdjacentHTML('beforeend', rowHtml);"
        "}",
        row_html,
    )


# Re-export for type hint - imported at module bottom to avoid the
# fetcher -> source -> fetcher circular import on module load.
from jobai.fetcher.browser import PageScript  # noqa: E402


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
