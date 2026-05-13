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
#: Results per page on the SA iworkfor portal. The site's pagination
#: control (``Next`` link) jumps the offset by 20 per click via the
#: ``jnext_prev(N)`` JS function.
_PAGE_SIZE = 20
#: Hard cap on pagination hops per scrape. The walker stops on the
#: first page that yields no new ids; the cap is just a guard
#: against runaway loops on a UI change.
_DEFAULT_MAX_PAGES = 100


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
            msg = (
                "SAIWorkForSource requires a fetcher with run_in_page; got "
                f"{type(fetcher).__name__}"
            )
            raise TypeError(msg)
        url = f"{_BASE_URL}/{self.account.lstrip('/')}"
        page_script = _walk_all_pages(self._max_pages)
        response = await run_in_page(url, page_script=page_script)
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


def _walk_all_pages(max_pages: int) -> PageScript:
    """Build a Playwright page-script that walks every paginated page.

    Returns a coroutine that drives SA's iworkfor pagination control:
    click Search to get page 1, then loop calling ``jnext_prev(N)``
    (the site's own pagination JS function) for each subsequent
    offset. Each page's ``<tr>`` rows are appended into the current
    DOM so the final ``page.content()`` snapshot contains every row
    across all pages.
    """

    async def script(page: Page) -> None:  # pragma: no cover - drives real Playwright
        # Integration-only: submits the search form, walks paginated
        # results via JS-driven form submits, and appends each page's
        # rows into the final DOM snapshot. Exercised end-to-end by
        # docker compose + the browser tier; not reachable in unit tests.
        # Step 1: click Search to get to page 1 of results.
        with contextlib.suppress(Exception):
            await page.click("#brsSearchBtn", timeout=15_000)
        with contextlib.suppress(Exception):
            await page.wait_for_selector(
                "table.Report tr.oddrow, table.Report tr.evenrow",
                timeout=20_000,
            )

        # Step 2: walk every page, accumulating row HTML in Python.
        # Append once at the end so we don't pollute the current DOM
        # mid-walk (which would break the seen-ids early-exit check).
        all_chunks: list[str] = []
        seen_ids: set[str] = set()
        for hop in range(max_pages):
            current_ids = await page.eval_on_selector_all(
                'td[data-fieldname="Reference No"]',
                "xs => xs.map(x => x.textContent.trim())",
            )
            new_ids = {str(x) for x in current_ids if x and x not in seen_ids}
            if not new_ids:
                # Current page has no new ids - exhausted or looped.
                break
            seen_ids |= new_ids

            chunk = await page.eval_on_selector_all(
                "tr.oddrow, tr.evenrow",
                "xs => xs.map(x => x.outerHTML).join('')",
            )
            if chunk:
                all_chunks.append(chunk)

            # Navigate to the NEXT page if we haven't hit the cap.
            if hop + 1 < max_pages:
                offset = (hop + 1) * _PAGE_SIZE
                try:
                    await page.evaluate(f"jnext_prev({offset})")
                except Exception:  # noqa: BLE001 - end of pagination
                    break
                with contextlib.suppress(Exception):
                    await page.wait_for_selector(
                        "table.Report tr.oddrow, table.Report tr.evenrow",
                        timeout=20_000,
                    )

        # Step 3: append everything captured into the final DOM so
        # ``page.content()`` snapshots all rows.
        if all_chunks:
            with contextlib.suppress(Exception):
                await _append_rows_to_first_table(page, "".join(all_chunks))

    return script


async def _append_rows_to_first_table(  # pragma: no cover - drives real Playwright
    page: Page, row_html: str
) -> None:
    """Append captured ``<tr>`` HTML to the current page's results tbody.

    Integration-only: page.evaluate runs JS inside the browser tier,
    not reachable in unit tests without spawning real Chromium.
    """
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
