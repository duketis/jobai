"""VIC Government — jobs.careers.vic.gov.au source.

VIC's central government job board runs on NGA Talent Solutions'
``jobtools`` platform — same family as QLD smartjobs, but VIC's
deployment doesn't accept a direct GET to the search-results URL:
the form must be POST-submitted, with session state set up by the
initial form GET.

We use :meth:`BrowserFetcher.run_in_page` to drive Playwright
through the workflow:

1. Navigate to the search form (sets the session cookie).
2. Click the ``input[name="in_searchBut"]`` Search button.
3. Wait for the rendered results table (``tr.odd, tr.even`` rows
   under the form's results frame).
4. Snapshot the resulting HTML and parse it like a regular tier-1
   response.

Per-row structure on the results page:

* ``<tr class="odd|even">`` — one row per job
* ``td > input[name="in_select"][value="{job_id}"]`` — stable id
* ``td > a[href^="/jobs/VG-"]`` — title + apply URL slug (the
  numeric portion is **not** part of the URL — the checkbox value
  is what we dedup on)
* ``td`` 2..7 — occupation, salary, agency, employment type,
  location, closing date (positional; structure has been stable)
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

_BASE_URL = "https://jobs.careers.vic.gov.au"
_SEARCH_PATH = "/jobtools/jncustomsearch.jobsearch"
_DEFAULT_ORGID = "14123"
#: Results per page on jobtools — the JS pagination control advances
#: the form's ``in_pg`` field by 20 per click, so each page contains
#: at most 20 ``<tr>`` job rows.
_PAGE_SIZE = 20
#: Hard cap on pagination clicks per scrape cycle. The walker stops
#: short when a page yields zero new rows; the cap is just a guard
#: against runaway loops on a UI change.
_DEFAULT_MAX_PAGES = 100

#: VIC publishes salaries inline in the table (e.g., "$70,000 - $90,000")
#: or as ``"See Advertisement"`` for ranges that don't fit. We extract
#: from the literal text only — no inline JS like QLD's smartjobs.
_DASH_CLASS = "-\N{EN DASH}\N{EM DASH}"
_SALARY_RANGE_RE = re.compile(
    rf"\$?\s*([\d,]+)\s*(?:[{_DASH_CLASS}]|to)\s*\$?\s*([\d,]+)",
    re.IGNORECASE,
)


class VICCareersFetchError(RuntimeError):
    """Raised when jobs.careers.vic.gov.au returns no usable results."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"vic_careers:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class VICCareersSource(BaseSource):
    """One jobs.careers.vic.gov.au search-results listing.

    ``account`` is the NGA organisation id; default ``"14123"`` is
    the Victorian Public Service umbrella org.
    """

    kind = "vic_careers"

    def __init__(
        self,
        account: str = _DEFAULT_ORGID,
        *,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        super().__init__(account or _DEFAULT_ORGID)
        if max_pages < 1:
            msg = f"max_pages must be >= 1, got {max_pages}"
            raise ValueError(msg)
        self._max_pages = max_pages

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        run_in_page = getattr(fetcher, "run_in_page", None)
        if run_in_page is None:
            msg = (
                "VICCareersSource requires a fetcher with run_in_page "
                "(BrowserFetcher or a wrapper that forwards it); got "
                f"{type(fetcher).__name__}"
            )
            raise TypeError(msg)
        url = f"{_BASE_URL}{_SEARCH_PATH}?in_organid={self.account or _DEFAULT_ORGID}"
        page_script = _walk_all_pages(self._max_pages)
        response = await run_in_page(url, page_script=page_script)
        if not response.is_ok:
            raise VICCareersFetchError(self.account, response.status_code)

        # The walker concatenates every page's rows into one HTML
        # blob, so a single parse pass yields jobs from all pages.
        # ``seen_ids`` here is belt-and-braces - the walker stops
        # when a page repeats - but defends against any JS-driven
        # duplication if the form ever lands on the same page twice.
        seen_ids: set[str] = set()
        tree = HTMLParser(response.text)
        for row in tree.css("tr.odd, tr.even"):
            job = _parse_row(row)
            if job is None or job.source_external_id in seen_ids:
                continue
            seen_ids.add(job.source_external_id)
            yield job


def _walk_all_pages(max_pages: int) -> PageScript:
    """Build a Playwright page-script that walks every paginated page.

    Returns a coroutine that drives the form: click Search, snapshot
    page 1, then loop ``max_pages-1`` times setting ``in_pg`` via JS
    and re-submitting the form. Each page's ``<tbody>`` rows are
    appended to the first page's body so the final ``page.content()``
    snapshot contains every row across all pages.
    """

    async def script(page: Page) -> None:  # pragma: no cover - drives real Playwright
        # Integration-only: Playwright-driven form submit + paginated
        # click loop. Exercised end-to-end by the docker-compose soak,
        # not reachable in unit tests without spawning real Chromium.
        # Step 1: click Search to get to page 1 of results.
        try:
            await page.click('input[name="in_searchBut"]', timeout=15_000)
        except Exception:  # noqa: BLE001 - missing button = empty results, not an error
            return
        with contextlib.suppress(Exception):
            await page.wait_for_selector(
                "tr.odd, tr.even, .no-results",
                timeout=20_000,
            )

        # Step 2: walk every page, accumulating row HTML in Python.
        # Append once at the end - mid-walk DOM appends would pollute
        # the seen-ids check (current page's IDs would include
        # already-appended ones, killing the 'still finding new ids'
        # signal that gates the loop).
        all_chunks: list[str] = []
        seen_ids: set[str] = set()
        for hop in range(max_pages):
            current_ids = await page.eval_on_selector_all(
                'input[name="in_select"]',
                "xs => xs.map(x => x.value)",
            )
            new_ids = {str(x) for x in current_ids if x and x not in seen_ids}
            if not new_ids:
                break
            seen_ids |= new_ids

            chunk = await page.eval_on_selector_all(
                "tr.odd, tr.even",
                "xs => xs.map(x => x.outerHTML).join('')",
            )
            if chunk:
                all_chunks.append(chunk)

            # Navigate to the NEXT page if we haven't hit the cap.
            if hop + 1 < max_pages:
                offset = (hop + 1) * _PAGE_SIZE
                # ``document.resultsform.submit()`` triggers a full
                # navigation, so we have to wait for it BEFORE the
                # next eval (otherwise the execution context gets
                # destroyed mid-eval and Playwright raises).
                try:
                    async with page.expect_navigation(
                        wait_until="domcontentloaded",
                        timeout=20_000,
                    ):
                        await page.evaluate(
                            f"document.resultsform.in_pg.value={offset};"
                            "document.resultsform.in_nav.value='next_set';"
                            "document.resultsform.submit()"
                        )
                except Exception:  # noqa: BLE001 - end of pagination
                    break
                with contextlib.suppress(Exception):
                    await page.wait_for_selector("tr.odd, tr.even", timeout=20_000)

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

    The accumulating snapshot strategy: every paginated page's rows
    get appended to whatever the current page is rendering, so when
    ``run_in_page`` snapshots ``page.content()`` at the end, all rows
    from all pages are present in a single HTML document.
    """
    if not row_html:
        return
    await page.evaluate(
        "rowHtml => {"
        "  const tbody = document.querySelector('tr.odd, tr.even')?.parentElement;"
        "  if (tbody) tbody.insertAdjacentHTML('beforeend', rowHtml);"
        "}",
        row_html,
    )


# Re-export for type hint.
from jobai.fetcher.browser import PageScript  # noqa: E402  - circular avoidance


def _parse_row(row: Node) -> NormalizedJob | None:
    """Map one ``<tr>`` results row onto a :class:`NormalizedJob`."""
    cells = row.css("td")
    if len(cells) < 7:
        return None

    checkbox = cells[0].css_first('input[name="in_select"]')
    job_id = checkbox.attributes.get("value") if checkbox is not None else None
    if not job_id:
        return None

    title_anchor = cells[1].css_first('a[href^="/jobs/"]') or cells[1].css_first("a[href]")
    if title_anchor is None:
        return None
    title = title_anchor.text(strip=True)
    apply_path = title_anchor.attributes.get("href")
    if not title or not apply_path:
        return None

    occupation = _cell_text(cells, 2)
    salary_text = _cell_text(cells, 3)
    agency = _cell_text(cells, 4) or "Victorian Government"
    employment_type = _cell_text(cells, 5)
    location = _cell_text(cells, 6)
    closing_date = _cell_text(cells, 7) if len(cells) > 7 else None

    salary_min, salary_max, salary_currency = _parse_salary(salary_text)

    return NormalizedJob(
        source_external_id=str(job_id),
        title=title,
        company=agency.strip(),
        apply_url=urljoin(_BASE_URL, apply_path),
        raw_data={
            "title": title,
            "occupation": occupation,
            "salary_text": salary_text,
            "agency": agency,
            "employment_type": employment_type,
            "location": location,
            "closing_date": closing_date,
        },
        location_raw=location,
        location_country="Australia",
        location_city=_first_segment(location),
        employment_type=employment_type,
        posted_at=closing_date,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        extra_tags=tuple(t for t in (occupation,) if t),
    )


def _cell_text(cells: list[Node], idx: int) -> str | None:
    if idx >= len(cells):
        return None
    text = cells[idx].text(strip=True)
    return text or None


def _first_segment(value: str | None) -> str | None:
    if not value:
        return None
    head = value.split(",")[0].strip()
    return head or None


def _parse_salary(text: str | None) -> tuple[int | None, int | None, str | None]:
    if not text or "advertisement" in text.lower():
        return None, None, None
    match = _SALARY_RANGE_RE.search(text)
    if match is None:
        return None, None, None
    low = _to_int(match.group(1))
    high = _to_int(match.group(2))
    if low is None or high is None:
        return None, None, None
    return low, high, "AUD"


def _to_int(token: str) -> int | None:
    cleaned = token.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    value = int(cleaned)
    if value < 1000:
        return value * 1000
    return value


__all__: list[str] = ["VICCareersFetchError", "VICCareersSource"]
