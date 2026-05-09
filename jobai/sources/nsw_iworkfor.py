"""NSW Government — iworkfor.nsw.gov.au source.

Cloudflare strict-challenge mode (May 2026 onwards) blocks every
plain-HTTP request with a ``Just a moment...`` interstitial. The
source declares ``needs_persistent_session = True`` so the runner
builds a tier-3 stealth fetcher with a long-lived browser context;
the Patchright stealth patches + a clean UA + ``goto(wait_until=
'networkidle')`` solves CF once at the start of each scrape cycle
and the cleared session walks every paginated page.

Pagination is the SPA's own Ant Design ``ant-pagination`` widget.
We drive it via ``run_in_page``: click ``[aria-label="Go to next
page"]``, wait for the leading card to swap (signals the SPA's XHR
+ repaint completed), capture each page's ``article.search-job-card``
HTML, accumulate in Python, and inject everything into the final DOM
so a single ``page.content()`` snapshot contains every page's cards.
URL ``?page=N`` does NOT work — the Angular app ignores the param.

The discover loop still detects the CF interstitial and raises
:class:`NSWIWorkForBlockedError` so a successful CF block gets
flagged as a real failure rather than a silent zero-card success
(the regression that hid for weeks before).

Card structure (post-redesign):

* root: ``article.search-job-card`` with ``aria-labelledby="job-title-{id}"``
* title: ``.search-job-card__title``
* organization (employer agency): ``.search-job-card__organization``
* apply URL: ``a[href^="/job/"]``
* metadata in ``dl.job-card-info`` keyed by ``<dt>`` labels
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

from playwright.async_api import Page
from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.fetcher.browser import PageScript
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://iworkfor.nsw.gov.au"

#: CSS selector the browser tier waits for after navigation. The
#: Next.js SPA renders job links as ``a[href^="/job/"]`` once the
#: backing API call completes; we wait on the link rather than the
#: old ``article.search-job-card`` selector (the post-CF redesign
#: dropped that wrapper).
_WAIT_SELECTOR = 'a[href^="/job/"]'

#: NSW iworkfor renders ~15 results per page; five pages is enough
#: depth without burning the browser fetcher on a long tail.
DEFAULT_MAX_PAGES = 100

#: Pull the trailing numeric id off ``/job/{slug}-{id}`` URLs.
_JOB_ID_RE = re.compile(r"-(\d+)$")


class NSWIWorkForFetchError(RuntimeError):
    """Raised when the iworkfor.nsw.gov.au search returns non-2xx."""

    def __init__(self, slug: str, status_code: int) -> None:
        super().__init__(f"nsw_iworkfor:{slug} returned HTTP {status_code}")
        self.slug = slug
        self.status_code = status_code


class NSWIWorkForBlockedError(RuntimeError):
    """Raised when iworkfor.nsw.gov.au serves the Cloudflare challenge.

    A 200 response with the ``Just a moment...`` interstitial is the
    site's way of saying "I refused you" while still using a 200 status
    code. Treating it as a successful 0-card scrape silently masks the
    block; raising surfaces it as a real failure on the run.
    """

    def __init__(self, slug: str) -> None:
        super().__init__(
            f"nsw_iworkfor:{slug} blocked by Cloudflare challenge "
            "(no jobs accessible without paid CF bypass)"
        )
        self.slug = slug


# Markers that identify the Cloudflare interstitial. We check the
# title and a CF-specific challenge-platform CSS class so a normal
# page that happens to contain the phrase "Just a moment" in body copy
# doesn't trigger.
_CLOUDFLARE_CHALLENGE_MARKERS: tuple[str, ...] = (
    "<title>Just a moment...</title>",
    "challenge-platform",
    "cf-mitigated",
)


def _is_cloudflare_challenge(text: str) -> bool:
    """True if ``text`` is the Cloudflare 'Just a moment...' interstitial.

    Two markers must hit (the title alone matches some legitimate
    pages with the phrase in copy; combining title + CF-specific
    asset path is unambiguous).
    """
    hits = sum(1 for marker in _CLOUDFLARE_CHALLENGE_MARKERS if marker in text)
    return hits >= 2


class NSWIWorkForSource(BaseSource):
    """One iworkfor.nsw.gov.au search-results page (paginated).

    ``account`` is the URL path fragment after ``iworkfor.nsw.gov.au/``,
    e.g. ``"jobs/all-keywords/all-agencies/all-organisations-entities/
    all-categories/all-locations/all-worktypes"`` for the unfiltered
    listing. Subsets (a specific agency, category, locale) are encoded
    in the path. Add slugs to companies.yaml as separate source rows
    for breadth.
    """

    kind = "nsw_iworkfor"
    # iworkfor.nsw.gov.au is fronted by Cloudflare's strict challenge
    # mode. The challenge resolves on a real browser context and
    # binds ``cf_clearance`` to that context's TLS handshake; a
    # per-fetch context would re-trigger the challenge on every
    # request. ``needs_persistent_session=True`` keeps one context
    # alive across all NSW fetches so we solve CF once per scrape
    # cycle and ride it through pagination.
    needs_persistent_session = True

    def __init__(self, account: str = "", *, max_pages: int = DEFAULT_MAX_PAGES) -> None:
        super().__init__(account)
        if max_pages < 1:
            msg = f"max_pages must be >= 1, got {max_pages}"
            raise ValueError(msg)
        self._max_pages = max_pages

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        # NSW iworkfor's Angular SPA paginates client-side via XHR
        # against api.ad-core04.com. URL ``?page=N`` does NOT work
        # (the SPA ignores the param), so plain HTTP-style pagination
        # is impossible - we must drive the in-browser Next button.
        run_in_page = getattr(fetcher, "run_in_page", None)
        if run_in_page is None:
            msg = (
                "NSWIWorkForSource requires a fetcher with run_in_page; got "
                f"{type(fetcher).__name__}"
            )
            raise TypeError(msg)
        url = f"{_BASE_URL}/{self.account.lstrip('/')}" if self.account else _BASE_URL
        page_script = _walk_all_pages(self._max_pages, account=self.account)
        response = await run_in_page(url, page_script=page_script)
        if not response.is_ok:
            raise NSWIWorkForFetchError(self.account, response.status_code)
        if _is_cloudflare_challenge(response.text):
            raise NSWIWorkForBlockedError(self.account)

        tree = HTMLParser(response.text)
        seen_ids: set[str] = set()
        for card in tree.css("article.search-job-card"):
            job = _parse_card(card)
            if job is None or job.source_external_id in seen_ids:
                continue
            seen_ids.add(job.source_external_id)
            yield job


# JS snippets used by the pagination walker. Hoisted out of the
# inner function so each is one logical block (and so ruff's
# E501 / line-length rule has fewer overlong-line opportunities).
_NEXT_BUTTON_STATE_JS = """
() => {
    const btn = document.querySelector('[aria-label="Go to next page"]');
    if (!btn) return 'absent';
    const li = btn.closest('.ant-pagination-next');
    const cls = li ? li.classList : null;
    if (cls && cls.contains('ant-pagination-disabled')) return 'disabled';
    return 'clickable';
}
"""

_LEADING_HREF_CHANGED_JS = """
(prevHref) => {
    const a = document.querySelector('article.search-job-card a[href^="/job/"]');
    return a && a.getAttribute('href') !== prevHref;
}
"""

_INJECT_CARDS_JS = """
(html) => {
    const target = document.querySelector('#search-results-section') || document.body;
    target.insertAdjacentHTML('beforeend',
        '<div data-jobai-paginated>' + html + '</div>');
}
"""


def _walk_all_pages(max_pages: int, *, account: str) -> PageScript:
    """Drive the Ant Design pagination control inside the SPA.

    NSW renders its pagination as Ant's ``ant-pagination`` widget;
    the next-page button has ``aria-label="Go to next page"``. The
    walker clicks it once per page, waits for the card list to swap
    (detected via a different leading job-id appearing), captures
    the page's ``article.search-job-card`` outerHTML, and accumulates
    everything in Python. At the end the captured cards are injected
    into the live DOM so the final ``page.content()`` snapshot
    contains every page's cards in one parseable document.
    """

    async def script(page: Page) -> None:
        # Wait for the SPA to render the first page of cards.
        try:
            await page.wait_for_selector(
                "article.search-job-card",
                timeout=30_000,
            )
        except Exception:  # noqa: BLE001 - blocked / empty / changed UI
            return

        all_chunks: list[str] = []
        seen_ids: set[str] = set()

        for hop in range(max_pages):
            # Capture the current page's job ids + card HTML.
            current_ids = await page.eval_on_selector_all(
                'article.search-job-card a[href^="/job/"]',
                "xs => xs.map(a => a.getAttribute('href'))",
            )
            new_ids = {str(href) for href in current_ids if href and href not in seen_ids}
            if not new_ids:
                break
            seen_ids |= new_ids

            chunk = await page.eval_on_selector_all(
                "article.search-job-card",
                "xs => xs.map(c => c.outerHTML).join('')",
            )
            if chunk:
                all_chunks.append(chunk)

            # Click the Next button if it exists and is enabled.
            if hop + 1 < max_pages:
                next_btn_state = await page.evaluate(_NEXT_BUTTON_STATE_JS)
                if next_btn_state != "clickable":
                    break
                # The Ant pagination widget renders top + bottom on
                # wide viewports; using ``.first`` avoids Playwright's
                # strict-mode violation when more than one Next button
                # matches the aria-label.
                try:
                    await page.locator(
                        '[aria-label="Go to next page"]'
                    ).first.click(timeout=10_000)
                except Exception:  # noqa: BLE001 - end of pagination
                    break
                # Wait for the leading card to change (signals the
                # SPA finished its XHR + repaint).
                first_new_id = next(iter(new_ids))
                try:
                    await page.wait_for_function(
                        _LEADING_HREF_CHANGED_JS,
                        arg=first_new_id,
                        timeout=20_000,
                    )
                except Exception:  # noqa: BLE001 - SPA didn't repaint, stop
                    break

        # Inject everything captured so the snapshot contains every
        # page's cards in one document.
        if all_chunks:
            with contextlib.suppress(Exception):
                await page.evaluate(
                    _INJECT_CARDS_JS,
                    "".join(all_chunks),
                )
        # Suppress unused-arg warning when the account-derived URL is
        # taken at the discover() level rather than in the script.
        _ = account

    return script


def _page_url(slug: str, page: int) -> str:
    base = f"{_BASE_URL}/{slug}" if slug else _BASE_URL
    if page == 1:
        return base
    sep = "&" if "?" in slug else "?"
    return f"{base}{sep}page={page}"


def _parse_card(card: Node) -> NormalizedJob | None:
    """Translate one search-job-card into a :class:`NormalizedJob`.

    Returns ``None`` when the card is missing the minimum required
    fields (id, title, apply path).
    """
    title_node = card.css_first(".search-job-card__title")
    if title_node is None:
        return None
    title = title_node.text(strip=True)
    if not title:
        return None

    apply_path = _href(card, ".search-job-card__title-link") or _href(card, 'a[href^="/job/"]')
    if not apply_path:
        return None

    job_id = _extract_job_id(apply_path, card)
    if job_id is None:
        return None

    org = _text(card, ".search-job-card__organization") or "NSW Government"
    occupation = _text(card, ".search-job-card__occupation")
    info = _parse_info_dl(card)

    location = info.get("Location")
    salary_text = info.get("Salary")
    salary_min, salary_max, salary_currency = _parse_salary(salary_text)

    return NormalizedJob(
        source_external_id=job_id,
        title=title,
        company=org.strip(),
        apply_url=urljoin(_BASE_URL, apply_path),
        raw_data={
            "title": title,
            "organization": org,
            "occupation": occupation,
            "info": info,
        },
        location_raw=location,
        location_country="Australia",
        location_city=_first_segment(location),
        employment_type=info.get("Work Type") or info.get("Employment Type"),
        posted_at=info.get("Listed On") or info.get("Closing Date"),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        extra_tags=tuple(t for t in (occupation,) if t),
    )


def _href(card: Node, selector: str) -> str | None:
    node = card.css_first(selector)
    if node is None:
        return None
    return node.attributes.get("href")


def _text(card: Node, selector: str) -> str | None:
    node = card.css_first(selector)
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


def _extract_job_id(apply_path: str, card: Node) -> str | None:
    """Pull the integer id from ``/job/{slug}-{id}`` or from
    ``aria-labelledby="job-title-{id}"`` as a fallback."""
    match = _JOB_ID_RE.search(apply_path.split("?", 1)[0].rstrip("/"))
    if match:
        return match.group(1)
    aria = card.attributes.get("aria-labelledby", "") or ""
    if aria.startswith("job-title-"):
        candidate = aria.removeprefix("job-title-")
        if candidate.isdigit():
            return candidate
    return None


def _parse_info_dl(card: Node) -> dict[str, str]:
    """Walk the ``dl.job-card-info`` table into ``{label: value}``.

    NSW emits each row as ``<div><dt><strong>Label</strong></dt><dd>value</dd></div>``;
    we strip the strong tag and collapse whitespace.
    """
    info: dict[str, str] = {}
    dl = card.css_first("dl.job-card-info")
    if dl is None:
        return info
    for row in dl.css("div"):
        dt = row.css_first("dt")
        dd = row.css_first("dd")
        if dt is None or dd is None:
            continue
        label = dt.text(strip=True)
        value = dd.text(strip=True)
        if label and value:
            info[label] = value
    return info


def _first_segment(value: str | None) -> str | None:
    if not value:
        return None
    head = value.split(",")[0].strip()
    return head or None


_DASH_CLASS = "-\N{EN DASH}\N{EM DASH}"
_SALARY_RANGE_RE = re.compile(
    rf"\$?\s*([\d,]+)\s*(?:[{_DASH_CLASS}]|to)\s*\$?\s*([\d,]+)",
    re.IGNORECASE,
)
_SALARY_SINGLE_RE = re.compile(r"\$?\s*([\d,]+)")


def _parse_salary(text: str | None) -> tuple[int | None, int | None, str | None]:
    if not text:
        return None, None, None
    match = _SALARY_RANGE_RE.search(text)
    if match:
        low = _to_int(match.group(1))
        high = _to_int(match.group(2))
        if low is not None and high is not None:
            return low, high, "AUD"
    single = _SALARY_SINGLE_RE.search(text)
    if single:
        value = _to_int(single.group(1))
        if value is not None:
            return value, None, "AUD"
    return None, None, None


def _to_int(token: str) -> int | None:
    cleaned = token.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    value = int(cleaned)
    if value < 1000:
        return value * 1000
    return value


__all__: list[str] = ["NSWIWorkForFetchError", "NSWIWorkForSource"]
# raw_data uses Any internally
_ = Any
