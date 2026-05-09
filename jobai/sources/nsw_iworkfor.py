"""NSW Government — iworkfor.nsw.gov.au source.

**Status: blocked by Cloudflare strict challenge mode (2026-05).**

The NSW board moved its old HTML-render pipeline behind Cloudflare's
strict challenge interstitial — every plain-HTTP request gets a
``Just a moment...`` page, and our browser tier (Patchright) gets
the same 403 the bare client does. Bypassing CF reliably needs paid
proxy services or residential IPs, which the project rules out.

The discover loop now detects the CF challenge HTML and raises
:class:`NSWIWorkForBlockedError` so the run is recorded as failed
(instead of silently succeeding with zero jobs, which is what the
old code did and which hid the regression for weeks). The expected
operator response is to disable the NSW source rows in the DB until
the situation changes.

Old card-structure notes (kept for when we can fetch again):

* root: ``article.search-job-card`` with ``aria-labelledby="job-title-{id}"``
* title: ``.search-job-card__title``
* organization (employer agency): ``.search-job-card__organization``
* apply URL: ``a[href^="/job/"]``
* metadata in ``dl.job-card-info`` keyed by ``<dt>`` labels
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
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
        seen_ids: set[str] = set()
        for page in range(1, self._max_pages + 1):
            url = _page_url(self.account, page)
            response = await fetcher.fetch(url, wait_for_selector=_WAIT_SELECTOR)
            if not response.is_ok:
                if page == 1:
                    raise NSWIWorkForFetchError(self.account, response.status_code)
                return
            # Cloudflare returns the challenge page with HTTP 200 - we
            # have to inspect the body to know we were actually blocked.
            # On page 1 raise so the run fails loudly; on later pages
            # the source has already yielded earlier results, so just
            # stop walking.
            if _is_cloudflare_challenge(response.text):
                if page == 1:
                    raise NSWIWorkForBlockedError(self.account)
                return

            tree = HTMLParser(response.text)
            cards = tree.css("article.search-job-card")
            page_yielded = 0
            for card in cards:
                job = _parse_card(card)
                if job is None or job.source_external_id in seen_ids:
                    continue
                seen_ids.add(job.source_external_id)
                page_yielded += 1
                yield job
            if page_yielded == 0:
                return


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
