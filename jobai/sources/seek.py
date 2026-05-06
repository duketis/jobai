"""Seek source — Australia's dominant job board.

Seek has no public API and serves a Cloudflare-fronted React SPA.
Plain HTTP gets 403; the fetcher layer escalates to Chromium for us.

The DOM is parsed via stable ``data-automation`` attributes that
Seek exposes for testing/automation tools. They're effectively a
contract — selectors targeting them survive design refactors that
would break a CSS-class-based scraper. Per-card root is
``article[data-automation="normalJob"]`` carrying the canonical job
id in ``data-job-id``.

A :class:`SeekSource` instance covers one search slug — the path
fragment from a Seek URL, e.g. ``"python-jobs/in-Melbourne-VIC"``.
``companies.yaml`` lists the slugs we care about.

Pagination is intentionally not handled in this first cut: each page
returns ~22 results which is plenty for a freshness signal. A
follow-up phase walks ``?page=N`` once we want depth.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://www.seek.com.au"

#: Hard cap on pages walked per scrape cycle. Five pages of ~22 jobs
#: each gives ~100 results per slug, which is plenty of depth without
#: burning half an hour against the browser tier per cadence tick.
DEFAULT_MAX_PAGES = 5


class SeekSource(BaseSource):
    """Pulls jobs from one Seek search-result page (paginated)."""

    kind = "seek"

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
            response = await fetcher.fetch(url)

            if not response.is_ok:
                if page == 1:
                    raise SeekFetchError(self.account, response.status_code)
                # Mid-walk failure on later pages: stop short rather
                # than fail the whole run; we still return everything
                # already yielded.
                return

            tree = HTMLParser(response.text)
            cards = tree.css('article[data-automation="normalJob"]')
            page_yielded = 0
            for card in cards:
                job = _parse_card(card)
                if job is None or job.source_external_id in seen_ids:
                    continue
                seen_ids.add(job.source_external_id)
                page_yielded += 1
                yield job
            # Last page reached when Seek serves zero new cards (the
            # site sometimes pads with already-seen jobs near the end).
            if page_yielded == 0:
                return


def _page_url(slug: str, page: int) -> str:
    """Append ``?page=N`` (or ``&page=N``) to ``account``-derived URL.

    Page 1 is fetched without ``?page=1`` to mirror the canonical
    landing URL exactly — Seek serves the same data either way but
    response caching keys can differ.
    """
    base = f"{_BASE_URL}/{slug}"
    if page == 1:
        return base
    sep = "&" if "?" in slug else "?"
    return f"{base}{sep}page={page}"


class SeekFetchError(RuntimeError):
    """Raised when a Seek search page returns a non-2xx status."""

    def __init__(self, slug: str, status_code: int) -> None:
        super().__init__(f"seek:{slug} returned HTTP {status_code}")
        self.slug = slug
        self.status_code = status_code


def _parse_card(card: Node) -> NormalizedJob | None:
    """Translate one job card into a :class:`NormalizedJob`.

    Returns ``None`` when the entry is missing the minimum required
    fields (id, title, an apply URL we can derive). Every other field
    is best-effort — Seek varies what's populated by listing type.
    """
    job_id = card.attributes.get("data-job-id")
    if not job_id:
        return None

    title_node = _automation(card, "jobTitle")
    if title_node is None:
        return None
    title = title_node.text(strip=True)
    if not title:
        return None

    apply_path = title_node.attributes.get("href") or _href_from_overlay(card)
    if not apply_path:
        return None
    apply_url = urljoin(_BASE_URL, _strip_query_anchors(apply_path))

    company = _text(_automation(card, "jobCompany")) or "Unknown"
    location = _text(_automation(card, "jobCardLocation")) or _text(
        _automation(card, "jobLocation")
    )
    salary_text = _text(_automation(card, "jobSalary"))
    salary_min, salary_max, salary_currency = _parse_salary(salary_text)
    posted_label = _text(_automation(card, "jobListingDate"))
    teaser = _text(_automation(card, "jobShortDescription"))
    classification = _text(_automation(card, "jobClassification"))
    sub_classification = _text(_automation(card, "jobSubClassification"))
    employment_type = _employment_type_from_card(card)

    return NormalizedJob(
        source_external_id=str(job_id),
        title=title,
        company=company.strip(),
        apply_url=apply_url,
        raw_data={
            "title": title,
            "company": company,
            "location": location,
            "salary_text": salary_text,
            "posted_label": posted_label,
            "classification": classification,
            "sub_classification": sub_classification,
            "teaser": teaser,
        },
        location_raw=location,
        location_country="Australia",
        location_city=_first_segment(location),
        remote_type=_infer_remote_type(location),
        employment_type=employment_type,
        posted_at=posted_label,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        description_text=teaser,
        extra_tags=tuple(t for t in (classification, sub_classification) if t),
    )


def _automation(card: Node, name: str) -> Node | None:
    return card.css_first(f'[data-automation="{name}"]')


def _text(node: Node | None) -> str | None:
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


def _href_from_overlay(card: Node) -> str | None:
    overlay = _automation(card, "job-list-item-link-overlay") or _automation(
        card, "job-list-view-job-link"
    )
    if overlay is None:
        return None
    return overlay.attributes.get("href")


def _strip_query_anchors(path: str) -> str:
    """Drop tracking params and fragment so dedup keys stay stable.

    Seek's apply URLs include ``ref``, ``origin``, ``#sol=...``
    fragments that change per visit; a canonical job has the same
    ``/job/{id}?type=standard`` regardless.
    """
    parsed = urlparse(path)
    if not parsed.path:
        return path
    return parsed.path


def _employment_type_from_card(card: Node) -> str | None:
    """Seek embeds ``This is a Full time job`` in plain text on each card."""
    for node in card.css("p"):
        text = node.text(strip=True)
        if text.lower().startswith("this is a"):
            cleaned = text.removeprefix("This is a ").removesuffix(" job").strip()
            return cleaned or None
    return None


def _first_segment(value: str | None) -> str | None:
    if not value:
        return None
    head = value.split(",")[0].strip()
    return head or None


def _infer_remote_type(location: str | None) -> str | None:
    """Seek surfaces remote/hybrid in the location field on flexible roles."""
    if not location:
        return None
    lower = location.lower()
    if "remote" in lower:
        return "remote"
    if "hybrid" in lower:
        return "hybrid"
    return None


# Seek uses Unicode dashes in salary ranges alongside plain hyphens.
_DASH_CLASS = "-\N{EN DASH}\N{EM DASH}"
_SALARY_RANGE_RE = re.compile(
    rf"\$?\s*([\d,]+)\s*(?:[{_DASH_CLASS}]|to)\s*\$?\s*([\d,]+)",
    re.IGNORECASE,
)
_SALARY_SINGLE_RE = re.compile(r"\$?\s*([\d,]+)")


def _parse_salary(text: str | None) -> tuple[int | None, int | None, str | None]:
    if not text:
        return None, None, None

    range_match = _SALARY_RANGE_RE.search(text)
    if range_match:
        low = _to_int(range_match.group(1))
        high = _to_int(range_match.group(2))
        if low is not None and high is not None:
            return low, high, "AUD"

    single_match = _SALARY_SINGLE_RE.search(text)
    if single_match:
        value = _to_int(single_match.group(1))
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


__all__: list[str] = ["SeekFetchError", "SeekSource"]
