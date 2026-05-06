"""APS Jobs (Australian Public Service) source — currently disabled.

.. warning::

   APS Jobs migrated to a Salesforce Lightning SPA in 2026. The Atom
   feed at ``/s/search.atom`` no longer exists (returns 404), and the
   new search-results page (``/s/job-search``) lazy-loads jobs via
   Salesforce Aura/Lightning XHR calls that this parser doesn't yet
   speak. ``companies.yaml`` ships the entries with ``enabled: false``
   so the scheduler skips them.

   Until a Salesforce-aware parser lands, this module's
   :class:`APSJobsSource` is dead code; the file is preserved
   (with passing tests against a representative legacy Atom fixture)
   so the migration to the new endpoint is a parser swap, not a
   greenfield addition.

Original (legacy) behaviour:

The site used to expose an Atom feed under ``/s/search.atom``; each
``<entry>`` carried a stable ``<id>`` containing ``ItemID=<digits>``
which became the ``source_external_id``.

The ``account`` was the RSS query string (URL-encoded), e.g.
``"Keywords=software"``; empty returned the full open feed.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://www.apsjobs.gov.au"
_FEED_PATH = "/s/search.atom"
_ITEM_ID_RE = re.compile(r"ItemID=(\d+)", re.IGNORECASE)


class APSJobsSource(BaseSource):
    """Pulls listings from the APS Jobs central feed."""

    kind = "apsjobs"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = f"{_BASE_URL}{_FEED_PATH}"
        if self.account:
            url = f"{url}?{self.account}"
        response = await fetcher.fetch(url)
        if not response.is_ok:
            raise APSJobsFetchError(self.account, response.status_code)

        for entry in _iterate_entries(response.text):
            job = _parse_entry(entry)
            if job is not None:
                yield job


class APSJobsFetchError(RuntimeError):
    """Raised when the APS Jobs feed returns a non-2xx status."""

    def __init__(self, query: str, status_code: int) -> None:
        super().__init__(f"apsjobs:{query} returned HTTP {status_code}")
        self.query = query
        self.status_code = status_code


def _iterate_entries(xml: str) -> list[Node]:
    """Walk the Atom feed and return every ``<entry>`` node."""
    tree = HTMLParser(xml)
    return list(tree.css("entry") or [])


def _parse_entry(entry: Node) -> NormalizedJob | None:
    title = _text(entry, "title")
    link = _attr(entry, "link", "href")
    if title is None or link is None:
        return None

    job_id = _extract_job_id(entry, link)
    if job_id is None:
        return None

    summary = _text(entry, "summary")
    posted_at = _text(entry, "updated") or _text(entry, "published")
    company = _extract_agency(summary) or "Australian Public Service"
    location = _extract_location(summary)
    salary_min, salary_max = _extract_salary(summary)

    return NormalizedJob(
        source_external_id=job_id,
        title=title.strip(),
        company=company.strip(),
        apply_url=link,
        raw_data={"summary": summary or "", "link": link},
        location_raw=location,
        location_country="Australia",
        location_city=_first_segment(location),
        posted_at=posted_at,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency="AUD" if salary_min or salary_max else None,
        description_text=summary,
    )


def _text(parent: Node, selector: str) -> str | None:
    node = parent.css_first(selector)
    if node is None:
        return None
    return node.text(strip=True) or None


def _attr(parent: Node, selector: str, attribute: str) -> str | None:
    node = parent.css_first(selector)
    if node is None:
        return None
    return node.attributes.get(attribute)


def _extract_job_id(entry: Node, link: str) -> str | None:
    """Pull a stable id from ``<id>`` or the ItemID query param."""
    raw_id = _text(entry, "id")
    if raw_id:
        match = _ITEM_ID_RE.search(raw_id)
        if match:
            return match.group(1)
    match = _ITEM_ID_RE.search(link)
    if match:
        return match.group(1)
    if raw_id:
        return raw_id
    return None


# APS Jobs summaries follow a stable convention: the agency name is
# everything up to ``" is hiring"``. Falling back to a coarser
# Department/Bureau match catches the rare entry that omits the
# verb. Tighter selectolax-style queries on the underlying HTML
# would be more brittle than this convention-based parse.
_AGENCY_PREFIX_RE = re.compile(r"^(.+?)\s+is\s+hiring", re.IGNORECASE)
_AGENCY_FALLBACK_RE = re.compile(
    r"\b(?:Department of [A-Z][\w &]+|[A-Z][\w &]+ (?:Bureau|Authority|Commission|Agency|Office))"
)
_SALARY_RE = re.compile(r"\$([\d,]+)\s*-\s*\$([\d,]+)")


def _extract_agency(summary: str | None) -> str | None:
    if not summary:
        return None
    match = _AGENCY_PREFIX_RE.match(summary.strip())
    if match:
        return match.group(1).strip()
    fallback = _AGENCY_FALLBACK_RE.search(summary)
    if fallback:
        return fallback.group(0).strip()
    return None


def _extract_location(summary: str | None) -> str | None:
    """APS summaries embed location after a ``Location:`` tag.

    Stops at the first sentence break (``.``) or ``;`` since other
    fields (Salary, Closes) follow on the same line in this format.
    """
    if not summary:
        return None
    match = re.search(r"Location:\s*([^.;\n]+)", summary, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_salary(summary: str | None) -> tuple[int | None, int | None]:
    if not summary:
        return None, None
    match = _SALARY_RE.search(summary)
    if match is None:
        return None, None
    return _to_int(match.group(1)), _to_int(match.group(2))


def _to_int(token: str) -> int | None:
    cleaned = token.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def _first_segment(value: str | None) -> str | None:
    if not value:
        return None
    head = value.split(",")[0].strip()
    return head or None


__all__: list[str] = ["APSJobsFetchError", "APSJobsSource"]
