"""LinkedIn (guest mode) source.

LinkedIn fingerprints aggressively. Logged-in scraping violates the
ToS; the *guest* search-results page is publicly accessible and is
what we target here. The ``/jobs/search`` endpoint returns an HTML
page with semantic markup the parser walks via :mod:`selectolax`.

The account encodes the search query: a string of the form
``"keywords=python&location=Australia"`` (URL-encoded). It plugs
straight into the public guest URL.

Description bodies are not on the listing page — only title, company,
location, and apply URL. Filling in descriptions would mean a per-job
detail fetch (one extra round trip per result), which we defer to a
later phase: the agent layer can summarise from title + company alone
when description is null.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://www.linkedin.com"
_SEARCH_PATH = "/jobs/search"

#: Match the integer job id LinkedIn embeds in `data-entity-urn` and
#: in the per-card detail URL (``/jobs/view/<id>``). Used as the
#: ``source_external_id`` so dedup across runs is stable.
_JOB_ID_RE = re.compile(r"jobPosting:(\d+)")
_VIEW_ID_RE = re.compile(r"/jobs/view/[^/]*-(\d+)")


class LinkedInSource(BaseSource):
    """One LinkedIn guest-mode jobs search.

    ``account`` is the URL-encoded query string passed to
    ``/jobs/search``, e.g. ``"keywords=python&location=Australia"``.
    """

    kind = "linkedin"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = f"{_BASE_URL}{_SEARCH_PATH}?{self.account}"
        response = await fetcher.fetch(url)
        if not response.is_ok:
            raise LinkedInFetchError(self.account, response.status_code)

        for card in _iterate_cards(response.text):
            job = _parse_card(card)
            if job is not None:
                yield job


class LinkedInFetchError(RuntimeError):
    """Raised when a LinkedIn search returns a non-2xx status."""

    def __init__(self, query: str, status_code: int) -> None:
        super().__init__(f"linkedin:{query} returned HTTP {status_code}")
        self.query = query
        self.status_code = status_code


def build_query(*, keywords: str, location: str = "Australia") -> str:
    """Build the URL-encoded query for a :class:`LinkedInSource` account.

    Provided as a convenience so callers don't have to remember the
    exact parameter names LinkedIn expects.
    """
    return urlencode({"keywords": keywords, "location": location})


def _iterate_cards(html: str) -> list[Node]:
    """Return every job-card node on the LinkedIn guest search page.

    Selectors target the stable ``base-card`` and ``job-search-card``
    classes LinkedIn uses for guest results. Returns an empty list
    when the page is fronted by a sign-in wall (no job-card nodes).
    """
    tree = HTMLParser(html)
    cards = tree.css("li div.base-card") or tree.css("div.job-search-card")
    return list(cards or [])


def _parse_card(card: Node) -> NormalizedJob | None:
    """Map one LinkedIn job-card node onto a :class:`NormalizedJob`.

    Returns ``None`` if the card lacks a stable id or title — every
    other field has a sensible fallback.
    """
    job_id = _extract_job_id(card)
    title = _text(card, "h3.base-search-card__title")
    apply_url = _href(card, "a.base-card__full-link")
    if job_id is None or title is None or apply_url is None:
        return None

    company = _text(card, "h4.base-search-card__subtitle a") or _text(
        card, "h4.base-search-card__subtitle"
    )
    location = _text(card, "span.job-search-card__location")
    posted_at = _attr(card, "time.job-search-card__listdate", "datetime") or _attr(
        card, "time.job-search-card__listdate--new", "datetime"
    )

    return NormalizedJob(
        source_external_id=str(job_id),
        title=title.strip(),
        company=(company or "Unknown").strip(),
        apply_url=_canonical_apply_url(apply_url),
        raw_data=_card_to_raw(card),
        location_raw=location.strip() if location else None,
        location_country=_country_from(location),
        location_city=_city_from(location),
        remote_type=_remote_from(location),
        posted_at=posted_at,
    )


def _canonical_apply_url(href: str) -> str:
    """Normalise to ``host + /jobs/view/{slug}-{id}`` without tracking.

    Guest-mode LinkedIn URLs include rotating ``refId`` /
    ``trackingId`` / ``position`` params that bloat the dedup index
    with new-looking rows on every scrape. Stripping the query keeps
    the canonical id surfaced in the path stable across runs.
    """
    full = urljoin(_BASE_URL, href)
    parsed = urlparse(full)
    return urlunparse(parsed._replace(query="", fragment=""))


def _extract_job_id(card: Node) -> str | None:
    """Pull a stable job id from the card's URN or apply URL."""
    urn = card.attributes.get("data-entity-urn") or ""
    if urn:
        match = _JOB_ID_RE.search(urn)
        if match:
            return match.group(1)
    href = _href(card, "a.base-card__full-link") or ""
    match = _VIEW_ID_RE.search(href)
    if match:
        return match.group(1)
    return None


def _text(card: Node, selector: str) -> str | None:
    node = card.css_first(selector)
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


def _href(card: Node, selector: str) -> str | None:
    node = card.css_first(selector)
    if node is None:
        return None
    return node.attributes.get("href")


def _attr(card: Node, selector: str, attribute: str) -> str | None:
    node = card.css_first(selector)
    if node is None:
        return None
    return node.attributes.get(attribute)


def _card_to_raw(card: Node) -> dict[str, Any]:
    """Snapshot the card's outer HTML so re-parses can run later
    without re-fetching."""
    return {"html": card.html or ""}


def _country_from(location: str | None) -> str | None:
    """Best-effort country extraction from LinkedIn's free-text location.

    Defaults to Australia (the agent's primary market) when the
    location string contains a recognisable Australian city.
    """
    if not location:
        return None
    lower = location.lower()
    if any(city in lower for city in ("australia", "sydney", "melbourne", "brisbane", "perth")):
        return "Australia"
    if "united states" in lower or "usa" in lower:
        return "United States"
    if "united kingdom" in lower or " uk" in lower:
        return "United Kingdom"
    return None


def _city_from(location: str | None) -> str | None:
    """First comma-separated segment is the city in LinkedIn's format."""
    if not location:
        return None
    head = location.split(",")[0].strip()
    return head or None


def _remote_from(location: str | None) -> str | None:
    if not location:
        return None
    lower = location.lower()
    if "remote" in lower:
        return "remote"
    if "hybrid" in lower:
        return "hybrid"
    return None
