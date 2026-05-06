"""Seek source — Australia's dominant job board.

Seek has no public API. Its search-results page is a Next.js React
app that embeds the full result set as JSON in a
``<script id="__NEXT_DATA__">`` block. Parsing that island is far
more reliable than scraping the rendered DOM: the JSON is a stable
interface (it has to feed the React app on hydration), while CSS
class names change with every release.

A :class:`SeekSource` instance covers one search slug — the path
fragment from a Seek URL, e.g. ``"python-jobs/in-Melbourne-VIC"``
for ``https://www.seek.com.au/python-jobs/in-Melbourne-VIC``.
``companies.yaml`` (or its successor) lists the slugs we care about.

Pagination is intentionally not handled in this first cut: the
default Seek page returns ~20 results, which is enough for the
freshness signal the agent surfaces. Phase 6.x will add ``?page=N``
walks once we wire the source to the scheduler and want depth.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://www.seek.com.au"

#: Matches the Next.js data-island script tag and captures its JSON
#: body. ``re.DOTALL`` is required because the JSON spans many lines.
_NEXT_DATA_RE = re.compile(
    r'<script\s+id="__NEXT_DATA__"\s+type="application/json"[^>]*>(.*?)</script>',
    re.DOTALL,
)


class SeekSource(BaseSource):
    """Pulls jobs from one Seek search-result page.

    The ``account`` is the URL path fragment after the ``seek.com.au/``
    prefix, e.g. ``"python-jobs/in-Melbourne-VIC"``.
    """

    kind = "seek"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = f"{_BASE_URL}/{self.account}"
        response = await fetcher.fetch(url)

        if not response.is_ok:
            raise SeekFetchError(self.account, response.status_code)

        results = _extract_results(response.text)
        for result in results:
            job = _parse_job(result)
            if job is not None:
                yield job


class SeekFetchError(RuntimeError):
    """Raised when a Seek search page returns a non-2xx status."""

    def __init__(self, slug: str, status_code: int) -> None:
        super().__init__(f"seek:{slug} returned HTTP {status_code}")
        self.slug = slug
        self.status_code = status_code


def _extract_results(html: str) -> list[dict[str, Any]]:
    """Pull the search results array out of the Next.js data island.

    Returns an empty list if the island is missing or malformed —
    those are recoverable conditions a runner should treat as "no
    new jobs this cycle", not a hard failure.
    """
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    page_props = payload.get("props", {}).get("pageProps", {}) if isinstance(payload, dict) else {}
    search_results = page_props.get("searchResults") if isinstance(page_props, dict) else None
    if not isinstance(search_results, dict):
        return []
    results = search_results.get("results")
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


def _parse_job(result: dict[str, Any]) -> NormalizedJob | None:
    """Translate one Seek result dict to a :class:`NormalizedJob`.

    Returns ``None`` when the entry is missing the minimum required
    fields (id, title, an apply URL we can derive). The parser is
    defensive on every other field — Seek's payload shape varies
    between regions and listing types, so we ``.get()`` everything
    and fall back to ``None`` rather than raising.
    """
    job_id = result.get("id")
    title = result.get("title")
    if job_id is None or not isinstance(title, str) or not title:
        return None

    apply_url = _derive_apply_url(result, job_id)
    if apply_url is None:
        return None

    advertiser = result.get("advertiser") or {}
    company = (advertiser.get("description") if isinstance(advertiser, dict) else None) or "Unknown"

    location_raw = _first_str(
        result.get("location"),
        result.get("jobLocation"),
        _first_label(result.get("locations")),
    )

    salary_min, salary_max, salary_currency = _parse_salary(result)

    return NormalizedJob(
        source_external_id=str(job_id),
        title=title.strip(),
        company=str(company).strip(),
        apply_url=apply_url,
        raw_data=result,
        location_raw=location_raw,
        location_country=_first_str(
            _country_from_locations(result.get("locations")),
            "Australia",
        ),
        location_city=_city_from_locations(result.get("locations")),
        remote_type=_infer_remote_type(result),
        employment_type=_normalise_str(result.get("workType")),
        posted_at=_normalise_str(result.get("listingDate")),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        description_text=_normalise_str(result.get("teaser")),
    )


def _derive_apply_url(result: dict[str, Any], job_id: Any) -> str | None:
    """Return the canonical Seek apply URL for a result entry."""
    explicit = result.get("url")
    if isinstance(explicit, str) and explicit.startswith("http"):
        return explicit
    return f"{_BASE_URL}/job/{job_id}"


def _first_str(*candidates: Any) -> str | None:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    return None


def _first_label(value: Any) -> str | None:
    """Return the ``label`` of the first dict in a list, else None."""
    if not isinstance(value, list):
        return None
    for entry in value:
        if isinstance(entry, dict):
            label = entry.get("label")
            if isinstance(label, str) and label.strip():
                return label.strip()
    return None


def _country_from_locations(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for entry in value:
        if isinstance(entry, dict):
            country = entry.get("country")
            if isinstance(country, str) and country.strip():
                return country.strip()
    return None


def _city_from_locations(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for entry in value:
        if isinstance(entry, dict):
            city = entry.get("city") or entry.get("suburb")
            if isinstance(city, str) and city.strip():
                return city.strip()
    return None


def _normalise_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _infer_remote_type(result: dict[str, Any]) -> str | None:
    """Map Seek's work-arrangements entries onto remote/hybrid/onsite."""
    arrangements = result.get("workArrangements")
    if isinstance(arrangements, dict):
        data = arrangements.get("data")
        if isinstance(data, list):
            labels = [entry.get("label", "").lower() for entry in data if isinstance(entry, dict)]
            if "remote" in labels:
                return "remote"
            if "hybrid" in labels:
                return "hybrid"
            if "on-site" in labels or "onsite" in labels:
                return "onsite"
    return None


# Seek uses Unicode dashes in salary ranges alongside plain hyphens.
# Built via \N escapes so the source file stays ASCII (avoids ruff
# RUF001 ambiguous-character warnings on en/em-dashes).
_DASH_CLASS = "-\N{EN DASH}\N{EM DASH}"
_SALARY_RANGE_RE = re.compile(
    rf"\$?\s*([\d,]+)\s*(?:[{_DASH_CLASS}]|to)\s*\$?\s*([\d,]+)",
    re.IGNORECASE,
)
_SALARY_SINGLE_RE = re.compile(r"\$?\s*([\d,]+)")


def _parse_salary(result: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    """Parse Seek's free-text salary string into ``(min, max, currency)``.

    Seek emits salary as a string like ``"$140,000 - $160,000"`` or
    ``"$80k+"``. We look for two integers (range) or one (single
    number). Any failure → all-Nones, preserving the raw text in
    ``raw_data``.
    """
    salary = result.get("salary")
    if not isinstance(salary, str) or not salary.strip():
        return None, None, None

    range_match = _SALARY_RANGE_RE.search(salary)
    if range_match:
        low = _to_int(range_match.group(1))
        high = _to_int(range_match.group(2))
        if low is not None and high is not None:
            return low, high, "AUD"

    single_match = _SALARY_SINGLE_RE.search(salary)
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
    # Detect "80k" style by checking length — anything under 1000 in a
    # salary context is almost certainly thousands and we should multiply.
    if value < 1000:
        return value * 1000
    return value
