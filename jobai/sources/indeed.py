"""Indeed (au.indeed.com) source.

Indeed serves a JS-heavy SPA, but the search-result page also embeds
a server-rendered ``window._initialData`` JSON blob with the full
result set. Parsing that island is more reliable than CSS-selectoring
the rendered DOM (Indeed reshuffles class names per A/B variant).

The ``account`` carries the URL query parameters: ``"q=python&l=Melbourne"``
plugs into ``https://au.indeed.com/jobs?q=python&l=Melbourne``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode, urljoin

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://au.indeed.com"
_SEARCH_PATH = "/jobs"

#: Match the inline data island the SSR pass emits before the React
#: bundle hydrates. The JSON is single-line for the modern Indeed
#: build but the regex stays DOTALL-tolerant in case that changes.
_INITIAL_DATA_RE = re.compile(
    r"window\._initialData\s*=\s*(\{.*?\})\s*;\s*window\.",
    re.DOTALL,
)

#: Older Indeed variants emit ``window.mosaic.providerData["mosaic-provider-jobcards"]``
#: instead. We try this fallback when ``_initialData`` is missing.
_MOSAIC_DATA_RE = re.compile(
    r'mosaic-provider-jobcards["\']\s*\]\s*=\s*(\{.*?\})\s*;\s*window\.',
    re.DOTALL,
)


class IndeedSource(BaseSource):
    """One Indeed (au.indeed.com) jobs search.

    ``account`` is the URL-encoded query string for ``/jobs``,
    e.g. ``"q=python&l=Melbourne&fromage=7"``.
    """

    kind = "indeed"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = f"{_BASE_URL}{_SEARCH_PATH}?{self.account}"
        response = await fetcher.fetch(url)
        if not response.is_ok:
            raise IndeedFetchError(self.account, response.status_code)

        for entry in _extract_results(response.text):
            job = _parse_job(entry)
            if job is not None:
                yield job


class IndeedFetchError(RuntimeError):
    """Raised when Indeed returns a non-2xx status."""

    def __init__(self, query: str, status_code: int) -> None:
        super().__init__(f"indeed:{query} returned HTTP {status_code}")
        self.query = query
        self.status_code = status_code


def build_query(*, keywords: str, location: str = "Australia", recency_days: int = 7) -> str:
    """Build the URL-encoded query for an :class:`IndeedSource` account."""
    return urlencode({"q": keywords, "l": location, "fromage": str(recency_days)})


def _extract_results(html: str) -> list[dict[str, Any]]:
    """Walk the embedded data island and return the per-job dicts."""
    payload = _parse_initial_data(html) or _parse_mosaic_data(html)
    if not isinstance(payload, dict):
        return []

    # `_initialData` shape: {"jobsearch": {"results": [...]}}.
    candidates = (
        _safe_dig(payload, "jobsearch", "results"),
        _safe_dig(payload, "metaData", "mosaicProviderJobCardsModel", "results"),
        _safe_dig(payload, "results"),
    )
    for results in candidates:
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
    return []


def _parse_initial_data(html: str) -> Any:
    match = _INITIAL_DATA_RE.search(html)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _parse_mosaic_data(html: str) -> Any:
    match = _MOSAIC_DATA_RE.search(html)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _safe_dig(payload: Any, *keys: str) -> Any:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _parse_job(entry: dict[str, Any]) -> NormalizedJob | None:
    job_key = entry.get("jobkey") or entry.get("jobKey") or entry.get("id")
    title = entry.get("title") or entry.get("displayTitle")
    if job_key is None or not isinstance(title, str) or not title.strip():
        return None

    apply_url = _resolve_apply_url(entry, job_key)
    company = entry.get("company") or entry.get("companyName") or "Unknown"
    location = entry.get("formattedLocation") or entry.get("locationName")
    salary_text = (
        (entry.get("salarySnippet") or {}).get("text")
        if isinstance(entry.get("salarySnippet"), dict)
        else entry.get("salaryText")
    )
    posted_at = entry.get("formattedRelativeTime") or entry.get("createDate")

    return NormalizedJob(
        source_external_id=str(job_key),
        title=title.strip(),
        company=str(company).strip(),
        apply_url=apply_url,
        raw_data=entry,
        location_raw=str(location).strip() if isinstance(location, str) else None,
        location_country="Australia",
        location_city=_city_from(location),
        remote_type=_remote_from(entry, location),
        employment_type=_normalise_str(entry.get("jobTypes")),
        posted_at=str(posted_at) if posted_at else None,
        salary_min=_parse_salary_min(salary_text),
        salary_max=_parse_salary_max(salary_text),
        salary_currency="AUD" if salary_text else None,
        description_text=_normalise_str(entry.get("snippet")),
    )


def _resolve_apply_url(entry: dict[str, Any], job_key: Any) -> str:
    explicit = entry.get("viewJobLink") or entry.get("link")
    if isinstance(explicit, str) and explicit:
        return urljoin(_BASE_URL, explicit)
    return f"{_BASE_URL}/viewjob?jk={job_key}"


def _city_from(location: Any) -> str | None:
    if not isinstance(location, str) or not location:
        return None
    head = location.split(",")[0].strip()
    return head or None


def _remote_from(entry: dict[str, Any], location: Any) -> str | None:
    remote_attr = entry.get("remoteWorkModel") or entry.get("workArrangementType")
    if isinstance(remote_attr, str):
        lower = remote_attr.lower()
        if "remote" in lower:
            return "remote"
        if "hybrid" in lower:
            return "hybrid"
    if isinstance(location, str):
        lower = location.lower()
        if "remote" in lower:
            return "remote"
        if "hybrid" in lower:
            return "hybrid"
    return None


def _normalise_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        joined = ", ".join(str(v) for v in value if isinstance(v, str))
        return joined or None
    return None


_SALARY_INT_RE = re.compile(r"\$?\s*([\d,]+)")


def _parse_salary_min(salary_text: Any) -> int | None:
    if not isinstance(salary_text, str):
        return None
    matches = _SALARY_INT_RE.findall(salary_text)
    if not matches:
        return None
    return _to_int(matches[0])


def _parse_salary_max(salary_text: Any) -> int | None:
    if not isinstance(salary_text, str):
        return None
    matches = _SALARY_INT_RE.findall(salary_text)
    if len(matches) < 2:
        return None
    return _to_int(matches[1])


def _to_int(token: str) -> int | None:
    cleaned = token.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    value = int(cleaned)
    if value < 1000:
        return value * 1000
    return value
