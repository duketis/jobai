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

#: Cap on pages walked per scrape cycle. Indeed serves 10 cards per
#: page so a 5-page walk yields ~50 results — enough for a freshness
#: signal while staying under any per-IP rate limits we'd otherwise
#: hit on the browser tier.
DEFAULT_MAX_PAGES = 5

#: Indeed's ``start`` query param is a 0-indexed offset into the
#: result set, advancing by ``_PAGE_SIZE`` per page.
_PAGE_SIZE = 10

#: Locate the start of each candidate data island. The JSON object
#: that follows is extracted via brace balancing in
#: :func:`_extract_balanced_object` since regex backtracking on a
#: ``\{.*?\}`` against multi-MB pages is both slow and fragile (a
#: single ``}`` inside a string literal trips a non-greedy match).
_INITIAL_DATA_START_RE = re.compile(r"window\._initialData\s*=\s*(?=\{)")
_MOSAIC_DATA_START_RE = re.compile(r'mosaic-provider-jobcards["\']\s*\]\s*=\s*(?=\{)')


class IndeedSource(BaseSource):
    """One Indeed (au.indeed.com) jobs search (paginated).

    ``account`` is the URL-encoded query string for ``/jobs``,
    e.g. ``"q=python&l=Melbourne&fromage=7"``.
    """

    kind = "indeed"

    def __init__(self, account: str = "", *, max_pages: int = DEFAULT_MAX_PAGES) -> None:
        super().__init__(account)
        if max_pages < 1:
            msg = f"max_pages must be >= 1, got {max_pages}"
            raise ValueError(msg)
        self._max_pages = max_pages

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        seen_ids: set[str] = set()
        for page in range(self._max_pages):
            url = _page_url(self.account, page)
            response = await fetcher.fetch(url)
            if not response.is_ok:
                if page == 0:
                    raise IndeedFetchError(self.account, response.status_code)
                return

            page_yielded = 0
            for entry in _extract_results(response.text):
                job = _parse_job(entry)
                if job is None or job.source_external_id in seen_ids:
                    continue
                seen_ids.add(job.source_external_id)
                page_yielded += 1
                yield job
            if page_yielded == 0:
                return


def _page_url(account: str, page: int) -> str:
    base = f"{_BASE_URL}{_SEARCH_PATH}?{account}"
    if page == 0:
        return base
    sep = "&" if account else ""
    return f"{base}{sep}start={page * _PAGE_SIZE}"


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
    """Walk the embedded data island and return the per-job dicts.

    Modern Indeed renders both ``window._initialData`` (page-level
    state) and ``window.mosaic.providerData["mosaic-provider-jobcards"]``
    (the search-results provider). The job list lives in different
    paths in each. We try every known path against every payload so
    a renamed root key in one doesn't drag down the whole parser.
    """
    payloads = [
        p for p in (_parse_initial_data(html), _parse_mosaic_data(html)) if isinstance(p, dict)
    ]
    if not payloads:
        return []

    paths = (
        ("jobsearch", "results"),
        ("metaData", "mosaicProviderJobCardsModel", "results"),
        ("results",),
    )
    for payload in payloads:
        for path in paths:
            results = _safe_dig(payload, *path)
            if isinstance(results, list) and results:
                return [r for r in results if isinstance(r, dict)]
    return []


def _parse_initial_data(html: str) -> Any:
    return _extract_payload(html, _INITIAL_DATA_START_RE)


def _parse_mosaic_data(html: str) -> Any:
    return _extract_payload(html, _MOSAIC_DATA_START_RE)


def _extract_payload(html: str, anchor_re: re.Pattern[str]) -> Any:
    match = anchor_re.search(html)
    if match is None:
        return None
    raw = _extract_balanced_object(html, match.end())
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_balanced_object(text: str, start: int) -> str | None:
    """Return the JSON object literal that begins at ``text[start]``.

    Walks the string counting brace depth, respecting string literals
    and escapes so a ``}`` embedded in a string doesn't close the
    object early. Returns ``None`` if the literal is malformed or
    truncated (which we treat as "no data" rather than raising).
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
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
    """Return a canonical ``/viewjob?jk={key}`` apply URL.

    Indeed's ``viewJobLink`` field comes back with multi-KB tracking
    payloads (``advn``, ``adid``, ``xkcb``, ``continueUrl``, ...).
    Anchoring on ``jk`` keeps dedup keys stable across runs and the
    DB rows readable. The ``link`` fallback is used only when no
    ``jk`` is available.
    """
    if job_key is not None:
        return f"{_BASE_URL}/viewjob?jk={job_key}"
    explicit = entry.get("viewJobLink") or entry.get("link")
    if isinstance(explicit, str) and explicit:
        return urljoin(_BASE_URL, explicit)
    return _BASE_URL


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
