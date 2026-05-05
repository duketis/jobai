"""Ashby ATS source.

Ashby exposes a public job-board endpoint at::

    https://api.ashbyhq.com/posting-api/job-board/{account}?includeCompensation=true

The response wraps a flat ``jobs`` array with structured per-job
fields: ``descriptionHtml`` / ``descriptionPlain``, ``workplaceType``
('Remote' | 'Hybrid' | 'OnSite'), ``employmentType``, ``location``,
``compensation``, ``address.postalAddress.addressCountry``. Single
fetch returns the entire board; no pagination needed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BOARD_URL_TEMPLATE = (
    "https://api.ashbyhq.com/posting-api/job-board/{account}?includeCompensation=true"
)


class AshbyFetchError(RuntimeError):
    """Raised when an Ashby board returns a non-2xx status."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"ashby:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class AshbySource(BaseSource):
    """Pulls jobs from one Ashby-hosted job board."""

    kind = "ashby"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = _BOARD_URL_TEMPLATE.format(account=self.account)
        response = await fetcher.fetch(url)

        if not response.is_ok:
            raise AshbyFetchError(self.account, response.status_code)

        payload = json.loads(response.body)
        if not isinstance(payload, dict):
            raise AshbyFetchError(self.account, response.status_code)

        for job in payload.get("jobs", []):
            if isinstance(job, dict) and job.get("isListed") is not False:
                yield _parse_job(job, account=self.account)


def _parse_job(job: dict[str, Any], *, account: str) -> NormalizedJob:
    """Map one Ashby posting to :class:`NormalizedJob`."""
    salary_min, salary_max, salary_currency = _extract_compensation(job.get("compensation"))

    return NormalizedJob(
        source_external_id=str(job["id"]),
        title=str(job.get("title") or ""),
        company=account,
        apply_url=str(job.get("applyUrl") or job.get("jobUrl") or ""),
        raw_data=job,
        location_raw=_first_str(job.get("location"), default=None),
        location_country=_extract_country(job),
        remote_type=_normalise_workplace_type(
            job.get("workplaceType"),
            is_remote=job.get("isRemote"),
        ),
        employment_type=_normalise_employment_type(job.get("employmentType")),
        posted_at=_first_str(job.get("publishedAt"), default=None),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        description_html=job.get("descriptionHtml"),
        description_text=job.get("descriptionPlain"),
    )


def _first_str(value: Any, *, default: str | None) -> str | None:
    """Coerce a value to a non-empty string, falling back to ``default``."""
    if isinstance(value, str) and value:
        return value
    return default


def _normalise_workplace_type(raw: Any, *, is_remote: Any) -> str | None:
    """Map Ashby's PascalCase workplace strings + isRemote bool to remote_type."""
    if isinstance(raw, str):
        lowered = raw.lower()
        if "remote" in lowered:
            return "remote"
        if "hybrid" in lowered:
            return "hybrid"
        if "onsite" in lowered or "on-site" in lowered or "office" in lowered:
            return "onsite"
    if is_remote is True:
        return "remote"
    return None


def _normalise_employment_type(raw: Any) -> str | None:
    """Convert FullTime / PartTime / Contract / Intern to lower-case-hyphenated."""
    if not isinstance(raw, str):
        return None
    mapping = {
        "fulltime": "full-time",
        "parttime": "part-time",
        "contract": "contract",
        "contractor": "contract",
        "intern": "internship",
        "internship": "internship",
        "temporary": "temporary",
    }
    return mapping.get(raw.lower(), raw.lower())


def _extract_country(job: dict[str, Any]) -> str | None:
    address = job.get("address") or {}
    if not isinstance(address, dict):
        return None
    postal = address.get("postalAddress") or {}
    if not isinstance(postal, dict):
        return None
    country = postal.get("addressCountry")
    return country if isinstance(country, str) else None


def _extract_compensation(
    comp: Any,
) -> tuple[int | None, int | None, str | None]:
    """Best-effort pull of (min, max, currency) from Ashby's compensation block.

    Ashby's compensation shape varies; we read the most common
    structure and quietly skip values we don't recognise. Missing
    salary information is the rule, not the exception.
    """
    if not isinstance(comp, dict):
        return (None, None, None)
    summary_components = comp.get("summaryComponents")
    if not isinstance(summary_components, list):
        return (None, None, None)
    for entry in summary_components:
        if not isinstance(entry, dict):
            continue
        if entry.get("compensationType") != "Salary":
            continue
        min_value = entry.get("minValue")
        max_value = entry.get("maxValue")
        currency = entry.get("currencyCode")
        return (
            int(min_value) if isinstance(min_value, (int, float)) else None,
            int(max_value) if isinstance(max_value, (int, float)) else None,
            currency if isinstance(currency, str) else None,
        )
    return (None, None, None)
