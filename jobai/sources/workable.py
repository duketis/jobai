"""Workable ATS source.

Workable exposes a public widget endpoint at::

    https://apply.workable.com/api/v1/widget/accounts/{account}

This returns the board's listing-level fields per job (title,
shortcode, employment_type, telecommuting, department, country, city,
state, locations[]) but **not the description**. The per-job page at
``apply.workable.com/{account}/j/{shortcode}`` is a JS-rendered SPA
and the documented v3 ``/jobs`` endpoint has been gated; no plain-HTTP
path returns the description text.

Per the data-completeness invariant (ARCHITECTURE §4.4), this is a
known gap. Phase 5 (browser-tier fetcher) will render the SPA and
extract the description; until then NormalizedJob.description_html /
description_text are ``None`` for Workable rows. The full listing
payload is preserved in raw_data so re-extraction is free once the
browser tier ships — no re-scrape needed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BOARD_URL_TEMPLATE = "https://apply.workable.com/api/v1/widget/accounts/{account}"


class WorkableFetchError(RuntimeError):
    """Raised when a Workable account returns a non-2xx status."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"workable:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class WorkableSource(BaseSource):
    """Pulls listing-level data from one Workable-hosted board.

    Description fields will be backfilled by the browser-tier fetcher
    (Phase 5). Listing fields are complete now.
    """

    kind = "workable"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = _BOARD_URL_TEMPLATE.format(account=self.account)
        response = await fetcher.fetch(url)

        if not response.is_ok:
            raise WorkableFetchError(self.account, response.status_code)

        payload = json.loads(response.body)
        if not isinstance(payload, dict):
            raise WorkableFetchError(self.account, response.status_code)

        company_name = str(payload.get("name") or self.account)
        for job in payload.get("jobs", []):
            if isinstance(job, dict):
                yield _parse_job(job, company=company_name)


def _parse_job(job: dict[str, Any], *, company: str) -> NormalizedJob:
    shortcode = str(job.get("shortcode") or job.get("id") or "")
    return NormalizedJob(
        source_external_id=shortcode,
        title=str(job.get("title") or ""),
        company=company,
        apply_url=str(job.get("application_url") or job.get("shortlink") or job.get("url") or ""),
        raw_data=job,
        location_raw=_format_location(job),
        location_country=_first_str(job.get("country"), default=None),
        location_city=_first_str(job.get("city"), default=None),
        remote_type=_remote_type_from_telecommuting(job.get("telecommuting")),
        employment_type=_normalise_employment_type(job.get("employment_type")),
        posted_at=_first_str(job.get("published_on") or job.get("created_at"), default=None),
    )


def _first_str(value: Any, *, default: str | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    return default


def _format_location(job: dict[str, Any]) -> str | None:
    """Compose a free-text location string from city / state / country."""
    parts = [job.get("city"), job.get("state"), job.get("country")]
    cleaned = [p for p in parts if isinstance(p, str) and p]
    return ", ".join(cleaned) if cleaned else None


def _remote_type_from_telecommuting(value: Any) -> str | None:
    """Workable models remote-vs-onsite as a single bool, ``telecommuting``.

    True -> 'remote'; False -> 'onsite'; anything else (None, missing) -> None
    so we don't fabricate a value when the field is absent.
    """
    if value is True:
        return "remote"
    if value is False:
        return "onsite"
    return None


def _normalise_employment_type(raw: Any) -> str | None:
    """Workable uses values like 'Full-time' / 'Part-time' / 'Contract'."""
    if not isinstance(raw, str) or not raw:
        return None
    lowered = raw.lower().strip()
    mapping = {
        "full-time": "full-time",
        "fulltime": "full-time",
        "full time": "full-time",
        "part-time": "part-time",
        "parttime": "part-time",
        "part time": "part-time",
        "contract": "contract",
        "contractor": "contract",
        "intern": "internship",
        "internship": "internship",
        "temporary": "temporary",
    }
    return mapping.get(lowered, lowered)
