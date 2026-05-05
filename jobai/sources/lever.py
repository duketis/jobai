"""Lever ATS source.

Lever exposes a public postings API at::

    https://api.lever.co/v0/postings/{account}?mode=json

The endpoint returns a flat JSON array of job objects (no wrapper, no
pagination). Each object includes a structured ``categories`` block
(location, team, department, commitment), ``descriptionHtml`` /
``descriptionPlain``, and ``workplaceType`` ('remote' | 'hybrid' |
'on-site') — all the fields a human sees on jobs.lever.co/{account}.

Single-stage fetch; no per-job detail required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BOARD_URL_TEMPLATE = "https://api.lever.co/v0/postings/{account}?mode=json"


class LeverFetchError(RuntimeError):
    """Raised when a Lever board returns a non-2xx status."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"lever:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class LeverSource(BaseSource):
    """Pulls jobs from one Lever-hosted postings board."""

    kind = "lever"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        url = _BOARD_URL_TEMPLATE.format(account=self.account)
        response = await fetcher.fetch(url)

        if not response.is_ok:
            raise LeverFetchError(self.account, response.status_code)

        payload = json.loads(response.body)
        if not isinstance(payload, list):
            raise LeverFetchError(self.account, response.status_code)

        for job in payload:
            if isinstance(job, dict):
                yield _parse_job(job, account=self.account)


def _parse_job(job: dict[str, Any], *, account: str) -> NormalizedJob:
    """Map one Lever posting to :class:`NormalizedJob`."""
    categories = job.get("categories") or {}
    location_raw = categories.get("location") if isinstance(categories, dict) else None

    return NormalizedJob(
        source_external_id=str(job["id"]),
        title=str(job.get("text") or ""),
        company=account,
        apply_url=str(job.get("applyUrl") or job.get("hostedUrl") or ""),
        raw_data=job,
        location_raw=location_raw,
        remote_type=_normalise_workplace_type(job.get("workplaceType")),
        employment_type=_extract_commitment(categories),
        posted_at=_extract_created_at(job),
        description_html=job.get("descriptionHtml") or job.get("description"),
        description_text=job.get("descriptionPlain"),
    )


def _normalise_workplace_type(raw: Any) -> str | None:
    """Map Lever's workplace strings to our remote_type vocabulary."""
    if not isinstance(raw, str):
        return None
    lowered = raw.lower().replace("-", "").replace(" ", "")
    if "remote" in lowered:
        return "remote"
    if "hybrid" in lowered:
        return "hybrid"
    if "onsite" in lowered or "office" in lowered:
        return "onsite"
    return None


def _extract_commitment(categories: Any) -> str | None:
    """Read the employment-type-ish field Lever calls 'commitment'."""
    if not isinstance(categories, dict):
        return None
    value = categories.get("commitment")
    return str(value) if isinstance(value, str) else None


def _extract_created_at(job: dict[str, Any]) -> str | None:
    """Lever stores createdAt as a millisecond Unix timestamp; convert to ISO."""
    raw = job.get("createdAt")
    if not isinstance(raw, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(raw / 1000, tz=UTC).isoformat()
    except (ValueError, OSError, OverflowError):
        return None
