"""SmartRecruiters ATS source.

SmartRecruiters exposes two public endpoints we use:

1. Listing (paginated)::

     GET https://api.smartrecruiters.com/v1/companies/{company}/postings
         ?limit={limit}&offset={offset}

   Returns ``{offset, limit, totalFound, content: [...]}``. Each
   listing entry has structured location, industry, department,
   experienceLevel, and typeOfEmployment, but **no description**.

2. Detail (per job)::

     GET https://api.smartrecruiters.com/v1/companies/{company}/postings/{uuid}

   Returns the same listing fields plus a ``jobAd.sections`` block
   with ``companyDescription``, ``jobDescription``, ``qualifications``,
   ``additionalInformation`` — the full text content.

This is a two-stage source per ARCHITECTURE §4.4: listing fetches
discover UUIDs, per-job fetches fill in the description. The runner
sees both fetches transparently because they go through the same
fetcher / RecordingFetcher chain.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from jobai.fetcher.base import Fetcher, Response
from jobai.sources.base import BaseSource, NormalizedJob

_LIST_URL_TEMPLATE = (
    "https://api.smartrecruiters.com/v1/companies/{company}/postings?limit={limit}&offset={offset}"
)
_DETAIL_URL_TEMPLATE = "https://api.smartrecruiters.com/v1/companies/{company}/postings/{uuid}"
_LISTING_PAGE_SIZE = 100


class SmartRecruitersFetchError(RuntimeError):
    """Raised when SmartRecruiters returns a non-2xx for a listing or detail page."""

    def __init__(self, account: str, status_code: int, *, stage: str) -> None:
        super().__init__(f"smartrecruiters:{account} returned HTTP {status_code} during {stage}")
        self.account = account
        self.status_code = status_code
        self.stage = stage


class SmartRecruitersSource(BaseSource):
    """Pulls jobs from one SmartRecruiters-hosted company board.

    Iterates the paginated listing endpoint, then for each posting
    fetches the detail endpoint to attach description fields. Yields
    one NormalizedJob per posting.
    """

    kind = "smartrecruiters"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        async for listing_job in self._iter_listing(fetcher):
            uuid = listing_job.get("uuid") or listing_job.get("id")
            if not uuid:
                continue
            detail = await self._fetch_detail(fetcher, str(uuid))
            yield _parse_job(listing_job, detail, account=self.account)

    async def _iter_listing(self, fetcher: Fetcher) -> AsyncIterator[dict[str, Any]]:
        offset = 0
        while True:
            url = _LIST_URL_TEMPLATE.format(
                company=self.account,
                limit=_LISTING_PAGE_SIZE,
                offset=offset,
            )
            response = await fetcher.fetch(url)
            if not response.is_ok:
                raise SmartRecruitersFetchError(self.account, response.status_code, stage="listing")
            payload = _parse_json_dict(response, account=self.account, stage="listing")

            content = payload.get("content")
            if not isinstance(content, list):
                return
            for entry in content:
                if isinstance(entry, dict):
                    yield entry

            total = payload.get("totalFound")
            if not isinstance(total, int):
                return
            offset += len(content)
            if offset >= total or not content:
                return

    async def _fetch_detail(self, fetcher: Fetcher, uuid: str) -> dict[str, Any]:
        url = _DETAIL_URL_TEMPLATE.format(company=self.account, uuid=uuid)
        response = await fetcher.fetch(url)
        if not response.is_ok:
            raise SmartRecruitersFetchError(self.account, response.status_code, stage="detail")
        return _parse_json_dict(response, account=self.account, stage="detail")


def _parse_json_dict(response: Response, *, account: str, stage: str) -> dict[str, Any]:
    parsed = json.loads(response.body)
    if not isinstance(parsed, dict):
        raise SmartRecruitersFetchError(account, response.status_code, stage=stage)
    return parsed


def _parse_job(
    listing: dict[str, Any],
    detail: dict[str, Any],
    *,
    account: str,
) -> NormalizedJob:
    """Combine the listing record and detail record into one NormalizedJob."""
    company_block = listing.get("company")
    company = (company_block.get("name") if isinstance(company_block, dict) else None) or account

    description_html = _compose_description_html(detail)

    return NormalizedJob(
        source_external_id=str(listing.get("uuid") or listing.get("id") or ""),
        title=str(listing.get("name") or ""),
        company=str(company),
        apply_url=str(detail.get("applyUrl") or listing.get("applyUrl") or ""),
        raw_data={"listing": listing, "detail": detail},
        location_raw=_format_location(listing.get("location")),
        location_country=_extract_location_field(listing, "country"),
        location_city=_extract_location_field(listing, "city"),
        remote_type=_remote_type_from_location(listing.get("location")),
        employment_type=_extract_label(listing.get("typeOfEmployment")),
        posted_at=_first_str(listing.get("releasedDate"), default=None),
        description_html=description_html,
    )


def _first_str(value: Any, *, default: str | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    return default


def _extract_label(field: Any) -> str | None:
    """SmartRecruiters wraps enum-ish values as ``{id, label}``; return label."""
    if isinstance(field, dict):
        label = field.get("label")
        if isinstance(label, str):
            return label.lower().replace(" ", "-")
    return None


def _format_location(location: Any) -> str | None:
    if isinstance(location, dict):
        full = location.get("fullLocation")
        if isinstance(full, str) and full:
            return full
        parts = [location.get("city"), location.get("region"), location.get("country")]
        cleaned = [str(p) for p in parts if isinstance(p, str) and p]
        if cleaned:
            return ", ".join(cleaned)
    return None


def _extract_location_field(listing: dict[str, Any], field: str) -> str | None:
    location = listing.get("location")
    if isinstance(location, dict):
        value = location.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _remote_type_from_location(location: Any) -> str | None:
    """Use SR's structured remote/hybrid bools."""
    if not isinstance(location, dict):
        return None
    if location.get("remote") is True:
        return "remote"
    if location.get("hybrid") is True:
        return "hybrid"
    if location.get("remote") is False and location.get("hybrid") is False:
        return "onsite"
    return None


def _compose_description_html(detail: dict[str, Any]) -> str | None:
    """Stitch the four jobAd sections into one HTML string."""
    job_ad = detail.get("jobAd")
    if not isinstance(job_ad, dict):
        return None
    sections = job_ad.get("sections")
    if not isinstance(sections, dict):
        return None

    chunks: list[str] = []
    for key in ("companyDescription", "jobDescription", "qualifications", "additionalInformation"):
        section = sections.get(key)
        if not isinstance(section, dict):
            continue
        title = section.get("title")
        text = section.get("text")
        if isinstance(title, str) and title:
            chunks.append(f"<h2>{title}</h2>")
        if isinstance(text, str) and text:
            chunks.append(text)
    return "\n".join(chunks) if chunks else None
