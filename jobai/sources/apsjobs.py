"""APS Jobs (Australian Public Service) source.

The APS Jobs site (``apsjobs.gov.au``) migrated from a static Atom
feed to a Salesforce Lightning community in 2026. The new SPA loads
jobs through Aura ApexAction calls — specifically the
``aps_jobSearchController.retrieveJobListings`` Apex method invoked
through the ``/s/sfsites/aura`` endpoint.

This source replays that XHR over plain HTTP. The Aura context — the
versioned ``fwuid`` token and the
``APPLICATION@markup://siteforce:communityApp`` build token — is
extracted from a bootstrap GET of ``/s/job-search``. Both tokens roll
when Salesforce ships an update, so we re-extract on every scrape
cycle rather than caching.

The Apex action returns a single JSON page of 15 listings plus the
total count and the next offset; we paginate by replaying the same
request with ``offset += newOffset`` until an empty page terminates
the walk. The endpoint is unauthenticated — ``aura.token=null``.

The ``account`` field is a free-text search string (mapped to
``filter.searchString``); empty means the full open-jobs feed.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast
from urllib.parse import unquote

from jobai.fetcher.base import Fetcher, Response
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://www.apsjobs.gov.au"
_BOOTSTRAP_PATH = "/s/job-search"
_AURA_PATH = "/s/sfsites/aura"
_DETAIL_URL_TEMPLATE = f"{_BASE_URL}/s/job-details?jobId={{job_id}}"
_AURA_APP = "siteforce:communityApp"

#: Apex action descriptor — same string Salesforce's own JS sends.
_ACTION_DESCRIPTOR = "aura://ApexActionController/ACTION$execute"
_APEX_CLASS = "aps_jobSearchController"
_APEX_METHOD = "retrieveJobListings"

#: Page-size guard — APS returns 15 per call but the server is the
#: source of truth (uses ``newOffset``); 200 pages is a hard ceiling
#: against runaway loops if the API ever stops decrementing.
_MAX_PAGES = 200

#: Regex extracting ``fwuid`` from the inline JS Aura config block.
_FWUID_RE = re.compile(r'"fwuid":"([^"]+)"')

#: Regex extracting the ``APPLICATION@markup://siteforce:communityApp``
#: build token. The ``@`` and ``//`` make a literal-anchored regex
#: simpler than a CSS lookup, and the value is a stable opaque token.
_APP_TOKEN_RE = re.compile(
    r'"APPLICATION@markup://siteforce:communityApp":"([^"]+)"',
)

#: Empty-filter shape Salesforce's JS sends when no facets are picked.
#: The ``searchString`` slot becomes the user's keyword query.
_EMPTY_FILTER: dict[str, Any] = {
    "searchString": None,
    "salaryFrom": None,
    "salaryTo": None,
    "closingDate": None,
    "positionInitiative": None,
    "classification": None,
    "securityClearance": None,
    "officeArrangement": None,
    "duration": None,
    "department": None,
    "category": None,
    "opportunityType": None,
    "employmentStatus": None,
    "state": None,
    "sortBy": None,
    "offset": 0,
    "offsetIsLimit": False,
    "lastVisitedId": None,
    "daysInPast": None,
    "name": None,
    "type": None,
    "notificationsEnabled": None,
    "savedSearchId": None,
}


class APSJobsFetchError(RuntimeError):
    """Raised when an APS Jobs Aura call fails."""

    def __init__(self, account: str, status_code: int, *, stage: str) -> None:
        super().__init__(
            f"apsjobs:{account or '<all>'} {stage} returned HTTP {status_code}",
        )
        self.account = account
        self.status_code = status_code
        self.stage = stage


class APSJobsSource(BaseSource):
    """APS Jobs Salesforce Lightning source.

    ``account`` is a free-text search string. The empty string means
    "all currently open positions".
    """

    kind = "apsjobs"

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        bootstrap = await fetcher.fetch(f"{_BASE_URL}{_BOOTSTRAP_PATH}")
        if not bootstrap.is_ok:
            raise APSJobsFetchError(
                self.account,
                bootstrap.status_code,
                stage="bootstrap",
            )
        fwuid, app_token = _extract_aura_context(bootstrap)
        seen_ids: set[str] = set()
        offset = 0
        for page_index in range(_MAX_PAGES):
            response = await _fetch_page(
                fetcher,
                fwuid=fwuid,
                app_token=app_token,
                offset=offset,
                search_string=self.account or None,
                request_index=page_index,
            )
            if not response.is_ok:
                raise APSJobsFetchError(
                    self.account,
                    response.status_code,
                    stage=f"page-{page_index}",
                )

            payload = _decode_aura_response(response)
            listings = payload["jobListings"]
            if not listings:
                return
            for listing in listings:
                job = _parse_listing(listing)
                if job is None or job.source_external_id in seen_ids:
                    continue
                seen_ids.add(job.source_external_id)
                yield job
            offset = int(payload["newOffset"])


async def _fetch_page(
    fetcher: Fetcher,
    *,
    fwuid: str,
    app_token: str,
    offset: int,
    search_string: str | None,
    request_index: int,
) -> Response:
    """POST one page of the ``retrieveJobListings`` Apex action."""
    aura_filter = dict(_EMPTY_FILTER)
    aura_filter["offset"] = offset
    if search_string:
        aura_filter["searchString"] = search_string

    actions = {
        "actions": [
            {
                # The ``id`` is just an echo handle Salesforce uses to
                # match request to response. Any unique-per-request
                # value works; we use a sequential one for log clarity.
                "id": f"{request_index};a",
                "descriptor": _ACTION_DESCRIPTOR,
                "callingDescriptor": "UNKNOWN",
                "params": {
                    "namespace": "",
                    "classname": _APEX_CLASS,
                    "method": _APEX_METHOD,
                    "params": {"filter": json.dumps(aura_filter)},
                    "cacheable": False,
                    "isContinuation": False,
                },
            },
        ],
    }
    aura_context = {
        "mode": "PROD",
        "fwuid": fwuid,
        "app": _AURA_APP,
        "loaded": {f"APPLICATION@markup://{_AURA_APP}": app_token},
        "dn": [],
        "globals": {},
        "uad": True,
    }
    body: Mapping[str, str] = {
        "message": json.dumps(actions),
        "aura.context": json.dumps(aura_context),
        "aura.pageURI": _BOOTSTRAP_PATH,
        "aura.token": "null",
    }
    # The query string echoes the action — Salesforce's instrumentation
    # uses it in server logs. Functionally any value works; staying
    # close to the live shape avoids a fingerprintable divergence.
    url = f"{_BASE_URL}{_AURA_PATH}?r={request_index + 1}&aura.ApexAction.execute=1"
    return await fetcher.fetch(url, method="POST", data=body)


def _extract_aura_context(response: Response) -> tuple[str, str]:
    """Pull ``fwuid`` and the application build token from the bootstrap.

    Both tokens are required to call any Apex action. The ``Link``
    response header carries them as URL-encoded JSON for every
    render path; the inline ``<script>`` block carries them only on
    the larger first-render HTML. Search the header first because
    it's the more reliable surface, then fall back to the body so
    we tolerate either shape.
    """
    haystack = unquote(response.headers.get("link", "")) + "\n" + response.text
    fwuid_m = _FWUID_RE.search(haystack)
    app_m = _APP_TOKEN_RE.search(haystack)
    if not fwuid_m or not app_m:
        msg = (
            "apsjobs bootstrap missing aura context "
            f"(fwuid={fwuid_m is not None}, app_token={app_m is not None})"
        )
        raise APSJobsFetchError("", 200, stage="bootstrap-parse") from RuntimeError(
            msg,
        )
    return fwuid_m.group(1), app_m.group(1)


def _decode_aura_response(response: Response) -> dict[str, Any]:
    """Unwrap the Apex returnValue from the Aura envelope.

    Aura's response shape is
    ``{actions: [{state, returnValue: {returnValue: <apex_payload>}}]}``;
    the doubly-nested ``returnValue`` is Salesforce's, not ours.
    """
    body = json.loads(response.text)
    action = body["actions"][0]
    if action.get("state") != "SUCCESS":
        errors = action.get("error") or []
        msg = f"apsjobs Aura action returned state={action.get('state')!r}: {errors}"
        raise APSJobsFetchError(
            "",
            response.status_code,
            stage="apex",
        ) from RuntimeError(msg)
    return cast("dict[str, Any]", action["returnValue"]["returnValue"])


def _parse_listing(listing: dict[str, Any]) -> NormalizedJob | None:
    """Translate one Apex listing into a :class:`NormalizedJob`."""
    job_id = listing.get("jobId")
    title = listing.get("jobName")
    if not job_id or not title:
        return None

    department = listing.get("departmentName") or "Australian Public Service"
    location = listing.get("jobLocation") or None
    application_url = listing.get("applicationURL")
    apply_url = _DETAIL_URL_TEMPLATE.format(job_id=job_id)

    salary_min = _to_int(listing.get("jobSalaryFrom"))
    salary_max = _to_int(listing.get("jobSalaryTo"))
    salary_currency = "AUD" if salary_min is not None or salary_max is not None else None

    description = listing.get("jobDescription")
    duties = listing.get("jobDuties")
    description_html = _join_html(description, duties)

    return NormalizedJob(
        source_external_id=str(job_id),
        title=str(title).strip(),
        company=str(department).strip(),
        apply_url=apply_url,
        raw_data={
            "listing": listing,
            "application_url": application_url,
            "vacancy_number": listing.get("vacancyNumber"),
        },
        location_raw=location,
        location_country="Australia",
        location_city=_first_segment(location),
        remote_type=_normalise_arrangement(listing.get("officeArrangement")),
        employment_type=listing.get("jobEmploymentType") or listing.get("jobType"),
        posted_at=listing.get("jobPostedDate"),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        description_html=description_html,
        extra_tags=tuple(t for t in (listing.get("jobClassification"),) if t),
    )


def _join_html(*parts: str | None) -> str | None:
    """Concatenate non-empty HTML chunks with a separator. ``None`` if all empty."""
    cleaned = [p for p in parts if p]
    if not cleaned:
        return None
    return "\n".join(cleaned)


def _first_segment(value: str | None) -> str | None:
    if not value:
        return None
    head = value.split(",")[0].strip()
    return head or None


def _to_int(value: Any) -> int | None:
    """Round-and-cast a Salesforce float-or-int salary to a plain int.

    APS returns salaries as JSON numbers (``125820.0``); ``NormalizedJob``
    stores them as ``int``. ``None`` and unparseable strings collapse
    to ``None`` so downstream code can branch on truthiness.
    """
    if value is None:
        return None
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return None


def _normalise_arrangement(value: str | None) -> str | None:
    """Map APS ``officeArrangement`` strings to a coarse remote_type.

    APS uses ``On Site``, ``Hybrid``, ``Flexible``, or
    semicolon-joined combinations. We collapse to one of
    ``onsite``/``hybrid``/``remote`` so downstream filters don't
    need to know the APS-specific vocabulary.
    """
    if not value:
        return None
    tokens = {t.strip().lower() for t in value.split(";") if t.strip()}
    if "flexible" in tokens or "remote" in tokens:
        return "remote"
    if "hybrid" in tokens:
        return "hybrid"
    if "on site" in tokens or "onsite" in tokens:
        return "onsite"
    return None


__all__: list[str] = ["APSJobsFetchError", "APSJobsSource"]
