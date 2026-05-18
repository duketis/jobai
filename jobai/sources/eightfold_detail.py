"""Fetch + parse an Eightfold (Microsoft Careers) job into JD text.

`apply.careers.microsoft.com/careers/job/<id>` is an Eightfold ATS
single-page app: the raw URL returns a ~640KB JavaScript shell with
no parseable job description or company, so a sibling that fetches
the page directly tailors *blind* (this is exactly why tailor run
#49 produced generic output and couldn't even tell the company was
Microsoft).

The job body is served as JSON from the Eightfold apply API at
``/api/apply/v2/jobs/<id>`` (verified live 2026-05-18: HTTP 200,
``job_description`` ≈ 6.2KB of HTML). :func:`eightfold_jd_url`
rewrites a careers URL onto that endpoint so the tailor JD-resolver
hands the siblings real text — same role
:mod:`jobai.sources.seek_detail` /
:mod:`jobai.sources.linkedin_detail` play for their boards.

Scoped to ``careers.microsoft.com`` (the only Eightfold tenant
calibrated against live); the URL shape is the generic Eightfold
pattern, so extending to other tenants is a host-list change.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from selectolax.parser import HTMLParser

from jobai.fetcher.base import Fetcher

#: Eightfold detail path: ``/careers/job/<numeric id>``. The id is the
#: only handle into the JSON API rewrite.
_JOB_PATH_RE = re.compile(r"/careers/job/(\d+)")

#: Host substring that identifies the Microsoft-careers Eightfold
#: tenant. Kept narrow on purpose — only this tenant is verified.
EIGHTFOLD_HOST = "careers.microsoft.com"


def eightfold_jd_url(url: str) -> str | None:
    """Rewrite a Microsoft-careers job URL onto the Eightfold JSON API.

    Returns ``None`` for non-Microsoft-careers hosts or URLs with no
    ``/careers/job/<id>`` segment (caller then defers to the sibling).
    Query/tracking params (``?utm_source=linkedin&src=LinkedIn``) are
    dropped — the API keys off the numeric id only.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if EIGHTFOLD_HOST not in host:
        return None
    match = _JOB_PATH_RE.search(parsed.path)
    if match is None:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/api/apply/v2/jobs/{match.group(1)}"


def parse_eightfold_description(body: str) -> str | None:
    """Pull plain JD text out of an Eightfold job JSON payload.

    The ``job_description`` field is HTML; it's stripped to text.
    Returns ``None`` when the body isn't JSON, the field is absent,
    or it's blank — caller treats that the same as a failed fetch
    rather than tailoring against an empty JD.
    """
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    html = payload.get("job_description")
    if not isinstance(html, str) or not html.strip():
        return None
    text = HTMLParser(html).text(strip=True)
    return text or None


async def fetch_eightfold_jd_text(url: str, fetcher: Fetcher) -> str | None:
    """Fetch a Microsoft-careers job's JD text via the Eightfold API.

    Non-Microsoft-careers URLs and id-less URLs short-circuit to
    ``None`` without a fetch. Any non-2xx response or unparsable body
    yields ``None`` (soft failure: the tailor chain falls back to the
    plain URL path) rather than raising. The caller owns the fetcher.
    """
    target = eightfold_jd_url(url)
    if target is None:
        return None
    response = await fetcher.fetch(target)
    if not response.is_ok:
        return None
    return parse_eightfold_description(response.text)
