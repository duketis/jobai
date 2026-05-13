"""QLD Government — smartjobs.qld.gov.au source.

QLD's central government job board runs on NGA Talent Solutions'
``jobtools`` platform. The default search-results page (a bare GET
to ``jncustomsearch.searchResults?in_organid=14904``) returns the
full unfiltered listing as classic server-rendered HTML — no JS
required, plain HTTP works.

The HTML is an ``<ol class="search-results jobs">`` of ``<li>``
items. Each li contains:

* ``h3 > a[href^="/jobs/QLD-"]`` — title link with apply URL
* ``span.result-title`` — title (sometimes followed by a comma and
  the agency name in plain text inside the same anchor)
* ``span.type`` — employment type ("Fixed Term Temporary Full-time")
* ``ul.location > li > strong.locality`` — locale name
* ``div.search-description`` — short description
* ``div.meta > strong.grade`` — pay grade label
* ``div.meta > span.salary > script`` — salary range, set as
  JavaScript variables ``sal1`` / ``sal2`` (annual range) and
  ``sal3`` / ``sal4`` (fortnightly). We extract from the inline
  script text rather than try to evaluate the JS.

Apply URLs use the form ``/jobs/QLD-{id}-{year}`` (year sometimes
omitted on cross-year ads). The ``QLD-{id}`` portion is the stable
``source_external_id``.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

from selectolax.parser import HTMLParser, Node

from jobai.fetcher.base import Fetcher
from jobai.sources.base import BaseSource, NormalizedJob

_BASE_URL = "https://smartjobs.qld.gov.au"
_SEARCH_PATH = "/jobtools/jncustomsearch.searchResults"
_DEFAULT_ORGID = "14904"
#: Results per page on jncustomsearch. The JS pagination control
#: advances the form's ``in_pg`` field by 20 per click.
_PAGE_SIZE = 20
#: Hard cap on pages walked per scrape cycle. The early-exit on a
#: zero-yield page short-circuits when smartjobs runs out of unique
#: cards, so a generous cap doesn't waste cycles - it just gives the
#: walker headroom for the natural ceiling (currently ~27 pages /
#: ~540 QLD Government roles, but other org-ids vary widely).
_DEFAULT_MAX_PAGES = 100

#: Match the ID portion of ``/jobs/QLD-{id}[-{year}]`` URLs.
_JOB_ID_RE = re.compile(r"/jobs/QLD-(\d+)")
#: Pull annual-salary numerics out of the inline ``sal1``/``sal2`` JS.
_SALARY_VAR_RE = re.compile(r"sal([12])\s*=\s*'(\d+)'")


class QLDSmartJobsFetchError(RuntimeError):
    """Raised when smartjobs.qld.gov.au returns a non-2xx status."""

    def __init__(self, account: str, status_code: int) -> None:
        super().__init__(f"qld_smartjobs:{account} returned HTTP {status_code}")
        self.account = account
        self.status_code = status_code


class QLDSmartJobsSource(BaseSource):
    """Pulls listings from one smartjobs.qld.gov.au org id.

    ``account`` is the NGA organisation id; the default ``"14904"``
    is the umbrella Queensland Government org. Other agencies
    occasionally publish under their own ids; pass them via the
    constructor when seeding additional rows.
    """

    kind = "qld_smartjobs"

    def __init__(
        self,
        account: str = _DEFAULT_ORGID,
        *,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        super().__init__(account or _DEFAULT_ORGID)
        if max_pages < 1:
            msg = f"max_pages must be >= 1, got {max_pages}"
            raise ValueError(msg)
        self._max_pages = max_pages

    async def discover(self, fetcher: Fetcher) -> AsyncIterator[NormalizedJob]:
        seen_ids: set[str] = set()
        for page in range(self._max_pages):
            offset = page * _PAGE_SIZE
            url = (
                f"{_BASE_URL}{_SEARCH_PATH}"
                f"?in_organid={self.account or _DEFAULT_ORGID}"
                f"&in_pg={offset}"
            )
            response = await fetcher.fetch(url)
            if not response.is_ok:
                if page == 0:
                    raise QLDSmartJobsFetchError(self.account, response.status_code)
                # Mid-walk failure: stop short rather than fail the
                # whole run; everything yielded so far is preserved.
                return

            tree = HTMLParser(response.text)
            page_yielded = 0
            for li in tree.css("li"):
                if li.css_first('a[href*="/jobs/QLD-"]') is None:
                    continue
                job = _parse_card(li)
                if job is None or job.source_external_id in seen_ids:
                    continue
                seen_ids.add(job.source_external_id)
                page_yielded += 1
                yield job
            # Last page reached when the offset overruns the result
            # set (smartjobs serves the empty results template).
            if page_yielded == 0:
                return


def _parse_card(card: Node) -> NormalizedJob | None:
    """Translate one ``<li>`` job-row into a :class:`NormalizedJob`."""
    apply_anchor = card.css_first('a[href*="/jobs/QLD-"]')
    if apply_anchor is None:
        return None
    apply_path = apply_anchor.attributes.get("href")
    # The ``a[href*="/jobs/QLD-"]`` selector above guarantees a non-empty
    # href; this guard exists as belt-and-braces only.
    if not apply_path:  # pragma: no cover
        return None

    match = _JOB_ID_RE.search(apply_path)
    if match is None:
        return None
    job_id = f"QLD-{match.group(1)}"

    title, company = _split_title_company(apply_anchor)
    if not title:
        return None

    description = _text(card, "div.search-description")
    employment_type = _text(card, "span.type")
    locality = _text(card, "ul.location strong.locality")
    grade = _text(card, "strong.grade")
    salary_min, salary_max, salary_currency = _parse_salary(card)

    return NormalizedJob(
        source_external_id=job_id,
        title=title,
        company=(company or "Queensland Government").strip(),
        apply_url=urljoin(_BASE_URL, apply_path),
        raw_data={
            "title": title,
            "company": company,
            "employment_type": employment_type,
            "locality": locality,
            "grade": grade,
            "description": description,
        },
        location_raw=locality,
        location_country="Australia",
        location_city=locality,
        employment_type=employment_type,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        description_text=description,
        extra_tags=tuple(t for t in (grade,) if t),
    )


def _split_title_company(anchor: Node) -> tuple[str | None, str | None]:
    """The title link wraps ``<span.result-title><strong>Title</strong></span>, Agency``.

    QLD packs the agency name as plain text *after* the strong tag.
    Splitting on the first comma after the inner strong gives us
    the title cleanly and a plain agency name.
    """
    strong = anchor.css_first("span.result-title strong") or anchor.css_first("strong")
    title = strong.text(strip=True) if strong else None

    full = anchor.text(strip=True)
    company = None
    if title and full and full.startswith(title):
        rest = full[len(title) :].strip().strip(",").strip()
        if rest:
            company = rest
    return title or None, company


def _text(card: Node, selector: str) -> str | None:
    node = card.css_first(selector)
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


def _parse_salary(card: Node) -> tuple[int | None, int | None, str | None]:
    """Pull ``sal1`` / ``sal2`` annual values out of the inline script."""
    salary_node = card.css_first("span.salary")
    if salary_node is None:
        return None, None, None
    raw = salary_node.html or ""
    pairs = dict(_SALARY_VAR_RE.findall(raw))
    low = _to_int(pairs.get("1"))
    high = _to_int(pairs.get("2"))
    if low is None and high is None:
        return None, None, None
    return low, high, "AUD"


def _to_int(token: str | None) -> int | None:
    if not token:
        return None
    cleaned = token.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)


__all__: list[str] = ["QLDSmartJobsFetchError", "QLDSmartJobsSource"]
# raw_data dict types
_ = Any
