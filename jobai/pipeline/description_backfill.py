"""Per-job description backfill.

Listing pages give us the bones: title, company, location, salary
where available, posted date. The actual description — what the
agent needs to summarise the role — lives on the per-job detail
page and would cost an extra round-trip per result if we fetched
it during normal scraping.

This module decouples description fetch from the scrape cycle. A
periodic job walks ``jobs`` rows that still have ``description_text
IS NULL``, fetches each one's apply URL, and parses the description
out of the detail-page DOM. Running it on a slower cadence than the
listing scrape keeps the scrape fast (fresh listings every cycle)
while descriptions fill in over the next few minutes.

Each kind has a :class:`DescriptionRecipe` registering three things:
the URL to actually fetch (sometimes a transform of the apply URL,
e.g. Indeed's Cloudflare-bypass via the search-page side panel),
the CSS selector the browser tier should wait for, and the parser
that turns the rendered HTML into the description text.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from selectolax.parser import HTMLParser

from jobai.fetcher.base import Fetcher

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Summary returned by :func:`backfill_descriptions`."""

    attempted: int
    filled: int
    skipped: int  # detail-page returned non-2xx or had no parsable description


#: A function that turns a detail-page HTML body into description text.
DescriptionParser = Callable[[str], str | None]

#: A function that maps a job's stored ``apply_url`` to the URL the
#: backfill should actually fetch. Most sources fetch the apply URL
#: directly (identity); Indeed needs a side-panel rewrite to dodge
#: Cloudflare on ``/viewjob``.
UrlTransform = Callable[[str], str]


def _identity_url(apply_url: str) -> str:
    return apply_url


@dataclass(frozen=True, slots=True)
class DescriptionRecipe:
    """Per-source-kind recipe for fetching and parsing a description.

    ``fetch_url`` defaults to identity; sources that gate the apply
    URL itself override it to point at an equivalent endpoint that
    actually serves the description body. ``wait_selector`` is
    forwarded to the fetcher so browser-tier renders block until
    the description block is in the DOM (HTTP-tier ignores it).
    """

    parse: DescriptionParser
    fetch_url: UrlTransform = _identity_url
    wait_selector: str | None = None


# ---------------------------------------------------------------------------
# LinkedIn
# ---------------------------------------------------------------------------


def _parse_linkedin_description(html: str) -> str | None:
    """Pull the description out of a LinkedIn ``/jobs/view/<id>`` page.

    LinkedIn renders the description into ``div.description__text``
    on the public guest detail page. ``show-more-less-html__markup``
    is its inner wrapper; either selector reaches the same content.
    """
    tree = HTMLParser(html)
    node = tree.css_first("div.description__text") or tree.css_first(
        "div.show-more-less-html__markup",
    )
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


# ---------------------------------------------------------------------------
# Indeed
# ---------------------------------------------------------------------------

#: Indeed apply URLs follow ``/viewjob?jk=<hex>``. The job key is the
#: only stable handle into the search-page side-panel rewrite below.
_INDEED_JOBKEY_RE = re.compile(r"[?&]jk=([A-Za-z0-9]+)")

#: The container Indeed renders the full description into on the
#: results page when ``vjk=<key>`` is set. ``data-testid`` is the
#: stable hook even when the surrounding class names rename.
_INDEED_DESCRIPTION_SELECTOR = (
    "#jobDescriptionText, [data-testid='jobsearch-JobComponent-description']"
)


def _indeed_side_panel_url(apply_url: str) -> str:
    """Map ``/viewjob?jk=KEY`` to a search-page URL that side-loads the job.

    Indeed Cloudflare-protects direct ``/viewjob`` hits even from
    Patchright; the public ``/jobs?...&vjk=KEY`` surface renders the
    same description into the right-hand side panel without a
    challenge. ``q=`` must be present (empty is fine); ``l=`` keeps
    Indeed's locale picker on Australia. The rewrite preserves the
    original URL when no ``jk`` param is found so a malformed row
    fails the per-job fetch loudly rather than silently misleading
    Cloudflare.
    """
    match = _INDEED_JOBKEY_RE.search(apply_url)
    if match is None:
        return apply_url
    return f"https://au.indeed.com/jobs?q=&l=Australia&vjk={match.group(1)}"


def _parse_indeed_description(html: str) -> str | None:
    """Extract the description text from an Indeed side-panel render."""
    tree = HTMLParser(html)
    node = tree.css_first("#jobDescriptionText") or tree.css_first(
        "[data-testid='jobsearch-JobComponent-description']",
    )
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


# ---------------------------------------------------------------------------
# Recipe registry
# ---------------------------------------------------------------------------


#: Kind -> recipe. The backfill only touches kinds present here so a
#: new source doesn't accidentally hit a detail-page request storm
#: against an endpoint nobody has calibrated against.
RECIPES: dict[str, DescriptionRecipe] = {
    "linkedin": DescriptionRecipe(parse=_parse_linkedin_description),
    "indeed": DescriptionRecipe(
        parse=_parse_indeed_description,
        fetch_url=_indeed_side_panel_url,
        wait_selector=_INDEED_DESCRIPTION_SELECTOR,
    ),
}


def select_pending_jobs(
    conn: sqlite3.Connection,
    *,
    kinds: tuple[str, ...] = tuple(RECIPES),
    limit: int = 50,
) -> list[tuple[int, str, str]]:
    """Return ``(job_id, source_kind, apply_url)`` rows needing a description.

    Picks jobs whose ``description_text`` is NULL and that have at
    least one source-link in ``kinds``. Newest-seen jobs first so
    the agent gets fresh roles' descriptions before stale ones.
    """
    if not kinds:
        return []
    # ``placeholders`` is built from the literal "?" string repeated
    # by the caller-supplied count of kinds — not user input — so
    # this is not a SQL-injection vector; values are still bound.
    placeholders = ",".join("?" for _ in kinds)
    sql = f"""
        SELECT j.id, s.kind, j.apply_url
        FROM jobs j
        JOIN job_sources js ON js.job_id = j.id
        JOIN sources s ON s.id = js.source_id
        WHERE (j.description_text IS NULL OR j.description_text = '')
          AND s.kind IN ({placeholders})
        GROUP BY j.id
        ORDER BY j.last_seen_at DESC
        LIMIT ?
        """  # noqa: S608  - placeholders are "?" literals; values are bound
    rows = conn.execute(sql, (*kinds, limit)).fetchall()
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


async def backfill_descriptions(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    *,
    limit: int = 50,
    recipes: dict[str, DescriptionRecipe] | None = None,
) -> BackfillResult:
    """Walk pending jobs and fill ``description_text`` from detail pages.

    Args:
        conn: open SQLite connection. Used for both the job-selection
            query and the per-job UPDATE.
        fetcher: a :class:`Fetcher` already configured for the tier
            most pending jobs need (typically tier-3 stealth — both
            LinkedIn guest detail and Indeed side-panel are
            fingerprint-checked). Caller owns the lifecycle.
        limit: max jobs touched in one call. Stops a runaway backfill
            from monopolising the scheduler / fetcher pool.
        recipes: override the default :data:`RECIPES` map (tests).

    Returns:
        :class:`BackfillResult` with attempt / fill / skip counts.
    """
    recipe_map = recipes if recipes is not None else RECIPES
    kinds = tuple(recipe_map.keys())
    pending = select_pending_jobs(conn, kinds=kinds, limit=limit)

    filled = 0
    skipped = 0
    for job_id, kind, apply_url in pending:
        recipe = recipe_map.get(kind)
        if recipe is None:
            skipped += 1
            continue
        fetch_url = recipe.fetch_url(apply_url)
        try:
            response = await fetcher.fetch(
                fetch_url,
                wait_for_selector=recipe.wait_selector,
            )
        except Exception as exc:  # noqa: BLE001 - any fetch failure ends one job, not the run
            _log.info(
                "description_backfill_fetch_failed",
                extra={
                    "job_id": job_id,
                    "kind": kind,
                    "url": fetch_url,
                    "error_class": type(exc).__name__,
                },
            )
            skipped += 1
            continue

        if not response.is_ok:
            _log.info(
                "description_backfill_non_ok",
                extra={"job_id": job_id, "status": response.status_code},
            )
            skipped += 1
            continue

        description = recipe.parse(response.text)
        if not description:
            skipped += 1
            continue

        conn.execute(
            "UPDATE jobs SET description_text = ? WHERE id = ?",
            (description, job_id),
        )
        conn.commit()
        filled += 1

    return BackfillResult(attempted=len(pending), filled=filled, skipped=skipped)
