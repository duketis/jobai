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

Currently parses LinkedIn detail pages only. Indeed protects
``viewjob`` pages with session-bound checks that even Patchright
can't reliably get past without cookie persistence; that's a future
session-aware fetch path. Seek and ATS sources already populate
``description_text`` from the listing payload, so they don't need a
backfill.
"""

from __future__ import annotations

import logging
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


#: Per-source-kind parser. Each receives the detail-page HTML and
#: returns the description text, or ``None`` if the page didn't
#: contain a parsable description block.
DescriptionParser = Callable[[str], str | None]


def _parse_linkedin_description(html: str) -> str | None:
    """Pull the description out of a LinkedIn ``/jobs/view/<id>`` page.

    LinkedIn renders the description into ``div.description__text``
    on the public guest detail page. ``show-more-less-html__markup``
    is its inner wrapper; either selector reaches the same content.
    """
    tree = HTMLParser(html)
    node = tree.css_first("div.description__text") or tree.css_first(
        "div.show-more-less-html__markup"
    )
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


#: Mapping ``source.kind`` -> :data:`DescriptionParser`. The backfill
#: only touches kinds present in this dict; everything else is
#: skipped so we don't issue request-storm against sources that
#: already populate descriptions or that block detail-page access.
PARSERS: dict[str, DescriptionParser] = {
    "linkedin": _parse_linkedin_description,
}


def select_pending_jobs(
    conn: sqlite3.Connection,
    *,
    kinds: tuple[str, ...] = tuple(PARSERS),
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
    parsers: dict[str, DescriptionParser] | None = None,
) -> BackfillResult:
    """Walk pending jobs and fill ``description_text`` from detail pages.

    Args:
        conn: open SQLite connection. Used for both the job-selection
            query and the per-job UPDATE.
        fetcher: a :class:`Fetcher` already configured for the tier
            most pending jobs need (typically tier-3 stealth for
            LinkedIn). Caller owns the lifecycle.
        limit: max jobs touched in one call. Stops a runaway backfill
            from monopolising the scheduler / fetcher pool.
        parsers: override the default :data:`PARSERS` map (tests).

    Returns:
        :class:`BackfillResult` with attempt / fill / skip counts.
    """
    parser_map = parsers if parsers is not None else PARSERS
    kinds = tuple(parser_map.keys())
    pending = select_pending_jobs(conn, kinds=kinds, limit=limit)

    filled = 0
    skipped = 0
    for job_id, kind, apply_url in pending:
        parser = parser_map.get(kind)
        if parser is None:
            skipped += 1
            continue
        try:
            response = await fetcher.fetch(apply_url)
        except Exception as exc:  # noqa: BLE001 - any fetch failure ends one job, not the run
            _log.info(
                "description_backfill_fetch_failed",
                extra={
                    "job_id": job_id,
                    "kind": kind,
                    "url": apply_url,
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

        description = parser(response.text)
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
