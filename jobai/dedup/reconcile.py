"""Cross-source fuzzy reconciliation pass.

Runs after a scrape cycle (or on demand) to merge canonical jobs that
the deterministic pass missed. Walks every (company_norm,
location_country) group of recent jobs, finds title pairs above the
fuzzy similarity threshold, and merges the younger job into the
older one — preserving job_sources links so search results still
point at every URL.

This is **idempotent**: re-running on already-merged data produces
the same result. It's also **bounded**: only jobs whose
last_seen_at is within ``window_days`` are considered, so the pass
runs in seconds against a large jobs table.

Why a separate pass and not inline in :mod:`promote`:

* Inline would be O(N) per insert (compare each new job to every
  existing job in the same group). Reconcile is O(N) per group per
  pass — same total work, but spread across explicit invocations
  rather than hot-path latency.
* Reconcile policy can change (threshold, window) without touching
  the runner. Promote stays narrow.
* A separate pass can be re-run after a bug fix to rebuild dedup
  state; an inline policy can't.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from jobai.dedup.fuzzy import DEFAULT_SIMILARITY_THRESHOLD, find_similar_match
from jobai.observability.logging import get_logger

_log = get_logger(__name__)

DEFAULT_WINDOW_DAYS = 14


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Summary of one ``reconcile_fuzzy_duplicates`` invocation."""

    groups_examined: int
    pairs_merged: int


def reconcile_fuzzy_duplicates(
    conn: sqlite3.Connection,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
) -> ReconcileResult:
    """Merge fuzzy-duplicate canonical jobs within ``window_days``.

    For each ``(company_norm, location_country)`` group of jobs whose
    ``last_seen_at`` is within the window, find pairs whose
    ``token_sort_ratio`` >= ``threshold`` and merge the younger job
    into the older one (older = lower ``id``, the first to appear).

    Merge transfers all ``job_sources`` rows to the surviving job and
    deletes the duplicate row from ``jobs`` (the FTS5 delete trigger
    cleans the search index automatically).
    """
    groups = _collect_recent_groups(conn, window_days=window_days)
    pairs_merged = 0

    for (company_norm, country), candidates in groups.items():
        if len(candidates) < 2:
            continue
        pairs_merged += _merge_within_group(
            conn,
            candidates=candidates,
            threshold=threshold,
            company_norm=company_norm,
            country=country,
        )

    conn.commit()
    _log.info(
        "fuzzy_reconcile_complete",
        groups_examined=len(groups),
        pairs_merged=pairs_merged,
        window_days=window_days,
        threshold=threshold,
    )
    return ReconcileResult(groups_examined=len(groups), pairs_merged=pairs_merged)


def _collect_recent_groups(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Group recent jobs by (company_norm, country). Values are (id, title)."""
    cursor = conn.execute(
        "SELECT id, title, company_norm, COALESCE(location_country, '') AS country "
        "FROM jobs "
        "WHERE last_seen_at >= datetime('now', '-' || ? || ' days') "
        "ORDER BY id ASC",  # ascending id => older jobs come first within each group
        (window_days,),
    )
    groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in cursor:
        key = (str(row[2]), str(row[3]))
        groups[key].append((int(row[0]), str(row[1])))
    return groups


def _merge_within_group(
    conn: sqlite3.Connection,
    *,
    candidates: list[tuple[int, str]],
    threshold: int,
    company_norm: str,
    country: str,
) -> int:
    """Within one (company_norm, country) bucket, merge fuzzy duplicates.

    Walks left-to-right (older first); for each candidate, looks for
    a fuzzy match among already-processed (older) candidates. If
    found, merges the current job into the older one.

    Returns the number of merges performed.
    """
    merged = 0
    survivors: list[tuple[int, str]] = []

    for cand_id, cand_title in candidates:
        match = find_similar_match(cand_title, survivors, threshold=threshold)
        if match is None:
            survivors.append((cand_id, cand_title))
            continue

        survivor_id, score = match
        _merge_jobs(conn, survivor_id=survivor_id, duplicate_id=cand_id)
        merged += 1
        _log.info(
            "fuzzy_merge",
            survivor_id=survivor_id,
            duplicate_id=cand_id,
            score=score,
            company_norm=company_norm,
            country=country,
            survivor_title=next(t for sid, t in survivors if sid == survivor_id),
            duplicate_title=cand_title,
        )

    return merged


def _merge_jobs(
    conn: sqlite3.Connection,
    *,
    survivor_id: int,
    duplicate_id: int,
) -> None:
    """Move job_sources rows from ``duplicate_id`` to ``survivor_id`` and delete the duplicate.

    ``job_sources`` has a composite primary key
    ``(job_id, source_id, jobs_raw_id)``. Two duplicates from the
    same source could conflict on (source_id, jobs_raw_id) when both
    re-target survivor_id, so we use INSERT OR IGNORE on a copy then
    DELETE the originals — idempotent under repeated runs.
    """
    conn.execute(
        "INSERT OR IGNORE INTO job_sources (job_id, source_id, jobs_raw_id, apply_url) "
        "SELECT ?, source_id, jobs_raw_id, apply_url "
        "FROM job_sources WHERE job_id = ?",
        (survivor_id, duplicate_id),
    )
    conn.execute("DELETE FROM job_sources WHERE job_id = ?", (duplicate_id,))
    conn.execute("DELETE FROM jobs WHERE id = ?", (duplicate_id,))
