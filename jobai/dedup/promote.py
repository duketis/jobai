"""Promote a NormalizedJob into the canonical ``jobs`` table.

The runner calls :func:`promote_to_canonical_jobs` immediately after
upserting a row into ``jobs_raw``. The canonical row is the
cross-source merged view; the ``job_sources`` join records which
``(source, jobs_raw)`` instances surfaced this canonical job, so
multiple sources surfacing the same role contribute apply URLs and
metadata without duplicating the job in search results.

This module performs only the deterministic match (by ``dedup_key``).
The fuzzy reconciliation pass is a separate maintenance step in
``jobai.dedup.reconcile`` — running it inline per upsert would be
O(N) per insert and a major hot-path cost. The reconcile pass runs
periodically and is cheap to re-run.

The FTS5 sync table (``jobs_fts``) is kept in lock-step with ``jobs``
by triggers defined in the initial schema migration; we don't touch
it directly here.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from jobai.dedup.best_of import merge_canonical_fields, mergeable_fields
from jobai.dedup.hashing import (
    compute_dedup_key,
    normalize_company,
    normalize_title,
)
from jobai.observability.logging import get_logger
from jobai.sources.base import NormalizedJob

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Outcome of promoting one NormalizedJob to the canonical ``jobs`` table."""

    job_id: int | None
    was_new: bool
    skipped_reason: str | None = None

    @property
    def was_skipped(self) -> bool:
        return self.skipped_reason is not None


def promote_to_canonical_jobs(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    jobs_raw_id: int,
    job: NormalizedJob,
) -> PromotionResult:
    """Upsert a NormalizedJob into ``jobs`` and link it via ``job_sources``.

    Steps:

    1. Validate that ``company`` and ``title`` are non-empty (otherwise
       the dedup key would collide across many bad rows).
    2. Compute the deterministic ``dedup_key``.
    3. INSERT a new row if no job has this key, or fetch the existing
       row's id.
    4. Always UPSERT the ``job_sources`` link for this
       ``(job, source, jobs_raw)`` triple.
    5. Update ``last_seen_at`` on existing rows; refresh mutable
       fields (description, salary, location) so re-scrapes pull in
       the most recent data.
    """
    if not job.company.strip() or not job.title.strip():
        _log.warning(
            "promote_skipped_empty_required_field",
            source_id=source_id,
            jobs_raw_id=jobs_raw_id,
            company=job.company,
            title=job.title,
        )
        return PromotionResult(
            job_id=None,
            was_new=False,
            skipped_reason="empty company or title",
        )

    dedup_key = compute_dedup_key(
        company=job.company,
        title=job.title,
        country=job.location_country,
    )
    company_norm = normalize_company(job.company)
    fingerprint = json.dumps(
        {
            "dedup_key": dedup_key,
            "company_norm": company_norm,
            "title_norm": normalize_title(job.title),
        },
        sort_keys=True,
    )
    now = _now_iso()

    existing_id = _find_existing_job_id(conn, dedup_key)

    if existing_id is None:
        job_id = _insert_new_canonical_job(
            conn,
            dedup_key=dedup_key,
            company_norm=company_norm,
            fingerprint=fingerprint,
            now=now,
            job=job,
        )
        was_new = True
    else:
        _update_existing_canonical_job(
            conn,
            job_id=existing_id,
            now=now,
            job=job,
        )
        job_id = existing_id
        was_new = False

    _upsert_job_source_link(
        conn,
        job_id=job_id,
        source_id=source_id,
        jobs_raw_id=jobs_raw_id,
        apply_url=job.apply_url,
    )
    conn.commit()

    return PromotionResult(job_id=job_id, was_new=was_new, skipped_reason=None)


def _find_existing_job_id(conn: sqlite3.Connection, dedup_key: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM jobs WHERE dedup_key = ?",
        (dedup_key,),
    ).fetchone()
    return int(row[0]) if row is not None else None


def _insert_new_canonical_job(
    conn: sqlite3.Connection,
    *,
    dedup_key: str,
    company_norm: str,
    fingerprint: str,
    now: str,
    job: NormalizedJob,
) -> int:
    cursor = conn.execute(
        "INSERT INTO jobs ("
        "  dedup_key, title, company, company_norm, "
        "  location_raw, location_country, location_city, "
        "  remote_type, employment_type, posted_at, "
        "  salary_min, salary_max, salary_currency, "
        "  description_text, description_html, apply_url, "
        "  first_seen_at, last_seen_at, fingerprint_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dedup_key,
            job.title,
            job.company,
            company_norm,
            job.location_raw,
            job.location_country,
            job.location_city,
            job.remote_type,
            job.employment_type,
            job.posted_at,
            job.salary_min,
            job.salary_max,
            job.salary_currency,
            job.description_text,
            job.description_html,
            job.apply_url,
            now,
            now,
            fingerprint,
        ),
    )
    last_id = cursor.lastrowid
    if last_id is None:
        raise RuntimeError("INSERT INTO jobs returned no lastrowid")
    return int(last_id)


def _update_existing_canonical_job(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    now: str,
    job: NormalizedJob,
) -> None:
    """Refresh mutable fields on a previously-seen canonical job.

    Reads the existing row, computes the best-of merge against the
    incoming ``NormalizedJob``, and writes the merged fields back.
    ``last_seen_at`` always advances; the merger handles every other
    mutable field (see :mod:`jobai.dedup.best_of` for the per-field
    rules). title / company / company_norm are immutable — they're
    part of the dedup key.
    """
    fields = mergeable_fields()
    # ruff S608: column names come from the static `_MERGEABLE_FIELDS`
    # tuple in best_of.py — never user input — so the f-string SQL is safe.
    select_sql = f"SELECT {', '.join(fields)} FROM jobs WHERE id = ?"  # noqa: S608
    existing_row = conn.execute(select_sql, (job_id,)).fetchone()
    existing: dict[str, object | None] = dict(zip(fields, existing_row, strict=True))
    incoming: dict[str, object | None] = {
        "location_raw": job.location_raw,
        "location_country": job.location_country,
        "location_city": job.location_city,
        "remote_type": job.remote_type,
        "employment_type": job.employment_type,
        "posted_at": job.posted_at,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_currency": job.salary_currency,
        "description_text": job.description_text,
        "description_html": job.description_html,
        "apply_url": job.apply_url or None,
    }
    merged = merge_canonical_fields(existing, incoming)

    set_clause = ", ".join(f"{f} = ?" for f in fields)
    # ruff S608: see select_sql comment above — fields are static.
    conn.execute(
        f"UPDATE jobs SET last_seen_at = ?, {set_clause} WHERE id = ?",  # noqa: S608
        (now, *(merged[f] for f in fields), job_id),
    )


def _upsert_job_source_link(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    source_id: int,
    jobs_raw_id: int,
    apply_url: str,
) -> None:
    """Idempotent insert into ``job_sources``.

    The PRIMARY KEY ``(job_id, source_id, jobs_raw_id)`` makes
    re-running on the same triple a no-op via INSERT OR REPLACE.
    """
    conn.execute(
        "INSERT OR REPLACE INTO job_sources "
        "(job_id, source_id, jobs_raw_id, apply_url) "
        "VALUES (?, ?, ?, ?)",
        (job_id, source_id, jobs_raw_id, apply_url),
    )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
