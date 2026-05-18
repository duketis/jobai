"""Single-source scrape runner.

For one source, the runner:

1. Inserts a ``scrape_runs`` row with status='running'.
2. Wraps the supplied fetcher in a :class:`RecordingFetcher` so every
   HTTP response is persisted to ``raw_responses`` automatically.
3. Iterates ``source.discover(fetcher)``, upserting each
   :class:`NormalizedJob` into ``jobs_raw`` (insert if new, update
   ``last_seen_at`` and the row contents if the SHA-256 of the
   serialised job has changed).
4. Updates the ``scrape_runs`` row with the final status and counts.

A single ``Exception`` from the source body marks the run failed and
records the error class + message in ``error_summary``; everything up
to the failure point is preserved in ``jobs_raw``. ``BaseException``
subclasses (``KeyboardInterrupt``, ``SystemExit``) propagate so a
process shutdown is not swallowed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jobai.dedup.promote import promote_to_canonical_jobs
from jobai.fetcher.base import Fetcher
from jobai.fetcher.recording import RecordingFetcher
from jobai.observability.logging import get_logger
from jobai.pipeline.posted_at_normalisation import normalise_posted_at
from jobai.pipeline.remote_inference import infer_remote_type
from jobai.pipeline.salary_inference import infer_salary, text_from_html
from jobai.pipeline.schema_change import (
    FieldChange,
    FieldStats,
    detect_changes,
    empty_stats,
    update_stats,
)
from jobai.sources.base import BaseSource, NormalizedJob
from jobai.sources.repository import SourceRow

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary of one scrape cycle, returned from :func:`run_source`."""

    run_id: int
    status: str  # 'success' | 'partial' | 'failed'
    items_seen: int
    items_new: int
    items_updated: int
    error_summary: str | None = None
    schema_changes: tuple[FieldChange, ...] = ()


async def run_source(
    *,
    conn: sqlite3.Connection,
    source: BaseSource,
    source_row: SourceRow,
    fetcher: Fetcher,
) -> RunResult:
    """Run one source end-to-end.

    The caller is responsible for opening the database connection and
    constructing the source instance and the fetcher (so the runner
    stays testable without filesystem or network).
    """
    run_id = _start_run(conn, source_id=source_row.id, tier=source_row.default_tier)
    recorder = RecordingFetcher(
        fetcher,
        conn=conn,
        run_id=run_id,
        source_id=source_row.id,
    )

    items_seen = 0
    items_new = 0
    items_updated = 0
    status = "success"
    error_summary: str | None = None
    field_stats: FieldStats = empty_stats()

    try:
        async for job in source.discover(recorder):
            items_seen += 1
            field_stats = update_stats(field_stats, job)
            jobs_raw_id, raw_was_new = _upsert_job_raw(
                conn,
                source_id=source_row.id,
                job=job,
            )
            if raw_was_new:
                items_new += 1
            else:
                items_updated += 1
            # Make sure the canonical row always lands with a
            # remote_type set. Sources that already populate it (most
            # ATS APIs, APS Jobs) pass through untouched; everyone
            # else gets the heuristic over title + description +
            # location, defaulting to onsite when nothing matches.
            promotable = _ensure_remote_type(job)
            # Mirror the same pattern for salary: if the source already
            # surfaced a structured salary, leave it alone; otherwise
            # try to infer one from the title + description text.
            promotable = _ensure_salary(promotable)
            # Boards put anything from real ISO timestamps to "8d ago"
            # / "Just posted" in posted_at. Normalise to ISO-8601 UTC
            # (or NULL) here so the canonical row never carries the raw
            # label — that label broke posted_newest sorting and the
            # UI's relative-time formatter.
            promotable = _normalise_posted_at(promotable)
            promote_to_canonical_jobs(
                conn,
                source_id=source_row.id,
                jobs_raw_id=jobs_raw_id,
                job=promotable,
            )
    except Exception as exc:  # noqa: BLE001  - runner finalises on any failure
        status = "failed"
        error_summary = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "scrape_run_failed",
            source=source_row.name,
            run_id=run_id,
            error_class=type(exc).__name__,
            error=str(exc),
        )

    schema_changes: tuple[FieldChange, ...] = ()
    if status == "success":
        previous_stats = _load_previous_stats(conn, source_id=source_row.id, before_run_id=run_id)
        schema_changes = tuple(detect_changes(previous_stats, field_stats))
        for change in schema_changes:
            _log.warning(
                "schema_change_detected",
                source=source_row.name,
                run_id=run_id,
                field=change.field,
                prev_null_rate=round(change.prev_null_rate, 3),
                curr_null_rate=round(change.curr_null_rate, 3),
                delta=round(change.delta, 3),
            )

    _finish_run(
        conn,
        run_id=run_id,
        status=status,
        items_seen=items_seen,
        items_new=items_new,
        items_updated=items_updated,
        error_summary=error_summary,
        field_stats=field_stats,
    )

    _log.info(
        "scrape_run_complete",
        source=source_row.name,
        run_id=run_id,
        status=status,
        items_seen=items_seen,
        items_new=items_new,
        items_updated=items_updated,
    )

    return RunResult(
        run_id=run_id,
        status=status,
        items_seen=items_seen,
        items_new=items_new,
        items_updated=items_updated,
        error_summary=error_summary,
        schema_changes=schema_changes,
    )


def _ensure_remote_type(job: NormalizedJob) -> NormalizedJob:
    """Return ``job`` with ``remote_type`` set to a heuristic value if absent."""
    if job.remote_type:
        return job
    inferred = infer_remote_type(
        title=job.title,
        description=job.description_text,
        location=job.location_raw,
    )
    return dataclasses.replace(job, remote_type=inferred)


def _ensure_salary(job: NormalizedJob) -> NormalizedJob:
    """Return ``job`` with salary fields filled from text when absent.

    Sources that surface a structured salary (Ashby's ``compensation``
    block, APS Jobs' ``jobSalaryFrom`` / ``jobSalaryTo``, vic_careers'
    salary cell) pass through untouched. Everyone else gets the
    regex inference over title + description; rejection (no signal,
    funding mention, hourly rate) leaves the fields null.
    """
    if job.salary_min is not None or job.salary_max is not None:
        return job
    # Fall back to stripped description_html when the source only
    # populates one description column (Greenhouse, SmartRecruiters,
    # APS Jobs all hand us HTML-only).
    description = job.description_text
    if not description and job.description_html:
        description = text_from_html(job.description_html)
    salary_min, salary_max, currency = infer_salary(
        title=job.title,
        description=description,
    )
    if salary_min is None and salary_max is None:
        return job
    return dataclasses.replace(
        job,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=currency,
    )


def _normalise_posted_at(job: NormalizedJob) -> NormalizedJob:
    """Return ``job`` with ``posted_at`` normalised to ISO-8601 UTC.

    Relative labels ("8d ago", "Just posted") are resolved against
    the scrape instant — that's "now" at the moment we observe the
    listing. ISO inputs are reformatted idempotently; unparseable
    text becomes ``None`` (NULL sorts last, raw text sorts randomly).
    """
    normalised = normalise_posted_at(job.posted_at, now=datetime.now(UTC))
    if normalised == job.posted_at:
        return job
    return dataclasses.replace(job, posted_at=normalised)


def _start_run(conn: sqlite3.Connection, *, source_id: int, tier: int) -> int:
    cursor = conn.execute(
        "INSERT INTO scrape_runs (source_id, started_at, status, tier_used) "
        "VALUES (?, ?, 'running', ?)",
        (source_id, _now_iso(), tier),
    )
    conn.commit()
    last_id = cursor.lastrowid
    # SQLite always returns lastrowid on a successful INSERT; defensive only.
    if last_id is None:  # pragma: no cover
        raise RuntimeError("INSERT INTO scrape_runs returned no lastrowid")
    return int(last_id)


def _finish_run(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    status: str,
    items_seen: int,
    items_new: int,
    items_updated: int,
    error_summary: str | None,
    field_stats: FieldStats,
) -> None:
    conn.execute(
        "UPDATE scrape_runs "
        "SET finished_at = ?, status = ?, "
        "    items_seen = ?, items_new = ?, items_updated = ?, "
        "    error_summary = ?, field_stats_json = ? "
        "WHERE id = ?",
        (
            _now_iso(),
            status,
            items_seen,
            items_new,
            items_updated,
            error_summary,
            field_stats.to_json(),
            run_id,
        ),
    )
    conn.commit()


def _load_previous_stats(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    before_run_id: int,
) -> FieldStats | None:
    """Return the field-stats from this source's most recent successful run.

    Used to compare against the current run's stats for schema-change
    detection. Returns ``None`` if there is no prior successful run
    or its stats column is null/malformed.
    """
    row = conn.execute(
        "SELECT field_stats_json FROM scrape_runs "
        "WHERE source_id = ? AND id < ? AND status = 'success' "
        "  AND field_stats_json IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (source_id, before_run_id),
    ).fetchone()
    if row is None:
        return None
    return FieldStats.from_json(row[0])


def _upsert_job_raw(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    job: NormalizedJob,
) -> tuple[int, bool]:
    """Upsert a NormalizedJob into ``jobs_raw``.

    Returns ``(jobs_raw_id, was_new)`` so callers can promote the row
    into the canonical ``jobs`` table without a follow-up SELECT.
    """
    raw_json = _serialize_job(job)
    raw_sha256 = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    now = _now_iso()

    existing = conn.execute(
        "SELECT id, raw_sha256 FROM jobs_raw WHERE source_id = ? AND source_external_id = ?",
        (source_id, job.source_external_id),
    ).fetchone()

    if existing is None:
        cursor = conn.execute(
            "INSERT INTO jobs_raw "
            "(source_id, source_external_id, raw_json, raw_sha256, "
            " first_seen_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, job.source_external_id, raw_json, raw_sha256, now, now),
        )
        conn.commit()
        last_id = cursor.lastrowid
        # SQLite always returns lastrowid on a successful INSERT; defensive only.
        if last_id is None:  # pragma: no cover
            raise RuntimeError("INSERT INTO jobs_raw returned no lastrowid")
        return (int(last_id), True)

    existing_id = int(existing[0])
    existing_sha = str(existing[1])

    if existing_sha != raw_sha256:
        conn.execute(
            "UPDATE jobs_raw SET raw_json = ?, raw_sha256 = ?, last_seen_at = ? WHERE id = ?",
            (raw_json, raw_sha256, now, existing_id),
        )
    else:
        conn.execute(
            "UPDATE jobs_raw SET last_seen_at = ? WHERE id = ?",
            (now, existing_id),
        )
    conn.commit()
    return (existing_id, False)


def _serialize_job(job: NormalizedJob) -> str:
    """Serialise a NormalizedJob to a deterministic JSON string."""
    payload: dict[str, Any] = dataclasses.asdict(job)
    # extra_tags is a tuple — asdict turns it into a list, which is
    # what we want for JSON. raw_data may contain non-string keys in
    # exotic providers, but ATS payloads are pure JSON so default works.
    return json.dumps(payload, sort_keys=True, default=str)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
