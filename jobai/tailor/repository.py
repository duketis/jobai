"""SQL CRUD for ``tailor_runs``.

The orchestrator and the routes both go through this module so the
state machine lives in one place: callers ask the repository for a
state transition, not for an UPDATE statement.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from jobai.tailor.models import (
    TERMINAL_STATUSES,
    QAAssessment,
    QAStatus,
    TailorRunRecord,
    TailorRunStatus,
)


def _now() -> str:
    """Return an ISO 8601 UTC timestamp string (matches schema defaults)."""
    return datetime.now(tz=UTC).isoformat()


def create_tailor_run(
    conn: sqlite3.Connection,
    *,
    job_id: int | None = None,
    jd_url: str | None = None,
) -> TailorRunRecord:
    """Insert a fresh tailor_run in ``pending`` status.

    Exactly one of ``job_id`` (catalogue path) or ``jd_url`` (one-off
    URL path) must be set. The DB-level CHECK constraint enforces
    this; we surface a clear error here too so the caller doesn't
    rely on the SQL exception message.
    """
    if (job_id is None) == (jd_url is None):
        msg = "create_tailor_run requires exactly one of job_id / jd_url"
        raise ValueError(msg)
    now = _now()
    cursor = conn.execute(
        "INSERT INTO tailor_runs (job_id, jd_url, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (job_id, jd_url, TailorRunStatus.PENDING.value, now, now),
    )
    conn.commit()
    new_id = int(cursor.lastrowid or 0)
    return TailorRunRecord(
        id=new_id,
        job_id=job_id,
        jd_url=jd_url,
        status=TailorRunStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


_SELECT_COLUMNS = (
    "id, job_id, jd_url, status, resume_run_id, resume_status, "
    "letter_run_id, letter_status, qa_status, qa_assessment_json, "
    "qa_attempts, resume_filename, letter_filename, error, "
    "created_at, updated_at, finished_at"
)


def get_tailor_run(conn: sqlite3.Connection, tailor_run_id: int) -> TailorRunRecord | None:
    """Return the run record for ``tailor_run_id`` or ``None`` if not found."""
    row = conn.execute(
        f"SELECT {_SELECT_COLUMNS} FROM tailor_runs WHERE id = ?",  # noqa: S608 - column list is a module-level literal
        (tailor_run_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def list_tailor_runs(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    job_id: int | None = None,
    status: TailorRunStatus | None = None,
) -> list[TailorRunRecord]:
    """Return tailor runs newest-first, optionally filtered."""
    clauses: list[str] = []
    params: list[object] = []
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(job_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status.value)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    # ``where`` is built from a closed set of literal clauses above; all
    # user-supplied values are bound via params.
    base = f"SELECT {_SELECT_COLUMNS} FROM tailor_runs "  # noqa: S608 - column list is a module-level literal
    sql = f"{base}{where} ORDER BY created_at DESC LIMIT ?"
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_record(row) for row in rows]


def update_status(
    conn: sqlite3.Connection,
    tailor_run_id: int,
    *,
    status: TailorRunStatus,
    resume_run_id: str | None = None,
    resume_status: str | None = None,
    letter_run_id: str | None = None,
    letter_status: str | None = None,
    qa_status: QAStatus | None = None,
    qa_assessment: QAAssessment | None = None,
    qa_attempts: int | None = None,
    resume_filename: str | None = None,
    letter_filename: str | None = None,
    error: str | None = None,
) -> None:
    """Persist a state-machine transition.

    Only fields that are not ``None`` overwrite existing values, so callers
    can advance one slice (e.g. just resume_status) without clobbering
    others. ``finished_at`` is set automatically when ``status`` is
    terminal. ``qa_assessment`` is JSON-serialised on the way in.
    """
    now = _now()
    sets = ["status = ?", "updated_at = ?"]
    params: list[object] = [status.value, now]

    def _maybe(column: str, value: object | None) -> None:
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(value)

    _maybe("resume_run_id", resume_run_id)
    _maybe("resume_status", resume_status)
    _maybe("letter_run_id", letter_run_id)
    _maybe("letter_status", letter_status)
    if qa_status is not None:
        sets.append("qa_status = ?")
        params.append(qa_status.value)
    if qa_assessment is not None:
        sets.append("qa_assessment_json = ?")
        params.append(qa_assessment.model_dump_json())
    if qa_attempts is not None:
        sets.append("qa_attempts = ?")
        params.append(qa_attempts)
    _maybe("resume_filename", resume_filename)
    _maybe("letter_filename", letter_filename)
    _maybe("error", error)

    if status in TERMINAL_STATUSES:
        sets.append("finished_at = ?")
        params.append(now)

    params.append(tailor_run_id)
    sql = f"UPDATE tailor_runs SET {', '.join(sets)} WHERE id = ?"  # noqa: S608 - column names are literals
    conn.execute(sql, params)
    conn.commit()


def _row_to_record(row: sqlite3.Row) -> TailorRunRecord:
    qa_status_raw = row["qa_status"]
    qa_assessment_raw = row["qa_assessment_json"]
    raw_job_id = row["job_id"]
    return TailorRunRecord(
        id=int(row["id"]),
        job_id=int(raw_job_id) if raw_job_id is not None else None,
        jd_url=row["jd_url"],
        status=TailorRunStatus(row["status"]),
        resume_run_id=row["resume_run_id"],
        resume_status=row["resume_status"],
        letter_run_id=row["letter_run_id"],
        letter_status=row["letter_status"],
        qa_status=QAStatus(qa_status_raw) if qa_status_raw else None,
        qa_assessment=(
            QAAssessment.model_validate(json.loads(qa_assessment_raw))
            if qa_assessment_raw
            else None
        ),
        qa_attempts=int(row["qa_attempts"]) if row["qa_attempts"] is not None else 0,
        resume_filename=row["resume_filename"],
        letter_filename=row["letter_filename"],
        error=row["error"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        finished_at=row["finished_at"],
    )
