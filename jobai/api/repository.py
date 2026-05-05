"""SQL queries shared across API routes.

Keeping queries in one module means: routes stay focused on HTTP
concerns (validation, status codes, response shaping), schema
changes are contained to one place, and we can test queries
independently of the FastAPI machinery.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from typing import Any

from jobai.api.models import JobDetail, JobsListResponse, JobSourceLink, JobSummary

# Columns selected for the summary view. Listed once to keep the search
# and detail queries in lockstep on column ordering.
_SUMMARY_COLUMNS = (
    "j.id, j.title, j.company, j.location_raw, j.location_country, j.location_city, "
    "j.remote_type, j.employment_type, j.posted_at, "
    "j.salary_min, j.salary_max, j.salary_currency, "
    "j.apply_url, j.first_seen_at, j.last_seen_at"
)
_DETAIL_EXTRA = ", j.description_text, j.description_html, j.company_norm, j.fingerprint_json"

#: Filter values for ``remote_type`` we accept on search.
_VALID_REMOTE_TYPES = {"remote", "hybrid", "onsite"}

#: Cap on per-page items so an unbounded query can't blow up memory.
MAX_LIMIT = 100
DEFAULT_LIMIT = 20


def search_jobs(
    conn: sqlite3.Connection,
    *,
    q: str | None = None,
    location: str | None = None,
    remote_type: str | None = None,
    employment_type: str | None = None,
    posted_since: str | None = None,
    company: str | None = None,
    source_kind: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> JobsListResponse:
    """Filter / search canonical jobs and return a paginated list.

    When ``q`` is provided, the FTS5 index drives ranking; otherwise
    results are sorted by ``last_seen_at DESC`` (freshest first).
    """
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)

    where, params, fts_join = _build_where(
        q=q,
        location=location,
        remote_type=remote_type,
        employment_type=employment_type,
        posted_since=posted_since,
        company=company,
        source_kind=source_kind,
    )

    order_by = "ORDER BY fts.rank" if q else "ORDER BY j.last_seen_at DESC"

    base_query = f"FROM jobs j {fts_join} {where}"
    total = int(conn.execute(f"SELECT COUNT(DISTINCT j.id) {base_query}", params).fetchone()[0])

    rows = conn.execute(
        f"SELECT DISTINCT {_SUMMARY_COLUMNS} {base_query} {order_by} LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()

    job_ids = [int(r[0]) for r in rows]
    sources_by_job = _load_source_links(conn, job_ids)

    items = [_row_to_summary(r, sources_by_job.get(int(r[0]), [])) for r in rows]

    return JobsListResponse(total=total, limit=limit, offset=offset, items=items)


def get_job_detail(conn: sqlite3.Connection, job_id: int) -> JobDetail | None:
    """Return one canonical job's full detail, or ``None`` if not found."""
    sql = f"SELECT {_SUMMARY_COLUMNS}{_DETAIL_EXTRA} FROM jobs j WHERE j.id = ?"  # noqa: S608  - column lists are module-level constants
    row = conn.execute(sql, (job_id,)).fetchone()
    if row is None:
        return None

    source_links = _load_source_links(conn, [job_id]).get(job_id, [])
    return _row_to_detail(row, source_links)


def _build_where(
    *,
    q: str | None,
    location: str | None,
    remote_type: str | None,
    employment_type: str | None,
    posted_since: str | None,
    company: str | None,
    source_kind: str | None,
) -> tuple[str, list[Any], str]:
    """Compose WHERE / params / optional FTS join from filter args."""
    clauses: list[str] = []
    params: list[Any] = []
    fts_join = ""

    if q:
        sanitized = sanitize_fts_query(q)
        if sanitized:
            fts_join = "JOIN jobs_fts fts ON fts.rowid = j.id"
            clauses.append("jobs_fts MATCH ?")
            params.append(sanitized)

    if location:
        clauses.append(
            "(j.location_raw LIKE ? OR j.location_city LIKE ? OR j.location_country LIKE ?)"
        )
        params.extend([f"%{location}%"] * 3)

    if remote_type:
        if remote_type not in _VALID_REMOTE_TYPES:
            raise ValueError(f"unknown remote_type {remote_type!r}")
        clauses.append("j.remote_type = ?")
        params.append(remote_type)

    if employment_type:
        clauses.append("j.employment_type = ?")
        params.append(employment_type)

    if posted_since:
        clauses.append("(j.posted_at IS NULL OR j.posted_at >= ?)")
        params.append(posted_since)

    if company:
        clauses.append("j.company_norm LIKE ?")
        params.append(f"%{company.lower()}%")

    if source_kind:
        # Restrict via job_sources -> sources join.
        clauses.append(
            "j.id IN (SELECT js.job_id FROM job_sources js "
            "JOIN sources s ON s.id = js.source_id WHERE s.kind = ?)"
        )
        params.append(source_kind)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params, fts_join


def _load_source_links(
    conn: sqlite3.Connection,
    job_ids: Sequence[int],
) -> dict[int, list[JobSourceLink]]:
    """Bulk-fetch the (source_name, apply_url) pairs for many jobs in one query."""
    if not job_ids:
        return {}
    placeholders = ",".join("?" for _ in job_ids)
    # placeholders is "?,?,?" derived from len(job_ids); ids bound via params.
    sql = (
        "SELECT js.job_id, s.kind, s.account, js.apply_url "  # noqa: S608
        "FROM job_sources js JOIN sources s ON s.id = js.source_id "
        f"WHERE js.job_id IN ({placeholders})"
    )
    rows = conn.execute(sql, list(job_ids)).fetchall()
    grouped: dict[int, list[JobSourceLink]] = {}
    for job_id, kind, account, apply_url in rows:
        name = f"{kind}:{account}" if account else str(kind)
        grouped.setdefault(int(job_id), []).append(
            JobSourceLink(source_name=name, apply_url=str(apply_url))
        )
    return grouped


_FTS_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def sanitize_fts_query(raw: str) -> str:
    """Convert a free-text user input into a safe FTS5 ``MATCH`` query.

    Strategy: extract word-character tokens (Unicode letters / digits)
    and quote each so FTS5 treats them as literal phrase parts. That
    rules out injection of FTS5 operators (``OR``, ``NOT``, column
    filters, etc.) from user input.
    """
    tokens = _FTS_TOKEN_RE.findall(raw)
    return " ".join(f'"{t}"' for t in tokens)


def _row_to_summary(
    row: sqlite3.Row | tuple[Any, ...],
    sources: list[JobSourceLink],
) -> JobSummary:
    return JobSummary(
        id=int(row[0]),
        title=str(row[1]),
        company=str(row[2]),
        location_raw=_optional_str(row[3]),
        location_country=_optional_str(row[4]),
        location_city=_optional_str(row[5]),
        remote_type=_optional_str(row[6]),
        employment_type=_optional_str(row[7]),
        posted_at=_optional_str(row[8]),
        salary_min=_optional_int(row[9]),
        salary_max=_optional_int(row[10]),
        salary_currency=_optional_str(row[11]),
        apply_url=str(row[12]),
        first_seen_at=str(row[13]),
        last_seen_at=str(row[14]),
        sources=sources,
    )


def _row_to_detail(
    row: sqlite3.Row | tuple[Any, ...],
    sources: list[JobSourceLink],
) -> JobDetail:
    summary = _row_to_summary(row, sources)
    return JobDetail(
        **summary.model_dump(),
        description_text=_optional_str(row[15]),
        description_html=_optional_str(row[16]),
        company_norm=str(row[17]),
        fingerprint_json=str(row[18]),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
