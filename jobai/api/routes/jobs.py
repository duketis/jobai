"""Jobs endpoints: search, detail, user state.

Search supports filters (q, location, remote, employment_type,
posted_since, company, source_kind) plus pagination. Detail returns
the full canonical row with description and source links.

The user-state endpoint persists how the human / agent has triaged
this job (saved, applied, dismissed, rejected) into the
``jobs_user_state`` table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from jobai.api.dependencies import ConnDep
from jobai.api.models import (
    JobDetail,
    JobsListResponse,
    JobStateResponse,
    JobStateUpdate,
)
from jobai.api.repository import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    get_job_detail,
    search_jobs,
)

router = APIRouter()

_VALID_STATES = {"new", "saved", "applied", "dismissed", "rejected"}


@router.get(
    "",
    response_model=JobsListResponse,
    summary="Search jobs with filters and pagination",
)
def list_jobs(
    conn: ConnDep,
    q: Annotated[
        str | None,
        Query(
            description="Free-text search over title, company, description, location (FTS5).",
            max_length=500,
        ),
    ] = None,
    location: Annotated[
        str | None,
        Query(description="Substring match on location (city / country / raw)."),
    ] = None,
    remote: Annotated[
        str | None,
        Query(
            description="Filter by remote type: 'remote', 'hybrid', or 'onsite'.",
            pattern="^(remote|hybrid|onsite)$",
        ),
    ] = None,
    employment_type: Annotated[
        str | None,
        Query(description="Filter by employment type, e.g. 'full-time'."),
    ] = None,
    posted_since: Annotated[
        str | None,
        Query(
            description="Return jobs posted on or after this ISO 8601 date/time.",
            max_length=64,
        ),
    ] = None,
    company: Annotated[
        str | None,
        Query(description="Substring match on normalised company name."),
    ] = None,
    source_kind: Annotated[
        str | None,
        Query(description="Restrict to jobs surfaced by this source kind."),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_LIMIT, description=f"Max items per page (1-{MAX_LIMIT})."),
    ] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
) -> JobsListResponse:
    """Return paginated jobs matching the supplied filters."""
    return search_jobs(
        conn,
        q=q,
        location=location,
        remote_type=remote,
        employment_type=employment_type,
        posted_since=posted_since,
        company=company,
        source_kind=source_kind,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{job_id}",
    response_model=JobDetail,
    summary="Fetch one canonical job's full detail",
)
def get_job(conn: ConnDep, job_id: int) -> JobDetail:
    """Return full job detail or 404 if no canonical row matches."""
    job = get_job_detail(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@router.post(
    "/{job_id}/state",
    response_model=JobStateResponse,
    summary="Update the user's triage state for a job",
)
def update_job_state(
    conn: ConnDep,
    job_id: int,
    body: JobStateUpdate,
) -> JobStateResponse:
    """Persist a state transition for ``job_id``.

    Returns 404 if the job does not exist; 422 if ``state`` is not
    one of the recognised values.
    """
    if body.state not in _VALID_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"state must be one of {sorted(_VALID_STATES)}",
        )

    if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")

    now = datetime.now(tz=UTC).isoformat()
    conn.execute(
        "INSERT INTO jobs_user_state (job_id, state, notes, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(job_id) DO UPDATE SET "
        "  state = excluded.state, "
        "  notes = excluded.notes, "
        "  updated_at = excluded.updated_at",
        (job_id, body.state, body.notes, now),
    )
    conn.commit()

    return JobStateResponse(
        job_id=job_id,
        state=body.state,
        notes=body.notes,
        updated_at=now,
    )
