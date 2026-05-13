"""Pydantic models for the tailor surface.

Two layers live here:

* Response shapes for ``/api/tailor`` — what the frontend consumes.
* Request shapes for the sibling services — what we send to resumeai /
  coverletterai. Keeping them next to the routes' response models means
  every wire-format change happens in one file.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TailorRunStatus(StrEnum):
    """Lifecycle states the orchestrator walks per chain.

    Matches the CHECK constraint on ``tailor_runs.status`` (migration
    0005). Stored as a TEXT column so the value is human-readable in
    ad-hoc SQLite browsing.
    """

    PENDING = "pending"
    RESUME_RUNNING = "resume_running"
    LETTER_RUNNING = "letter_running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: Terminal states — the orchestrator never updates a row that is in one of
#: these and the scheduler never re-kicks it.
TERMINAL_STATUSES: frozenset[TailorRunStatus] = frozenset(
    {TailorRunStatus.SUCCEEDED, TailorRunStatus.FAILED},
)


class TailorRunRecord(BaseModel):
    """One row from ``tailor_runs`` shaped for HTTP responses.

    ``job_id`` is the canonical jobai job id; the sibling run ids
    (``resume_run_id`` / ``letter_run_id``) are opaque strings the
    siblings hand back. ``error`` is non-null only when ``status`` is
    ``failed`` — surface it directly so the UI can render the cause
    without an extra round-trip.
    """

    id: int
    job_id: int
    status: TailorRunStatus
    resume_run_id: str | None = None
    resume_status: str | None = None
    letter_run_id: str | None = None
    letter_status: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None


class TailorRunsListResponse(BaseModel):
    """Paginated list of tailor runs, newest first."""

    items: list[TailorRunRecord]


class KickOneResponse(BaseModel):
    """Response to ``POST /api/tailor/jobs/{job_id}`` — one new run created."""

    tailor_run_id: int = Field(description="The jobai-internal tailor_runs.id.")
    job_id: int
    status: TailorRunStatus


class KickBatchRequest(BaseModel):
    """Request body for ``POST /api/tailor/batch``."""

    job_ids: list[int] = Field(
        min_length=1,
        max_length=100,
        description=(
            "Canonical jobai job ids to tailor. Each id becomes one row in "
            "tailor_runs; chains run concurrently up to the configured cap."
        ),
    )


class KickBatchResponse(BaseModel):
    """Response to ``POST /api/tailor/batch`` — list of per-job rows created."""

    items: list[KickOneResponse]


class ResumeaiTailorRequest(BaseModel):
    """Request body for ``POST /api/tailor`` on resumeai.

    Mirrors the request shape resumeai (and coverletterai) accept. Exactly
    one of ``jd_url`` / ``jd_text`` must be set; we always populate
    ``jd_url`` from the job's ``apply_url``.
    """

    jd_url: str | None = None
    jd_text: str | None = None
    model: str | None = None


class CoverletteraiTailorRequest(BaseModel):
    """Request body for ``POST /api/tailor`` on coverletterai.

    Extends the resumeai shape with ``resume_run_id`` so the cover-letter
    pipeline can fetch the matching tailored resume by reference. We
    never use ``resume_payload`` — by-ref keeps the integration thin.
    """

    jd_url: str | None = None
    jd_text: str | None = None
    model: str | None = None
    resume_run_id: str | None = None


class SiblingRunSnapshot(BaseModel):
    """The status fields we care about when polling a sibling service.

    Both resumeai and coverletterai return a much larger record per run;
    the orchestrator only needs ``id`` + ``status`` to drive the state
    machine. ``model_config`` ignores extra fields so changes elsewhere
    in the sibling response don't break parsing.
    """

    id: str
    status: str

    model_config = {"extra": "ignore"}
