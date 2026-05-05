"""Pydantic response models shared across API routes.

Defining models here (rather than per-route) lets us evolve the
public response shape independently from the SQL repository
internals: a column rename in the schema is contained to the
repository code; adding a field is a one-line change here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobSourceLink(BaseModel):
    """One source's record of a canonical job.

    A canonical job can have multiple links — Greenhouse and LinkedIn
    might both surface the same role; we keep both apply URLs.
    """

    source_name: str = Field(description="The source identifier, e.g. 'greenhouse:atlassian'.")
    apply_url: str = Field(description="Where this source sends applicants.")


class JobSummary(BaseModel):
    """Compact job representation for search results."""

    id: int
    title: str
    company: str
    location_raw: str | None = None
    location_country: str | None = None
    location_city: str | None = None
    remote_type: str | None = Field(
        default=None, description="'remote' | 'hybrid' | 'onsite' | null."
    )
    employment_type: str | None = None
    posted_at: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    apply_url: str
    first_seen_at: str
    last_seen_at: str
    sources: list[JobSourceLink] = Field(default_factory=list)


class JobDetail(JobSummary):
    """Full job representation including description and dedup fingerprint."""

    description_text: str | None = None
    description_html: str | None = None
    company_norm: str
    fingerprint_json: str = Field(
        description="Audit metadata for the dedup decision (deterministic key, normalised inputs).",
    )


class JobsListResponse(BaseModel):
    """Paginated list of :class:`JobSummary` items."""

    total: int = Field(description="Total matching items across all pages.")
    limit: int
    offset: int
    items: list[JobSummary]


class JobStateUpdate(BaseModel):
    """Request body for ``POST /api/jobs/{id}/state``."""

    state: str = Field(
        description="One of: 'new', 'saved', 'applied', 'dismissed', 'rejected'.",
    )
    notes: str | None = None


class JobStateResponse(BaseModel):
    """Response to a state update — echoes the persisted row."""

    job_id: int
    state: str
    notes: str | None
    updated_at: str
