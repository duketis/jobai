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


class JobIdsResponse(BaseModel):
    """Every job id matching the search filters (cross-page select-all).

    The UI requests this when the user clicks "Select all N matching"
    so the batch-tailor submit can carry the full id list, not just
    the current page's. Capped at :data:`jobai.api.repository.MAX_IDS`
    (default 1000) -- callers compare ``len(ids)`` to ``total`` to
    detect truncation.
    """

    ids: list[int] = Field(description="Matching job ids, in the requested sort order.")
    total: int = Field(description="Total matching count regardless of any cap.")


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


class SourceSummary(BaseModel):
    """One configured source plus its current runtime health."""

    id: int
    name: str = Field(description="The source identifier, e.g. 'greenhouse:atlassian'.")
    kind: str
    account: str
    display_name: str
    default_tier: int
    enabled: bool
    cadence_seconds: int
    current_tier: int | None = None
    last_success_at: str | None = None
    last_error_at: str | None = None
    last_error_class: str | None = None
    consecutive_failures: int = 0
    cooldown_until: str | None = None


class SourcesListResponse(BaseModel):
    """List of configured sources for the dashboard / agent."""

    items: list[SourceSummary]


class NotificationItem(BaseModel):
    """One in-app notification row."""

    id: int
    kind: str
    severity: str = Field(description="'info' | 'warn' | 'error'.")
    title: str
    body: str | None = None
    created_at: str
    read_at: str | None = None


class NotificationsListResponse(BaseModel):
    """Paginated notifications, optionally filtered to unread."""

    total: int
    unread_count: int
    items: list[NotificationItem]


class NotificationReadResponse(BaseModel):
    """Echo of the read state after marking a notification."""

    id: int
    read_at: str


class AgentChatRequest(BaseModel):
    """Request body for ``POST /api/agent/chat``.

    A ``conversation_id`` of ``None`` starts a new conversation; the
    server titles it from the first message. Otherwise the message is
    appended to the existing thread.
    """

    conversation_id: int | None = Field(
        default=None,
        description="Existing conversation to continue, or null to start a new one.",
    )
    message: str = Field(
        min_length=1,
        max_length=10_000,
        description="The user's message for this turn.",
    )


class ConversationItem(BaseModel):
    """A single conversation in a list view."""

    id: int
    title: str
    created_at: str
    updated_at: str


class ConversationsListResponse(BaseModel):
    """Paginated list of conversations, newest activity first."""

    items: list[ConversationItem]


class ConversationMessageItem(BaseModel):
    """One stored message returned over the wire.

    ``content`` is JSON-decoded — either a plain string (user prompt)
    or the polymorphic Anthropic content array (text + tool_use +
    tool_result blocks).
    """

    id: int
    role: str
    content: list[dict[str, object]] | str
    created_at: str


class ConversationDetailResponse(BaseModel):
    """A conversation with all its messages, oldest first."""

    id: int
    title: str
    created_at: str
    updated_at: str
    messages: list[ConversationMessageItem]
