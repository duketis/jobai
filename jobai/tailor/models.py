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

    Matches the CHECK constraint on ``tailor_runs.status`` (migrations
    0005, 0006, 0008). Stored as a TEXT column so the value is human-
    readable in ad-hoc SQLite browsing.

    ``qa_retry_running`` fires when QA returned must-fix issues and the
    orchestrator is re-kicking the cover letter with the QA feedback
    appended to the JD prompt. Distinct from ``letter_running`` so the
    UI can show "QA fix attempt 2/2" rather than treating a retry as
    a fresh chain.
    """

    PENDING = "pending"
    RESUME_RUNNING = "resume_running"
    LETTER_RUNNING = "letter_running"
    QA_RUNNING = "qa_running"
    QA_RETRY_RUNNING = "qa_retry_running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: Terminal states — the orchestrator never updates a row that is in one of
#: these and the scheduler never re-kicks it.
TERMINAL_STATUSES: frozenset[TailorRunStatus] = frozenset(
    {TailorRunStatus.SUCCEEDED, TailorRunStatus.FAILED},
)


class QAStatus(StrEnum):
    """Outcome of the final cross-artefact QA pass.

    Not the same axis as :class:`TailorRunStatus` -- a row's
    ``qa_status`` is ``running`` while the LLM is thinking and one of
    pass / concerns / fail when the assessment returns. The row's
    overall ``status`` is ``qa_running`` during this stage and lands at
    ``succeeded`` regardless of the QA outcome (we surface the
    assessment, we don't gatekeep the artefact).
    """

    RUNNING = "running"
    PASS = "pass"  # noqa: S105 - 'pass' is a QA verdict label, not a credential
    CONCERNS = "concerns"
    FAIL = "fail"


class QAIssue(BaseModel):
    """One structured issue from the cross-artefact QA agent."""

    severity: str = Field(
        description=(
            "'must_fix' for application-breaking issues (contradictions, "
            "missing must-have keywords) or 'nice_to_fix' for polish."
        ),
    )
    category: str = Field(
        description=(
            "'coverage' (JD requirements not addressed), 'consistency' "
            "(resume/letter contradict), 'format' (visual / tonal "
            "inconsistency between the two), or 'content' (weak prose / "
            "buzzwords)."
        ),
    )
    summary: str = Field(description="One-line description of the issue.")
    detail: str | None = Field(
        default=None,
        description="Longer explanation including the offending text or location.",
    )


class QAAssessment(BaseModel):
    """Structured output the QA agent returns.

    Scores are 0-100. ``status`` is the headline derived from the
    issues + scores (pass ≥ 80 with no must-fix; concerns 60-79 or
    any nice-to-fix; fail < 60 or any must-fix).
    """

    status: QAStatus
    coverage_score: int = Field(ge=0, le=100)
    consistency_score: int = Field(ge=0, le=100)
    format_score: int = Field(ge=0, le=100)
    must_fix_issues: list[QAIssue] = Field(default_factory=list)
    nice_to_fix_issues: list[QAIssue] = Field(default_factory=list)
    summary: str = Field(
        description="One-paragraph human-readable verdict shown in the UI tooltip.",
    )


class TailorRunRecord(BaseModel):
    """One row from ``tailor_runs`` shaped for HTTP responses.

    ``job_id`` is the canonical jobai job id when the chain was kicked
    against a catalogue row; ``jd_url`` carries the URL directly when
    the chain came in via ``POST /api/tailor/url`` for a JD jobai
    never scraped. Exactly one of the two is set per row (DB-level
    CHECK enforces). The sibling run ids (``resume_run_id`` /
    ``letter_run_id``) are opaque strings the siblings hand back.
    ``error`` is non-null only when ``status`` is ``failed`` — surface
    it directly so the UI can render the cause without an extra
    round-trip. ``qa_status`` + ``qa_assessment`` are populated by
    the final cross-artefact QA pass.
    """

    id: int
    job_id: int | None = None
    jd_url: str | None = None
    status: TailorRunStatus
    resume_run_id: str | None = None
    resume_status: str | None = None
    letter_run_id: str | None = None
    letter_status: str | None = None
    qa_status: QAStatus | None = None
    qa_assessment: QAAssessment | None = None
    qa_attempts: int = Field(
        default=0,
        description=(
            "How many QA passes have run for this chain. 0 means QA hasn't "
            "fired yet; 1 is the initial verdict; 2 means a retry happened "
            "(orchestrator re-kicked the letter with QA feedback)."
        ),
    )
    resume_filename: str | None = Field(
        default=None,
        description=(
            "The descriptive PDF filename the resume should download as "
            "(e.g. `Jane_Doe-Software_Engineer-Acme-Resume.pdf`). Populated "
            "by the orchestrator at terminal SUCCESS so the frontend can "
            "render it as a link label + `<a download=...>` attribute. "
            "Null for runs that finished before v1.15.0 -- the PDF route "
            "falls back to live computation in that case."
        ),
    )
    letter_filename: str | None = Field(
        default=None,
        description=("Descriptive cover-letter filename, see `resume_filename`."),
    )
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


class KickByUrlRequest(BaseModel):
    """Request body for ``POST /api/tailor/url``.

    The endpoint accepts a bare JD URL and tries to resolve it to an
    existing catalogue job first (so the run lands on the normal
    catalogue path with full metadata + tracking). When no match is
    found, the chain still runs -- the URL is forwarded directly to
    resumeai and the row carries it on ``tailor_runs.jd_url``.
    """

    jd_url: str = Field(min_length=1, max_length=2000)


class KickByUrlResponse(BaseModel):
    """Response to ``POST /api/tailor/url``.

    Tells the caller which path the kick took so the UI can show
    "matched existing job" vs "tailoring directly from the URL".
    """

    tailor_run_id: int
    status: TailorRunStatus
    matched_job_id: int | None = Field(
        default=None,
        description=(
            "Set when the URL matched a catalogue job; the chain ran "
            "the normal path. Null when no match was found and the "
            "chain is using the bare-URL fallback."
        ),
    )
    matched_count: int = Field(
        default=0,
        description=(
            "How many catalogue rows matched the URL (only the first "
            "is used to kick the chain). Useful when the same JD was "
            "scraped from multiple boards."
        ),
    )


class KickBatchRequest(BaseModel):
    """Request body for ``POST /api/tailor/batch``."""

    job_ids: list[int] = Field(
        min_length=1,
        max_length=10_000,
        description=(
            "Canonical jobai job ids to tailor. Each id becomes one row in "
            "tailor_runs; chains run concurrently up to the configured cap "
            "(JOBAI_TAILOR_MAX_CONCURRENT, default 3) -- everything above "
            "the cap queues up and processes in order. Upper bound is a "
            "safety stop, not a quota."
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
