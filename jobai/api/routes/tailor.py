"""Tailor endpoints: kick chains, list / inspect runs, stream PDFs.

The orchestration runs in a background task pool owned by the app
lifespan (see :mod:`jobai.tailor.worker`); these routes are thin: they
validate input, write a tailor_runs row, hand the row id to the pool,
and return.

PDF endpoints proxy-stream from the sibling services. The siblings'
PDF routes live at slightly different paths (``/runs/{id}/pdf`` on
resumeai, ``/api/runs/{id}/pdf`` on coverletterai) — that quirk is
fully contained in :mod:`jobai.tailor.client`, so the route bodies
just call ``stream_pdf``.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jobai.api.dependencies import ConnDep, get_db_path
from jobai.api.repository import find_jobs_by_url
from jobai.api.runtime_settings import get_effective_agent_config
from jobai.tailor.client import CoverletteraiClient, ResumeaiClient
from jobai.tailor.filenames import build_pdf_filename
from jobai.tailor.models import (
    KickBatchRequest,
    KickBatchResponse,
    KickByUrlRequest,
    KickByUrlResponse,
    KickOneResponse,
    QAAssessment,
    TailorRunRecord,
    TailorRunsListResponse,
    TailorRunStatus,
)
from jobai.tailor.orchestrator import run_chain
from jobai.tailor.qa import QAClient, build_qa_client
from jobai.tailor.repository import (
    create_tailor_run,
    get_tailor_run,
    list_tailor_runs,
    set_applied,
)
from jobai.tailor.worker import TailorPool

router = APIRouter()


# -- DI helpers -----------------------------------------------------------


def get_tailor_pool(request: Request) -> TailorPool:
    """Pull the lifespan-owned :class:`TailorPool` off ``app.state``."""
    pool: TailorPool | None = getattr(request.app.state, "tailor_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="tailor pool not initialised (scheduler / lifespan disabled?)",
        )
    return pool


def get_resume_client(request: Request) -> ResumeaiClient:
    """Pull the lifespan-owned resumeai client off ``app.state``."""
    client: ResumeaiClient | None = getattr(request.app.state, "resume_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="resumeai client not initialised")
    return client


def get_letter_client(request: Request) -> CoverletteraiClient:
    """Pull the lifespan-owned coverletterai client off ``app.state``."""
    client: CoverletteraiClient | None = getattr(request.app.state, "letter_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="coverletterai client not initialised")
    return client


def get_qa_client(conn: ConnDep) -> QAClient | None:
    """Build a QA client from the live effective agent config.

    Rebuilt per request (not stashed on ``app.state``) so that the
    moment the user flips backend / pastes a key / paints an OAuth
    token via the Settings UI, the next tailor chain picks it up
    without a process restart. Returns ``None`` when no credentials
    are reachable; the orchestrator then skips the QA stage cleanly.
    """
    cfg = get_effective_agent_config(conn)
    return build_qa_client(cfg)


def get_resumeai_url(request: Request) -> str:
    """Return the resumeai base URL the lifespan stashed on ``app.state``.

    Each tailor chain uses this to refresh every project-scan context
    entry right before resumeai sees the JD, so the LLM's portfolio
    stats are always live.
    """
    url: str | None = getattr(request.app.state, "resumeai_url", None)
    if url is None:
        raise HTTPException(status_code=503, detail="resumeai url not initialised")
    return url


def get_tailor_output_dir(request: Request) -> Path:
    """Return the per-job snapshot output directory.

    Read off ``app.state.tailor_output_dir`` (set by the lifespan from
    ``Settings.tailor_output_dir`` env config). Each successful tailor
    run drops a folder under this path containing the PDFs + JD + QA +
    CHECKLIST.md the user opens to apply.
    """
    output: Path | None = getattr(request.app.state, "tailor_output_dir", None)
    if output is None:
        raise HTTPException(status_code=503, detail="tailor output dir not initialised")
    return output


PoolDep = Annotated[TailorPool, Depends(get_tailor_pool)]
ResumeDep = Annotated[ResumeaiClient, Depends(get_resume_client)]
LetterDep = Annotated[CoverletteraiClient, Depends(get_letter_client)]
QADep = Annotated["QAClient | None", Depends(get_qa_client)]
DbPathDep = Annotated[Path, Depends(get_db_path)]
ResumeaiUrlDep = Annotated[str, Depends(get_resumeai_url)]
TailorOutputDirDep = Annotated[Path, Depends(get_tailor_output_dir)]


# -- Helpers --------------------------------------------------------------


def _schedule_chain(
    *,
    pool: TailorPool,
    tailor_run_id: int,
    db_path: Path,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
    qa_client: QAClient | None,
    resumeai_url: str,
    snapshot_output_dir: Path,
) -> None:
    """Submit a chain coroutine to the pool with all collaborators bound."""
    from jobai.scheduler import refresh_project_scans  # noqa: PLC0415
    from jobai.tailor.qa import fetch_qa_context_summary  # noqa: PLC0415

    async def _refresh() -> None:
        # Discard the (refreshed, failed) counts -- the orchestrator
        # doesn't need them; the helper already logs at INFO. We just
        # need the side-effect of re-scanning every project entry.
        await refresh_project_scans(resumeai_url)

    async def _fetch_qa_context() -> str | None:
        return await fetch_qa_context_summary(resumeai_url)

    async def _resolve_jd_text(jd_url: str) -> str | None:
        # Seek catalogue rows only store the listing teaser and Seek's
        # detail page is Cloudflare-gated, so resumeai can't fetch it
        # (hard 403). Pull the full JD here on jobai's tier-3 stealth
        # fetcher and hand it to the siblings as text. Non-Seek URLs
        # fall through to the existing URL path (resumeai fetches).
        from urllib.parse import urlparse  # noqa: PLC0415

        if "seek.com" not in urlparse(jd_url).netloc.lower():
            return None
        from jobai.fetcher.dispatch import build_fetcher  # noqa: PLC0415
        from jobai.sources.seek_detail import fetch_seek_jd_text  # noqa: PLC0415

        fetcher = build_fetcher(tier=3)
        try:
            return await fetch_seek_jd_text(jd_url, fetcher)
        finally:
            with contextlib.suppress(Exception):
                await fetcher.aclose()

    async def _factory() -> None:
        await run_chain(
            tailor_run_id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            sleeper=asyncio.sleep,
            qa_client=qa_client,
            refresh_context_scans=_refresh,
            fetch_qa_context=_fetch_qa_context,
            resolve_jd_text=_resolve_jd_text,
            snapshot_output_dir=snapshot_output_dir,
        )

    pool.submit(_factory)


# -- Routes ---------------------------------------------------------------


@router.post(
    "/jobs/{job_id}",
    response_model=KickOneResponse,
    summary="Kick off a tailor chain for one job",
    status_code=202,
)
async def kick_one(
    conn: ConnDep,
    pool: PoolDep,
    resume_client: ResumeDep,
    letter_client: LetterDep,
    qa_client: QADep,
    db_path: DbPathDep,
    resumeai_url: ResumeaiUrlDep,
    snapshot_output_dir: TailorOutputDirDep,
    job_id: int,
) -> KickOneResponse:
    """Create a ``tailor_runs`` row, queue the chain, return the row id."""
    if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    record = create_tailor_run(conn, job_id=job_id)
    _schedule_chain(
        pool=pool,
        tailor_run_id=record.id,
        db_path=db_path,
        resume_client=resume_client,
        letter_client=letter_client,
        qa_client=qa_client,
        resumeai_url=resumeai_url,
        snapshot_output_dir=snapshot_output_dir,
    )
    # The catalogue-path create_tailor_run always sets job_id; the
    # assert is for mypy's benefit since the typed field is Optional
    # (URL-only runs leave it None).
    assert record.job_id is not None  # noqa: S101
    return KickOneResponse(
        tailor_run_id=record.id,
        job_id=record.job_id,
        status=record.status,
    )


@router.post(
    "/batch",
    response_model=KickBatchResponse,
    summary="Kick off tailor chains for many jobs at once",
    status_code=202,
)
async def kick_batch(
    conn: ConnDep,
    pool: PoolDep,
    resume_client: ResumeDep,
    letter_client: LetterDep,
    qa_client: QADep,
    db_path: DbPathDep,
    resumeai_url: ResumeaiUrlDep,
    snapshot_output_dir: TailorOutputDirDep,
    body: KickBatchRequest,
) -> KickBatchResponse:
    """Create one ``tailor_runs`` row per job in ``body.job_ids``.

    Unknown job ids cause a 404 with the offending id listed. Duplicates
    in ``job_ids`` produce duplicate runs (deliberately — the user
    re-kicking on the same job mid-batch is valid).
    """
    # Chunk the existence check so a 10k-id submit doesn't trip
    # SQLite's SQLITE_MAX_VARIABLE_NUMBER (default 32_766 in modern
    # builds, but lower on some platforms). 500-per-batch keeps every
    # query well under any sane limit and still ships in 1-2 round
    # trips for any realistic batch size.
    chunk_size = 500
    found: set[int] = set()
    for start in range(0, len(body.job_ids), chunk_size):
        chunk = body.job_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT id FROM jobs WHERE id IN ({placeholders})",  # noqa: S608 - placeholders count matches params length
            chunk,
        ):
            found.add(int(row[0]))
    missing = sorted(set(body.job_ids) - found)
    if missing:
        # Truncate the 404 detail when the list is huge -- otherwise the
        # response body balloons to MBs of integers for a typo submit.
        preview = missing[:25]
        suffix = f" (+ {len(missing) - 25} more)" if len(missing) > 25 else ""
        raise HTTPException(
            status_code=404,
            detail=f"jobs not found: {preview}{suffix}",
        )
    items: list[KickOneResponse] = []
    for job_id in body.job_ids:
        record = create_tailor_run(conn, job_id=job_id)
        _schedule_chain(
            pool=pool,
            tailor_run_id=record.id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            qa_client=qa_client,
            resumeai_url=resumeai_url,
            snapshot_output_dir=snapshot_output_dir,
        )
        # Batch always uses the catalogue path; narrow for mypy.
        assert record.job_id is not None  # noqa: S101
        items.append(
            KickOneResponse(
                tailor_run_id=record.id,
                job_id=record.job_id,
                status=record.status,
            )
        )
    return KickBatchResponse(items=items)


@router.post(
    "/url",
    response_model=KickByUrlResponse,
    summary="Kick off a tailor chain for a bare JD URL (catalogue or one-off)",
    status_code=202,
)
async def kick_by_url(
    conn: ConnDep,
    pool: PoolDep,
    resume_client: ResumeDep,
    letter_client: LetterDep,
    qa_client: QADep,
    db_path: DbPathDep,
    resumeai_url: ResumeaiUrlDep,
    snapshot_output_dir: TailorOutputDirDep,
    body: KickByUrlRequest,
) -> KickByUrlResponse:
    """Kick a tailor chain for a JD URL.

    Resolution order:

    1. Try to match the URL against the catalogue (exact + query-
       string-stripped). If found, kick the chain on that job_id --
       the normal catalogue path, full metadata, run shows up in
       /tailor-runs joined to the job row.
    2. No match? Kick the chain with the URL on the row directly.
       The siblings still get the JD URL; the run shows up in
       /tailor-runs without a catalogue job (jd_url visible
       instead). Useful for JDs jobai never scraped (LinkedIn DMs,
       recruiter emails, anything off-network).
    """
    matches = find_jobs_by_url(conn, body.jd_url)
    if matches:
        # Catalogue hit -- use the normal path so the run is fully
        # tracked against the existing job row.
        record = create_tailor_run(conn, job_id=matches[0].id)
        _schedule_chain(
            pool=pool,
            tailor_run_id=record.id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            qa_client=qa_client,
            resumeai_url=resumeai_url,
            snapshot_output_dir=snapshot_output_dir,
        )
        return KickByUrlResponse(
            tailor_run_id=record.id,
            status=record.status,
            matched_job_id=matches[0].id,
            matched_count=len(matches),
        )

    # No catalogue match -- fall back to the bare-URL path.
    record = create_tailor_run(conn, jd_url=body.jd_url)
    _schedule_chain(
        pool=pool,
        tailor_run_id=record.id,
        db_path=db_path,
        resume_client=resume_client,
        letter_client=letter_client,
        qa_client=qa_client,
        resumeai_url=resumeai_url,
        snapshot_output_dir=snapshot_output_dir,
    )
    return KickByUrlResponse(
        tailor_run_id=record.id,
        status=record.status,
        matched_job_id=None,
        matched_count=0,
    )


@router.get(
    "/runs",
    response_model=TailorRunsListResponse,
    summary="List tailor runs newest-first",
)
def list_runs(
    conn: ConnDep,
    job_id: Annotated[int | None, Query(description="Filter to a single job.")] = None,
    status: Annotated[
        TailorRunStatus | None,
        Query(description="Filter to one of pending/resume_running/.../failed."),
    ] = None,
    applied: Annotated[
        bool | None,
        Query(description="True = only applied; False = only not-yet-applied; null = both."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> TailorRunsListResponse:
    """Return the most recent tailor runs, optionally filtered."""
    runs = list_tailor_runs(
        conn,
        limit=limit,
        job_id=job_id,
        status=status,
        applied=applied,
    )
    return TailorRunsListResponse(items=runs)


@router.get(
    "/runs/{tailor_run_id}",
    response_model=TailorRunRecord,
    summary="Inspect one tailor run",
)
def get_run(conn: ConnDep, tailor_run_id: int) -> TailorRunRecord:
    """Return full state for ``tailor_run_id`` or 404 if not found."""
    record = get_tailor_run(conn, tailor_run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"tailor run {tailor_run_id} not found")
    return record


class AppliedUpdateRequest(BaseModel):
    """Body for PATCH /api/tailor/runs/{id}/applied.

    ``applied=true`` stamps ``applied_at`` with the current UTC
    timestamp; ``applied=false`` clears it back to NULL. The same
    endpoint handles both directions so the UI's Mark / Unmark
    toggle is a single round trip.
    """

    applied: bool


@router.patch(
    "/runs/{tailor_run_id}/applied",
    response_model=TailorRunRecord,
    summary="Mark or unmark a tailor run as applied (sets applied_at).",
)
def set_run_applied(
    conn: ConnDep,
    tailor_run_id: int,
    body: AppliedUpdateRequest,
) -> TailorRunRecord:
    """Toggle the applied flag on a tailor run."""
    record = set_applied(conn, tailor_run_id, applied=body.applied)
    if record is None:
        raise HTTPException(status_code=404, detail=f"tailor run {tailor_run_id} not found")
    return record


class TailorRunExport(BaseModel):
    """Full bundle of an applied (or applied-able) tailor run.

    Generic export envelope -- any external tool that wants to load a
    jobai application's artefacts can fetch this one URL and have
    everything it needs: the parsed JD, the resume + cover-letter PDF
    locations, the QA verdict, and the bookkeeping metadata. The
    PDFs are returned as URLs (not bytes) so the caller streams them
    independently when it needs to.

    The user copies the export URL from the /tailor-runs page and
    pastes it into whatever downstream tool consumes it. No
    consumer-specific fields live here -- the schema is stable and
    deliberately minimal.
    """

    tailor_run_id: int = Field(description="jobai-internal run id.")
    job_id: int | None = Field(description="Catalogue job id, if matched.")
    jd_url: str | None = Field(description="Job description URL the run targeted.")
    apply_url: str | None = Field(description="Apply URL for the catalogue job, if known.")
    title: str = Field(description="Job title (from catalogue or sibling-parsed JD).")
    company: str = Field(description="Company name (from catalogue or sibling-parsed JD).")
    jd_markdown: str = Field(description="Parsed JD as markdown for downstream consumers.")
    resume_pdf_url: str = Field(description="Stable URL to stream the tailored resume PDF.")
    letter_pdf_url: str = Field(description="Stable URL to stream the tailored cover-letter PDF.")
    resume_filename: str | None = Field(description="Suggested filename for the resume PDF.")
    letter_filename: str | None = Field(description="Suggested filename for the cover-letter PDF.")
    qa_assessment: QAAssessment | None = Field(description="Full QA verdict, if available.")
    applied_at: str | None = Field(description="ISO 8601 UTC when the user marked applied.")
    created_at: str = Field(description="ISO 8601 UTC when the chain was kicked.")
    finished_at: str | None = Field(description="ISO 8601 UTC when the chain reached terminal.")


@router.get(
    "/runs/{tailor_run_id}/export",
    response_model=TailorRunExport,
    summary="Export an applied tailor run as a single JSON bundle.",
)
async def export_run(
    request: Request,
    conn: ConnDep,
    resume_client: ResumeDep,
    tailor_run_id: int,
) -> TailorRunExport:
    """Return the full bundle of artefacts for a tailor run.

    Stable schema for any external tool to consume. The PDFs come
    back as URLs so the caller streams them on demand rather than
    paying the base64 cost up front. JD is rendered as markdown
    (same shape the on-disk ``jd.md`` carries).
    """
    record = get_tailor_run(conn, tailor_run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"tailor run {tailor_run_id} not found")

    # Resolve title + company + apply URL, preferring the jobs row
    # for catalogue runs and the sibling's parsed requirements for
    # bare-URL runs. Same precedence the snapshot module uses.
    title = "Job"
    company = "Company"
    apply_url: str | None = record.jd_url
    if record.job_id is not None:
        row = conn.execute(
            "SELECT title, company, apply_url FROM jobs WHERE id = ?",
            (record.job_id,),
        ).fetchone()
        if row is not None:  # pragma: no branch - jobs FK CASCADE means the run is gone too
            title = row["title"] or title
            company = row["company"] or company
            apply_url = row["apply_url"] or apply_url

    # Pull the sibling's parsed JD + tailored payload. Fail-soft: a
    # sibling outage degrades the JD markdown to a stub but the rest
    # of the bundle is still useful.
    resume_record: dict[str, object] = {}
    if record.resume_run_id:
        try:
            resume_record = await resume_client.get_run(record.resume_run_id)
        except Exception:  # noqa: BLE001 - export is best-effort
            resume_record = {}
    if record.job_id is None and isinstance(resume_record, dict):
        reqs = resume_record.get("requirements") if resume_record else None
        if isinstance(reqs, dict):
            t = reqs.get("title")
            c = reqs.get("company")
            if isinstance(t, str) and t:
                title = t
            if isinstance(c, str) and c:
                company = c

    # JD markdown reuses the same builder as the on-disk snapshot so
    # the export and the file match byte-for-byte.
    from jobai.tailor.snapshot import _build_jd_markdown  # noqa: PLC0415

    jd_markdown = _build_jd_markdown(record, resume_record, title=title, company=company)

    # Build absolute URLs so the export bundle works when copied to
    # another host (the user pastes it into a tool running outside
    # the browser tab that fetched the export).
    base = str(request.base_url).rstrip("/")
    return TailorRunExport(
        tailor_run_id=record.id,
        job_id=record.job_id,
        jd_url=record.jd_url,
        apply_url=apply_url,
        title=title,
        company=company,
        jd_markdown=jd_markdown,
        resume_pdf_url=f"{base}/api/tailor/runs/{record.id}/resume.pdf",
        letter_pdf_url=f"{base}/api/tailor/runs/{record.id}/letter.pdf",
        resume_filename=record.resume_filename,
        letter_filename=record.letter_filename,
        qa_assessment=record.qa_assessment,
        applied_at=record.applied_at,
        created_at=record.created_at,
        finished_at=record.finished_at,
    )


@router.get(
    "/runs/{tailor_run_id}/resume.pdf",
    summary="Stream the tailored resume PDF from resumeai",
)
async def download_resume_pdf(
    conn: ConnDep,
    resume_client: ResumeDep,
    tailor_run_id: int,
) -> StreamingResponse:
    """Proxy ``GET /runs/{resume_run_id}/pdf`` on resumeai for this row."""
    resume_run_id = _require_artefact(conn, tailor_run_id, kind="resume")
    filename = await _resolve_pdf_filename(
        conn=conn,
        resume_client=resume_client,
        tailor_run_id=tailor_run_id,
        kind="resume",
    )
    response = await resume_client.stream_pdf(resume_run_id)
    return await _proxy_pdf(response, filename=filename)


@router.get(
    "/runs/{tailor_run_id}/letter.pdf",
    summary="Stream the tailored cover-letter PDF from coverletterai",
)
async def download_letter_pdf(
    conn: ConnDep,
    resume_client: ResumeDep,
    letter_client: LetterDep,
    tailor_run_id: int,
) -> StreamingResponse:
    """Proxy ``GET /api/runs/{letter_run_id}/pdf`` on coverletterai for this row."""
    letter_run_id = _require_artefact(conn, tailor_run_id, kind="letter")
    filename = await _resolve_pdf_filename(
        conn=conn,
        resume_client=resume_client,
        tailor_run_id=tailor_run_id,
        kind="letter",
    )
    response = await letter_client.stream_pdf(letter_run_id)
    return await _proxy_pdf(response, filename=filename)


# -- Internal helpers -----------------------------------------------------


def _require_artefact(conn: sqlite3.Connection, tailor_run_id: int, *, kind: str) -> str:
    """Return the sibling run id for ``kind`` ("resume" or "letter")."""
    record = get_tailor_run(conn, tailor_run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"tailor run {tailor_run_id} not found")
    sibling_run_id = record.resume_run_id if kind == "resume" else record.letter_run_id
    if sibling_run_id is None:
        label = "resume" if kind == "resume" else "cover letter"
        raise HTTPException(
            status_code=409,
            detail=f"tailor run {tailor_run_id} has not produced a {label} yet",
        )
    return sibling_run_id


async def _proxy_pdf(
    response: httpx.Response,
    *,
    filename: str | None = None,
) -> StreamingResponse:
    """Adapt a streaming :class:`httpx.Response` to a FastAPI StreamingResponse.

    Surfaces sibling-side 4xx/5xx as 502 so the caller doesn't confuse a
    sibling-missing-file with a missing tailor_run row.

    When ``filename`` is provided, set ``Content-Disposition: inline``
    so the browser opens the PDF inline but uses the suggested name on
    "Save as" -- key for batch tailor runs where the user otherwise
    has to rename ``resume.pdf`` / ``resume (1).pdf`` etc. by hand.
    """
    if response.status_code >= 400:
        status_code = response.status_code
        try:
            await response.aread()
        finally:
            await response.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"sibling PDF endpoint returned {status_code}",
        )

    async def _iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()

    media_type = response.headers.get("content-type", "application/pdf")
    headers: dict[str, str] = {}
    if filename:
        # Content-Disposition needs ASCII; the helper that builds the
        # filename already strips non-ASCII so a plain ``filename=``
        # parameter is safe. We deliberately use ``inline`` rather than
        # ``attachment`` so the browser still previews the PDF -- the
        # user only needs the right name when they hit "Save as".
        headers["Content-Disposition"] = f'inline; filename="{filename}"'
    return StreamingResponse(_iter(), media_type=media_type, headers=headers)


async def _resolve_pdf_filename(
    *,
    conn: sqlite3.Connection,
    resume_client: ResumeaiClient,
    tailor_run_id: int,
    kind: str,
) -> str:
    """Return the cached filename from the row, or compute live if absent.

    v1.15.0 added the cache columns so the list-runs response can serve
    filenames without an N+1 sibling fetch. Rows that finished before
    the cache landed still have NULL filenames; fall back to the live
    builder so old runs keep working.
    """
    record = get_tailor_run(conn, tailor_run_id)
    if record is not None:
        cached = record.resume_filename if kind == "resume" else record.letter_filename
        if cached:
            return cached
    return await build_pdf_filename(
        conn=conn,
        resume_client=resume_client,
        tailor_run_id=tailor_run_id,
        kind=kind,
    )
