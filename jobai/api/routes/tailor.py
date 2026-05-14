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
import re
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from jobai.api.dependencies import ConnDep, get_db_path
from jobai.api.repository import find_jobs_by_url
from jobai.api.runtime_settings import get_effective_agent_config
from jobai.tailor.client import CoverletteraiClient, ResumeaiClient
from jobai.tailor.models import (
    KickBatchRequest,
    KickBatchResponse,
    KickByUrlRequest,
    KickByUrlResponse,
    KickOneResponse,
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


PoolDep = Annotated[TailorPool, Depends(get_tailor_pool)]
ResumeDep = Annotated[ResumeaiClient, Depends(get_resume_client)]
LetterDep = Annotated[CoverletteraiClient, Depends(get_letter_client)]
QADep = Annotated["QAClient | None", Depends(get_qa_client)]
DbPathDep = Annotated[Path, Depends(get_db_path)]
ResumeaiUrlDep = Annotated[str, Depends(get_resumeai_url)]


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
) -> None:
    """Submit a chain coroutine to the pool with all collaborators bound."""
    from jobai.scheduler import refresh_project_scans  # noqa: PLC0415

    async def _refresh() -> None:
        # Discard the (refreshed, failed) counts -- the orchestrator
        # doesn't need them; the helper already logs at INFO. We just
        # need the side-effect of re-scanning every project entry.
        await refresh_project_scans(resumeai_url)

    async def _factory() -> None:
        await run_chain(
            tailor_run_id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            sleeper=asyncio.sleep,
            qa_client=qa_client,
            refresh_context_scans=_refresh,
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
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> TailorRunsListResponse:
    """Return the most recent tailor runs, optionally filtered."""
    runs = list_tailor_runs(conn, limit=limit, job_id=job_id, status=status)
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
    filename = await _build_pdf_filename(
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
    filename = await _build_pdf_filename(
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


_FILENAME_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_FILENAME_WHITESPACE = re.compile(r"\s+")


def _sanitize_filename_part(text: str | None, *, fallback: str) -> str:
    """Strip path-illegal characters, collapse whitespace, ASCII-fold.

    Returns ``fallback`` when the input is empty / None or sanitises
    down to nothing -- keeps the final filename non-empty even for
    bare-URL runs the sibling hasn't tagged with a title yet.
    """
    if not text:
        return fallback
    # Drop characters illegal on Windows + macOS, then collapse runs
    # of whitespace to a single space, then ASCII-fold so the
    # downloaded filename works across browsers and shells.
    cleaned = _FILENAME_BAD_CHARS.sub(" ", text)
    cleaned = _FILENAME_WHITESPACE.sub(" ", cleaned).strip(" .")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or fallback


async def _build_pdf_filename(
    *,
    conn: sqlite3.Connection,
    resume_client: ResumeaiClient,
    tailor_run_id: int,
    kind: str,
) -> str:
    """Compose ``<Name>-<JobTitle>-<Company>-<Resume|CoverLetter>.pdf``.

    Identity (applicant name) comes from the sibling's tailored payload
    -- resumeai owns the identity. Title + company come from the
    ``jobs`` row for catalogue runs, or from the sibling's parsed
    requirements for bare-URL runs. Every field has a fallback so a
    partially-populated row still produces a usable filename.

    ``kind`` is ``"resume"`` or ``"letter"``.
    """
    record = get_tailor_run(conn, tailor_run_id)
    # _require_artefact already validated the row exists when the route
    # reached us; this is the defensive fallback for direct callers.
    if record is None:  # pragma: no cover - the route guard runs first
        suffix = "Resume" if kind == "resume" else "CoverLetter"
        return f"Applicant-Job-Company-{suffix}.pdf"

    title = "Job"
    company = "Company"
    if record.job_id is not None:
        row = conn.execute(
            "SELECT title, company FROM jobs WHERE id = ?",
            (record.job_id,),
        ).fetchone()
        if row is not None:
            title = row[0] or title
            company = row[1] or company

    # Pull the applicant name (and, for URL-only runs, title+company)
    # from the sibling's tailored record. Wrapped in a defensive
    # try/except: a sibling outage here only degrades the filename, it
    # doesn't break the actual PDF stream the user is waiting on.
    name = "Applicant"
    if record.resume_run_id:
        try:
            resume_rec = await resume_client.get_run(record.resume_run_id)
        except Exception:  # noqa: BLE001 - filename is best-effort
            resume_rec = {}
        tailored = resume_rec.get("tailored") if isinstance(resume_rec, dict) else None
        if isinstance(tailored, dict):
            name_val = tailored.get("name")
            if isinstance(name_val, str):
                name = name_val
        if record.job_id is None:
            reqs = resume_rec.get("requirements") if isinstance(resume_rec, dict) else None
            if isinstance(reqs, dict):
                t = reqs.get("title")
                c = reqs.get("company")
                if isinstance(t, str):
                    title = t
                if isinstance(c, str):
                    company = c

    name_s = _sanitize_filename_part(name, fallback="Applicant")
    title_s = _sanitize_filename_part(title, fallback="Job")
    company_s = _sanitize_filename_part(company, fallback="Company")
    suffix = "Resume" if kind == "resume" else "CoverLetter"
    # Replace spaces inside each part with underscores so the final
    # filename reads as one hyphen-separated token per field.
    parts = [p.replace(" ", "_") for p in (name_s, title_s, company_s, suffix)]
    return "-".join(parts) + ".pdf"
