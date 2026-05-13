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
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from jobai.api.dependencies import ConnDep, get_db_path
from jobai.tailor.client import CoverletteraiClient, ResumeaiClient
from jobai.tailor.models import (
    KickBatchRequest,
    KickBatchResponse,
    KickOneResponse,
    TailorRunRecord,
    TailorRunsListResponse,
    TailorRunStatus,
)
from jobai.tailor.orchestrator import run_chain
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


PoolDep = Annotated[TailorPool, Depends(get_tailor_pool)]
ResumeDep = Annotated[ResumeaiClient, Depends(get_resume_client)]
LetterDep = Annotated[CoverletteraiClient, Depends(get_letter_client)]
DbPathDep = Annotated[Path, Depends(get_db_path)]


# -- Helpers --------------------------------------------------------------


def _schedule_chain(
    *,
    pool: TailorPool,
    tailor_run_id: int,
    db_path: Path,
    resume_client: ResumeaiClient,
    letter_client: CoverletteraiClient,
) -> None:
    """Submit a chain coroutine to the pool with all collaborators bound."""

    async def _factory() -> None:
        await run_chain(
            tailor_run_id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
            sleeper=asyncio.sleep,
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
    db_path: DbPathDep,
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
    )
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
    db_path: DbPathDep,
    body: KickBatchRequest,
) -> KickBatchResponse:
    """Create one ``tailor_runs`` row per job in ``body.job_ids``.

    Unknown job ids cause a 404 with the offending id listed. Duplicates
    in ``job_ids`` produce duplicate runs (deliberately — the user
    re-kicking on the same job mid-batch is valid).
    """
    placeholders = ",".join("?" for _ in body.job_ids)
    found = {
        int(row[0])
        for row in conn.execute(
            f"SELECT id FROM jobs WHERE id IN ({placeholders})",  # noqa: S608 - placeholders count matches params length
            body.job_ids,
        )
    }
    missing = sorted(set(body.job_ids) - found)
    if missing:
        raise HTTPException(status_code=404, detail=f"jobs not found: {missing}")
    items: list[KickOneResponse] = []
    for job_id in body.job_ids:
        record = create_tailor_run(conn, job_id=job_id)
        _schedule_chain(
            pool=pool,
            tailor_run_id=record.id,
            db_path=db_path,
            resume_client=resume_client,
            letter_client=letter_client,
        )
        items.append(
            KickOneResponse(
                tailor_run_id=record.id,
                job_id=record.job_id,
                status=record.status,
            )
        )
    return KickBatchResponse(items=items)


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
    response = await resume_client.stream_pdf(resume_run_id)
    return await _proxy_pdf(response)


@router.get(
    "/runs/{tailor_run_id}/letter.pdf",
    summary="Stream the tailored cover-letter PDF from coverletterai",
)
async def download_letter_pdf(
    conn: ConnDep,
    letter_client: LetterDep,
    tailor_run_id: int,
) -> StreamingResponse:
    """Proxy ``GET /api/runs/{letter_run_id}/pdf`` on coverletterai for this row."""
    letter_run_id = _require_artefact(conn, tailor_run_id, kind="letter")
    response = await letter_client.stream_pdf(letter_run_id)
    return await _proxy_pdf(response)


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


async def _proxy_pdf(response: httpx.Response) -> StreamingResponse:
    """Adapt a streaming :class:`httpx.Response` to a FastAPI StreamingResponse.

    Surfaces sibling-side 4xx/5xx as 502 so the caller doesn't confuse a
    sibling-missing-file with a missing tailor_run row.
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
    return StreamingResponse(_iter(), media_type=media_type)
