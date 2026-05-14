"""Context-pool endpoints -- proxy the resumeai sibling's pool through jobai.

jobai is the user's primary surface; resumeai is the source of truth
for the context pool that feeds tailoring. Rather than make the user
bounce between two ports to manage their context, these routes proxy
the resumeai endpoints so the entire workflow (jobs -> tailor ->
context) lives behind one URL.

The proxy is intentionally thin: we don't mirror the pool locally,
and we don't add a second source of truth. If resumeai is down the
context page reports it -- the user runs both siblings as a pair
anyway when they're using the tailor chain.
"""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from jobai.context.client import (
    ContextClient,
    ContextFile,
    ProjectScanCreate,
    SnippetCreate,
)

router = APIRouter()


def get_context_client(request: Request) -> ContextClient:
    """Pull the lifespan-owned :class:`ContextClient` off ``app.state``."""
    client: ContextClient | None = getattr(request.app.state, "context_client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="context client not initialised (lifespan disabled?)",
        )
    return client


ContextClientDep = Annotated[ContextClient, Depends(get_context_client)]


@router.get(
    "",
    response_model=list[ContextFile],
    summary="List every entry in the shared user-context pool",
)
async def list_context(client: ContextClientDep) -> list[ContextFile]:
    """Forward to ``resumeai /api/context``."""
    try:
        return await client.list_files()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai context lookup failed: {exc}",
        ) from exc


@router.get(
    "/{file_id}",
    response_model=ContextFile,
    summary="Inspect one context entry",
)
async def get_context(file_id: str, client: ContextClientDep) -> ContextFile:
    try:
        return await client.get_file(file_id)
    except httpx.HTTPStatusError as exc:
        # 404 from resumeai stays a 404 to the caller; anything else
        # comes back as a 502 so the UI can distinguish "not found"
        # from "sibling unavailable".
        status = exc.response.status_code if exc.response is not None else 502
        if status == 404:
            raise HTTPException(status_code=404, detail="context entry not found") from exc
        raise HTTPException(
            status_code=502,
            detail=f"resumeai context lookup failed: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai context lookup failed: {exc}",
        ) from exc


@router.post(
    "/snippet",
    response_model=ContextFile,
    status_code=201,
    summary="Add a free-text snippet to the user-context pool",
)
async def add_snippet(
    client: ContextClientDep,
    name: Annotated[str, Form(..., min_length=1, max_length=200)],
    text: Annotated[str, Form(..., min_length=1)],
    tags: Annotated[str, Form()] = "",
    note: Annotated[str, Form()] = "",
) -> ContextFile:
    """Forward the multipart form to resumeai's snippet endpoint."""
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    payload = SnippetCreate(
        name=name,
        text=text,
        tags=parsed_tags,
        note=note or None,
    )
    try:
        return await client.add_snippet(payload)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai snippet create failed: {exc}",
        ) from exc


@router.post(
    "/file",
    response_model=ContextFile,
    status_code=201,
    summary="Upload a PDF / CSV / markdown / text file to the user-context pool",
)
async def upload_file(
    client: ContextClientDep,
    upload: Annotated[UploadFile, File(...)],
    tags: Annotated[str, Form()] = "",
    note: Annotated[str, Form()] = "",
) -> ContextFile:
    """Forward the upload (filename + content-type + body) to resumeai.

    Single-file only on the wire (matches resumeai's surface). The
    frontend folder/multi-file picker fans out into N requests to this
    endpoint rather than batching them, so each upload either succeeds
    or fails independently and the UI can show per-file progress.
    """
    body = await upload.read()
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    try:
        return await client.upload_file(
            filename=upload.filename or "upload.bin",
            content_type=upload.content_type or "application/octet-stream",
            body=body,
            tags=parsed_tags,
            note=note or None,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai upload failed: {exc}",
        ) from exc


@router.post(
    "/project",
    response_model=ContextFile,
    status_code=201,
    summary="Scan a local git repo and ingest the summary as a context entry",
)
async def scan_project(
    client: ContextClientDep,
    path: Annotated[str, Form(..., min_length=1)],
    name: Annotated[str, Form()] = "",
    author_email: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
    note: Annotated[str, Form()] = "",
) -> ContextFile:
    """Forward the project-scan form to resumeai.

    ``path`` is an absolute host path; resumeai resolves it via its
    own ``/host/personal`` read-only mount. ``author_email`` filters
    the git log to commits by that author (defaults to the whole
    repo summary when blank).
    """
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    payload = ProjectScanCreate(
        path=path,
        name=name or None,
        author_email=author_email or None,
        tags=parsed_tags,
        note=note or None,
    )
    try:
        return await client.scan_project(payload)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai project scan failed: {exc}",
        ) from exc


@router.post(
    "/{file_id}/refresh",
    response_model=ContextFile,
    summary="Re-scan a project-scan entry from its embedded path",
)
async def refresh_project(file_id: str, client: ContextClientDep) -> ContextFile:
    """Refresh a stale project-scan entry.

    Project scans are point-in-time snapshots, so an entry created
    today doesn't reflect commits / coverage stats / file additions
    the user pushed afterwards. This endpoint walks the same path
    again and replaces the stale row.

    Only works on entries whose ``tags`` contain ``source:local_project``;
    a 400 is returned for file uploads and snippets (which have no
    'source' to re-walk).
    """
    try:
        return await client.refresh_project(file_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        if status == 404:
            raise HTTPException(status_code=404, detail="context entry not found") from exc
        raise HTTPException(
            status_code=502,
            detail=f"resumeai refresh failed: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai refresh failed: {exc}",
        ) from exc


@router.delete(
    "/{file_id}",
    status_code=204,
    summary="Remove one context entry",
)
async def delete_context(file_id: str, client: ContextClientDep) -> Response:
    try:
        await client.delete_file(file_id)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        if status == 404:
            raise HTTPException(status_code=404, detail="context entry not found") from exc
        raise HTTPException(
            status_code=502,
            detail=f"resumeai delete failed: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"resumeai delete failed: {exc}",
        ) from exc
    return Response(status_code=204)
