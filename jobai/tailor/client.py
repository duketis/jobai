"""HTTP clients for the resumeai + coverletterai sibling services.

The orchestrator depends on a thin :class:`Protocol` per sibling rather
than on the concrete httpx-backed implementation. Tests inject a fake
that returns scripted responses; production wires the httpx classes via
the FastAPI dependency injection layer.

Each sibling exposes a *kick-off* endpoint that returns a run id and a
*poll* endpoint that returns the run record. The PDF download endpoints
diverge between the two siblings (resumeai uses ``/runs/{id}/pdf``;
coverletterai uses ``/api/runs/{id}/pdf``) — that shape is encoded here
so the rest of the package never has to think about it.
"""

from __future__ import annotations

from typing import Any, Final, Protocol

import httpx

from jobai.tailor.models import (
    CoverletteraiTailorRequest,
    ResumeaiTailorRequest,
    SiblingRunSnapshot,
)

#: Default service URLs used inside the ``ai-tailor-network`` docker network.
#: From the host, both are also reachable on ``localhost`` at the same ports —
#: tests rely on that for live captures; production talks via the docker name.
DEFAULT_RESUMEAI_URL: Final[str] = "http://resumeai:8765"
DEFAULT_COVERLETTERAI_URL: Final[str] = "http://coverletterai:8766"

#: Per-request timeout in seconds for the sibling-service HTTP calls.
#: Generous because tailoring is LLM-bound (~60-180s for a resume); the
#: kick-off itself is fast but we use the same client for poll + kick.
_DEFAULT_TIMEOUT: Final[float] = 30.0


class ResumeaiClient(Protocol):
    """Wire surface for the resumeai sibling.

    The orchestrator and the PDF-proxy route both depend on this
    Protocol so tests inject fakes without touching httpx.
    """

    async def kick(self, request: ResumeaiTailorRequest) -> str:
        """POST ``/api/tailor`` and return the ``run_id``."""
        ...

    async def poll(self, run_id: str) -> SiblingRunSnapshot:
        """GET ``/api/runs/{run_id}`` and return ``id`` + ``status``."""
        ...

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """GET ``/api/runs/{run_id}`` and return the full record dict.

        Used by the QA step to pull the structured ``requirements``
        + ``tailored`` JSON the LLM needs to grade the application.
        """
        ...

    async def stream_pdf(self, run_id: str) -> httpx.Response:
        """GET the PDF endpoint and return the raw streaming response.

        The caller is responsible for ``aclose()`` on the returned
        response — the route handler hands it to FastAPI's StreamingResponse
        which closes it when the client disconnects.
        """
        ...


class CoverletteraiClient(Protocol):
    """Wire surface for the coverletterai sibling.

    Identical method names to :class:`ResumeaiClient` but the kick
    request carries the resume run id, and the PDF endpoint is at
    a different path.
    """

    async def kick(self, request: CoverletteraiTailorRequest) -> str:
        """POST ``/api/tailor`` and return the ``run_id``."""
        ...

    async def poll(self, run_id: str) -> SiblingRunSnapshot:
        """GET ``/api/runs/{run_id}`` and return ``id`` + ``status``."""
        ...

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """GET ``/api/runs/{run_id}`` and return the full record dict."""
        ...

    async def stream_pdf(self, run_id: str) -> httpx.Response:
        """GET the PDF endpoint and return the raw streaming response."""
        ...


class HttpxResumeaiClient:
    """httpx-backed :class:`ResumeaiClient` for production."""

    def __init__(self, base_url: str = DEFAULT_RESUMEAI_URL) -> None:
        self._base_url = base_url.rstrip("/")

    async def kick(self, request: ResumeaiTailorRequest) -> str:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self._base_url}/api/tailor",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["run_id"])

    async def poll(self, run_id: str) -> SiblingRunSnapshot:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.get(f"{self._base_url}/api/runs/{run_id}")
            response.raise_for_status()
            return SiblingRunSnapshot.model_validate(response.json())

    async def get_run(self, run_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.get(f"{self._base_url}/api/runs/{run_id}")
            response.raise_for_status()
            payload = response.json()
            return dict(payload) if isinstance(payload, dict) else {}

    async def stream_pdf(self, run_id: str) -> httpx.Response:
        # resumeai's PDF route lives at /runs/{id}/pdf, NOT under /api.
        # The /api/runs/{id}/pdf path returns 404 -- see live capture
        # in 2026-05-13 integration session.
        client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        request = client.build_request("GET", f"{self._base_url}/runs/{run_id}/pdf")
        return await client.send(request, stream=True)


class HttpxCoverletteraiClient:
    """httpx-backed :class:`CoverletteraiClient` for production."""

    def __init__(self, base_url: str = DEFAULT_COVERLETTERAI_URL) -> None:
        self._base_url = base_url.rstrip("/")

    async def kick(self, request: CoverletteraiTailorRequest) -> str:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self._base_url}/api/tailor",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["run_id"])

    async def poll(self, run_id: str) -> SiblingRunSnapshot:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.get(f"{self._base_url}/api/runs/{run_id}")
            response.raise_for_status()
            return SiblingRunSnapshot.model_validate(response.json())

    async def get_run(self, run_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.get(f"{self._base_url}/api/runs/{run_id}")
            response.raise_for_status()
            payload = response.json()
            return dict(payload) if isinstance(payload, dict) else {}

    async def stream_pdf(self, run_id: str) -> httpx.Response:
        # coverletterai's PDF route IS under /api (different from resumeai).
        client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        request = client.build_request("GET", f"{self._base_url}/api/runs/{run_id}/pdf")
        return await client.send(request, stream=True)
