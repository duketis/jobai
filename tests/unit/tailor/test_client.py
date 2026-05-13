"""HTTP-client coverage for jobai.tailor.client.

Uses respx to mock the sibling endpoints so the tests exercise the
real httpx code paths (URL construction, JSON parsing, stream entry)
without touching the wire.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from jobai.tailor.client import (
    HttpxCoverletteraiClient,
    HttpxResumeaiClient,
)
from jobai.tailor.models import CoverletteraiTailorRequest, ResumeaiTailorRequest


async def test_resume_kick_posts_to_api_tailor_and_returns_run_id() -> None:
    client = HttpxResumeaiClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        route = router.post("/api/tailor").mock(
            return_value=httpx.Response(202, json={"run_id": "rs_42", "status": "pending"}),
        )
        run_id = await client.kick(ResumeaiTailorRequest(jd_url="https://example/jd"))
    assert run_id == "rs_42"
    request_payload = route.calls[0].request.read()
    assert b"https://example/jd" in request_payload


async def test_resume_kick_raises_on_5xx() -> None:
    client = HttpxResumeaiClient(base_url="http://resumeai:8765")
    with (
        respx.mock(base_url="http://resumeai:8765") as router,
        pytest.raises(httpx.HTTPStatusError),
    ):
        router.post("/api/tailor").mock(return_value=httpx.Response(500))
        await client.kick(ResumeaiTailorRequest(jd_url="https://example/jd"))


async def test_resume_poll_returns_snapshot() -> None:
    client = HttpxResumeaiClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/api/runs/rs_1").mock(
            return_value=httpx.Response(
                200,
                json={"id": "rs_1", "status": "succeeded", "tailored": {"x": 1}},
            ),
        )
        snap = await client.poll("rs_1")
    assert snap.id == "rs_1"
    assert snap.status == "succeeded"


async def test_resume_stream_pdf_hits_non_api_path() -> None:
    """resumeai's PDF route is at ``/runs/{id}/pdf`` -- not under /api."""
    client = HttpxResumeaiClient(base_url="http://resumeai:8765")
    with respx.mock(base_url="http://resumeai:8765") as router:
        router.get("/runs/rs_1/pdf").mock(
            return_value=httpx.Response(
                200,
                content=b"%PDF-1.5 ok",
                headers={"content-type": "application/pdf"},
            ),
        )
        response = await client.stream_pdf("rs_1")
        assert response.status_code == 200
        # Drain so respx considers the call complete.
        body = b"".join([chunk async for chunk in response.aiter_bytes()])
        await response.aclose()
    assert body == b"%PDF-1.5 ok"


async def test_letter_kick_includes_resume_run_id() -> None:
    client = HttpxCoverletteraiClient(base_url="http://coverletterai:8766")
    with respx.mock(base_url="http://coverletterai:8766") as router:
        route = router.post("/api/tailor").mock(
            return_value=httpx.Response(202, json={"run_id": "ls_7"}),
        )
        run_id = await client.kick(
            CoverletteraiTailorRequest(
                jd_url="https://example/jd",
                resume_run_id="rs_42",
            ),
        )
        assert run_id == "ls_7"
        body = route.calls[0].request.read()
    assert b"resume_run_id" in body


async def test_letter_poll_returns_snapshot() -> None:
    client = HttpxCoverletteraiClient(base_url="http://coverletterai:8766")
    with respx.mock(base_url="http://coverletterai:8766") as router:
        router.get("/api/runs/ls_1").mock(
            return_value=httpx.Response(200, json={"id": "ls_1", "status": "tailoring"}),
        )
        snap = await client.poll("ls_1")
    assert snap.status == "tailoring"


async def test_letter_stream_pdf_hits_api_path() -> None:
    """coverletterai's PDF route IS under /api -- different from resumeai."""
    client = HttpxCoverletteraiClient(base_url="http://coverletterai:8766")
    with respx.mock(base_url="http://coverletterai:8766") as router:
        router.get("/api/runs/ls_1/pdf").mock(
            return_value=httpx.Response(
                200,
                content=b"%PDF-1.5 letter",
                headers={"content-type": "application/pdf"},
            ),
        )
        response = await client.stream_pdf("ls_1")
        body = b"".join([chunk async for chunk in response.aiter_bytes()])
        await response.aclose()
    assert body == b"%PDF-1.5 letter"


def test_client_strips_trailing_slash_in_base_url() -> None:
    """``http://x:8765/`` and ``http://x:8765`` should produce identical requests."""
    assert HttpxResumeaiClient("http://x:8765/")._base_url == "http://x:8765"
    assert HttpxCoverletteraiClient("http://x:8766/")._base_url == "http://x:8766"
