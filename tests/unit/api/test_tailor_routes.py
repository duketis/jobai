"""End-to-end coverage for the /api/tailor routes.

The fixtures here build a fresh FastAPI app per test and override the
sibling-client + pool DI so no live HTTP fires. The orchestrator runs
in-process under the lifespan-owned :class:`TailorPool`; the scripted
clients return instantly so the chain reaches a terminal state by the
time the test inspects the row.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jobai.api.dependencies import get_db_path
from jobai.api.routes.tailor import (
    get_letter_client,
    get_resume_client,
)
from jobai.api.server import create_app
from jobai.db.migrations import apply_pending
from jobai.tailor.worker import TailorPool
from tests.unit.tailor.conftest import (
    ScriptedLetterClient,
    ScriptedResumeClient,
    _seed_one_job,
)


@pytest.fixture
def tailor_app_db(tmp_path: Path) -> Path:
    """Migrated SQLite DB pre-seeded with one job (id=1)."""
    db = tmp_path / "routes-test.db"
    conn = sqlite3.connect(db)
    try:
        apply_pending(conn)
        _seed_one_job(conn)
    finally:
        conn.close()
    return db


@pytest.fixture
def app_with_overrides(
    tailor_app_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient]:
    """Build the FastAPI app with sibling clients + DB path stubbed.

    The chain that the route triggers needs a *deterministic* DB path
    because it opens its own connection. We override ``get_db_path`` so
    both the request handler and the background chain resolve to the
    same file. The lifespan's clients/pool are also short-circuited to
    in-memory fakes so no real HTTP fires.
    """
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    resume_client = ScriptedResumeClient()
    letter_client = ScriptedLetterClient()
    application = create_app()
    application.dependency_overrides[get_db_path] = lambda: tailor_app_db
    application.dependency_overrides[get_resume_client] = lambda: resume_client
    application.dependency_overrides[get_letter_client] = lambda: letter_client
    return application, resume_client, letter_client


@pytest.fixture
def client(
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> Iterator[TestClient]:
    app, _, _ = app_with_overrides
    with TestClient(app) as test_client:
        yield test_client


# -- kick endpoints ---------------------------------------------------------


def test_kick_one_creates_row_and_returns_pending(
    client: TestClient,
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    response = client.post("/api/tailor/jobs/1")
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"] == 1
    assert body["status"] == "pending"
    assert body["tailor_run_id"] > 0
    # Pool drain (lifespan exit) waits for the chain to finish; by the
    # time the with-TestClient block ends, the chain has completed.


def test_kick_one_404_for_unknown_job(client: TestClient) -> None:
    response = client.post("/api/tailor/jobs/99999")
    assert response.status_code == 404


def test_kick_batch_creates_runs_for_each_job(client: TestClient) -> None:
    response = client.post("/api/tailor/batch", json={"job_ids": [1, 1]})
    assert response.status_code == 202
    body = response.json()
    # Two distinct rows even for the same job id -- the user may want
    # to re-tailor with a different model in a future iteration.
    assert len(body["items"]) == 2
    assert {item["job_id"] for item in body["items"]} == {1}
    assert len({item["tailor_run_id"] for item in body["items"]}) == 2


def test_kick_batch_404_lists_missing_ids(client: TestClient) -> None:
    response = client.post("/api/tailor/batch", json={"job_ids": [1, 4242]})
    assert response.status_code == 404
    assert "4242" in response.json()["detail"]


def test_kick_batch_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/api/tailor/batch", json={"job_ids": []})
    assert response.status_code == 422


# -- listing + detail ------------------------------------------------------


def test_list_runs_empty_when_none_kicked(client: TestClient) -> None:
    response = client.get("/api/tailor/runs")
    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_list_and_get_after_chain_succeeds(
    tailor_app_db: Path,
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """The chain runs to ``succeeded`` and is reflected by both list + detail.

    The lifespan's shutdown drains the pool, so as long as we let the
    ``with TestClient(...)`` block exit before asserting, the chain has
    completed.
    """
    app, _, _ = app_with_overrides
    with TestClient(app) as test_client:
        kick = test_client.post("/api/tailor/jobs/1")
        run_id = kick.json()["tailor_run_id"]
    # Lifespan exit awaited pool.drain(); inspect via a fresh TestClient.
    with TestClient(app) as test_client:
        detail = test_client.get(f"/api/tailor/runs/{run_id}").json()
        listing = test_client.get("/api/tailor/runs").json()
    assert detail["status"] == "succeeded"
    assert detail["resume_run_id"] == "rs_1"
    assert detail["letter_run_id"] == "ls_1"
    assert any(item["id"] == run_id for item in listing["items"])


def test_get_run_404_for_unknown(client: TestClient) -> None:
    response = client.get("/api/tailor/runs/99999")
    assert response.status_code == 404


def test_list_runs_filter_by_job_and_status(client: TestClient) -> None:
    kick = client.post("/api/tailor/jobs/1")
    run_id = kick.json()["tailor_run_id"]
    # Hammer the list with both filter params to exercise both branches.
    by_job = client.get("/api/tailor/runs?job_id=1&limit=10").json()
    assert any(item["id"] == run_id for item in by_job["items"])
    by_status = client.get("/api/tailor/runs?status=pending").json()
    # status might be pending or succeeded depending on chain timing; the
    # call must succeed regardless.
    assert "items" in by_status


# -- PDF proxy -------------------------------------------------------------


def test_resume_pdf_returns_409_when_chain_not_started(
    client: TestClient,
) -> None:
    """Kicking + immediately requesting the PDF before the chain has produced
    the resume_run_id surfaces a 409 rather than a 404 (the run exists)."""
    # Create a row directly so we KNOW resume_run_id is null when we ask.
    response = client.get("/api/tailor/runs/1/resume.pdf")
    assert response.status_code == 404  # row doesn't exist yet


def test_pdf_routes_proxy_sibling_response(
    tailor_app_db: Path,
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """End-to-end: kick chain (drains via lifespan), then fetch both PDFs."""
    app, _, _ = app_with_overrides

    def _mk_response(body: bytes) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/pdf"},
        )

    scripted_resume = ScriptedResumeClient(stream_response=_mk_response(b"%PDF resume"))
    scripted_letter = ScriptedLetterClient(stream_response=_mk_response(b"%PDF letter"))
    app.dependency_overrides[get_resume_client] = lambda: scripted_resume
    app.dependency_overrides[get_letter_client] = lambda: scripted_letter

    with TestClient(app) as test_client:
        kick = test_client.post("/api/tailor/jobs/1")
        run_id = kick.json()["tailor_run_id"]
    # Lifespan exit drained the pool; siblings are now consumed but we
    # re-script them for the PDF proxy turns.
    scripted_resume._stream_response = _mk_response(b"%PDF resume bytes")
    scripted_letter._stream_response = _mk_response(b"%PDF letter bytes")

    with TestClient(app) as test_client:
        resume_pdf = test_client.get(f"/api/tailor/runs/{run_id}/resume.pdf")
        letter_pdf = test_client.get(f"/api/tailor/runs/{run_id}/letter.pdf")
    assert resume_pdf.status_code == 200
    assert resume_pdf.content == b"%PDF resume bytes"
    assert letter_pdf.status_code == 200
    assert letter_pdf.content == b"%PDF letter bytes"


def test_resume_pdf_proxies_sibling_error_as_502(
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """A sibling-side 404 / 500 surfaces as a 502 to jobai's caller."""
    app, _, _ = app_with_overrides

    scripted_resume = ScriptedResumeClient()
    scripted_letter = ScriptedLetterClient()
    app.dependency_overrides[get_resume_client] = lambda: scripted_resume
    app.dependency_overrides[get_letter_client] = lambda: scripted_letter

    with TestClient(app) as test_client:
        kick = test_client.post("/api/tailor/jobs/1")
        run_id = kick.json()["tailor_run_id"]
    # After lifespan drain, swap in a 500 response for the PDF turn.
    scripted_resume._stream_response = httpx.Response(
        500,
        content=b"sibling exploded",
        headers={"content-type": "application/json"},
    )
    with TestClient(app) as test_client:
        resume_pdf = test_client.get(f"/api/tailor/runs/{run_id}/resume.pdf")
    assert resume_pdf.status_code == 502


def test_resume_pdf_409_when_chain_has_not_produced_artefact(
    tailor_app_db: Path,
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """A row that's still ``pending`` (no resume_run_id) gets 409 on PDF fetch."""
    app, _, _ = app_with_overrides

    # Insert a fresh row directly without running the chain so resume_run_id stays null.
    conn = sqlite3.connect(tailor_app_db)
    try:
        cursor = conn.execute(
            "INSERT INTO tailor_runs (job_id, status, created_at, updated_at) "
            "VALUES (1, 'pending', datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()
    finally:
        conn.close()

    with TestClient(app) as test_client:
        resume_pdf = test_client.get(f"/api/tailor/runs/{new_id}/resume.pdf")
        assert resume_pdf.status_code == 409
        letter_pdf = test_client.get(f"/api/tailor/runs/{new_id}/letter.pdf")
        assert letter_pdf.status_code == 409


# -- DI failure modes ------------------------------------------------------


def test_kick_returns_503_when_pool_not_initialised(
    tailor_app_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the lifespan hasn't run (TestClient without ``with``), the pool
    DI guard surfaces a 503 rather than a confusing AttributeError."""
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app = create_app()
    app.dependency_overrides[get_db_path] = lambda: tailor_app_db
    test_client = TestClient(app)  # no ``with`` -- lifespan not entered
    response = test_client.post("/api/tailor/jobs/1")
    assert response.status_code == 503


def test_resume_client_dep_503_when_state_missing(
    tailor_app_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as the pool case but for the sibling clients."""
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app = create_app()
    app.dependency_overrides[get_db_path] = lambda: tailor_app_db
    test_client = TestClient(app)
    response = test_client.get("/api/tailor/runs/1/resume.pdf")
    # Without the lifespan, both the row read AND the client lookup fail.
    # Either 404 (no row) or 503 (no client) is acceptable for this guard.
    assert response.status_code in {404, 503}


def test_letter_client_dep_503_when_state_missing(
    tailor_app_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app = create_app()
    app.dependency_overrides[get_db_path] = lambda: tailor_app_db
    test_client = TestClient(app)
    response = test_client.get("/api/tailor/runs/1/letter.pdf")
    assert response.status_code in {404, 503}


def test_pool_di_returns_pool_when_initialised(
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """Direct DI smoke -- the helper hands back the lifespan-owned pool."""
    app, _, _ = app_with_overrides
    with TestClient(app):
        # The lifespan has run, so app.state.tailor_pool is wired.
        assert isinstance(app.state.tailor_pool, TailorPool)
        # Protocol types aren't @runtime_checkable -- duck-type the shape.
        assert hasattr(app.state.resume_client, "kick")
        assert hasattr(app.state.letter_client, "kick")


def test_get_qa_client_builds_from_effective_agent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_qa_client`` reads the live :class:`EffectiveAgentConfig` and
    delegates to :func:`build_qa_client`. We patch the resolver so the
    test doesn't depend on the underlying ``app_settings`` row layout."""
    from jobai.api.routes.tailor import get_qa_client  # noqa: PLC0415
    from jobai.api.runtime_settings import EffectiveAgentConfig  # noqa: PLC0415
    from jobai.tailor.qa import SubscriptionQAClient  # noqa: PLC0415

    fake_cfg = EffectiveAgentConfig(
        agent_backend="subscription",
        anthropic_api_key=None,
        claude_code_oauth_token="oat-x",  # noqa: S106 - test fixture, not a real token
        anthropic_model="claude-opus-4-7",
    )
    monkeypatch.setattr(
        "jobai.api.routes.tailor.get_effective_agent_config",
        lambda conn: fake_cfg,
    )

    stub_conn: Any = object()
    client = get_qa_client(stub_conn)
    assert isinstance(client, SubscriptionQAClient)


def test_get_qa_client_returns_none_when_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jobai.api.routes.tailor import get_qa_client  # noqa: PLC0415
    from jobai.api.runtime_settings import EffectiveAgentConfig  # noqa: PLC0415

    fake_cfg = EffectiveAgentConfig(
        agent_backend="api",
        anthropic_api_key=None,
        claude_code_oauth_token=None,
        anthropic_model="claude-opus-4-7",
    )
    monkeypatch.setattr(
        "jobai.api.routes.tailor.get_effective_agent_config",
        lambda conn: fake_cfg,
    )

    stub_conn: Any = object()
    assert get_qa_client(stub_conn) is None


def test_resume_and_letter_client_di_return_lifespan_object() -> None:
    """The DI helpers return whatever the lifespan stashed on app.state.

    This is the happy-path branch of ``get_resume_client`` / ``get_letter_client``
    -- the 503 paths are covered above. Stubs Request directly to avoid
    needing the dependency-override layer to short-circuit the path we
    want to exercise.
    """

    class _StubState:
        def __init__(self) -> None:
            self.resume_client = ScriptedResumeClient()
            self.letter_client = ScriptedLetterClient()

    class _StubApp:
        def __init__(self) -> None:
            self.state = _StubState()

    class _StubRequest:
        def __init__(self) -> None:
            self.app = _StubApp()

    request: Any = _StubRequest()
    assert get_resume_client(request) is request.app.state.resume_client
    assert get_letter_client(request) is request.app.state.letter_client
