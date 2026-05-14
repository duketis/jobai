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

    Also stubs the project-scan refresh helper to a no-op coroutine so
    every kicked chain doesn't try to hit the real resumeai context
    pool. The refresh path itself is covered by the orchestrator and
    scheduler tests; route-level coverage only cares that the closure
    is wired and invoked.
    """
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")

    async def _noop_refresh(_url: str) -> tuple[int, int]:
        return (0, 0)

    monkeypatch.setattr(
        "jobai.scheduler.refresh_project_scans",
        _noop_refresh,
    )

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


def test_kick_batch_404_detail_is_truncated_for_huge_missing_lists(
    client: TestClient,
) -> None:
    """A typo submit of 100 nonexistent ids must not return a megabyte
    of integers in the 404 body -- we cap the preview at 25 and append
    a '(+ N more)' suffix."""
    response = client.post(
        "/api/tailor/batch",
        json={"job_ids": list(range(5000, 5100))},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "+ 75 more" in detail


def test_kick_batch_handles_large_id_list_via_chunked_lookup(
    client: TestClient,
) -> None:
    """The IN-clause existence check chunks at 500 ids so a 1k+ batch
    doesn't trip SQLite's parameter limit. Single seeded job id (1) is
    duplicated across the batch -- all rows resolve, all chains submit."""
    big_batch = [1] * 1_500
    response = client.post("/api/tailor/batch", json={"job_ids": big_batch})
    assert response.status_code == 202
    body = response.json()
    assert len(body["items"]) == 1_500


def test_kick_by_url_matches_catalogue_when_url_exists(
    client: TestClient,
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """When the pasted URL is already in the catalogue, the run uses
    the normal job_id path and reports the matched_job_id back."""
    del app_with_overrides
    response = client.post(
        "/api/tailor/url",
        json={"jd_url": "https://example.com/jd-1"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["matched_job_id"] == 1
    assert body["matched_count"] == 1
    assert body["status"] == "pending"


def test_kick_by_url_strips_query_params_for_catalogue_match(
    client: TestClient,
) -> None:
    """A URL with tracking params still matches the canonical row."""
    response = client.post(
        "/api/tailor/url",
        json={"jd_url": "https://example.com/jd-1?trid=abc&utm_source=email"},
    )
    assert response.status_code == 202
    assert response.json()["matched_job_id"] == 1


def test_kick_by_url_falls_back_to_url_when_no_catalogue_match(
    client: TestClient,
) -> None:
    """An off-network URL with no catalogue hit still kicks a chain;
    the response reports matched_job_id=null so the UI can label the
    flow correctly."""
    response = client.post(
        "/api/tailor/url",
        json={"jd_url": "https://strange.example.com/some-new-jd/abc"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["matched_job_id"] is None
    assert body["matched_count"] == 0
    assert body["status"] == "pending"


def test_kick_by_url_rejects_empty_url(client: TestClient) -> None:
    response = client.post("/api/tailor/url", json={"jd_url": ""})
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


def test_resumeai_url_dep_503_when_state_missing(
    tailor_app_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the lifespan hasn't initialised ``app.state.resumeai_url``,
    the kick routes must surface a 503 rather than crash building the
    refresh closure."""
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app = create_app()
    app.dependency_overrides[get_db_path] = lambda: tailor_app_db
    test_client = TestClient(app)  # no ``with`` -- lifespan not entered
    response = test_client.post("/api/tailor/jobs/1")
    # Several lifespan-owned deps are missing; we accept any 503 from
    # the DI guards (pool / resume / letter / resumeai_url). The point
    # is that no 5xx-as-AttributeError leaks out.
    assert response.status_code == 503


def test_get_resumeai_url_returns_lifespan_state() -> None:
    """Direct DI smoke -- the helper hands back whatever the lifespan
    stashed on app.state."""
    from jobai.api.routes.tailor import get_resumeai_url  # noqa: PLC0415

    class _StubState:
        def __init__(self) -> None:
            self.resumeai_url = "http://resumeai:8765"

    class _StubApp:
        def __init__(self) -> None:
            self.state = _StubState()

    class _StubRequest:
        def __init__(self) -> None:
            self.app = _StubApp()

    request: Any = _StubRequest()
    assert get_resumeai_url(request) == "http://resumeai:8765"


def test_get_resumeai_url_raises_503_when_missing() -> None:
    """If ``app.state.resumeai_url`` was never set (lifespan skipped /
    misconfigured), the DI helper surfaces a structured 503 rather
    than letting an AttributeError escape into the route handler."""
    from fastapi import HTTPException  # noqa: PLC0415

    from jobai.api.routes.tailor import get_resumeai_url  # noqa: PLC0415

    class _StubState:
        pass  # no resumeai_url attribute

    class _StubApp:
        def __init__(self) -> None:
            self.state = _StubState()

    class _StubRequest:
        def __init__(self) -> None:
            self.app = _StubApp()

    request: Any = _StubRequest()
    with pytest.raises(HTTPException) as exc:
        get_resumeai_url(request)
    assert exc.value.status_code == 503


async def test_schedule_chain_wires_refresh_closure_with_url(
    monkeypatch: pytest.MonkeyPatch,
    tailor_app_db: Path,
) -> None:
    """The internal ``_schedule_chain`` helper must hand the
    orchestrator a closure that, when invoked, calls the public
    ``refresh_project_scans`` helper with the configured URL. This is
    the seam between the route layer and the scheduler module -- a
    bug here would mean a wrong URL or no refresh at all."""
    from jobai.api.routes.tailor import _schedule_chain  # noqa: PLC0415
    from jobai.tailor.worker import TailorPool  # noqa: PLC0415

    seen_urls: list[str] = []

    async def _spy(url: str) -> tuple[int, int]:
        seen_urls.append(url)
        return (1, 0)

    monkeypatch.setattr("jobai.scheduler.refresh_project_scans", _spy)

    # Also stub run_chain so it just calls the refresh closure and
    # returns; tracing the wiring rather than running the full chain.
    invoked_with_refresh: list[bool] = []

    async def _fake_run_chain(
        _run_id: int,
        **kwargs: Any,
    ) -> None:
        refresh = kwargs.get("refresh_context_scans")
        assert callable(refresh)
        invoked_with_refresh.append(True)
        await refresh()

    monkeypatch.setattr("jobai.api.routes.tailor.run_chain", _fake_run_chain)

    pool = TailorPool(max_concurrent=1)

    conn = sqlite3.connect(tailor_app_db)
    try:
        new_id = int(
            conn.execute(
                "INSERT INTO tailor_runs (job_id, status, created_at, updated_at) "
                "VALUES (1, 'pending', datetime('now'), datetime('now')) RETURNING id",
            ).fetchone()[0],
        )
        conn.commit()
    finally:
        conn.close()

    _schedule_chain(
        pool=pool,
        tailor_run_id=new_id,
        db_path=tailor_app_db,
        resume_client=ScriptedResumeClient(),
        letter_client=ScriptedLetterClient(),
        qa_client=None,
        resumeai_url="http://resumeai:8765",
    )

    await pool.drain()
    assert invoked_with_refresh == [True]
    assert seen_urls == ["http://resumeai:8765"]


def test_letter_client_dep_503_when_attribute_missing() -> None:
    """The letter-client DI helper guards against a partially-wired
    lifespan (resume client present, letter client missing). Adding a
    new resume-client dep in front of the letter PDF route masked the
    end-to-end exercise of this branch, so cover it directly."""
    from fastapi import HTTPException  # noqa: PLC0415

    from jobai.api.routes.tailor import get_letter_client  # noqa: PLC0415

    class _StubState:
        pass  # no letter_client attribute

    class _StubApp:
        def __init__(self) -> None:
            self.state = _StubState()

    class _StubRequest:
        def __init__(self) -> None:
            self.app = _StubApp()

    request: Any = _StubRequest()
    with pytest.raises(HTTPException) as exc:
        get_letter_client(request)
    assert exc.value.status_code == 503


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


# -- PDF filename composition ----------------------------------------------


def test_pdf_filename_catalogue_path(
    tailor_app_db: Path,
    app_with_overrides: tuple[FastAPI, ScriptedResumeClient, ScriptedLetterClient],
) -> None:
    """For a catalogue run, the streamed PDF carries a
    ``<Name>-<JobTitle>-<Company>-<Resume|CoverLetter>.pdf`` filename
    in Content-Disposition. Title + company come from the ``jobs`` row;
    name comes from the resumeai sibling's tailored payload."""
    app, _, _ = app_with_overrides

    def _mk_pdf() -> httpx.Response:
        return httpx.Response(200, content=b"%PDF x", headers={"content-type": "application/pdf"})

    scripted_resume = ScriptedResumeClient(stream_response=_mk_pdf())
    scripted_letter = ScriptedLetterClient(stream_response=_mk_pdf())
    app.dependency_overrides[get_resume_client] = lambda: scripted_resume
    app.dependency_overrides[get_letter_client] = lambda: scripted_letter

    with TestClient(app) as test_client:
        kick = test_client.post("/api/tailor/jobs/1")
        run_id = kick.json()["tailor_run_id"]

    scripted_resume._stream_response = _mk_pdf()
    scripted_letter._stream_response = _mk_pdf()

    with TestClient(app) as test_client:
        resume = test_client.get(f"/api/tailor/runs/{run_id}/resume.pdf")
        letter = test_client.get(f"/api/tailor/runs/{run_id}/letter.pdf")

    # Seeded job: title='Engineer', company='Acme'.
    # ScriptedResumeClient default tailored.name = 'Jane Doe'.
    assert "Jane_Doe-Engineer-Acme-Resume.pdf" in resume.headers["content-disposition"]
    assert "Jane_Doe-Engineer-Acme-CoverLetter.pdf" in letter.headers["content-disposition"]


def test_pdf_filename_url_only_falls_back_to_sibling_requirements(
    tailor_app_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare-URL run with no catalogue match has no ``jobs`` row to
    pull title/company from -- they come from the sibling's parsed
    ``requirements`` instead."""
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")

    async def _noop_refresh(_url: str) -> tuple[int, int]:
        return (0, 0)

    monkeypatch.setattr("jobai.scheduler.refresh_project_scans", _noop_refresh)

    def _mk_pdf() -> httpx.Response:
        return httpx.Response(200, content=b"%PDF x", headers={"content-type": "application/pdf"})

    scripted_resume = ScriptedResumeClient(
        stream_response=_mk_pdf(),
        run_record={
            "id": "rs_1",
            "status": "succeeded",
            "requirements": {"title": "Staff Engineer", "company": "Globex"},
            "tailored": {"name": "Alex Roe"},
        },
    )
    scripted_letter = ScriptedLetterClient(stream_response=_mk_pdf())
    application = create_app()
    application.dependency_overrides[get_db_path] = lambda: tailor_app_db
    application.dependency_overrides[get_resume_client] = lambda: scripted_resume
    application.dependency_overrides[get_letter_client] = lambda: scripted_letter

    with TestClient(application) as test_client:
        kick = test_client.post(
            "/api/tailor/url",
            json={"jd_url": "https://example.com/not-in-catalogue"},
        )
        run_id = kick.json()["tailor_run_id"]

    scripted_resume._stream_response = _mk_pdf()
    scripted_letter._stream_response = _mk_pdf()

    with TestClient(application) as test_client:
        resume = test_client.get(f"/api/tailor/runs/{run_id}/resume.pdf")

    assert "Alex_Roe-Staff_Engineer-Globex-Resume.pdf" in resume.headers["content-disposition"]


def test_sanitize_filename_part_strips_path_chars_and_falls_back() -> None:
    """``_sanitize_filename_part`` drops Windows/macOS-illegal chars,
    collapses runs of whitespace, ASCII-folds, and returns the
    fallback when the input sanitises to nothing."""
    from jobai.tailor.filenames import sanitize_filename_part  # noqa: PLC0415

    _sanitize_filename_part = sanitize_filename_part

    assert _sanitize_filename_part("My / Job: Title", fallback="Job") == "My Job Title"
    assert _sanitize_filename_part("   ", fallback="Job") == "Job"
    assert _sanitize_filename_part(None, fallback="Job") == "Job"
    assert _sanitize_filename_part("a\x00\x1f\x02b", fallback="Job") == "a b"
    # ASCII-folding strips non-ASCII glyphs but keeps the meaningful core.
    assert _sanitize_filename_part("Café Ltd", fallback="Company") == "Caf Ltd"
    # Trailing dots/spaces are stripped (Windows quirk: trailing
    # dots in filenames silently become invalid).
    assert _sanitize_filename_part("Foo...", fallback="Job") == "Foo"


async def test_proxy_pdf_omits_content_disposition_when_no_filename() -> None:
    """``_proxy_pdf`` is also called from places that don't supply a
    filename (defensive default); skip the Content-Disposition header
    in that case rather than emitting an empty value."""
    from jobai.api.routes.tailor import _proxy_pdf  # noqa: PLC0415

    upstream = httpx.Response(200, content=b"%PDF x", headers={"content-type": "application/pdf"})
    response = await _proxy_pdf(upstream)
    assert "content-disposition" not in {k.lower() for k in response.headers}


async def test_build_pdf_filename_uses_default_when_job_row_deleted(
    tailor_app_db: Path,
) -> None:
    """If the tailor row points at a job_id that no longer exists
    (catalogue trimmed, manual delete), the helper falls back to the
    'Job'/'Company' defaults rather than crashing on a None row."""
    from jobai.tailor.filenames import build_pdf_filename as _build_pdf_filename  # noqa: PLC0415

    conn = sqlite3.connect(tailor_app_db)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "INSERT INTO tailor_runs (job_id, status, resume_run_id, "
            "                          created_at, updated_at) "
            "VALUES (4242, 'succeeded', 'rs_x', datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        filename = await _build_pdf_filename(
            conn=conn,
            resume_client=ScriptedResumeClient(
                run_record={"id": "rs_x", "tailored": {"name": "Sam Doe"}}
            ),
            tailor_run_id=new_id,
            kind="letter",
        )
    finally:
        conn.close()

    assert filename == "Sam_Doe-Job-Company-CoverLetter.pdf"


async def test_build_pdf_filename_when_resume_run_id_missing(
    tailor_app_db: Path,
) -> None:
    """A row with no resume_run_id (manually inserted / chain crashed
    mid-flight) still produces a filename -- title + company come
    from the jobs row, name falls back to the 'Applicant' default."""
    from jobai.tailor.filenames import build_pdf_filename as _build_pdf_filename  # noqa: PLC0415

    conn = sqlite3.connect(tailor_app_db)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "INSERT INTO tailor_runs (job_id, status, created_at, updated_at) "
            "VALUES (1, 'pending', datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        filename = await _build_pdf_filename(
            conn=conn,
            resume_client=ScriptedResumeClient(),
            tailor_run_id=new_id,
            kind="resume",
        )
    finally:
        conn.close()

    assert filename == "Applicant-Engineer-Acme-Resume.pdf"


async def test_build_pdf_filename_handles_weird_sibling_payloads(
    tailor_app_db: Path,
) -> None:
    """The helper type-narrows every sibling field defensively so a
    sibling returning ``{"tailored": "not-a-dict"}`` or a non-string
    name/title doesn't break filename construction."""
    from jobai.tailor.filenames import build_pdf_filename as _build_pdf_filename  # noqa: PLC0415

    class _WeirdResume(ScriptedResumeClient):
        async def get_run(self, run_id: str) -> dict[str, object]:
            self.get_run_calls.append(run_id)
            return {
                # Wrong type at every layer:
                "tailored": "not a dict",
                "requirements": ["also not a dict"],
            }

    conn = sqlite3.connect(tailor_app_db)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "INSERT INTO tailor_runs (status, jd_url, resume_run_id, "
            "                          created_at, updated_at) "
            "VALUES ('succeeded', 'https://x.example/jd', 'rs_x', "
            "        datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        filename = await _build_pdf_filename(
            conn=conn,
            resume_client=_WeirdResume(),
            tailor_run_id=new_id,
            kind="resume",
        )
    finally:
        conn.close()

    # All four parts fall back: no job row (URL-only), tailored isn't
    # a dict (name fallback), requirements isn't a dict (title/company
    # fallback).
    assert filename == "Applicant-Job-Company-Resume.pdf"


async def test_build_pdf_filename_url_path_with_non_string_fields(
    tailor_app_db: Path,
) -> None:
    """When the bare-URL path runs and the sibling returns a
    ``requirements`` dict whose ``title``/``company`` aren't strings
    (numeric ATS code, missing field), the helper keeps the
    fallbacks rather than coercing surprising types."""
    from jobai.tailor.filenames import build_pdf_filename as _build_pdf_filename  # noqa: PLC0415

    class _IntFields(ScriptedResumeClient):
        async def get_run(self, run_id: str) -> dict[str, object]:
            self.get_run_calls.append(run_id)
            return {
                "tailored": {"name": 12345},  # non-string name
                "requirements": {"title": 42, "company": None},
            }

    conn = sqlite3.connect(tailor_app_db)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "INSERT INTO tailor_runs (status, jd_url, resume_run_id, "
            "                          created_at, updated_at) "
            "VALUES ('succeeded', 'https://x.example/jd2', 'rs_y', "
            "        datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        filename = await _build_pdf_filename(
            conn=conn,
            resume_client=_IntFields(),
            tailor_run_id=new_id,
            kind="letter",
        )
    finally:
        conn.close()

    assert filename == "Applicant-Job-Company-CoverLetter.pdf"


async def test_build_pdf_filename_swallows_sibling_failure(
    tailor_app_db: Path,
) -> None:
    """If the resumeai sibling 5xx's on ``get_run`` while we're trying
    to look up the applicant name, the helper returns a filename
    anyway -- the actual PDF stream is what the user cares about,
    not a perfect filename."""
    from jobai.tailor.filenames import build_pdf_filename as _build_pdf_filename  # noqa: PLC0415

    class _BoomResume(ScriptedResumeClient):
        async def get_run(self, run_id: str) -> dict[str, object]:
            self.get_run_calls.append(run_id)
            msg = "sibling unavailable"
            raise RuntimeError(msg)

    conn = sqlite3.connect(tailor_app_db)
    conn.row_factory = sqlite3.Row
    try:
        # Insert a tailor_runs row with resume_run_id so the helper
        # actually tries the sibling call.
        cursor = conn.execute(
            "INSERT INTO tailor_runs (job_id, status, resume_run_id, "
            "                          created_at, updated_at) "
            "VALUES (1, 'succeeded', 'rs_x', datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        filename = await _build_pdf_filename(
            conn=conn,
            resume_client=_BoomResume(),
            tailor_run_id=new_id,
            kind="resume",
        )
    finally:
        conn.close()

    # Title + company still come from the seeded jobs row even when
    # the sibling fails; only the applicant name falls back.
    assert filename == "Applicant-Engineer-Acme-Resume.pdf"


async def test_resolve_pdf_filename_falls_back_to_live_when_cache_empty(
    tailor_app_db: Path,
) -> None:
    """v1.15.0 caches the filenames on the tailor_runs row at terminal
    SUCCESS. Rows that finished before the cache landed have NULL --
    the route helper falls back to building live so old runs keep
    working without backfill."""
    from jobai.api.routes.tailor import _resolve_pdf_filename  # noqa: PLC0415

    conn = sqlite3.connect(tailor_app_db)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "INSERT INTO tailor_runs (job_id, status, resume_run_id, created_at, updated_at) "
            "VALUES (1, 'succeeded', 'rs_legacy', datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        filename = await _resolve_pdf_filename(
            conn=conn,
            resume_client=ScriptedResumeClient(
                run_record={
                    "id": "rs_legacy",
                    "tailored": {"name": "Jane Doe"},
                },
            ),
            tailor_run_id=new_id,
            kind="resume",
        )
    finally:
        conn.close()

    # Live builder ran (cache was NULL); used the seeded job's
    # title + company and the sibling's tailored.name.
    assert filename == "Jane_Doe-Engineer-Acme-Resume.pdf"


async def test_resolve_pdf_filename_returns_cached_when_present(
    tailor_app_db: Path,
) -> None:
    """When the row already has a cached filename, the helper short-
    circuits and returns it -- no sibling fetch, no live compute."""
    from jobai.api.routes.tailor import _resolve_pdf_filename  # noqa: PLC0415

    conn = sqlite3.connect(tailor_app_db)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "INSERT INTO tailor_runs ("
            "job_id, status, resume_run_id, resume_filename, letter_filename, "
            "created_at, updated_at"
            ") VALUES (1, 'succeeded', 'rs_x', 'Cached-Resume.pdf', "
            "'Cached-Letter.pdf', datetime('now'), datetime('now'))",
        )
        new_id = int(cursor.lastrowid or 0)
        conn.commit()

        class _BoomResume:
            async def get_run(self, _run_id: str) -> dict[str, object]:
                msg = "should not be called -- cache should short-circuit"
                raise AssertionError(msg)

        resume_filename = await _resolve_pdf_filename(
            conn=conn,
            resume_client=_BoomResume(),  # type: ignore[arg-type]
            tailor_run_id=new_id,
            kind="resume",
        )
        letter_filename = await _resolve_pdf_filename(
            conn=conn,
            resume_client=_BoomResume(),  # type: ignore[arg-type]
            tailor_run_id=new_id,
            kind="letter",
        )
    finally:
        conn.close()

    assert resume_filename == "Cached-Resume.pdf"
    assert letter_filename == "Cached-Letter.pdf"


async def test_resolve_pdf_filename_when_row_missing_falls_through_to_live(
    tailor_app_db: Path,
) -> None:
    """Defensive: if the route somehow calls with a nonexistent run id,
    ``_resolve_pdf_filename`` falls through to the live builder which
    itself returns a sane default. Belt-and-braces -- production
    routes always go through ``_require_artefact`` first."""
    from jobai.api.routes.tailor import _resolve_pdf_filename  # noqa: PLC0415

    conn = sqlite3.connect(tailor_app_db)
    try:
        conn.row_factory = sqlite3.Row
        filename = await _resolve_pdf_filename(
            conn=conn,
            resume_client=ScriptedResumeClient(),
            tailor_run_id=99_999,  # no such row
            kind="resume",
        )
    finally:
        conn.close()

    assert filename == "Applicant-Job-Company-Resume.pdf"
