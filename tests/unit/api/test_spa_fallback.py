"""Tests for the SPA static-mount + fallback route in :mod:`jobai.api.server`.

The frontend ships as a Vite-built bundle under ``jobai/api/static/``.
Two behaviours need to hold:

1. Requests for real built assets (``/assets/*``, ``index.html``)
   resolve to the file on disk.
2. Any other GET that doesn't match an ``/api/*`` route falls back
   to ``index.html`` so React Router can render its client-side path.

Tests fabricate a minimal static dir on disk via monkeypatch so they
don't depend on the frontend actually having been built.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jobai.api.server import create_app


@pytest.fixture
def static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the SPA mount at a fabricated static directory."""
    fake_static = tmp_path / "static"
    (fake_static / "assets").mkdir(parents=True)
    (fake_static / "index.html").write_text(
        "<!doctype html><html><body>jobai-spa</body></html>",
        encoding="utf-8",
    )
    (fake_static / "assets" / "index.css").write_text(
        "body{color:red}",
        encoding="utf-8",
    )
    monkeypatch.setattr("jobai.api.server._STATIC_DIR", fake_static)
    return fake_static


@pytest.fixture
def client(
    static_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    del static_dir  # ordering-only dependency on the static-dir fixture
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app: FastAPI = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_index_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "jobai-spa" in response.text


def test_assets_path_resolves_to_real_file(client: TestClient) -> None:
    response = client.get("/assets/index.css")
    assert response.status_code == 200
    assert "color:red" in response.text


def test_unknown_route_falls_back_to_index_html(client: TestClient) -> None:
    """React Router routes (``/jobs``, ``/chat/:id``) must serve the SPA."""
    for path in ("/jobs", "/jobs/123", "/chat", "/chat/456"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "jobai-spa" in response.text, path


def test_api_routes_take_precedence(client: TestClient) -> None:
    """The SPA fallback must not shadow ``/api/*`` endpoints.

    /openapi.json is FastAPI's auto-generated schema — it lives at
    /openapi.json, not /api/*, but it must still resolve before
    falling through to the SPA. Using it here keeps the test free of
    DB dependencies (the /api/health endpoint reads `jobs` and would
    need a migrated SQLite file).
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    # And confirm an /api/* path that returns 422 without a DB still
    # returns JSON (not the SPA HTML), proving the API router caught it.
    response = client.post("/api/agent/chat", json={})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


def test_no_static_dir_means_no_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh checkout (no built SPA) still boots without raising.

    Verified via /openapi.json (no DB dependency) plus the missing /
    route returning 404 instead of trying to serve a non-existent
    index.html.
    """
    monkeypatch.setattr("jobai.api.server._STATIC_DIR", tmp_path / "missing")
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app = create_app()
    with TestClient(app) as test_client:
        assert test_client.get("/openapi.json").status_code == 200
        assert test_client.get("/").status_code == 404


def test_top_level_file_under_static_served_via_spa_fallback(
    client: TestClient, static_dir: Path
) -> None:
    """A file at the root of static/ (not under /assets/) should be
    served directly by the SPA fallback, not rewritten to index.html.
    Common case: favicon.ico, robots.txt."""
    (static_dir / "favicon.ico").write_bytes(b"\x00\x00\x01\x00\x01\x00")
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    # Real file body, not the index.html shell.
    assert response.content.startswith(b"\x00\x00\x01\x00")


def test_index_html_without_assets_dir_still_serves_spa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the built SPA has an index.html but no /assets/ subdir (e.g.
    a hand-rolled static page during local hacking), boot is still
    clean and / serves the index. Covers the false branch of
    ``if assets_dir.is_dir():``."""
    fake_static = tmp_path / "static"
    fake_static.mkdir()
    (fake_static / "index.html").write_text("<!doctype html>plain", encoding="utf-8")
    monkeypatch.setattr("jobai.api.server._STATIC_DIR", fake_static)
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    app = create_app()
    with TestClient(app) as test_client:
        assert "plain" in test_client.get("/").text


def test_build_qa_client_returns_none_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an Anthropic API key anywhere, the QA client is None
    and the tailor chain skips the QA stage rather than crashing."""
    from jobai.api.server import _build_qa_client  # noqa: PLC0415

    class _FakeSettings:
        anthropic_api_key = None
        anthropic_model = "claude-opus-4-7"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _build_qa_client(_FakeSettings()) is None


def test_build_qa_client_constructs_anthropic_adapter_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the boot config carries a key, the helper instantiates the
    Anthropic-backed adapter against that key + the configured model."""
    from jobai.api.server import _build_qa_client  # noqa: PLC0415
    from jobai.tailor.qa import AnthropicQAClient  # noqa: PLC0415

    class _FakeSettings:
        anthropic_api_key = "sk-ant-test"
        anthropic_model = "claude-opus-4-7"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _build_qa_client(_FakeSettings())
    assert isinstance(client, AnthropicQAClient)
    assert client._default_model == "claude-opus-4-7"


def test_lifespan_starts_and_stops_scheduler_when_not_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler-on path of the lifespan (no JOBAI_DISABLE_SCHEDULER)
    builds + starts + shuts down the scheduler without raising. The
    boot uses an empty DB so no source actually fires while the app
    is up; we exercise the wiring, not source execution."""
    from jobai.db.migrations import apply_pending  # noqa: PLC0415

    monkeypatch.delenv("JOBAI_DISABLE_SCHEDULER", raising=False)
    db_path = tmp_path / "scheduler-test.db"
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(db_path)
    try:
        apply_pending(conn)
    finally:
        conn.close()
    monkeypatch.setenv("JOBAI_DB_PATH", str(db_path))
    # Re-cache jobai.config so the new env value is picked up.
    from jobai.config import get_settings  # noqa: PLC0415

    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        assert test_client.get("/openapi.json").status_code == 200
        # The lifespan should have wired the scheduler onto app.state.
        assert app.state.scheduler is not None
