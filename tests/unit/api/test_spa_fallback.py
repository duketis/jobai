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
