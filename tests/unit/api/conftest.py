"""Shared API test fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jobai.api.dependencies import get_db_path
from jobai.api.server import create_app
from jobai.db.migrations import apply_pending


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a fresh migrated SQLite file per test."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    try:
        apply_pending(conn)
    finally:
        conn.close()
    return path


@pytest.fixture
def app(db_path: Path) -> FastAPI:
    """Build a FastAPI app with the test DB injected via dependency override."""
    application = create_app()
    application.dependency_overrides[get_db_path] = lambda: db_path
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Sync TestClient for issuing requests in tests."""
    with TestClient(app) as test_client:
        yield test_client
