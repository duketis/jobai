"""Tests for the FastAPI lifespan / scheduler boot integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jobai.api.server import create_app
from jobai.config import get_settings
from jobai.db.migrations import apply_pending
from jobai.sources.repository import upsert_source


@pytest.fixture
def seeded_db_path(tmp_path: Path) -> Path:
    path = tmp_path / "lifespan.db"
    conn = sqlite3.connect(path)
    try:
        apply_pending(conn)
        upsert_source(
            conn,
            kind="greenhouse",
            account="atlassian",
            display_name="Atlassian",
            cadence_seconds=3600,
        )
    finally:
        conn.commit()
        conn.close()
    return path


def test_disable_scheduler_env_skips_boot(
    monkeypatch: pytest.MonkeyPatch,
    seeded_db_path: Path,
) -> None:
    """``JOBAI_DISABLE_SCHEDULER=1`` makes the lifespan a no-op."""
    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    monkeypatch.setenv("JOBAI_DB_PATH", str(seeded_db_path))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
    assert app.state.scheduler is None


def test_scheduler_boots_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    seeded_db_path: Path,
) -> None:
    """Without the disable flag, the scheduler starts and registers the
    seeded source as a job."""
    monkeypatch.delenv("JOBAI_DISABLE_SCHEDULER", raising=False)
    monkeypatch.setenv("JOBAI_DB_PATH", str(seeded_db_path))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        scheduler = app.state.scheduler
        assert scheduler is not None
        assert scheduler.running
        # Greenhouse:atlassian was seeded → expect one job registered.
        assert len(scheduler.get_jobs()) == 1

    # Lifespan exit should have stopped the scheduler. ``running`` is
    # a property that re-reads state on each access, so this is a
    # genuine post-shutdown check, not a no-op narrowing artifact.
    assert not scheduler.running
    get_settings.cache_clear()
