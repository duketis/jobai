"""Tests for the FastAPI lifespan / scheduler boot integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jobai.api.server import create_app
from jobai.config import get_settings
from jobai.db.connection import connect
from jobai.db.migrations import apply_pending
from jobai.sources.repository import upsert_source
from jobai.tailor.models import TailorRunStatus
from jobai.tailor.repository import (
    ORPHAN_ERROR,
    create_tailor_run,
    get_tailor_run,
    update_status,
)


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
        # One source-cadence job for the seeded greenhouse:atlassian
        # plus the singleton description-backfill job.
        job_ids = sorted(j.id for j in scheduler.get_jobs())
        assert any(jid.startswith("jobai.source.") for jid in job_ids)
        assert "jobai.backfill.descriptions" in job_ids

    # Lifespan exit should have stopped the scheduler. ``running`` is
    # a property that re-reads state on each access, so this is a
    # genuine post-shutdown check, not a no-op narrowing artifact.
    assert not scheduler.running
    get_settings.cache_clear()


def test_lifespan_reaps_orphaned_tailor_runs_on_boot(
    monkeypatch: pytest.MonkeyPatch,
    seeded_db_path: Path,
) -> None:
    """A tailor_run left mid-flight by a previous process is failed by
    the startup reaper the moment the app boots — the fix for runs
    hanging 'running' forever after a restart."""
    with connect(seeded_db_path) as conn:
        record = create_tailor_run(conn, jd_url="https://example.com/jd")
        update_status(conn, record.id, status=TailorRunStatus.LETTER_RUNNING)

    monkeypatch.setenv("JOBAI_DISABLE_SCHEDULER", "1")
    monkeypatch.setenv("JOBAI_DB_PATH", str(seeded_db_path))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200

    with connect(seeded_db_path) as conn:
        reaped = get_tailor_run(conn, record.id)
    assert reaped is not None
    assert reaped.status is TailorRunStatus.FAILED
    assert reaped.error == ORPHAN_ERROR
    get_settings.cache_clear()
