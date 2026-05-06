"""Tests for the APScheduler integration."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from jobai.db.migrations import apply_pending
from jobai.scheduler import (
    _JOB_ID_PREFIX,
    build_scheduler,
    register_sources,
    run_source_by_id,
    shutdown,
)
from jobai.sources.repository import upsert_source


@pytest.fixture
def db_path(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    try:
        apply_pending(conn)
    finally:
        conn.close()
    yield path


@pytest.fixture
def seeded_db(db_path: Path) -> Path:
    conn = sqlite3.connect(db_path)
    try:
        upsert_source(
            conn,
            kind="greenhouse",
            account="atlassian",
            display_name="Atlassian",
            cadence_seconds=600,
        )
        upsert_source(
            conn,
            kind="lever",
            account="palantir",
            display_name="Palantir",
            cadence_seconds=900,
        )
        upsert_source(
            conn,
            kind="greenhouse",
            account="canva",
            display_name="Canva",
            cadence_seconds=1200,
            enabled=False,  # disabled — should not be registered
        )
    finally:
        conn.commit()
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# build_scheduler / register_sources
# ---------------------------------------------------------------------------


def test_build_scheduler_returns_unstarted_async_scheduler() -> None:
    scheduler = build_scheduler()
    assert isinstance(scheduler, AsyncIOScheduler)
    assert scheduler.running is False


def test_register_sources_skips_disabled_sources(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    count = register_sources(
        scheduler,
        db_path=seeded_db,
        job_factory=lambda _id, _path: _noop_job,
    )
    assert count == 2  # canva is disabled
    job_ids = sorted(j.id for j in scheduler.get_jobs())
    assert all(j.startswith(_JOB_ID_PREFIX) for j in job_ids)


def test_register_sources_uses_per_source_cadence(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    register_sources(
        scheduler,
        db_path=seeded_db,
        job_factory=lambda _id, _path: _noop_job,
    )
    intervals = sorted(j.trigger.interval.total_seconds() for j in scheduler.get_jobs())
    assert intervals == [600, 900]


def test_register_sources_replaces_existing_jobs(seeded_db: Path) -> None:
    """Re-registering on a started scheduler should not double-add."""
    scheduler = build_scheduler()
    first = register_sources(
        scheduler,
        db_path=seeded_db,
        job_factory=lambda _id, _path: _noop_job,
    )
    second = register_sources(
        scheduler,
        db_path=seeded_db,
        job_factory=lambda _id, _path: _noop_job,
    )
    assert first == second == 2
    assert len(scheduler.get_jobs()) == 2


def test_register_sources_passes_id_and_path_to_factory(seeded_db: Path) -> None:
    seen: list[tuple[int, Path]] = []

    def factory(source_id: int, path: Path) -> Callable[[], Awaitable[None]]:
        seen.append((source_id, path))
        return _noop_job

    scheduler = build_scheduler()
    register_sources(scheduler, db_path=seeded_db, job_factory=factory)
    assert len(seen) == 2
    assert all(p == seeded_db for _, p in seen)
    # Source ids are positive ints assigned by SQLite — exact values
    # depend on insert order but we know there are two distinct ones.
    assert len({sid for sid, _ in seen}) == 2


# ---------------------------------------------------------------------------
# run_source_by_id
# ---------------------------------------------------------------------------


async def test_run_source_by_id_returns_none_for_unknown_kind(db_path: Path) -> None:
    """If the source kind isn't registered, the runner skips gracefully."""
    conn = sqlite3.connect(db_path)
    try:
        # Insert a source with a kind not in the registry (bypassing
        # the loader's validation by going straight to SQL).
        cursor = conn.execute(
            "INSERT INTO sources "
            "(kind, account, display_name, default_tier, enabled, cadence_seconds) "
            "VALUES ('mystery', 'x', 'X', 1, 1, 60)",
        )
        conn.commit()
        source_id = cursor.lastrowid
        assert source_id is not None
    finally:
        conn.close()

    result = await run_source_by_id(int(source_id), db_path)
    assert result is None


async def test_run_source_by_id_returns_none_for_disabled_source(seeded_db: Path) -> None:
    conn = sqlite3.connect(seeded_db)
    try:
        row = conn.execute("SELECT id FROM sources WHERE account = 'canva'").fetchone()
    finally:
        conn.close()
    result = await run_source_by_id(int(row[0]), seeded_db)
    assert result is None


async def test_run_source_by_id_returns_none_for_missing_id(db_path: Path) -> None:
    result = await run_source_by_id(99_999, db_path)
    assert result is None


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_is_idempotent_on_unstarted_scheduler() -> None:
    scheduler = build_scheduler()
    # Should not raise even though scheduler never started.
    await shutdown(scheduler)
    assert scheduler.running is False


async def test_shutdown_stops_running_scheduler() -> None:
    scheduler = build_scheduler()
    scheduler.start()
    assert scheduler.running is True
    await shutdown(scheduler)
    assert scheduler.running is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_job() -> None:
    return None
