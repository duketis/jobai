"""Tests for the APScheduler integration."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from jobai.db.migrations import apply_pending
from jobai.scheduler import (
    _BACKFILL_JOB_ID,
    _DISCOVERY_JOB_ID,
    _JOB_ID_PREFIX,
    build_scheduler,
    register_ats_discovery,
    register_description_backfill,
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
# register_description_backfill
# ---------------------------------------------------------------------------


def test_register_description_backfill_adds_singleton_job(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    register_description_backfill(scheduler, db_path=seeded_db)
    assert scheduler.get_job(_BACKFILL_JOB_ID) is not None


def test_register_description_backfill_is_idempotent(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    register_description_backfill(scheduler, db_path=seeded_db)
    register_description_backfill(scheduler, db_path=seeded_db)
    backfill_jobs = [j for j in scheduler.get_jobs() if j.id == _BACKFILL_JOB_ID]
    assert len(backfill_jobs) == 1


def test_register_description_backfill_uses_configurable_interval(
    seeded_db: Path,
) -> None:
    scheduler = build_scheduler()
    register_description_backfill(scheduler, db_path=seeded_db, interval_seconds=900)
    job = scheduler.get_job(_BACKFILL_JOB_ID)
    assert job is not None
    assert job.trigger.interval.total_seconds() == 900


# ---------------------------------------------------------------------------
# register_ats_discovery
# ---------------------------------------------------------------------------


def test_register_ats_discovery_adds_singleton_job(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    register_ats_discovery(scheduler, db_path=seeded_db)
    assert scheduler.get_job(_DISCOVERY_JOB_ID) is not None


def test_register_ats_discovery_is_idempotent(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    register_ats_discovery(scheduler, db_path=seeded_db)
    register_ats_discovery(scheduler, db_path=seeded_db)
    jobs = [j for j in scheduler.get_jobs() if j.id == _DISCOVERY_JOB_ID]
    assert len(jobs) == 1


def test_register_ats_discovery_uses_configurable_interval(seeded_db: Path) -> None:
    scheduler = build_scheduler()
    register_ats_discovery(scheduler, db_path=seeded_db, interval_seconds=3600)
    job = scheduler.get_job(_DISCOVERY_JOB_ID)
    assert job is not None
    assert job.trigger.interval.total_seconds() == 3600


async def test_register_ats_discovery_inner_run_invokes_discovery(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inner ``_run`` coroutine ``register_ats_discovery`` schedules
    just delegates to ``_run_ats_discovery``. Pulling the job out of
    the scheduler and invoking its target covers that line."""
    from jobai import scheduler as scheduler_mod  # noqa: PLC0415

    seen: list[Path] = []

    async def fake_run(db_path: Path) -> None:
        seen.append(db_path)

    monkeypatch.setattr(scheduler_mod, "_run_ats_discovery", fake_run)
    scheduler = build_scheduler()
    register_ats_discovery(scheduler, db_path=seeded_db, interval_seconds=60)
    job = scheduler.get_job(_DISCOVERY_JOB_ID)
    assert job is not None
    await job.func()
    assert seen == [seeded_db]


async def test_run_ats_discovery_upserts_new_slugs(
    seeded_db: Path,
) -> None:
    """The scheduled tick body finds ATS slugs in apply_url and
    upserts each as an enabled source. Seeds one Greenhouse + one
    Lever apply URL the seed doesn't cover, then asserts both are
    registered after a single tick."""
    from jobai import scheduler as scheduler_mod  # noqa: PLC0415

    conn = sqlite3.connect(seeded_db)
    try:
        cursor = conn.execute(
            "INSERT INTO sources (kind, account, display_name, cadence_seconds) "
            "VALUES ('seek', 'jobs/in-AU', 'Seek (Aus)', 3600)",
        )
        source_id = int(cursor.lastrowid or 0)
        for ext, url in (
            ("g1", "https://boards.greenhouse.io/newco/jobs/1"),
            ("l1", "https://jobs.lever.co/anotherco/role-x"),
        ):
            conn.execute(
                "INSERT INTO jobs_raw (source_id, source_external_id, raw_json, raw_sha256, "
                "first_seen_at, last_seen_at) "
                "VALUES (?, ?, '{}', 'sha', datetime('now'), datetime('now'))",
                (source_id, ext),
            )
            conn.execute(
                "INSERT INTO jobs (dedup_key, title, company, company_norm, apply_url, "
                "first_seen_at, last_seen_at) "
                "VALUES (?, 't', 'c', 'c', ?, datetime('now'), datetime('now'))",
                (ext, url),
            )
        conn.commit()
    finally:
        conn.close()

    await scheduler_mod._run_ats_discovery(seeded_db)

    conn = sqlite3.connect(seeded_db)
    try:
        accounts = {(row[0], row[1]) for row in conn.execute("SELECT kind, account FROM sources")}
    finally:
        conn.close()
    # The discovery tick added these two from the apply URLs we seeded.
    assert ("greenhouse", "newco") in accounts
    assert ("lever", "anotherco") in accounts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_job() -> None:
    return None


# ---------------------------------------------------------------------------
# Other branches: existing non-managed jobs, default factory, backfill body
# ---------------------------------------------------------------------------


def test_register_sources_preserves_jobs_outside_managed_namespace(
    seeded_db: Path,
) -> None:
    """Jobs whose id doesn't start with the managed prefix must survive
    re-registration. Covers the ``101 -> 100`` continue branch."""
    scheduler = build_scheduler()
    scheduler.add_job(_noop_job, id="unrelated-pre-existing", trigger="interval", seconds=60)
    register_sources(
        scheduler,
        db_path=seeded_db,
        job_factory=lambda _id, _path: _noop_job,
    )
    assert scheduler.get_job("unrelated-pre-existing") is not None


async def test_default_job_factory_invokes_run_source_by_id(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default factory returns a coroutine that calls run_source_by_id
    with the bound source_id + db_path."""
    from jobai import scheduler as scheduler_mod  # noqa: PLC0415

    seen: list[tuple[int, Path]] = []

    async def fake(source_id: int, db_path: Path) -> None:
        seen.append((source_id, db_path))

    monkeypatch.setattr(scheduler_mod, "run_source_by_id", fake)
    coro_factory = scheduler_mod._default_job_factory(42, seeded_db)
    await coro_factory()
    assert seen == [(42, seeded_db)]


async def test_run_description_backfill_calls_backfill_with_tier3_fetcher(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled backfill tick should build a tier-3 fetcher, hand
    it to backfill_descriptions, and close it on the way out."""
    from jobai import scheduler as scheduler_mod  # noqa: PLC0415

    class _FakeFetcher:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    captured: dict[str, object] = {}
    fake = _FakeFetcher()

    def fake_build(*, tier: int, **kwargs: object) -> _FakeFetcher:
        captured["tier"] = tier
        return fake

    from jobai.pipeline.description_backfill import BackfillResult  # noqa: PLC0415

    async def fake_backfill(conn: object, fetcher: object, *, limit: int) -> BackfillResult:
        del conn
        captured["limit"] = limit
        captured["fetcher_is_fake"] = fetcher is fake
        return BackfillResult(attempted=2, filled=1, skipped=1)

    monkeypatch.setattr(scheduler_mod, "build_fetcher", fake_build)
    monkeypatch.setattr(scheduler_mod, "backfill_descriptions", fake_backfill)

    await scheduler_mod._run_description_backfill(seeded_db, limit=5)
    assert captured["tier"] == 3
    assert captured["limit"] == 5
    assert captured["fetcher_is_fake"] is True
    assert fake.closed is True


async def test_register_description_backfill_inner_run_invokes_backfill(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inner ``_run`` coroutine that ``register_description_backfill``
    schedules just delegates to ``_run_description_backfill``. Pulling the
    job out of the scheduler and invoking its target covers that line."""
    from jobai import scheduler as scheduler_mod  # noqa: PLC0415

    seen: list[tuple[Path, int]] = []

    async def fake_run(db_path: Path, *, limit: int) -> None:
        seen.append((db_path, limit))

    monkeypatch.setattr(scheduler_mod, "_run_description_backfill", fake_run)
    scheduler = build_scheduler()
    register_description_backfill(
        scheduler, db_path=seeded_db, interval_seconds=60, limit_per_tick=7
    )
    job = scheduler.get_job(_BACKFILL_JOB_ID)
    assert job is not None
    await job.func()
    assert seen == [(seeded_db, 7)]


async def test_run_source_by_id_executes_runner_for_enabled_source(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: enabled source, registered kind. The function
    builds the source instance, the fetcher, and calls run_source.
    All three are monkey-patched out so no real HTTP / browser fires."""
    from jobai import scheduler as scheduler_mod  # noqa: PLC0415

    captured: dict[str, object] = {}

    class _FakeFetcher:
        async def aclose(self) -> None:
            captured["closed"] = True

    class _FakeSource:
        needs_persistent_session = False

        def __init__(self, *, account: str) -> None:
            captured["source_account"] = account

    def fake_get_source_class(kind: str) -> type[_FakeSource]:
        captured["kind_requested"] = kind
        return _FakeSource

    def fake_build_fetcher(**kwargs: object) -> _FakeFetcher:
        captured["fetcher_kwargs"] = kwargs
        return _FakeFetcher()

    from jobai.pipeline.runner import RunResult  # noqa: PLC0415

    async def fake_run_source(**kwargs: object) -> RunResult:
        del kwargs
        captured["run_source_called"] = True
        return RunResult(
            run_id=1,
            status="success",
            items_seen=0,
            items_new=0,
            items_updated=0,
        )

    monkeypatch.setattr(scheduler_mod, "get_source_class", fake_get_source_class)
    monkeypatch.setattr(scheduler_mod, "build_fetcher", fake_build_fetcher)
    monkeypatch.setattr(scheduler_mod, "run_source", fake_run_source)

    # seeded_db has greenhouse:atlassian as source_id=1.
    result = await run_source_by_id(1, seeded_db)
    assert result is not None
    assert result.status == "success"
    assert captured["run_source_called"] is True
    assert captured["closed"] is True
