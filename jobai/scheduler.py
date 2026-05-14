"""APScheduler integration: run each enabled source at its cadence.

The scheduler is the difference between "I have to remember to scrape"
and "the system runs itself in the background." For one user it could
be a cron job; we use APScheduler so the same code drives the API
process (lifespan-managed scheduler) and a future ``jobai scheduler``
standalone CLI without re-implementing the scheduling primitives.

We pick :class:`AsyncIOScheduler` rather than ``BackgroundScheduler``
because every scrape touches async fetchers (httpx / Playwright /
Patchright); running them on a thread-pool scheduler would force a
thread-to-event-loop hop per cycle and add a layer of foot-guns.

Job persistence is intentionally **not** wired up. APScheduler's
SQLAlchemyJobStore would let jobs survive process restarts, but
sources are seeded from the DB on boot via :func:`register_sources`,
so the same set is reconstructed every time. Adding a JobStore would
just give us two sources of truth.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from jobai.db.connection import connect
from jobai.fetcher.dispatch import build_fetcher
from jobai.pipeline.description_backfill import backfill_descriptions
from jobai.pipeline.runner import RunResult, run_source
from jobai.sources.registry import UnknownSourceKindError, get_source_class
from jobai.sources.repository import SourceRow, list_sources

_log = logging.getLogger(__name__)

#: Bound on the random delay before the FIRST run for each job, in
#: seconds. Spreads scheduler start-up load when many sources fire on
#: the same cadence (e.g. all the Seek slugs at 3600s).
_INITIAL_JITTER_SECONDS = 30

#: Job IDs follow this format so reconciling DB → scheduler is a
#: simple lookup. Source ids are stable (PK in the sources table).
_JOB_ID_PREFIX = "jobai.source."

#: ID for the singleton description-backfill job, distinct from
#: per-source IDs so registration logic doesn't accidentally clobber it.
_BACKFILL_JOB_ID = "jobai.backfill.descriptions"

#: Cadence and per-tick budget for the backfill. 600s = every 10 min;
#: 25 jobs per tick = ~150/hour, well below the per-IP rate-limit
#: floor for LinkedIn guest pages.
_BACKFILL_INTERVAL_SECONDS = 600
_BACKFILL_LIMIT_PER_TICK = 25

#: ID for the singleton ATS-discovery job. Mines apply URLs from every
#: scrape cycle for company slugs we don't yet have a direct ATS feed
#: for and upserts them as enabled sources.
_DISCOVERY_JOB_ID = "jobai.discovery.ats_slugs"

#: Discovery cadence in seconds. 24h is the right cadence: companies
#: don't switch ATS providers overnight, and the apply-URL corpus only
#: changes after a full ``--enabled`` cycle (~hourly) populates new
#: rows. Daily catches new entrants without spamming the DB.
_DISCOVERY_INTERVAL_SECONDS = 86_400

#: ID for the singleton context-pool refresh job. Re-scans every
#: project-scan entry against its source path so the pool reflects
#: the user's CURRENT repo state instead of the day it was uploaded.
_CONTEXT_REFRESH_JOB_ID = "jobai.context.refresh_projects"

#: Refresh cadence: daily. Projects don't get re-tailored that
#: often, and a daily refresh is enough to keep stats current
#: without spamming resumeai's scan endpoint.
_CONTEXT_REFRESH_INTERVAL_SECONDS = 86_400


def build_scheduler() -> AsyncIOScheduler:
    """Construct an unstarted :class:`AsyncIOScheduler`.

    Caller registers jobs via :func:`register_sources` then calls
    ``scheduler.start()`` (or wires it into a FastAPI lifespan).
    """
    return AsyncIOScheduler(timezone="UTC")


def register_sources(
    scheduler: AsyncIOScheduler,
    *,
    db_path: Path,
    job_factory: Callable[[int, Path], Callable[[], Awaitable[None]]] | None = None,
) -> int:
    """Add one interval job per enabled source.

    Args:
        scheduler: an unstarted (or running) :class:`AsyncIOScheduler`.
        db_path: SQLite path the job functions will reopen per run.
            Reopening is cheap and avoids holding a connection across
            cycles.
        job_factory: builds the coroutine each scheduled run invokes.
            Defaulted to :func:`_default_job_factory`; tests inject a
            stub.

    Returns:
        The number of jobs registered.
    """
    factory = job_factory or _default_job_factory

    with connect(db_path) as conn:
        rows = list_sources(conn, enabled_only=True)

    # APScheduler's ``replace_existing`` only fires when the scheduler
    # is running; jobs added pre-start go on a separate pending list,
    # so a second registration would duplicate. Clearing managed jobs
    # first makes registration idempotent regardless of state.
    for existing in list(scheduler.get_jobs()):
        if existing.id.startswith(_JOB_ID_PREFIX):
            existing.remove()

    registered = 0
    for row in rows:
        scheduler.add_job(
            factory(row.id, db_path),
            trigger=IntervalTrigger(seconds=row.cadence_seconds),
            id=f"{_JOB_ID_PREFIX}{row.id}",
            name=row.name,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            jitter=_INITIAL_JITTER_SECONDS,
        )
        registered += 1
    return registered


def register_description_backfill(
    scheduler: AsyncIOScheduler,
    *,
    db_path: Path,
    interval_seconds: int = _BACKFILL_INTERVAL_SECONDS,
    limit_per_tick: int = _BACKFILL_LIMIT_PER_TICK,
) -> None:
    """Add the singleton description-backfill job to the scheduler.

    Walks ``jobs`` rows missing ``description_text`` and fetches
    each apply URL on the source's appropriate fetcher tier. Tier-3
    (stealth) is used for the only currently-supported parser
    (LinkedIn) since vanilla Playwright is more aggressively
    fingerprinted on detail pages.
    """
    # Drop any prior job by this id so re-registration is idempotent.
    if scheduler.get_job(_BACKFILL_JOB_ID) is not None:
        scheduler.remove_job(_BACKFILL_JOB_ID)

    async def _run() -> None:
        await _run_description_backfill(db_path, limit=limit_per_tick)

    scheduler.add_job(
        _run,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id=_BACKFILL_JOB_ID,
        name="description backfill",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        jitter=_INITIAL_JITTER_SECONDS,
    )


async def _run_description_backfill(db_path: Path, *, limit: int) -> None:
    """Backfill body, called by the scheduled job each tick."""
    fetcher = build_fetcher(tier=3)
    try:
        with connect(db_path) as conn:
            result = await backfill_descriptions(conn, fetcher, limit=limit)
        _log.info(
            "description_backfill_tick",
            extra={
                "attempted": result.attempted,
                "filled": result.filled,
                "skipped": result.skipped,
            },
        )
    finally:
        with suppress(Exception):
            await fetcher.aclose()


def register_ats_discovery(
    scheduler: AsyncIOScheduler,
    *,
    db_path: Path,
    interval_seconds: int = _DISCOVERY_INTERVAL_SECONDS,
) -> None:
    """Add the singleton ATS-slug discovery job to the scheduler.

    The job runs ``discover_slugs`` + ``diff_against_seeded`` against
    the live DB on a daily cadence and upserts every newly-observed
    slug as an enabled source. This means new Greenhouse / Lever /
    Ashby / SmartRecruiters / Workable employer pages get a direct
    feed automatically, without any operator action.
    """
    if scheduler.get_job(_DISCOVERY_JOB_ID) is not None:
        scheduler.remove_job(_DISCOVERY_JOB_ID)

    async def _run() -> None:
        await _run_ats_discovery(db_path)

    scheduler.add_job(
        _run,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id=_DISCOVERY_JOB_ID,
        name="ats slug discovery",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        jitter=_INITIAL_JITTER_SECONDS,
    )


async def _run_ats_discovery(db_path: Path) -> None:
    """Discovery body, called by the scheduled job each tick."""
    from jobai.sources.ats_discovery import (  # noqa: PLC0415
        diff_against_seeded,
        discover_slugs,
        load_seeded_accounts,
    )
    from jobai.sources.repository import upsert_source  # noqa: PLC0415

    with connect(db_path) as conn:
        discovered = discover_slugs(conn)
        seeded = load_seeded_accounts(conn)
        new = diff_against_seeded(discovered, seeded)
        for entry in new:
            upsert_source(
                conn,
                kind=entry.kind,
                account=entry.account,
                display_name=entry.account,
                cadence_seconds=3600,
            )
        _log.info(
            "ats_discovery_tick",
            extra={"new_slugs": len(new), "kinds": sorted({s.kind for s in new})},
        )


def register_context_refresh(
    scheduler: AsyncIOScheduler,
    *,
    resumeai_url: str,
    interval_seconds: int = _CONTEXT_REFRESH_INTERVAL_SECONDS,
) -> None:
    """Add the singleton project-scan refresh job to the scheduler.

    The job walks the context pool, picks every entry tagged
    ``source:local_project``, and re-runs the scan against the
    embedded path so the pool stays current. Without this, snapshots
    drift the moment the user makes another commit to the project
    and tailored documents end up citing yesterday's stats.
    """
    if scheduler.get_job(_CONTEXT_REFRESH_JOB_ID) is not None:
        scheduler.remove_job(_CONTEXT_REFRESH_JOB_ID)

    async def _run() -> None:
        await _run_context_refresh(resumeai_url)

    scheduler.add_job(
        _run,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id=_CONTEXT_REFRESH_JOB_ID,
        name="context pool project refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        jitter=_INITIAL_JITTER_SECONDS,
    )


async def _run_context_refresh(resumeai_url: str) -> None:
    """Refresh-every-project body, called by the scheduled job each tick."""
    from jobai.context.client import HttpxContextClient  # noqa: PLC0415

    client = HttpxContextClient(base_url=resumeai_url)
    refreshed = 0
    failed = 0
    try:
        entries = await client.list_files()
        for entry in entries:
            if "source:local_project" not in entry.tags:
                continue
            try:
                await client.refresh_project(entry.id)
                refreshed += 1
            except Exception:  # noqa: BLE001 - report-and-continue per entry
                failed += 1
                _log.warning(
                    "context_refresh_entry_failed",
                    extra={"file_id": entry.id, "entry_name": entry.name},
                    exc_info=True,
                )
        _log.info(
            "context_refresh_tick",
            extra={"refreshed": refreshed, "failed": failed, "total": len(entries)},
        )
    finally:
        await client.aclose()


def _default_job_factory(
    source_id: int,
    db_path: Path,
) -> Callable[[], Awaitable[None]]:
    """Return a coroutine that runs one scrape cycle for ``source_id``."""

    async def _run() -> None:
        await run_source_by_id(source_id, db_path)

    return _run


async def run_source_by_id(source_id: int, db_path: Path) -> RunResult | None:
    """Resolve, fetch, and run one source by its DB id.

    Returns ``None`` when the source is gone (deleted between
    registration and trigger), disabled, or its kind is unknown.
    Returns the :class:`RunResult` otherwise.
    """
    with connect(db_path) as conn:
        row = _load_source_row(conn, source_id)
        if row is None or not row.enabled:
            _log.info(
                "scheduler_skip_disabled_source",
                extra={"source_id": source_id},
            )
            return None

        try:
            source_class = get_source_class(row.kind)
        except UnknownSourceKindError as exc:
            _log.warning(
                "scheduler_unknown_source_kind",
                extra={"source_id": source_id, "kind": row.kind, "error": str(exc)},
            )
            return None

        source = source_class(account=row.account)
        fetcher = build_fetcher(
            tier=row.default_tier,
            persistent_session=getattr(source, "needs_persistent_session", False),
        )
        try:
            return await run_source(
                conn=conn,
                source=source,
                source_row=row,
                fetcher=fetcher,
            )
        finally:
            with suppress(Exception):
                await fetcher.aclose()


def _load_source_row(conn: sqlite3.Connection, source_id: int) -> SourceRow | None:
    """Look up a single source by id (used by the scheduled job)."""
    rows = list_sources(conn, enabled_only=False)
    for row in rows:
        if row.id == source_id:
            return row
    return None


async def shutdown(scheduler: AsyncIOScheduler) -> None:
    """Stop the scheduler and await job finalisation.

    Safe to call from a FastAPI lifespan ``finally`` block — never
    raises if the scheduler was never started or already stopped.
    """
    if not scheduler.running:
        return
    scheduler.shutdown(wait=False)
    # Yield once so any cancelled jobs can settle before the loop
    # closes; otherwise pending tasks log "Task was destroyed but it
    # is pending!" warnings on shutdown.
    await asyncio.sleep(0)
