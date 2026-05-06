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
from jobai.fetcher.http import HttpFetcher
from jobai.fetcher.retry import RetryingFetcher
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

        async with HttpFetcher() as http_fetcher:
            fetcher = RetryingFetcher(http_fetcher)
            try:
                return await run_source(
                    conn=conn,
                    source=source,
                    source_row=row,
                    fetcher=fetcher,
                )
            finally:
                # RetryingFetcher.aclose() closes the wrapped fetcher.
                # The async-with on http_fetcher would re-close it; the
                # idempotent aclose() in HttpFetcher tolerates that.
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
