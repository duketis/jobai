"""FastAPI application factory.

The ``create_app()`` factory builds a fully-configured FastAPI app
with all routers wired up. ``app`` is the ASGI callable Uvicorn (or
any other ASGI server) imports for ``jobai.api.server:app``.

Keeping construction in a factory means tests can build a fresh app
per test (with overridden dependencies) instead of mutating a global
singleton.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from jobai import __version__
from jobai.api.routes import agent, conversations, health, jobs, notifications, sources
from jobai.config import get_settings
from jobai.scheduler import build_scheduler, register_sources, shutdown

_log = logging.getLogger(__name__)

#: Env-var to disable the scheduler boot. Tests that need a hot
#: TestClient (or production deployments running the scheduler in a
#: separate process) set ``JOBAI_DISABLE_SCHEDULER=1``.
_DISABLE_FLAG = "JOBAI_DISABLE_SCHEDULER"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hook.

    Boots the APScheduler with one interval job per enabled source
    so the API process keeps the data layer fresh in the background.
    Shutdown stops the scheduler cleanly so in-flight jobs settle
    before the loop closes.

    Disabled by ``JOBAI_DISABLE_SCHEDULER=1`` for tests and for
    multi-process deployments where one worker owns the scheduler.
    """
    if os.environ.get(_DISABLE_FLAG):
        _log.info("scheduler_disabled_via_env", extra={"flag": _DISABLE_FLAG})
        app.state.scheduler = None
        yield
        return

    settings = get_settings()
    scheduler = build_scheduler()
    try:
        registered = register_sources(scheduler, db_path=settings.db_path)
        scheduler.start()
        _log.info("scheduler_started", extra={"jobs": registered})
        app.state.scheduler = scheduler
        yield
    finally:
        await shutdown(scheduler)
        _log.info("scheduler_stopped")


def create_app() -> FastAPI:
    """Build and return a configured FastAPI app."""
    application = FastAPI(
        title="jobai",
        description=(
            "Local-first AI job-hunting agent — data layer API. "
            "Auto-generated OpenAPI docs at /docs."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix="/api", tags=["health"])
    application.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
    application.include_router(sources.router, prefix="/api/sources", tags=["sources"])
    application.include_router(
        notifications.router,
        prefix="/api/notifications",
        tags=["notifications"],
    )
    application.include_router(agent.router, prefix="/api/agent", tags=["agent"])
    application.include_router(
        conversations.router,
        prefix="/api/conversations",
        tags=["conversations"],
    )
    return application


#: Module-level ASGI app for ``uvicorn jobai.api.server:app``.
app = create_app()
