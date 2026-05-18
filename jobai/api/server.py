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
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import Response

from jobai import __version__
from jobai.api.routes import (
    agent,
    context,
    conversations,
    health,
    jobs,
    notifications,
    sources,
    tailor,
)
from jobai.api.routes import (
    settings as settings_routes,
)
from jobai.config import get_settings
from jobai.context.client import HttpxContextClient
from jobai.scheduler import (
    build_scheduler,
    register_ats_discovery,
    register_context_refresh,
    register_description_backfill,
    register_sources,
    shutdown,
)
from jobai.tailor.client import (
    HttpxCoverletteraiClient,
    HttpxResumeaiClient,
)
from jobai.tailor.worker import TailorPool

#: Built-frontend directory. Vite emits ``index.html`` + ``assets/*`` here
#: when ``npm run build`` runs in ``frontend/``. The directory is gitignored
#: in production deploys but committed for repeatable builds.
_STATIC_DIR = Path(__file__).parent / "static"

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
    settings = get_settings()

    # Tailor wiring runs regardless of the scheduler flag — the chain
    # routes are useful in tests too, and the pool / clients are cheap.
    app.state.resume_client = HttpxResumeaiClient(base_url=settings.resumeai_url)
    app.state.letter_client = HttpxCoverletteraiClient(base_url=settings.coverletterai_url)
    # Stash the resumeai URL so the tailor routes can build a per-chain
    # context-refresh closure that re-scans every project entry right
    # before kicking the resume render -- keeps the LLM's stats fresh
    # without waiting for the daily scheduler tick.
    app.state.resumeai_url = settings.resumeai_url
    # Output directory for the on-disk snapshot every successful
    # tailor run drops (resume PDF, letter PDF, JD markdown, QA
    # verdict, application checklist, metadata). Anyone cloning the
    # repo can override via ``JOBAI_TAILOR_OUTPUT_DIR`` so the folders
    # land somewhere the host's file manager can reach.
    app.state.tailor_output_dir = Path(settings.tailor_output_dir)
    tailor_pool = TailorPool(max_concurrent=settings.tailor_max_concurrent)
    app.state.tailor_pool = tailor_pool
    # Tailor chains are in-process asyncio tasks. Any run still
    # non-terminal at boot was orphaned when the previous process
    # exited (nothing is left to drive it), so fail it now instead of
    # leaving it hanging "running" forever in the UI.
    from jobai.db.connection import connect  # noqa: PLC0415
    from jobai.tailor.repository import reap_orphaned_runs  # noqa: PLC0415

    with connect(settings.db_path) as _reap_conn:
        reaped = reap_orphaned_runs(_reap_conn)
    if reaped:
        _log.info("tailor_runs_reaped_on_startup", extra={"count": reaped})
    # The context pool lives in resumeai; jobai proxies through so the
    # whole job-hunt workflow (browse -> tailor -> manage context) is
    # behind one URL.
    context_client = HttpxContextClient(base_url=settings.resumeai_url)
    app.state.context_client = context_client

    if os.environ.get(_DISABLE_FLAG):
        _log.info("scheduler_disabled_via_env", extra={"flag": _DISABLE_FLAG})
        app.state.scheduler = None
        try:
            yield
        finally:
            await tailor_pool.drain()
            await context_client.aclose()
        return

    scheduler = build_scheduler()
    try:
        registered = register_sources(scheduler, db_path=settings.db_path)
        register_description_backfill(scheduler, db_path=settings.db_path)
        register_ats_discovery(scheduler, db_path=settings.db_path)
        register_context_refresh(scheduler, resumeai_url=settings.resumeai_url)
        scheduler.start()
        _log.info(
            "scheduler_started",
            extra={"jobs": registered, "backfill": "enabled"},
        )
        app.state.scheduler = scheduler
        yield
    finally:
        await shutdown(scheduler)
        await tailor_pool.drain()
        await context_client.aclose()
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
    application.include_router(
        settings_routes.router,
        prefix="/api/settings",
        tags=["settings"],
    )
    application.include_router(tailor.router, prefix="/api/tailor", tags=["tailor"])
    application.include_router(context.router, prefix="/api/context", tags=["context"])
    _mount_frontend(application)
    return application


def _mount_frontend(application: FastAPI) -> None:
    """Serve the React SPA from ``jobai/api/static`` if it exists.

    Mounting is conditional so a fresh checkout (where the frontend
    hasn't been built yet) still boots a working API. The ``/api/*``
    routes already declared above take precedence — Starlette's mount
    sits at the root and only catches paths that fall through.

    SPA fallback: any non-``/api/*`` GET that isn't a real file under
    ``static/`` returns ``index.html`` so React Router's client-side
    routes (``/jobs``, ``/chat/:id``, ...) render correctly on a
    direct page load or refresh.
    """
    if not (_STATIC_DIR / "index.html").is_file():
        _log.info(
            "frontend_static_dir_missing",
            extra={"path": str(_STATIC_DIR)},
        )
        return

    assets_dir = _STATIC_DIR / "assets"
    if assets_dir.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="frontend-assets",
        )

    @application.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(request: Request, full_path: str) -> Response:
        # /api/* never reaches here — those routers are matched first.
        candidate = _STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        # Otherwise hand back index.html and let React Router decide.
        del request
        return FileResponse(_STATIC_DIR / "index.html")


#: Module-level ASGI app for ``uvicorn jobai.api.server:app``.
app = create_app()
