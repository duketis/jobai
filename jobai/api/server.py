"""FastAPI application factory.

The ``create_app()`` factory builds a fully-configured FastAPI app
with all routers wired up. ``app`` is the ASGI callable Uvicorn (or
any other ASGI server) imports for ``jobai.api.server:app``.

Keeping construction in a factory means tests can build a fresh app
per test (with overridden dependencies) instead of mutating a global
singleton.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from jobai import __version__
from jobai.api.routes import agent, health, jobs, notifications, sources


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hook.

    Currently a no-op. Future phases plug in: scheduler boot/shutdown,
    notification dispatch worker, browser-tier warm-up.
    """
    yield


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
    return application


#: Module-level ASGI app for ``uvicorn jobai.api.server:app``.
app = create_app()
