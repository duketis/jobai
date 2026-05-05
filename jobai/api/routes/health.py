"""Aggregate health endpoint — answers 'is the data layer healthy?' in one call.

The response is the snapshot every dashboard / monitoring consumer
needs: how many jobs we have, how many were added recently, and how
many sources are currently failing. ``status`` is a coarse bucket
('ok' / 'degraded') derived from the source-failure count so callers
can light up a single indicator without parsing the breakdown.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from jobai.api.dependencies import ConnDep

router = APIRouter()


class HealthResponse(BaseModel):
    """Aggregate health snapshot of the data layer."""

    status: str = Field(description="Coarse bucket: 'ok' or 'degraded'.")
    jobs_total: int = Field(description="Count of canonical (deduplicated) jobs.")
    jobs_added_24h: int = Field(description="Jobs first seen within the last 24h.")
    sources_total: int = Field(description="Count of configured sources.")
    sources_enabled: int = Field(description="Of those, how many are enabled.")
    sources_failing: int = Field(
        description="Sources whose most recent attempt failed within 24h.",
    )
    timestamp: str = Field(description="ISO 8601 UTC timestamp of this snapshot.")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Aggregate health snapshot",
)
def get_health(conn: ConnDep) -> HealthResponse:
    """Return the data layer's aggregate health.

    Used by dashboards, the AI agent's startup self-check, and the
    ``jobai health`` CLI.
    """
    jobs_total = _scalar_int(conn, "SELECT COUNT(*) FROM jobs")
    jobs_added_24h = _scalar_int(
        conn,
        "SELECT COUNT(*) FROM jobs WHERE first_seen_at >= datetime('now', '-1 day')",
    )
    sources_total = _scalar_int(conn, "SELECT COUNT(*) FROM sources")
    sources_enabled = _scalar_int(conn, "SELECT COUNT(*) FROM sources WHERE enabled = 1")
    sources_failing = _scalar_int(
        conn,
        "SELECT COUNT(*) FROM source_runtime_state "
        "WHERE last_error_at IS NOT NULL "
        "AND last_error_at > COALESCE(last_success_at, '1970-01-01') "
        "AND last_error_at >= datetime('now', '-1 day')",
    )

    status = "ok" if sources_failing == 0 else "degraded"

    return HealthResponse(
        status=status,
        jobs_total=jobs_total,
        jobs_added_24h=jobs_added_24h,
        sources_total=sources_total,
        sources_enabled=sources_enabled,
        sources_failing=sources_failing,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


def _scalar_int(conn: ConnDep, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row is not None else 0
