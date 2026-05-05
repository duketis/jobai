"""Sources endpoint: read-only view of every configured source.

Reads ``sources`` LEFT JOIN ``source_runtime_state`` so callers see
both the static configuration (kind, account, default_tier,
cadence, enabled) and the live health (current tier, last success,
last error, cooldown). One round-trip per request.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Query

from jobai.api.dependencies import ConnDep
from jobai.api.models import SourcesListResponse, SourceSummary

router = APIRouter()


@router.get(
    "",
    response_model=SourcesListResponse,
    summary="List configured sources with runtime health",
)
def list_sources(
    conn: ConnDep,
    enabled_only: Annotated[
        bool,
        Query(description="Only return enabled sources."),
    ] = False,
) -> SourcesListResponse:
    sql = (
        "SELECT s.id, s.kind, s.account, s.display_name, s.default_tier, "
        "       s.enabled, s.cadence_seconds, s.config_json, "
        "       rs.current_tier, rs.last_success_at, rs.last_error_at, "
        "       rs.last_error_class, rs.consecutive_failures, rs.cooldown_until "
        "FROM sources s LEFT JOIN source_runtime_state rs ON rs.source_id = s.id"
    )
    if enabled_only:
        sql += " WHERE s.enabled = 1"
    sql += " ORDER BY s.kind, s.account"

    items = [_row_to_summary(row) for row in conn.execute(sql)]
    return SourcesListResponse(items=items)


def _row_to_summary(row: sqlite3.Row | tuple[Any, ...]) -> SourceSummary:
    kind = str(row[1])
    account = str(row[2])
    name = f"{kind}:{account}" if account else kind
    # config_json is reserved for future per-source overrides; we don't
    # surface it in the response yet but parsing here proves it's valid.
    json.loads(row[7]) if row[7] else {}
    return SourceSummary(
        id=int(row[0]),
        name=name,
        kind=kind,
        account=account,
        display_name=str(row[3]),
        default_tier=int(row[4]),
        enabled=bool(row[5]),
        cadence_seconds=int(row[6]),
        current_tier=_optional_int(row[8]),
        last_success_at=_optional_str(row[9]),
        last_error_at=_optional_str(row[10]),
        last_error_class=_optional_str(row[11]),
        consecutive_failures=int(row[12]) if row[12] is not None else 0,
        cooldown_until=_optional_str(row[13]),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
