"""In-app notifications endpoint.

The schema-change detector, source-failure handlers, and any future
producer write rows into ``notifications``. This route reads them
back, with an unread filter and a mark-read mutation.

The notifications table starts empty. Producers (Phase 6's
schema-change + source-failure detection) populate it; this layer
just exposes it.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from jobai.api.dependencies import ConnDep
from jobai.api.models import (
    NotificationItem,
    NotificationReadResponse,
    NotificationsListResponse,
)

router = APIRouter()


@router.get(
    "",
    response_model=NotificationsListResponse,
    summary="List notifications with unread filter and pagination",
)
def list_notifications(
    conn: ConnDep,
    unread_only: Annotated[
        bool,
        Query(description="Only return notifications that have not been marked read."),
    ] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> NotificationsListResponse:
    where = "WHERE read_at IS NULL" if unread_only else ""
    total = int(
        conn.execute(
            f"SELECT COUNT(*) FROM notifications {where}",  # noqa: S608  - 'where' is a literal
        ).fetchone()[0]
    )
    unread_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM notifications WHERE read_at IS NULL",
        ).fetchone()[0]
    )

    rows = conn.execute(
        f"SELECT id, kind, severity, title, body, created_at, read_at "  # noqa: S608
        f"FROM notifications {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()

    items = [_row_to_item(row) for row in rows]
    return NotificationsListResponse(total=total, unread_count=unread_count, items=items)


@router.post(
    "/{notification_id}/read",
    response_model=NotificationReadResponse,
    summary="Mark a notification as read",
)
def mark_notification_read(
    conn: ConnDep,
    notification_id: int,
) -> NotificationReadResponse:
    if (
        conn.execute(
            "SELECT 1 FROM notifications WHERE id = ?",
            (notification_id,),
        ).fetchone()
        is None
    ):
        raise HTTPException(status_code=404, detail=f"notification {notification_id} not found")

    now = datetime.now(tz=UTC).isoformat()
    conn.execute(
        "UPDATE notifications SET read_at = COALESCE(read_at, ?) WHERE id = ?",
        (now, notification_id),
    )
    conn.commit()

    row = conn.execute(
        "SELECT read_at FROM notifications WHERE id = ?",
        (notification_id,),
    ).fetchone()
    return NotificationReadResponse(id=notification_id, read_at=str(row[0]))


def _row_to_item(row: sqlite3.Row | tuple[Any, ...]) -> NotificationItem:
    return NotificationItem(
        id=int(row[0]),
        kind=str(row[1]),
        severity=str(row[2]),
        title=str(row[3]),
        body=_optional_str(row[4]),
        created_at=str(row[5]),
        read_at=_optional_str(row[6]),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
