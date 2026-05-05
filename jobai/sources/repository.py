"""CRUD operations on the ``sources`` and ``source_runtime_state`` tables.

The repository is the only module that knows the SQL schema for sources;
everywhere else (CLI, runner, scheduler) goes through these typed
helpers. That keeps schema changes contained: a column rename touches
this file, the migration, and nothing else.

Returns are :class:`SourceRow` instances, immutable so callers cannot
accidentally mutate state that has not been written back.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceRow:
    """A row of the ``sources`` table, plus convenience accessors."""

    id: int
    kind: str
    account: str
    display_name: str
    default_tier: int
    enabled: bool
    cadence_seconds: int
    config: dict[str, Any]
    created_at: str

    @property
    def name(self) -> str:
        """The CLI / log identifier, e.g. ``greenhouse:atlassian``."""
        return f"{self.kind}:{self.account}" if self.account else self.kind


class SourceNotFoundError(LookupError):
    """Raised when a lookup by name does not match a row."""

    def __init__(self, kind: str, account: str) -> None:
        super().__init__(f"source not found: {kind}:{account}")
        self.kind = kind
        self.account = account


def upsert_source(
    conn: sqlite3.Connection,
    *,
    kind: str,
    account: str,
    display_name: str,
    default_tier: int = 1,
    enabled: bool = True,
    cadence_seconds: int = 1800,
    config: dict[str, Any] | None = None,
) -> SourceRow:
    """Insert a row, or update mutable fields on the existing row.

    The ``UNIQUE(kind, account)`` constraint drives the upsert; the
    primary key, ``enabled`` flag, and ``created_at`` timestamp are
    preserved on update so toggling a source via :func:`set_enabled`
    is not undone by the next ``sync``.
    """
    conn.execute(
        "INSERT INTO sources "
        "(kind, account, display_name, default_tier, enabled, cadence_seconds, config_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(kind, account) DO UPDATE SET "
        "    display_name = excluded.display_name, "
        "    default_tier = excluded.default_tier, "
        "    cadence_seconds = excluded.cadence_seconds, "
        "    config_json = excluded.config_json",
        (
            kind,
            account,
            display_name,
            default_tier,
            int(enabled),
            cadence_seconds,
            json.dumps(config or {}),
        ),
    )
    conn.commit()
    return get_source_by_name(conn, kind=kind, account=account)


def get_source_by_name(
    conn: sqlite3.Connection,
    *,
    kind: str,
    account: str,
) -> SourceRow:
    """Look up a source by ``(kind, account)``."""
    row = conn.execute(
        "SELECT id, kind, account, display_name, default_tier, enabled, "
        "       cadence_seconds, config_json, created_at "
        "FROM sources WHERE kind = ? AND account = ?",
        (kind, account),
    ).fetchone()
    if row is None:
        raise SourceNotFoundError(kind, account)
    return _row_to_source(row)


def list_sources(
    conn: sqlite3.Connection,
    *,
    enabled_only: bool = False,
) -> list[SourceRow]:
    """Return every configured source, optionally filtered to enabled ones."""
    sql = (
        "SELECT id, kind, account, display_name, default_tier, enabled, "
        "       cadence_seconds, config_json, created_at FROM sources"
    )
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY kind, account"
    return [_row_to_source(r) for r in conn.execute(sql)]


def set_enabled(
    conn: sqlite3.Connection,
    *,
    kind: str,
    account: str,
    enabled: bool,
) -> None:
    """Toggle the enabled flag for a single source."""
    cursor = conn.execute(
        "UPDATE sources SET enabled = ? WHERE kind = ? AND account = ?",
        (int(enabled), kind, account),
    )
    if cursor.rowcount == 0:
        raise SourceNotFoundError(kind, account)
    conn.commit()


def _row_to_source(row: sqlite3.Row | tuple[Any, ...]) -> SourceRow:
    """Translate a row tuple from any of the SELECTs above into a SourceRow."""
    return SourceRow(
        id=int(row[0]),
        kind=str(row[1]),
        account=str(row[2]),
        display_name=str(row[3]),
        default_tier=int(row[4]),
        enabled=bool(row[5]),
        cadence_seconds=int(row[6]),
        config=json.loads(row[7]) if row[7] else {},
        created_at=str(row[8]),
    )
