"""SQL migration runner.

Applies numbered ``.sql`` files from this package directory in order,
recording each successful application in a ``_schema_migrations`` table.
Re-running is a no-op for migrations already recorded; a migration that
fails part-way through is not recorded as applied, so a subsequent run
can retry it.

The runner is deliberately small. It does not support down-migrations,
checksum validation, or branched histories. Those are features we do not
have a use case for at our scale; if and when we do, this module is the
right place to grow them.

The migration files live alongside this module file (``0001_initial.sql``,
``0002_*.sql``, etc.). The runner discovers them through
:data:`_DEFAULT_MIGRATIONS_DIR`, which resolves to the package directory.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_MIGRATION_FILENAME_RE = re.compile(r"^(?P<id>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")
_DEFAULT_MIGRATIONS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Migration:
    """A single migration: numeric id, slug name, raw SQL body."""

    id: int
    name: str
    sql: str


def discover_migrations(
    migrations_dir: Path = _DEFAULT_MIGRATIONS_DIR,
) -> list[Migration]:
    """Return migrations in ``migrations_dir`` sorted by numeric id.

    Files that do not match the ``NNNN_name.sql`` format are silently
    ignored, which lets us drop README files or other tooling alongside.
    """
    migrations: list[Migration] = []
    for path in sorted(migrations_dir.iterdir()):
        if not path.is_file():
            continue
        match = _MIGRATION_FILENAME_RE.match(path.name)
        if match is None:
            continue
        migrations.append(
            Migration(
                id=int(match.group("id")),
                name=match.group("name"),
                sql=path.read_text(encoding="utf-8"),
            )
        )
    return migrations


def applied_migration_ids(conn: sqlite3.Connection) -> set[int]:
    """Return the ids of migrations recorded as applied.

    Creates the bookkeeping table on first call so callers do not need to
    run a separate setup step.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _schema_migrations ("
        "    id INTEGER PRIMARY KEY,"
        "    name TEXT NOT NULL,"
        "    applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    cursor = conn.execute("SELECT id FROM _schema_migrations")
    return {int(row[0]) for row in cursor}


def apply_pending(
    conn: sqlite3.Connection,
    migrations_dir: Path = _DEFAULT_MIGRATIONS_DIR,
) -> list[Migration]:
    """Apply migrations not yet recorded as applied.

    Returns the migrations applied in this call, in the order they were
    applied. Raises :class:`sqlite3.Error` (or a subclass) if a migration
    fails; the failing migration is not recorded as applied, but partial
    schema changes from a multi-statement script may persist.
    """
    applied = applied_migration_ids(conn)
    pending = [m for m in discover_migrations(migrations_dir) if m.id not in applied]
    for migration in pending:
        conn.executescript(migration.sql)
        conn.execute(
            "INSERT INTO _schema_migrations (id, name) VALUES (?, ?)",
            (migration.id, migration.name),
        )
        conn.commit()
    return pending
