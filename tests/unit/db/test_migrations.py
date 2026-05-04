"""Unit tests for the SQL migration runner."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.db.migrations import apply_pending, applied_migration_ids, discover_migrations


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """An ephemeral in-memory database, closed at fixture teardown."""
    connection = sqlite3.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """Two trivial migration files plus an unrelated file the runner ignores."""
    (tmp_path / "0001_initial.sql").write_text(
        "CREATE TABLE first (id INTEGER PRIMARY KEY);"
    )
    (tmp_path / "0002_add_second.sql").write_text(
        "CREATE TABLE second (id INTEGER PRIMARY KEY);"
    )
    (tmp_path / "README.md").write_text("ignored, not a migration")
    return tmp_path


def test_discover_returns_migrations_in_id_order(migrations_dir: Path) -> None:
    discovered = discover_migrations(migrations_dir)
    assert [m.id for m in discovered] == [1, 2]
    assert [m.name for m in discovered] == ["initial", "add_second"]


def test_discover_ignores_files_that_do_not_match_the_pattern(
    migrations_dir: Path,
) -> None:
    discovered = discover_migrations(migrations_dir)
    assert all(m.name != "README" for m in discovered)
    assert len(discovered) == 2


def test_applied_migration_ids_returns_empty_set_on_fresh_db(
    conn: sqlite3.Connection,
) -> None:
    assert applied_migration_ids(conn) == set()


def test_applied_migration_ids_creates_bookkeeping_table(
    conn: sqlite3.Connection,
) -> None:
    applied_migration_ids(conn)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_migrations'"
    ).fetchall()
    assert len(rows) == 1


def test_apply_pending_runs_all_migrations_on_fresh_db(
    conn: sqlite3.Connection,
    migrations_dir: Path,
) -> None:
    applied = apply_pending(conn, migrations_dir)

    assert [m.id for m in applied] == [1, 2]

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "first" in tables
    assert "second" in tables


def test_apply_pending_is_idempotent(
    conn: sqlite3.Connection,
    migrations_dir: Path,
) -> None:
    first_pass = apply_pending(conn, migrations_dir)
    second_pass = apply_pending(conn, migrations_dir)

    assert len(first_pass) == 2
    assert second_pass == []


def test_apply_pending_records_each_migration_in_order(
    conn: sqlite3.Connection,
    migrations_dir: Path,
) -> None:
    apply_pending(conn, migrations_dir)
    rows = conn.execute(
        "SELECT id, name FROM _schema_migrations ORDER BY id"
    ).fetchall()
    assert rows == [(1, "initial"), (2, "add_second")]


def test_apply_pending_skips_already_applied_migrations(
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    (tmp_path / "0001_one.sql").write_text(
        "CREATE TABLE one (id INTEGER PRIMARY KEY);"
    )

    apply_pending(conn, tmp_path)

    (tmp_path / "0002_two.sql").write_text(
        "CREATE TABLE two (id INTEGER PRIMARY KEY);"
    )

    second_pass = apply_pending(conn, tmp_path)

    assert [m.id for m in second_pass] == [2]


def test_failing_migration_is_not_recorded_as_applied(tmp_path: Path) -> None:
    (tmp_path / "0001_good.sql").write_text(
        "CREATE TABLE good (id INTEGER PRIMARY KEY);"
    )
    (tmp_path / "0002_bad.sql").write_text(
        "INSERT INTO nonexistent_table VALUES (1);"
    )

    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(sqlite3.Error):
            apply_pending(connection, tmp_path)

        applied = applied_migration_ids(connection)
        assert 1 in applied
        assert 2 not in applied
    finally:
        connection.close()


def test_default_migrations_dir_loads_initial_schema() -> None:
    """The packaged migration set must be discoverable without an explicit path."""
    discovered = discover_migrations()

    assert any(m.id == 1 and m.name == "initial" for m in discovered), (
        "expected to find 0001_initial.sql in the packaged migrations directory"
    )
