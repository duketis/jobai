"""Unit tests for the SQLite connection helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jobai.db.connection import connect


def test_connect_uses_wal_journal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        result = conn.execute("PRAGMA journal_mode").fetchone()
    assert result[0].lower() == "wal"


def test_connect_enables_foreign_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        result = conn.execute("PRAGMA foreign_keys").fetchone()
    assert result[0] == 1


def test_connect_uses_row_factory_for_dict_style_access(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.execute("INSERT INTO t (a, b) VALUES (1, 'hi')")
        row = conn.execute("SELECT a, b FROM t").fetchone()

    assert row["a"] == 1
    assert row["b"] == "hi"


def test_connect_persists_data_across_invocations(tmp_path: Path) -> None:
    """Two sequential context entries on the same path see each other's writes."""
    db_path = tmp_path / "persist.db"

    with connect(db_path) as conn:
        conn.execute("CREATE TABLE t (a INTEGER)")
        conn.execute("INSERT INTO t (a) VALUES (42)")
        conn.commit()

    with connect(db_path) as conn:
        row = conn.execute("SELECT a FROM t").fetchone()

    assert row["a"] == 42


def test_connect_closes_connection_on_context_exit(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        leaked = conn

    with pytest.raises(sqlite3.ProgrammingError):
        leaked.execute("SELECT 1")


def test_connect_closes_connection_when_body_raises(tmp_path: Path) -> None:
    """Exception inside the ``with`` block must still close the connection."""
    db_path = tmp_path / "test.db"
    leaked: sqlite3.Connection | None = None

    with pytest.raises(RuntimeError, match="boom"), connect(db_path) as conn:
        leaked = conn
        raise RuntimeError("boom")

    assert leaked is not None
    with pytest.raises(sqlite3.ProgrammingError):
        leaked.execute("SELECT 1")
