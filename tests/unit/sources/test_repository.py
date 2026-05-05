"""Tests for the sources repository."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.db.migrations import apply_pending
from jobai.sources.repository import (
    SourceNotFoundError,
    get_source_by_name,
    list_sources,
    set_enabled,
    upsert_source,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """A migrated SQLite database against a fresh temp file."""
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


def test_upsert_inserts_new_row(conn: sqlite3.Connection) -> None:
    row = upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="Atlassian",
    )

    assert row.id > 0
    assert row.kind == "greenhouse"
    assert row.account == "atlassian"
    assert row.display_name == "Atlassian"
    assert row.default_tier == 1
    assert row.enabled is True
    assert row.cadence_seconds == 1800
    assert row.config == {}
    assert row.name == "greenhouse:atlassian"


def test_upsert_updates_mutable_fields_on_conflict(conn: sqlite3.Connection) -> None:
    first = upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="Atlassian",
        default_tier=1,
        cadence_seconds=1800,
    )

    updated = upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="Atlassian Pty Ltd",
        default_tier=2,
        cadence_seconds=900,
    )

    assert updated.id == first.id
    assert updated.display_name == "Atlassian Pty Ltd"
    assert updated.default_tier == 2
    assert updated.cadence_seconds == 900


def test_upsert_does_not_clobber_enabled_flag(conn: sqlite3.Connection) -> None:
    """An admin-disabled source must stay disabled across a re-sync."""
    upsert_source(conn, kind="greenhouse", account="atlassian", display_name="X")
    set_enabled(conn, kind="greenhouse", account="atlassian", enabled=False)

    upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="X",
    )

    row = get_source_by_name(conn, kind="greenhouse", account="atlassian")
    assert row.enabled is False


def test_get_source_by_name_raises_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(SourceNotFoundError):
        get_source_by_name(conn, kind="greenhouse", account="missing-co")


def test_list_sources_returns_all_in_kind_account_order(conn: sqlite3.Connection) -> None:
    upsert_source(conn, kind="greenhouse", account="atlassian", display_name="A")
    upsert_source(conn, kind="greenhouse", account="canva", display_name="B")
    upsert_source(conn, kind="lever", account="netflix", display_name="C")

    sources = list_sources(conn)

    assert [s.name for s in sources] == [
        "greenhouse:atlassian",
        "greenhouse:canva",
        "lever:netflix",
    ]


def test_list_sources_enabled_only(conn: sqlite3.Connection) -> None:
    upsert_source(conn, kind="greenhouse", account="a", display_name="A")
    upsert_source(conn, kind="greenhouse", account="b", display_name="B")
    set_enabled(conn, kind="greenhouse", account="b", enabled=False)

    sources = list_sources(conn, enabled_only=True)

    assert [s.name for s in sources] == ["greenhouse:a"]


def test_set_enabled_toggles_flag(conn: sqlite3.Connection) -> None:
    upsert_source(conn, kind="greenhouse", account="a", display_name="A")

    set_enabled(conn, kind="greenhouse", account="a", enabled=False)
    assert get_source_by_name(conn, kind="greenhouse", account="a").enabled is False

    set_enabled(conn, kind="greenhouse", account="a", enabled=True)
    assert get_source_by_name(conn, kind="greenhouse", account="a").enabled is True


def test_set_enabled_raises_when_source_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(SourceNotFoundError):
        set_enabled(conn, kind="greenhouse", account="ghost", enabled=False)
