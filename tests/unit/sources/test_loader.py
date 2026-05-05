"""Tests for the companies.yaml bulk loader."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.db.migrations import apply_pending
from jobai.sources.loader import (
    DEFAULT_COMPANIES_YAML,
    CompaniesYamlError,
    sync_companies_yaml,
)
from jobai.sources.registry import UnknownSourceKindError
from jobai.sources.repository import (
    get_source_by_name,
    list_sources,
    set_enabled,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


def _write_yaml(path: Path, content: str) -> Path:
    file = path / "companies.yaml"
    file.write_text(content, encoding="utf-8")
    return file


def test_sync_inserts_each_entry(conn: sqlite3.Connection, tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        greenhouse:
          - account: atlassian
            display_name: Atlassian
          - account: canva
            display_name: Canva
        """,
    )

    report = sync_companies_yaml(conn, path=yaml_file)

    assert report.upserted == 2
    assert report.skipped_unknown_kind == []
    sources = list_sources(conn)
    assert [s.name for s in sources] == ["greenhouse:atlassian", "greenhouse:canva"]


def test_sync_is_idempotent(conn: sqlite3.Connection, tmp_path: Path) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        greenhouse:
          - account: atlassian
            display_name: Atlassian
        """,
    )

    sync_companies_yaml(conn, path=yaml_file)
    second_report = sync_companies_yaml(conn, path=yaml_file)

    assert second_report.upserted == 1
    assert len(list_sources(conn)) == 1


def test_sync_preserves_disabled_flag_across_runs(
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        greenhouse:
          - account: atlassian
            display_name: Atlassian
        """,
    )

    sync_companies_yaml(conn, path=yaml_file)
    set_enabled(conn, kind="greenhouse", account="atlassian", enabled=False)

    sync_companies_yaml(conn, path=yaml_file)

    row = get_source_by_name(conn, kind="greenhouse", account="atlassian")
    assert row.enabled is False


def test_sync_skips_unknown_kind_by_default(
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        greenhouse:
          - account: atlassian
            display_name: Atlassian
        not_a_real_ats:
          - account: foo
            display_name: Foo
        """,
    )

    report = sync_companies_yaml(conn, path=yaml_file)

    assert report.upserted == 1
    assert "not_a_real_ats" in report.skipped_unknown_kind
    sources = list_sources(conn)
    assert {s.kind for s in sources} == {"greenhouse"}


def test_sync_strict_raises_on_unknown_kind(
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        not_a_real_ats:
          - account: foo
            display_name: Foo
        """,
    )

    with pytest.raises(UnknownSourceKindError):
        sync_companies_yaml(conn, path=yaml_file, strict=True)


def test_sync_rejects_invalid_top_level_shape(
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    yaml_file = _write_yaml(tmp_path, "- not a mapping\n- also not\n")

    with pytest.raises(CompaniesYamlError):
        sync_companies_yaml(conn, path=yaml_file)


def test_sync_rejects_entry_missing_required_fields(
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    yaml_file = _write_yaml(
        tmp_path,
        """
        greenhouse:
          - account: atlassian
        """,
    )

    with pytest.raises(CompaniesYamlError, match="display_name"):
        sync_companies_yaml(conn, path=yaml_file)


def test_default_companies_yaml_loads_against_real_schema(
    conn: sqlite3.Connection,
) -> None:
    """The packaged seed file must be valid against the real registry."""
    report = sync_companies_yaml(conn, path=DEFAULT_COMPANIES_YAML)

    assert report.upserted >= 5
    assert report.skipped_unknown_kind == []
