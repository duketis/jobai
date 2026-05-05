"""Tests for the Typer CLI."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jobai.cli import app
from jobai.config import get_settings
from jobai.db.migrations import apply_pending
from jobai.sources.repository import get_source_by_name, upsert_source

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@pytest.fixture(autouse=True)
def _isolated_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Each test gets its own DB; settings cache is cleared so the path takes."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("JOBAI_DB_PATH", str(db_path))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield db_path
    get_settings.cache_clear()


def _migrate_directly(db_path: Path) -> None:
    """Apply migrations without going through the CLI, for tests that need a
    pre-migrated DB before invoking other commands."""
    conn = sqlite3.connect(db_path)
    try:
        apply_pending(conn)
    finally:
        conn.close()


def test_migrate_command_creates_tables(_isolated_db: Path) -> None:
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(_isolated_db)
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()

    assert "sources" in tables
    assert "jobs_raw" in tables
    assert "raw_responses" in tables


def test_source_sync_loads_packaged_yaml(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])

    result = runner.invoke(app, ["source", "sync"])

    assert result.exit_code == 0, result.output
    assert "upserted" in result.output

    conn = sqlite3.connect(_isolated_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    finally:
        conn.close()
    assert count >= 5


def test_source_list_displays_synced_sources(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    runner.invoke(app, ["source", "sync"])

    result = runner.invoke(app, ["source", "list"])

    assert result.exit_code == 0
    assert "greenhouse:atlassian" in result.output
    assert "TIER" in result.output


def test_source_list_empty_message_when_no_sources(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["source", "list"])
    assert result.exit_code == 0
    assert "no sources configured" in result.output


def test_source_disable_then_list_enabled_only(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    runner.invoke(app, ["source", "sync"])

    runner.invoke(app, ["source", "disable", "greenhouse:atlassian"])
    result = runner.invoke(app, ["source", "list", "--enabled"])

    assert result.exit_code == 0
    assert "greenhouse:atlassian" not in result.output
    assert "greenhouse:canva" in result.output


def test_source_enable_re_enables_after_disable(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    runner.invoke(app, ["source", "sync"])

    runner.invoke(app, ["source", "disable", "greenhouse:atlassian"])
    runner.invoke(app, ["source", "enable", "greenhouse:atlassian"])

    conn = sqlite3.connect(_isolated_db)
    conn.row_factory = sqlite3.Row
    try:
        row = get_source_by_name(conn, kind="greenhouse", account="atlassian")
    finally:
        conn.close()
    assert row.enabled is True


def test_source_disable_with_invalid_name_format_fails(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["source", "disable", "not-a-valid-format"])
    assert result.exit_code != 0
    assert "invalid source name" in result.output.lower()


def test_run_requires_either_source_or_enabled(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "specify either" in result.output.lower()


def test_run_rejects_both_source_and_enabled(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["run", "--source", "greenhouse:atlassian", "--enabled"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_serve_help_lists_host_port_reload_flags(_isolated_db: Path) -> None:
    """The serve command must expose host/port/reload — verify via --help so
    the test does not actually start a long-running server.

    Rich's help formatter intersperses ANSI escape codes inside option
    names (so '--host' appears as '--' + style escapes + 'host'). We
    strip ANSI before substring-asserting, which is platform-stable.
    """
    result = runner.invoke(app, ["serve", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    plain = _ANSI_RE.sub("", result.output)
    assert "--host" in plain
    assert "--port" in plain
    assert "--reload" in plain


def test_reconcile_command_runs_against_empty_db(_isolated_db: Path) -> None:
    """`jobai reconcile` on an empty DB should report zero merges, not crash."""
    runner.invoke(app, ["migrate"])

    result = runner.invoke(app, ["reconcile"])

    assert result.exit_code == 0, result.output
    assert "merged 0 pair" in result.output


def test_reconcile_command_accepts_threshold_and_window_options(
    _isolated_db: Path,
) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["reconcile", "--window", "30", "--threshold", "90"])
    assert result.exit_code == 0, result.output


def test_run_unknown_source_fails_cleanly(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])

    # Pre-create a source row that points at a kind we don't have a class for.
    conn = sqlite3.connect(_isolated_db)
    try:
        upsert_source(
            conn,
            kind="not-a-real-ats",
            account="x",
            display_name="X",
        )
    finally:
        conn.close()

    result = runner.invoke(app, ["run", "--source", "not-a-real-ats:x"])
    assert result.exit_code != 0
