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
    assert "greenhouse:anthropic" in result.output
    assert "TIER" in result.output


def test_source_list_empty_message_when_no_sources(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["source", "list"])
    assert result.exit_code == 0
    assert "no sources configured" in result.output


def test_source_disable_then_list_enabled_only(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    runner.invoke(app, ["source", "sync"])

    runner.invoke(app, ["source", "disable", "greenhouse:anthropic"])
    result = runner.invoke(app, ["source", "list", "--enabled"])

    assert result.exit_code == 0
    assert "greenhouse:anthropic" not in result.output
    assert "greenhouse:stripe" in result.output


def test_source_enable_re_enables_after_disable(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    runner.invoke(app, ["source", "sync"])

    runner.invoke(app, ["source", "disable", "greenhouse:anthropic"])
    runner.invoke(app, ["source", "enable", "greenhouse:anthropic"])

    conn = sqlite3.connect(_isolated_db)
    conn.row_factory = sqlite3.Row
    try:
        row = get_source_by_name(conn, kind="greenhouse", account="anthropic")
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


def test_infer_salary_command_runs_against_empty_db(_isolated_db: Path) -> None:
    """``jobai infer-salary`` on an empty DB reports zero updates, not a crash."""
    runner.invoke(app, ["migrate"])

    result = runner.invoke(app, ["infer-salary"])

    assert result.exit_code == 0, result.output
    assert "infer-salary: inspected 0, updated 0" in result.output


def test_infer_salary_command_fills_in_salary_from_description(
    _isolated_db: Path,
) -> None:
    """End-to-end CLI smoke: seed a row with a description that carries
    a salary marker, run ``jobai infer-salary``, confirm the row
    landed with the parsed numbers."""
    runner.invoke(app, ["migrate"])
    conn = sqlite3.connect(_isolated_db)
    try:
        conn.execute(
            "INSERT INTO jobs ("
            "  dedup_key, title, company, company_norm, apply_url, "
            "  description_text, first_seen_at, last_seen_at, fingerprint_json"
            ") VALUES ('k1', 'Engineer', 'Atlassian', 'atlassian', "
            "'https://e.example/1', "
            "'Salary: $120,000 - $160,000 per annum + super', "
            "datetime('now'), datetime('now'), '{}')",
        )
        conn.commit()
    finally:
        conn.close()

    result = runner.invoke(app, ["infer-salary"])
    assert result.exit_code == 0, result.output
    assert "updated 1" in result.output

    conn = sqlite3.connect(_isolated_db)
    try:
        row = conn.execute("SELECT salary_min, salary_max, salary_currency FROM jobs").fetchone()
        assert row == (120_000, 160_000, "AUD")
    finally:
        conn.close()


def test_infer_salary_command_accepts_limit_option(_isolated_db: Path) -> None:
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["infer-salary", "--limit", "10"])
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


def test_serve_command_invokes_uvicorn_with_configured_host_port(
    _isolated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``jobai serve`` delegates to uvicorn.run; monkey-patch the call
    so the test doesn't actually start a server."""
    from jobai import cli as cli_mod  # noqa: PLC0415

    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    import uvicorn  # noqa: PLC0415

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(cli_mod, "uvicorn", uvicorn)
    result = runner.invoke(
        app,
        ["serve", "--host", "0.0.0.0", "--port", "8888"],  # noqa: S104
    )
    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert captured["args"] == ("jobai.api.server:app",)
    assert kwargs["host"] == "0.0.0.0"  # noqa: S104
    assert kwargs["port"] == 8888
    assert kwargs["reload"] is False


def test_infer_remote_command_walks_and_reports(_isolated_db: Path) -> None:
    """``infer-remote`` against an empty DB reports zero updates cleanly."""
    runner.invoke(app, ["migrate"])
    result = runner.invoke(app, ["infer-remote"])
    assert result.exit_code == 0, result.output
    assert "infer-remote" in result.output


def test_source_sync_reports_skipped_unknown_kinds(
    _isolated_db: Path,
    tmp_path: Path,
) -> None:
    """When companies.yaml lists a kind not in the registry, ``source
    sync`` prints a 'skipped unknown kinds' line in addition to the
    upsert count."""
    runner.invoke(app, ["migrate"])
    yaml_file = tmp_path / "companies.yaml"
    yaml_file.write_text(
        "greenhouse:\n"
        "  - account: atlassian\n"
        "    display_name: Atlassian\n"
        "not_a_real_ats:\n"
        "  - account: foo\n"
        "    display_name: Foo\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["source", "sync", "--file", str(yaml_file)])
    assert result.exit_code == 0, result.output
    assert "skipped unknown kinds" in result.output
    assert "not_a_real_ats" in result.output


def test_split_name_rejects_missing_kind_or_account() -> None:
    """``_split_name`` raises BadParameter for any of: missing colon,
    empty kind ('@:account'), or empty account ('kind:')."""
    import typer  # noqa: PLC0415

    from jobai.cli import _split_name  # noqa: PLC0415

    with pytest.raises(typer.BadParameter):
        _split_name("missing-colon")
    with pytest.raises(typer.BadParameter):
        _split_name(":no-kind")
    with pytest.raises(typer.BadParameter):
        _split_name("no-account:")


def test_run_one_source_command_executes_runner_and_echoes_result(
    _isolated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``jobai run --source kind:account`` instantiates the source,
    runs it, and echoes a one-line summary. We monkey-patch the
    fetcher + run_source so the test doesn't hit the network."""
    from jobai import cli as cli_mod  # noqa: PLC0415
    from jobai.pipeline.runner import RunResult  # noqa: PLC0415

    runner.invoke(app, ["migrate"])
    conn = sqlite3.connect(_isolated_db)
    try:
        upsert_source(conn, kind="greenhouse", account="atlassian", display_name="Atlassian")
    finally:
        conn.close()

    class _FakeFetcher:
        async def aclose(self) -> None:
            return None

    def fake_build_fetcher(**kwargs: object) -> _FakeFetcher:
        del kwargs
        return _FakeFetcher()

    async def fake_run_source(**kwargs: object) -> RunResult:
        del kwargs
        return RunResult(
            run_id=1,
            status="success",
            items_seen=3,
            items_new=2,
            items_updated=1,
        )

    monkeypatch.setattr(cli_mod, "build_fetcher", fake_build_fetcher)
    monkeypatch.setattr(cli_mod, "run_source", fake_run_source)

    result = runner.invoke(app, ["run", "--source", "greenhouse:atlassian"])
    assert result.exit_code == 0, result.output
    assert "greenhouse:atlassian" in result.output
    assert "status=success" in result.output


def test_run_enabled_walks_every_enabled_source(
    _isolated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``jobai run --enabled`` iterates every enabled source row,
    runs each, and echoes one summary line per source."""
    from jobai import cli as cli_mod  # noqa: PLC0415
    from jobai.pipeline.runner import RunResult  # noqa: PLC0415

    runner.invoke(app, ["migrate"])
    conn = sqlite3.connect(_isolated_db)
    try:
        upsert_source(conn, kind="greenhouse", account="atlassian", display_name="Atlassian")
        upsert_source(conn, kind="lever", account="palantir", display_name="Palantir")
    finally:
        conn.close()

    class _FakeFetcher:
        async def aclose(self) -> None:
            return None

    def fake_build_fetcher(**kwargs: object) -> _FakeFetcher:
        del kwargs
        return _FakeFetcher()

    async def fake_run_source(**kwargs: object) -> RunResult:
        del kwargs
        return RunResult(
            run_id=1,
            status="success",
            items_seen=0,
            items_new=0,
            items_updated=0,
            error_summary="non-fatal",  # triggers the "  error:" echo branch
        )

    monkeypatch.setattr(cli_mod, "build_fetcher", fake_build_fetcher)
    monkeypatch.setattr(cli_mod, "run_source", fake_run_source)

    result = runner.invoke(app, ["run", "--enabled"])
    assert result.exit_code == 0, result.output
    assert "greenhouse:atlassian" in result.output
    assert "lever:palantir" in result.output
    assert "error: non-fatal" in result.output
