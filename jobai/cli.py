"""jobai command-line interface.

Entry points (registered in pyproject.toml as the ``jobai`` script):

* ``jobai migrate`` — apply pending database migrations.
* ``jobai source sync [--file path]`` — upsert sources from companies.yaml.
* ``jobai source list [--enabled]`` — pretty-print configured sources.
* ``jobai source enable NAME`` / ``jobai source disable NAME``.
* ``jobai run --source NAME`` — run one source ad hoc.
* ``jobai run --enabled`` — run every enabled source sequentially.

Long-running supervision (``jobai serve`` with the scheduler) lands in
a later phase; this CLI is the authoring surface, not the deployment
surface.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from jobai.config import get_settings
from jobai.db.connection import connect
from jobai.db.migrations import apply_pending
from jobai.dedup.fuzzy import DEFAULT_SIMILARITY_THRESHOLD
from jobai.dedup.reconcile import DEFAULT_WINDOW_DAYS, reconcile_fuzzy_duplicates
from jobai.fetcher.http import HttpFetcher
from jobai.observability.logging import configure_logging, get_logger
from jobai.pipeline.runner import RunResult, run_source
from jobai.sources.base import BaseSource
from jobai.sources.loader import DEFAULT_COMPANIES_YAML, sync_companies_yaml
from jobai.sources.registry import get_source_class
from jobai.sources.repository import (
    SourceRow,
    get_source_by_name,
    list_sources,
    set_enabled,
)

app = typer.Typer(
    name="jobai",
    help="Local-first AI job-hunting agent — data layer CLI.",
    no_args_is_help=True,
)

source_app = typer.Typer(
    name="source",
    help="Manage configured sources.",
    no_args_is_help=True,
)
app.add_typer(source_app)


# ---------------------------------------------------------------------------
# top-level commands
# ---------------------------------------------------------------------------


@app.command()
def migrate() -> None:
    """Apply pending database migrations."""
    settings = get_settings()
    with connect(settings.db_path) as conn:
        applied = apply_pending(conn)
    typer.echo(f"applied {len(applied)} migration(s)")


@app.command()
def reconcile(
    window: int = typer.Option(
        DEFAULT_WINDOW_DAYS,
        "--window",
        help="Only consider jobs whose last_seen_at is within this many days.",
    ),
    threshold: int = typer.Option(
        DEFAULT_SIMILARITY_THRESHOLD,
        "--threshold",
        help="Fuzzy similarity threshold (0-100); higher = stricter.",
    ),
) -> None:
    """Run the cross-source fuzzy reconciliation pass."""
    settings = get_settings()
    configure_logging(level=settings.log_level)
    with connect(settings.db_path) as conn:
        result = reconcile_fuzzy_duplicates(
            conn,
            window_days=window,
            threshold=threshold,
        )
    typer.echo(
        f"reconcile: examined {result.groups_examined} group(s), "
        f"merged {result.pairs_merged} pair(s)"
    )


@app.command()
def run(
    source: str | None = typer.Option(
        None,
        "--source",
        help="Source name like 'greenhouse:atlassian'. Mutually exclusive with --enabled.",
    ),
    enabled: bool = typer.Option(
        False,
        "--enabled",
        help="Run every enabled source sequentially.",
    ),
) -> None:
    """Run one or more sources ad hoc."""
    if source is None and not enabled:
        raise typer.BadParameter("specify either --source NAME or --enabled")
    if source is not None and enabled:
        raise typer.BadParameter("--source and --enabled are mutually exclusive")

    settings = get_settings()
    configure_logging(level=settings.log_level)

    if source is not None:
        result = asyncio.run(_run_one_by_name(settings.db_path, name=source))
        _echo_result(source, result)
    else:
        results = asyncio.run(_run_all_enabled(settings.db_path))
        for name, result in results:
            _echo_result(name, result)


# ---------------------------------------------------------------------------
# source subcommands
# ---------------------------------------------------------------------------


@source_app.command("sync")
def source_sync(
    file: Path = typer.Option(
        DEFAULT_COMPANIES_YAML,
        "--file",
        "-f",
        help="Path to a companies.yaml file.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on unknown source kinds rather than skipping them.",
    ),
) -> None:
    """Upsert source rows from a companies.yaml file."""
    settings = get_settings()
    with connect(settings.db_path) as conn:
        report = sync_companies_yaml(conn, path=file, strict=strict)

    typer.echo(f"upserted {report.upserted} source(s)")
    if report.skipped_unknown_kind:
        typer.echo(f"skipped unknown kinds: {', '.join(report.skipped_unknown_kind)}")


@source_app.command("list")
def source_list(
    enabled_only: bool = typer.Option(
        False,
        "--enabled",
        help="Only list enabled sources.",
    ),
) -> None:
    """List configured sources."""
    settings = get_settings()
    with connect(settings.db_path) as conn:
        rows = list_sources(conn, enabled_only=enabled_only)

    if not rows:
        typer.echo("(no sources configured — run 'jobai source sync' first)")
        return

    typer.echo(f"{'NAME':<40} {'TIER':<5} {'CADENCE':<10} ENABLED")
    for row in rows:
        typer.echo(f"{row.name:<40} {row.default_tier:<5} {row.cadence_seconds:<10} {row.enabled}")


@source_app.command("enable")
def source_enable(name: str = typer.Argument(..., help="Source name like 'kind:account'.")) -> None:
    """Enable a source."""
    _toggle(name, enabled=True)
    typer.echo(f"enabled {name}")


@source_app.command("disable")
def source_disable(
    name: str = typer.Argument(..., help="Source name like 'kind:account'."),
) -> None:
    """Disable a source."""
    _toggle(name, enabled=False)
    typer.echo(f"disabled {name}")


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _toggle(name: str, *, enabled: bool) -> None:
    kind, account = _split_name(name)
    settings = get_settings()
    with connect(settings.db_path) as conn:
        set_enabled(conn, kind=kind, account=account, enabled=enabled)


def _split_name(name: str) -> tuple[str, str]:
    """Parse 'kind:account' into (kind, account)."""
    if ":" not in name:
        raise typer.BadParameter(
            f"invalid source name {name!r}: expected 'kind:account' (e.g. 'greenhouse:atlassian')"
        )
    kind, _, account = name.partition(":")
    if not kind or not account:
        raise typer.BadParameter(f"invalid source name {name!r}")
    return kind, account


def _instantiate_source(row: SourceRow) -> BaseSource:
    cls = get_source_class(row.kind)
    return cls(row.account)


async def _run_one_by_name(db_path: Path, *, name: str) -> RunResult:
    kind, account = _split_name(name)
    log = get_logger("jobai.cli")
    with connect(db_path) as conn:
        source_row = get_source_by_name(conn, kind=kind, account=account)
        source = _instantiate_source(source_row)
        async with HttpFetcher() as fetcher:
            result = await run_source(
                conn=conn,
                source=source,
                source_row=source_row,
                fetcher=fetcher,
            )
    log.info("cli_run_complete", source=name, result=result)
    return result


async def _run_all_enabled(db_path: Path) -> list[tuple[str, RunResult]]:
    log = get_logger("jobai.cli")
    results: list[tuple[str, RunResult]] = []
    with connect(db_path) as conn:
        rows = list_sources(conn, enabled_only=True)

    for source_row in rows:
        source = _instantiate_source(source_row)
        async with HttpFetcher() as fetcher:
            with connect(db_path) as conn:
                result = await run_source(
                    conn=conn,
                    source=source,
                    source_row=source_row,
                    fetcher=fetcher,
                )
        log.info("cli_run_complete", source=source_row.name, result=result)
        results.append((source_row.name, result))

    return results


def _echo_result(name: str, result: RunResult) -> None:
    typer.echo(
        f"{name}: status={result.status} "
        f"seen={result.items_seen} new={result.items_new} updated={result.items_updated}"
    )
    if result.error_summary:
        typer.echo(f"  error: {result.error_summary}")
