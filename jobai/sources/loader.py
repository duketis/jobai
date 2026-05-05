"""Bulk-load source rows from ``companies.yaml`` into the sources table.

The loader is what powers ``jobai source sync``. It reads the YAML
seed file (kind -> list of entries), validates each entry, and upserts
through the repository. Sync is idempotent: rerunning leaves enabled
flags untouched (see :func:`jobai.sources.repository.upsert_source`).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jobai.sources.registry import UnknownSourceKindError, get_source_class
from jobai.sources.repository import upsert_source

DEFAULT_COMPANIES_YAML = Path(__file__).parent / "companies.yaml"


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Result of one ``sync_companies_yaml`` invocation."""

    upserted: int
    skipped_unknown_kind: list[str]


class CompaniesYamlError(ValueError):
    """Raised when the YAML structure or per-entry shape is invalid."""


def sync_companies_yaml(
    conn: sqlite3.Connection,
    *,
    path: Path = DEFAULT_COMPANIES_YAML,
    strict: bool = False,
) -> SyncReport:
    """Read ``path`` and upsert each entry into the ``sources`` table.

    Args:
        conn: an open SQLite connection (typically from
            :func:`jobai.db.connection.connect`).
        path: location of the YAML file. Defaults to the packaged
            ``companies.yaml`` next to this module.
        strict: when True, an unknown kind raises immediately. When
            False (default), unknown kinds are skipped and reported in
            :attr:`SyncReport.skipped_unknown_kind` so adding a new
            kind to ``companies.yaml`` does not break sync until the
            class is registered.

    Returns:
        A :class:`SyncReport` describing the outcome.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CompaniesYamlError(
            f"top-level YAML must be a mapping of kind -> list[entry], got {type(raw).__name__}"
        )

    upserted = 0
    skipped: list[str] = []

    for kind, entries in raw.items():
        if not isinstance(kind, str):
            raise CompaniesYamlError(f"non-string kind in YAML: {kind!r}")

        try:
            get_source_class(kind)
        except UnknownSourceKindError:
            if strict:
                raise
            skipped.append(kind)
            continue

        if not isinstance(entries, list):
            raise CompaniesYamlError(
                f"entries for kind {kind!r} must be a list, got {type(entries).__name__}"
            )

        for entry in entries:
            _validate_entry(kind, entry)
            upsert_source(
                conn,
                kind=kind,
                account=entry["account"],
                display_name=entry["display_name"],
                default_tier=entry.get("default_tier", 1),
                enabled=entry.get("enabled", True),
                cadence_seconds=entry.get("cadence_seconds", 1800),
                config=entry.get("config"),
            )
            upserted += 1

    return SyncReport(upserted=upserted, skipped_unknown_kind=skipped)


def _validate_entry(kind: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise CompaniesYamlError(
            f"each entry under {kind!r} must be a mapping, got {type(entry).__name__}"
        )
    for required_key in ("account", "display_name"):
        if required_key not in entry:
            raise CompaniesYamlError(
                f"entry under {kind!r} missing required field {required_key!r}: {entry!r}"
            )
        if not isinstance(entry[required_key], str) or not entry[required_key]:
            raise CompaniesYamlError(
                f"entry under {kind!r} has invalid {required_key!r}: {entry[required_key]!r}"
            )
