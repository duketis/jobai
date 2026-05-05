"""Tests for promotion of NormalizedJob into the canonical jobs table.

The tests run against a real migrated SQLite DB so the FTS5 sync
triggers fire alongside the inserts and we can verify the search
index stays consistent.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.dedup.promote import promote_to_canonical_jobs
from jobai.sources.base import NormalizedJob
from jobai.sources.repository import upsert_source


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


@pytest.fixture
def source_id(conn: sqlite3.Connection) -> int:
    return upsert_source(
        conn,
        kind="greenhouse",
        account="atlassian",
        display_name="Atlassian",
    ).id


@pytest.fixture
def alt_source_id(conn: sqlite3.Connection) -> int:
    """A second source so we can test cross-source linkage."""
    return upsert_source(
        conn,
        kind="lever",
        account="atlassian-lever",
        display_name="Atlassian (Lever)",
    ).id


def _insert_jobs_raw(conn: sqlite3.Connection, source_id: int, external_id: str) -> int:
    cursor = conn.execute(
        "INSERT INTO jobs_raw "
        "(source_id, source_external_id, raw_json, raw_sha256, first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
        (source_id, external_id, "{}", "deadbeef"),
    )
    last_id = cursor.lastrowid
    assert last_id is not None
    conn.commit()
    return int(last_id)


def _make_job(**overrides: Any) -> NormalizedJob:
    base: dict[str, Any] = {
        "source_external_id": "1",
        "title": "Senior Backend Engineer",
        "company": "Atlassian",
        "apply_url": "https://example.com/apply/1",
        "raw_data": {"id": 1},
        "location_raw": "Sydney, Australia",
        "location_country": "Australia",
    }
    base.update(overrides)
    return NormalizedJob(**base)


# ---------------------------------------------------------------------------
# Basic insertion
# ---------------------------------------------------------------------------


def test_promote_inserts_new_canonical_job(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job()

    result = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )

    assert result.was_new is True
    assert result.job_id is not None

    rows = conn.execute("SELECT title, company FROM jobs").fetchall()
    assert len(rows) == 1
    assert rows[0]["title"] == "Senior Backend Engineer"


def test_promote_creates_job_sources_link(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job()

    result = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )

    rows = conn.execute(
        "SELECT job_id, source_id, jobs_raw_id, apply_url FROM job_sources",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["job_id"] == result.job_id
    assert rows[0]["source_id"] == source_id
    assert rows[0]["jobs_raw_id"] == raw_id


# ---------------------------------------------------------------------------
# Idempotency / dedup
# ---------------------------------------------------------------------------


def test_promote_is_idempotent_for_same_source(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """Re-running promotion for the same job_raw must not duplicate the job."""
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job()

    first = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )
    second = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )

    assert first.was_new is True
    assert second.was_new is False
    assert first.job_id == second.job_id

    job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM job_sources").fetchone()[0]
    assert job_count == 1
    assert link_count == 1  # PRIMARY KEY makes link upsert idempotent


def test_promote_merges_same_role_from_different_sources(
    conn: sqlite3.Connection,
    source_id: int,
    alt_source_id: int,
) -> None:
    """Same role on two sources -> one canonical job, two job_sources rows."""
    raw_a = _insert_jobs_raw(conn, source_id, "1")
    raw_b = _insert_jobs_raw(conn, alt_source_id, "x")

    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_a,
        job=_make_job(),
    )
    promote_to_canonical_jobs(
        conn,
        source_id=alt_source_id,
        jobs_raw_id=raw_b,
        job=_make_job(source_external_id="x", apply_url="https://lever.example/2"),
    )

    job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM job_sources").fetchone()[0]
    assert job_count == 1
    assert link_count == 2


def test_promote_collapses_company_suffix_variants(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """Atlassian, Atlassian Pty Ltd, ATLASSIAN INC. -> one canonical job."""
    raw_a = _insert_jobs_raw(conn, source_id, "1")
    raw_b = _insert_jobs_raw(conn, source_id, "2")
    raw_c = _insert_jobs_raw(conn, source_id, "3")

    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_a,
        job=_make_job(company="Atlassian"),
    )
    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_b,
        job=_make_job(source_external_id="2", company="Atlassian Pty Ltd"),
    )
    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_c,
        job=_make_job(source_external_id="3", company="ATLASSIAN INC."),
    )

    job_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert job_count == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_promote_skips_when_company_missing(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job(company="")

    result = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )

    assert result.was_skipped is True
    assert result.job_id is None
    assert result.skipped_reason is not None
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_promote_skips_when_title_missing(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job(title="")

    result = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )

    assert result.was_skipped is True
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Mutable field updates
# ---------------------------------------------------------------------------


def test_promote_updates_last_seen_at_on_repeat(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job()

    first = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )
    first_seen = conn.execute(
        "SELECT first_seen_at, last_seen_at FROM jobs WHERE id = ?",
        (first.job_id,),
    ).fetchone()

    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )
    refreshed = conn.execute(
        "SELECT first_seen_at, last_seen_at FROM jobs WHERE id = ?",
        (first.job_id,),
    ).fetchone()

    assert refreshed["first_seen_at"] == first_seen["first_seen_at"]
    assert refreshed["last_seen_at"] >= first_seen["last_seen_at"]


def test_promote_refreshes_salary_when_newly_disclosed(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")

    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=_make_job(),
    )
    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=_make_job(salary_min=120000, salary_max=180000, salary_currency="AUD"),
    )

    row = conn.execute(
        "SELECT salary_min, salary_max, salary_currency FROM jobs",
    ).fetchone()
    assert row["salary_min"] == 120000
    assert row["salary_max"] == 180000
    assert row["salary_currency"] == "AUD"


# ---------------------------------------------------------------------------
# FTS5 sync (verifies the schema's triggers fire on inserts/updates)
# ---------------------------------------------------------------------------


def test_promote_inserts_into_fts_index(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")
    job = _make_job(
        title="Python Backend Engineer",
        description_text="Build async Python services on AWS.",
    )

    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )

    matches = conn.execute(
        "SELECT j.title FROM jobs j "
        "JOIN jobs_fts fts ON fts.rowid = j.id "
        "WHERE jobs_fts MATCH 'python AND backend'"
    ).fetchall()
    assert len(matches) == 1
    assert matches[0]["title"] == "Python Backend Engineer"


def test_promote_updates_fts_index_when_description_changes(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    raw_id = _insert_jobs_raw(conn, source_id, "1")

    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=_make_job(description_text="Java Spring Boot service work."),
    )
    # An identical job (same dedup key) with a NEW description text.
    promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=_make_job(description_text="Migrate the service to Kotlin."),
    )

    java_matches = conn.execute("SELECT 1 FROM jobs_fts WHERE jobs_fts MATCH 'java'").fetchall()
    kotlin_matches = conn.execute("SELECT 1 FROM jobs_fts WHERE jobs_fts MATCH 'kotlin'").fetchall()
    assert len(java_matches) == 0  # old text is gone
    assert len(kotlin_matches) == 1
