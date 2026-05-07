"""Tests for the cross-source fuzzy reconciliation pass."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.dedup.promote import promote_to_canonical_jobs
from jobai.dedup.reconcile import reconcile_fuzzy_duplicates
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
    return upsert_source(
        conn,
        kind="lever",
        account="atlassian-lever",
        display_name="Atlassian (Lever)",
    ).id


def _insert_jobs_raw(conn: sqlite3.Connection, source_id: int, ext_id: str) -> int:
    cursor = conn.execute(
        "INSERT INTO jobs_raw "
        "(source_id, source_external_id, raw_json, raw_sha256, first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
        (source_id, ext_id, "{}", "deadbeef"),
    )
    last_id = cursor.lastrowid
    assert last_id is not None
    conn.commit()
    return int(last_id)


def _seed_canonical_job(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    ext_id: str,
    title: str,
    company: str = "Atlassian",
    country: str | None = "Australia",
    apply_url: str | None = None,
    **extras: Any,
) -> int:
    """Promote a synthetic job into the canonical table and return its id."""
    raw_id = _insert_jobs_raw(conn, source_id, ext_id)
    job = NormalizedJob(
        source_external_id=ext_id,
        title=title,
        company=company,
        apply_url=apply_url or f"https://example.com/{ext_id}",
        raw_data={"id": ext_id},
        location_country=country,
        **extras,
    )
    result = promote_to_canonical_jobs(
        conn,
        source_id=source_id,
        jobs_raw_id=raw_id,
        job=job,
    )
    assert result.job_id is not None
    return result.job_id


# ---------------------------------------------------------------------------
# Group selection
# ---------------------------------------------------------------------------


def test_reconcile_returns_zero_merges_when_only_one_job_in_group(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    _seed_canonical_job(conn, source_id=source_id, ext_id="1", title="Backend Engineer")

    result = reconcile_fuzzy_duplicates(conn)

    assert result.pairs_merged == 0
    assert result.groups_examined == 1


def test_reconcile_does_not_merge_unrelated_titles(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    _seed_canonical_job(conn, source_id=source_id, ext_id="1", title="Backend Engineer")
    _seed_canonical_job(conn, source_id=source_id, ext_id="2", title="Marketing Manager")

    result = reconcile_fuzzy_duplicates(conn)

    assert result.pairs_merged == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_reconcile_does_not_merge_different_seniority(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    _seed_canonical_job(conn, source_id=source_id, ext_id="1", title="Senior Software Engineer")
    _seed_canonical_job(conn, source_id=source_id, ext_id="2", title="Staff Software Engineer")

    result = reconcile_fuzzy_duplicates(conn)

    assert result.pairs_merged == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# Merging behaviour
# ---------------------------------------------------------------------------


def test_reconcile_merges_abbreviation_variants_within_one_company(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    older_id = _seed_canonical_job(
        conn, source_id=source_id, ext_id="1", title="Senior Backend Engineer"
    )
    duplicate_id = _seed_canonical_job(
        conn, source_id=source_id, ext_id="2", title="Sr. Backend Engineer"
    )

    result = reconcile_fuzzy_duplicates(conn)

    assert result.pairs_merged == 1
    remaining = {row[0] for row in conn.execute("SELECT id FROM jobs")}
    assert older_id in remaining
    assert duplicate_id not in remaining


def test_reconcile_preserves_job_sources_links_during_merge(
    conn: sqlite3.Connection,
    source_id: int,
    alt_source_id: int,
) -> None:
    """Each duplicate's source links must move to the survivor."""
    older_id = _seed_canonical_job(
        conn, source_id=source_id, ext_id="1", title="Senior Backend Engineer"
    )
    _seed_canonical_job(conn, source_id=alt_source_id, ext_id="x", title="Sr. Backend Engineer")

    pre_total_links = conn.execute("SELECT COUNT(*) FROM job_sources").fetchone()[0]
    assert pre_total_links == 2  # one per canonical job before merge

    reconcile_fuzzy_duplicates(conn)

    post_links = conn.execute(
        "SELECT job_id, source_id FROM job_sources",
    ).fetchall()
    job_ids = {row[0] for row in post_links}
    source_ids = {row[1] for row in post_links}
    assert job_ids == {older_id}
    assert source_ids == {source_id, alt_source_id}


def test_reconcile_does_not_merge_jobs_in_different_company_groups(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="1",
        title="Sr. Backend Engineer",
        company="Atlassian",
    )
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="2",
        title="Senior Backend Engineer",
        company="Canva",
    )

    result = reconcile_fuzzy_duplicates(conn)

    assert result.pairs_merged == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_reconcile_does_not_merge_jobs_in_different_countries(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="1",
        title="Sr. Backend Engineer",
        country="Australia",
    )
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="2",
        title="Senior Backend Engineer",
        country="United States",
    )

    result = reconcile_fuzzy_duplicates(conn)

    assert result.pairs_merged == 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_reconcile_is_idempotent(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    _seed_canonical_job(conn, source_id=source_id, ext_id="1", title="Senior Backend Engineer")
    _seed_canonical_job(conn, source_id=source_id, ext_id="2", title="Sr. Backend Engineer")

    first = reconcile_fuzzy_duplicates(conn)
    second = reconcile_fuzzy_duplicates(conn)

    assert first.pairs_merged == 1
    assert second.pairs_merged == 0
    # Final state: one canonical job, one job_sources row per source link
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def test_reconcile_skips_jobs_outside_window(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """A job whose last_seen_at is older than the window is not considered."""
    older_id = _seed_canonical_job(
        conn, source_id=source_id, ext_id="1", title="Senior Backend Engineer"
    )
    duplicate_id = _seed_canonical_job(
        conn, source_id=source_id, ext_id="2", title="Sr. Backend Engineer"
    )

    # Push the older job's last_seen_at far into the past.
    conn.execute(
        "UPDATE jobs SET last_seen_at = datetime('now', '-30 days') WHERE id = ?",
        (older_id,),
    )
    conn.commit()

    result = reconcile_fuzzy_duplicates(conn, window_days=14)

    # Only the duplicate is in the window; with no peer to compare against
    # within the window, no merge happens.
    assert result.pairs_merged == 0
    remaining = {row[0] for row in conn.execute("SELECT id FROM jobs")}
    assert older_id in remaining
    assert duplicate_id in remaining


# ---------------------------------------------------------------------------
# FTS5 sync
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Best-of field merging during reconcile
#
# The pre-merger reconcile pass kept the survivor's field values verbatim and
# threw away the duplicate's data on every merge. That meant a salary the
# duplicate uniquely had — but the survivor lacked — was lost forever. The
# tests below pin the new contract: the merger picks the best value per
# field across both rows before deleting the duplicate.
# ---------------------------------------------------------------------------


def test_reconcile_fills_in_missing_salary_from_duplicate(
    conn: sqlite3.Connection,
    source_id: int,
    alt_source_id: int,
) -> None:
    """Survivor has no salary; the fuzzy duplicate does. Merge must
    promote the duplicate's salary onto the survivor before deletion."""
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="1",
        title="Senior Backend Engineer",
    )
    _seed_canonical_job(
        conn,
        source_id=alt_source_id,
        ext_id="2",
        title="Sr. Backend Engineer",
        salary_min=150000,
        salary_max=180000,
        salary_currency="AUD",
    )

    reconcile_fuzzy_duplicates(conn)

    rows = conn.execute("SELECT salary_min, salary_max, salary_currency FROM jobs").fetchall()
    assert len(rows) == 1
    survivor = rows[0]
    assert survivor["salary_min"] == 150000
    assert survivor["salary_max"] == 180000
    assert survivor["salary_currency"] == "AUD"


def test_reconcile_keeps_richer_description_from_duplicate(
    conn: sqlite3.Connection,
    source_id: int,
    alt_source_id: int,
) -> None:
    """If the duplicate carries the full description and the survivor
    only has a teaser, reconcile must keep the longer text."""
    teaser = "Senior Python role at Atlassian. Apply now."
    full = "Senior Python Engineer at Atlassian working on async services. " * 10

    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="1",
        title="Senior Backend Engineer",
        description_text=teaser,
    )
    _seed_canonical_job(
        conn,
        source_id=alt_source_id,
        ext_id="2",
        title="Sr. Backend Engineer",
        description_text=full,
    )

    reconcile_fuzzy_duplicates(conn)

    row = conn.execute("SELECT description_text FROM jobs").fetchone()
    assert row["description_text"] == full


def test_reconcile_keeps_earliest_posted_at_from_duplicate(
    conn: sqlite3.Connection,
    source_id: int,
    alt_source_id: int,
) -> None:
    """When the duplicate carries an earlier posted_at than the survivor
    (e.g. the original posting that the survivor was a re-list of),
    reconcile must keep the earlier date."""
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="1",
        title="Senior Backend Engineer",
        posted_at="2026-05-07T00:00:00+00:00",
    )
    _seed_canonical_job(
        conn,
        source_id=alt_source_id,
        ext_id="2",
        title="Sr. Backend Engineer",
        posted_at="2026-05-01T00:00:00+00:00",
    )

    reconcile_fuzzy_duplicates(conn)

    row = conn.execute("SELECT posted_at FROM jobs").fetchone()
    assert row["posted_at"] == "2026-05-01T00:00:00+00:00"


def test_reconcile_keeps_fts_index_consistent_after_merge(
    conn: sqlite3.Connection,
    source_id: int,
) -> None:
    """The DELETE trigger must remove the merged-away row from jobs_fts."""
    _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="1",
        title="Senior Backend Engineer",
        description_text="Distinctive token alpharaptor",
    )
    duplicate_id = _seed_canonical_job(
        conn,
        source_id=source_id,
        ext_id="2",
        title="Sr. Backend Engineer",
        description_text="Distinctive token betacore",
    )

    reconcile_fuzzy_duplicates(conn)

    # The merged-away row's distinctive description token must NOT be in
    # the FTS index any more (DELETE trigger fired).
    duplicate_matches = conn.execute(
        "SELECT 1 FROM jobs_fts WHERE jobs_fts MATCH 'betacore'"
    ).fetchall()
    assert duplicate_matches == []

    # The survivor's description token IS still searchable.
    survivor_matches = conn.execute(
        "SELECT 1 FROM jobs_fts WHERE jobs_fts MATCH 'alpharaptor'"
    ).fetchall()
    assert len(survivor_matches) == 1

    # The merged-away job should not appear in jobs at all.
    duplicate_rows = conn.execute(
        "SELECT 1 FROM jobs WHERE id = ?",
        (duplicate_id,),
    ).fetchall()
    assert duplicate_rows == []
