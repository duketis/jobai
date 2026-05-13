"""Coverage for the ATS slug-discovery helper.

The helper walks ``jobs.apply_url`` and pulls company slugs out of
ATS-host-pointing URLs. Tests exercise every supported pattern + the
diff helper + the seeded-accounts loader.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.db.migrations import apply_pending
from jobai.sources.ats_discovery import (
    SlugCount,
    diff_against_seeded,
    discover_slugs,
    load_seeded_accounts,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = tmp_path / "discovery.db"
    connection = sqlite3.connect(db)
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


def _seed_job(conn: sqlite3.Connection, ext_id: str, apply_url: str) -> None:
    """Insert one ``sources`` + ``jobs_raw`` + ``jobs`` triple."""
    cursor = conn.execute(
        "INSERT INTO sources (kind, account, display_name, cadence_seconds) "
        "VALUES (?, ?, ?, 3600) ON CONFLICT (kind, account) DO NOTHING",
        ("greenhouse", "seed", "Seed"),
    )
    source_id = (
        cursor.lastrowid
        or conn.execute(
            "SELECT id FROM sources WHERE kind='greenhouse' AND account='seed'",
        ).fetchone()[0]
    )
    raw_cursor = conn.execute(
        "INSERT INTO jobs_raw (source_id, source_external_id, raw_json, raw_sha256, "
        "first_seen_at, last_seen_at) "
        "VALUES (?, ?, '{}', 'sha', datetime('now'), datetime('now'))",
        (source_id, ext_id),
    )
    conn.execute(
        "INSERT INTO jobs (dedup_key, title, company, company_norm, apply_url, "
        "first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        (ext_id, "Engineer", "Co", "co", apply_url),
    )
    conn.execute(
        "INSERT INTO job_sources (job_id, source_id, jobs_raw_id, apply_url) "
        "VALUES ((SELECT id FROM jobs WHERE dedup_key=?), ?, ?, ?)",
        (ext_id, source_id, raw_cursor.lastrowid, apply_url),
    )
    conn.commit()


def test_discover_slugs_returns_empty_when_no_apply_urls_match(
    conn: sqlite3.Connection,
) -> None:
    """No apply URLs at all -> empty list."""
    assert discover_slugs(conn) == []


def test_discover_slugs_picks_up_each_supported_ats(
    conn: sqlite3.Connection,
) -> None:
    """One URL per provider; helper returns one SlugCount per (kind, account)."""
    urls = {
        "smartrecruiters": "https://jobs.smartrecruiters.com/SEEK/744000125234569",
        "greenhouse": "https://boards.greenhouse.io/cloudflare/jobs/9001",
        "lever": "https://jobs.lever.co/palantir/abc-123",
        "ashby": "https://jobs.ashbyhq.com/openai/role-x",
        "workable": "https://apply.workable.com/example-co/j/ABC",
    }
    for i, (_, url) in enumerate(urls.items()):
        _seed_job(conn, f"ext-{i}", url)
    discovered = {(s.kind, s.account, s.count) for s in discover_slugs(conn)}
    assert discovered == {
        ("smartrecruiters", "SEEK", 1),
        ("greenhouse", "cloudflare", 1),
        ("lever", "palantir", 1),
        ("ashby", "openai", 1),
        ("workable", "example-co", 1),
    }


def test_discover_slugs_aggregates_counts_by_slug(conn: sqlite3.Connection) -> None:
    """The same slug across multiple distinct apply URLs aggregates."""
    _seed_job(conn, "a", "https://boards.greenhouse.io/anthropic/jobs/1")
    _seed_job(conn, "b", "https://boards.greenhouse.io/anthropic/jobs/2")
    _seed_job(conn, "c", "https://boards.greenhouse.io/figma/jobs/1")
    out = discover_slugs(conn)
    by_account = {s.account: s.count for s in out if s.kind == "greenhouse"}
    assert by_account == {"anthropic": 2, "figma": 1}


def test_discover_slugs_skips_unrelated_urls(conn: sqlite3.Connection) -> None:
    """A non-ATS URL contributes nothing to the discovery."""
    _seed_job(conn, "a", "https://example.com/jobs/123")
    _seed_job(conn, "b", "https://seek.com.au/job/55555")
    assert discover_slugs(conn) == []


def test_diff_against_seeded_drops_already_seeded_slugs() -> None:
    discovered = [
        SlugCount(kind="greenhouse", account="anthropic", count=5),
        SlugCount(kind="greenhouse", account="figma", count=2),
        SlugCount(kind="lever", account="palantir", count=3),
    ]
    seeded = {"greenhouse": {"anthropic"}, "lever": set()}
    new = diff_against_seeded(discovered, seeded)
    accounts = {(s.kind, s.account) for s in new}
    assert accounts == {("greenhouse", "figma"), ("lever", "palantir")}


def test_diff_against_seeded_is_case_insensitive() -> None:
    """ATS APIs resolve slugs case-insensitively (eg SR returns the same
    jobs for ``canva`` and ``Canva``). A capital-case slug discovered
    in apply URLs must NOT re-seed a lowercase entry already registered."""
    discovered = [
        SlugCount(kind="smartrecruiters", account="Canva", count=300),
        SlugCount(kind="smartrecruiters", account="Visa", count=21),
        SlugCount(kind="smartrecruiters", account="NewCo", count=5),
    ]
    seeded = {"smartrecruiters": {"canva", "visa"}}
    new = diff_against_seeded(discovered, seeded)
    assert [s.account for s in new] == ["NewCo"]


def test_load_seeded_accounts_groups_by_kind(conn: sqlite3.Connection) -> None:
    """The repository call shapes every ``sources`` row by kind so the
    diff helper can ignore already-registered slugs."""
    conn.execute(
        "INSERT INTO sources (kind, account, display_name, cadence_seconds) "
        "VALUES ('greenhouse', 'anthropic', 'Anthropic', 3600), "
        "       ('greenhouse', 'figma', 'Figma', 3600), "
        "       ('lever', 'palantir', 'Palantir', 3600)",
    )
    conn.commit()
    out = load_seeded_accounts(conn)
    assert out["greenhouse"] == {"anthropic", "figma"}
    assert out["lever"] == {"palantir"}
    assert out["ashby"] == set()
