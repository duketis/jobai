"""Tests for posted_at normalisation + backfill.

These nail down the contract that ``jobs.posted_at`` is *always*
ISO-8601 UTC (or NULL) once a row lands canonically — never the raw
"8d ago" / "4 days ago" / "Just posted" text the boards emit. That
raw text broke both the ``posted_newest`` sort (lexical string sort
over mixed garbage) and the UI's relative-time formatter (which fell
back to printing the raw string verbatim).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jobai.db.migrations import apply_pending
from jobai.pipeline.posted_at_normalisation import (
    backfill_posted_at,
    normalise_posted_at,
)

# A fixed reference instant so relative-string maths is deterministic.
NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "posted.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# normalise_posted_at — single-call parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_empty_inputs_return_none(raw: str | None) -> None:
    assert normalise_posted_at(raw, now=NOW) is None


@pytest.mark.parametrize(
    "raw",
    [
        "see job description",
        "ASAP",
        "Closing soon",
        "Refer to advert",
        "n/a",
        # All-digit but not a plausible epoch (10s / 13ms) — a bare
        # year or a short id, not a timestamp. We don't guess.
        "2026",
        "12345",
        "999999999999",
    ],
)
def test_unparseable_text_returns_none(raw: str) -> None:
    assert normalise_posted_at(raw, now=NOW) is None


def test_iso_with_offset_normalised_to_utc() -> None:
    out = normalise_posted_at("2026-05-14T10:21:48.831265+00:00", now=NOW)
    assert out == datetime(2026, 5, 14, 10, 21, 48, 831265, tzinfo=UTC).isoformat()


def test_iso_with_z_suffix() -> None:
    out = normalise_posted_at("2026-05-14T10:21:48Z", now=NOW)
    assert out == datetime(2026, 5, 14, 10, 21, 48, tzinfo=UTC).isoformat()


def test_iso_with_non_utc_offset_converted_to_utc() -> None:
    # +10:00 (AEST) → subtract 10h to land in UTC.
    out = normalise_posted_at("2026-05-14T20:00:00+10:00", now=NOW)
    assert out == datetime(2026, 5, 14, 10, 0, 0, tzinfo=UTC).isoformat()


def test_naive_iso_assumed_utc() -> None:
    out = normalise_posted_at("2026-05-14T10:21:48", now=NOW)
    assert out == datetime(2026, 5, 14, 10, 21, 48, tzinfo=UTC).isoformat()


def test_sqlite_space_separated_timestamp() -> None:
    # first_seen_at-style value (datetime('now') → "YYYY-MM-DD HH:MM:SS").
    out = normalise_posted_at("2026-05-15 01:15:00", now=NOW)
    assert out == datetime(2026, 5, 15, 1, 15, 0, tzinfo=UTC).isoformat()


def test_date_only_becomes_midnight_utc() -> None:
    out = normalise_posted_at("2026-05-14", now=NOW)
    assert out == datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC).isoformat()


def test_epoch_millis() -> None:
    ms = int(datetime(2026, 5, 14, 10, 21, 48, tzinfo=UTC).timestamp() * 1000)
    out = normalise_posted_at(str(ms), now=NOW)
    assert out == datetime(2026, 5, 14, 10, 21, 48, tzinfo=UTC).isoformat()


def test_epoch_seconds() -> None:
    secs = int(datetime(2026, 5, 14, 10, 21, 48, tzinfo=UTC).timestamp())
    out = normalise_posted_at(str(secs), now=NOW)
    assert out == datetime(2026, 5, 14, 10, 21, 48, tzinfo=UTC).isoformat()


@pytest.mark.parametrize(
    "raw",
    ["just now", "Just posted", "just listed", "Posted today", "today", "New", "NEW"],
)
def test_fresh_phrases_map_to_now(raw: str) -> None:
    assert normalise_posted_at(raw, now=NOW) == NOW.isoformat()


def test_yesterday() -> None:
    out = normalise_posted_at("Yesterday", now=NOW)
    assert out == datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC).isoformat()


@pytest.mark.parametrize(
    ("raw", "delta_seconds"),
    [
        ("9m ago", 9 * 60),
        ("1 minute ago", 60),
        ("30 minutes ago", 30 * 60),
        ("30 min ago", 30 * 60),
        ("9h ago", 9 * 3600),
        ("1 hour ago", 3600),
        ("9 hours ago", 9 * 3600),
        ("2 hr ago", 2 * 3600),
        ("8d ago", 8 * 86400),
        ("1 day ago", 86400),
        ("4 days ago", 4 * 86400),
        ("2w ago", 2 * 7 * 86400),
        ("1 week ago", 7 * 86400),
        ("3 weeks ago", 3 * 7 * 86400),
        ("1mo ago", 30 * 86400),
        ("3 months ago", 3 * 30 * 86400),
        ("1 month ago", 30 * 86400),
        ("1 year ago", 365 * 86400),
        ("2 years ago", 2 * 365 * 86400),
        ("30+ days ago", 30 * 86400),
        (" 8D AGO ", 8 * 86400),
    ],
)
def test_relative_offsets(raw: str, delta_seconds: int) -> None:
    out = normalise_posted_at(raw, now=NOW)
    expected = datetime.fromisoformat(out) if out else None
    assert expected is not None
    assert (NOW - expected).total_seconds() == pytest.approx(delta_seconds)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("14/05/2026", datetime(2026, 5, 14, tzinfo=UTC)),
        ("14-05-2026", datetime(2026, 5, 14, tzinfo=UTC)),
        ("14 May 2026", datetime(2026, 5, 14, tzinfo=UTC)),
        ("14 February 2026", datetime(2026, 2, 14, tzinfo=UTC)),
        ("1 Jan 2026", datetime(2026, 1, 1, tzinfo=UTC)),
    ],
)
def test_au_date_formats(raw: str, expected: datetime) -> None:
    assert normalise_posted_at(raw, now=NOW) == expected.isoformat()


def test_idempotent_on_already_iso_utc() -> None:
    once = normalise_posted_at("8d ago", now=NOW)
    assert once is not None
    twice = normalise_posted_at(once, now=NOW)
    assert twice == once


# ---------------------------------------------------------------------------
# backfill_posted_at — whole-catalogue repair
# ---------------------------------------------------------------------------


def _seed(
    conn: sqlite3.Connection,
    *,
    title: str,
    posted_at: str | None,
    first_seen_at: str = "2026-05-10T00:00:00+00:00",
) -> int:
    cur = conn.execute(
        "INSERT INTO jobs "
        "(dedup_key, title, company, company_norm, location_raw, "
        " remote_type, description_text, apply_url, posted_at, "
        " first_seen_at, last_seen_at, fingerprint_json) "
        "VALUES (?, ?, 'Acme', 'acme', 'Sydney', 'onsite', NULL, ?, ?, "
        " ?, datetime('now'), '{}')",
        (
            f"acme:{title.lower().replace(' ', '-')}",
            title,
            f"https://example.com/{title}",
            posted_at,
            first_seen_at,
        ),
    )
    job_id = cur.lastrowid
    conn.commit()
    assert job_id is not None
    return int(job_id)


def _posted(conn: sqlite3.Connection, job_id: int) -> str | None:
    row = conn.execute("SELECT posted_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
    value: str | None = row[0]
    return value


def test_backfill_normalises_relative_against_first_seen_at(
    conn: sqlite3.Connection,
) -> None:
    """The reference instant for a relative string is the row's
    ``first_seen_at`` — that's when "8d ago" was actually true, not
    whenever the backfill happens to run."""
    job = _seed(
        conn,
        title="Seek Engineer",
        posted_at="8d ago",
        first_seen_at="2026-05-10T00:00:00+00:00",
    )
    result = backfill_posted_at(conn)
    assert result.updated == 1
    assert result.parsed == 1
    assert result.nulled == 0
    expected = datetime(2026, 5, 2, 0, 0, 0, tzinfo=UTC).isoformat()
    assert _posted(conn, job) == expected


def test_backfill_nulls_unparseable_values(conn: sqlite3.Connection) -> None:
    job = _seed(conn, title="Mystery", posted_at="see advert")
    result = backfill_posted_at(conn)
    assert result.updated == 1
    assert result.nulled == 1
    assert result.parsed == 0
    assert _posted(conn, job) is None


def test_backfill_skips_null_and_empty(conn: sqlite3.Connection) -> None:
    a = _seed(conn, title="No Date", posted_at=None)
    b = _seed(conn, title="Blank Date", posted_at="")
    result = backfill_posted_at(conn)
    assert result.inspected == 0
    assert result.updated == 0
    assert _posted(conn, a) is None
    assert _posted(conn, b) == ""


def test_backfill_idempotent_on_iso(conn: sqlite3.Connection) -> None:
    iso = "2026-05-14T10:21:48.831265+00:00"
    job = _seed(conn, title="Greenhouse Job", posted_at=iso)
    first = backfill_posted_at(conn)
    assert first.updated == 0  # already canonical ISO — no write
    second = backfill_posted_at(conn)
    assert second.updated == 0
    assert _posted(conn, job) == iso


def test_backfill_falls_back_to_now_when_first_seen_unparseable(
    conn: sqlite3.Connection,
) -> None:
    job = _seed(
        conn,
        title="Weird Seen",
        posted_at="1 day ago",
        first_seen_at="not-a-timestamp",
    )
    result = backfill_posted_at(conn, now=NOW)
    assert result.parsed == 1
    expected = datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC).isoformat()
    assert _posted(conn, job) == expected


def test_backfill_respects_limit(conn: sqlite3.Connection) -> None:
    for i in range(5):
        _seed(conn, title=f"Job {i}", posted_at="3 days ago")
    result = backfill_posted_at(conn, limit=2)
    assert result.inspected == 2
    assert result.updated == 2


def test_backfill_restores_chronological_sort(conn: sqlite3.Connection) -> None:
    """After the pass, ``ORDER BY posted_at DESC`` is real chronology,
    not lexical garbage. This is the user-visible bug."""
    old = _seed(
        conn,
        title="Old Role",
        posted_at="8d ago",
        first_seen_at="2026-05-18T00:00:00+00:00",
    )
    mid = _seed(
        conn,
        title="Mid Role",
        posted_at="2 days ago",
        first_seen_at="2026-05-18T00:00:00+00:00",
    )
    fresh = _seed(
        conn,
        title="Fresh Role",
        posted_at="Just posted",
        first_seen_at="2026-05-18T00:00:00+00:00",
    )
    backfill_posted_at(conn)
    ordered = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM jobs ORDER BY posted_at DESC NULLS LAST",
        ).fetchall()
    ]
    assert ordered == [fresh, mid, old]
