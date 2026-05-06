"""Tests for the remote_type inference + backfill."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.db.migrations import apply_pending
from jobai.pipeline.remote_inference import (
    backfill_remote_types,
    infer_remote_type,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "remote.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# infer_remote_type — single-call classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "This is a fully remote role.",
        "We are a 100% remote team.",
        "Remote-first culture.",
        "Work from anywhere in Australia.",
        "You'll be working remotely with a small team.",
        "WFH friendly; no office expected.",
    ],
)
def test_explicit_remote_phrases_classify_remote(description: str) -> None:
    assert infer_remote_type(title="Engineer", description=description) == "remote"


@pytest.mark.parametrize(
    "description",
    [
        "Hybrid working from our Melbourne office.",
        "We offer flexible working arrangements.",
        "3 days in the office, 2 from home.",
        "Blended work — split between office and home.",
    ],
)
def test_explicit_hybrid_phrases_classify_hybrid(description: str) -> None:
    assert infer_remote_type(title="Engineer", description=description) == "hybrid"


@pytest.mark.parametrize(
    "description",
    [
        "On-site role at our Sydney HQ.",
        "Onsite presence required.",
        "Office-based position; no remote work.",
        "This is an in-person role.",
        "You must be located in Australia and work from our Brisbane office.",
    ],
)
def test_explicit_onsite_phrases_classify_onsite(description: str) -> None:
    assert infer_remote_type(title="Engineer", description=description) == "onsite"


def test_remote_signal_outranks_hybrid_signal() -> None:
    """If both fire, remote wins. A 'fully remote' role that mentions
    'we sometimes do hybrid offsites' is still a remote role."""
    desc = "This is a fully remote position. Occasional hybrid team offsites a few times a year."
    assert infer_remote_type(title="Engineer", description=desc) == "remote"


def test_hybrid_signal_outranks_onsite_signal() -> None:
    """A listing that mentions a Sydney office but explicitly hybrid
    schedule should classify as hybrid, not onsite."""
    desc = "Hybrid working arrangement. Our office in Sydney is open for the in-office days."
    assert infer_remote_type(title="Engineer", description=desc) == "hybrid"


def test_multi_city_location_falls_back_to_hybrid() -> None:
    """APS Jobs and similar feeds list every city the role can sit in
    when the answer is "any of them" — that's effectively hybrid."""
    assert (
        infer_remote_type(
            title="Senior Engineer",
            description=None,
            location="Adelaide SA, Brisbane QLD, Canberra ACT, Melbourne VIC",
        )
        == "hybrid"
    )


def test_default_is_onsite_when_nothing_matches() -> None:
    """A bone-dry listing falls through to onsite — the conservative
    default that protects the ``remote=true`` filter from false positives."""
    assert (
        infer_remote_type(
            title="Engineer",
            description="We build cool things.",
            location="Sydney NSW",
        )
        == "onsite"
    )


def test_empty_input_returns_default() -> None:
    """No title, description, or location → fall through to default."""
    assert infer_remote_type(title=None, description=None, location=None) == "onsite"


def test_explicit_default_override_is_respected() -> None:
    """Caller can override the fallback (e.g. set hybrid as default
    if their data has a different distribution)."""
    assert (
        infer_remote_type(
            title="Engineer",
            description="We build cool things.",
            default="hybrid",
        )
        == "hybrid"
    )


def test_word_boundary_avoids_false_positive_inside_other_words() -> None:
    """``remoter`` shouldn't trigger a remote match; it's a word fragment."""
    # "remoter" — a real-but-rare word; not a remote-work signal.
    assert (
        infer_remote_type(
            title="Engineer",
            description="The remoter regions of the world need our service.",
            location="Sydney NSW",
        )
        == "onsite"
    )


# ---------------------------------------------------------------------------
# backfill_remote_types — DB-touching pass
# ---------------------------------------------------------------------------


def _seed_canonical_job(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: str | None = None,
    location: str | None = None,
    remote_type: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO jobs "
        "(dedup_key, title, company, company_norm, location_raw, "
        " remote_type, description_text, apply_url, "
        " first_seen_at, last_seen_at, fingerprint_json) "
        "VALUES (?, ?, 'Acme', 'acme', ?, ?, ?, ?, "
        " datetime('now'), datetime('now'), '{}')",
        (
            f"acme:{title.lower().replace(' ', '-')}",
            title,
            location,
            remote_type,
            description,
            f"https://example.com/{title}",
        ),
    )
    job_id = cur.lastrowid
    conn.commit()
    assert job_id is not None
    return int(job_id)


def test_backfill_only_touches_null_or_empty_rows(conn: sqlite3.Connection) -> None:
    """Rows that already have a remote_type (set by the source) stay
    untouched — only the unset ones go through inference."""
    locked = _seed_canonical_job(
        conn,
        title="Locked Engineer",
        description="Anywhere in Australia.",
        remote_type="onsite",  # source explicitly said onsite; honour it
    )
    pending = _seed_canonical_job(
        conn,
        title="Remote Engineer",
        description="This is a fully remote role.",
        remote_type=None,
    )

    result = backfill_remote_types(conn)

    assert result.updated == 1
    assert result.by_value["remote"] == 1
    locked_after = conn.execute(
        "SELECT remote_type FROM jobs WHERE id = ?",
        (locked,),
    ).fetchone()[0]
    pending_after = conn.execute(
        "SELECT remote_type FROM jobs WHERE id = ?",
        (pending,),
    ).fetchone()[0]
    assert locked_after == "onsite"
    assert pending_after == "remote"


def test_backfill_treats_empty_string_as_null(conn: sqlite3.Connection) -> None:
    """Some legacy rows store ``''`` instead of NULL; the SQL filter
    has to catch both."""
    job_id = _seed_canonical_job(
        conn,
        title="Engineer",
        description="Hybrid working from our Melbourne office.",
        remote_type="",
    )
    result = backfill_remote_types(conn)
    assert result.updated == 1
    after = conn.execute(
        "SELECT remote_type FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()[0]
    assert after == "hybrid"


def test_backfill_distributes_across_buckets(conn: sqlite3.Connection) -> None:
    _seed_canonical_job(
        conn,
        title="Remote Engineer",
        description="Fully remote, Australia-wide.",
    )
    _seed_canonical_job(
        conn,
        title="Hybrid Engineer",
        description="Flexible working with 2 days in the office.",
    )
    _seed_canonical_job(
        conn,
        title="Onsite Engineer",
        description="On-site presence required at our Sydney HQ.",
    )
    _seed_canonical_job(
        conn,
        title="Mystery Engineer",
        description="A great team and good benefits.",
        location="Sydney NSW",
    )

    result = backfill_remote_types(conn)

    assert result.inspected == 4
    assert result.updated == 4
    assert result.by_value == {"remote": 1, "hybrid": 1, "onsite": 2}


def test_backfill_respects_limit(conn: sqlite3.Connection) -> None:
    for i in range(5):
        _seed_canonical_job(
            conn,
            title=f"Engineer {i}",
            description="Remote-first role.",
        )
    result = backfill_remote_types(conn, limit=2)
    assert result.inspected == 2
    assert result.updated == 2
    # The remaining 3 keep their NULL.
    null_count = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE remote_type IS NULL OR remote_type = ''",
    ).fetchone()[0]
    assert null_count == 3


def test_backfill_returns_zero_when_nothing_pending(conn: sqlite3.Connection) -> None:
    _seed_canonical_job(conn, title="Engineer", remote_type="remote")
    result = backfill_remote_types(conn)
    assert result.inspected == 0
    assert result.updated == 0
    assert result.by_value == {"remote": 0, "hybrid": 0, "onsite": 0}
