"""Tests for the per-job description backfill."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.fetcher.base import Response
from jobai.pipeline.description_backfill import (
    PARSERS,
    BackfillResult,
    _parse_linkedin_description,
    backfill_descriptions,
    select_pending_jobs,
)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "backfill.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


_seed_counter = 0


def _seed_job(
    conn: sqlite3.Connection,
    *,
    kind: str,
    title: str = "Senior Engineer",
    apply_url: str = "https://example.com/job/1",
    description: str | None = None,
) -> int:
    """Insert one source + raw + canonical job row, return job_id."""
    global _seed_counter  # noqa: PLW0603 - test-only fixture state
    _seed_counter += 1
    account = f"test{_seed_counter}"
    cur = conn.execute(
        "INSERT INTO sources "
        "(kind, account, display_name, default_tier, enabled, cadence_seconds) "
        "VALUES (?, ?, ?, ?, 1, 3600)",
        (kind, account, f"{kind} {account}", 3),
    )
    source_id = cur.lastrowid
    raw = conn.execute(
        "INSERT INTO jobs_raw (source_id, source_external_id, raw_json, raw_sha256, "
        "first_seen_at, last_seen_at) VALUES (?, ?, '{}', 'x', datetime('now'), datetime('now'))",
        (source_id, apply_url),
    )
    raw_id = raw.lastrowid
    job = conn.execute(
        "INSERT INTO jobs "
        "(dedup_key, title, company, company_norm, apply_url, description_text, "
        " first_seen_at, last_seen_at, fingerprint_json) "
        "VALUES (?, ?, 'Co', 'co', ?, ?, datetime('now'), datetime('now'), '{}')",
        (apply_url, title, apply_url, description),
    )
    job_id = job.lastrowid
    conn.execute(
        "INSERT INTO job_sources (job_id, source_id, jobs_raw_id, apply_url) VALUES (?, ?, ?, ?)",
        (job_id, source_id, raw_id, apply_url),
    )
    conn.commit()
    assert job_id is not None
    return int(job_id)


class _ScriptedFetcher:
    """Per-URL canned-response fetcher."""

    def __init__(self, responses: dict[str, Response | BaseException]) -> None:
        self._responses = dict(responses)
        self.calls: list[str] = []

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109
        wait_for_selector: str | None = None,
    ) -> Response:
        self.calls.append(url)
        item = self._responses.get(url)
        if item is None:
            return Response(
                url=url,
                status_code=404,
                headers={},
                body=b"",
                fetched_at=datetime.now(tz=UTC),
            )
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        return None


def _resp(status: int, body: bytes = b"") -> Response:
    return Response(
        url="https://example.com",
        status_code=status,
        headers={},
        body=body,
        fetched_at=datetime.now(tz=UTC),
    )


# ---------------------------------------------------------------------------
# select_pending_jobs
# ---------------------------------------------------------------------------


def test_select_pending_jobs_returns_only_null_descriptions(
    conn: sqlite3.Connection,
) -> None:
    needs = _seed_job(conn, kind="linkedin", apply_url="https://l.in/1")
    _seed_job(
        conn,
        kind="linkedin",
        apply_url="https://l.in/2",
        description="already filled",
    )
    rows = select_pending_jobs(conn, kinds=("linkedin",))
    assert [r[0] for r in rows] == [needs]


def test_select_pending_jobs_filters_by_kind(conn: sqlite3.Connection) -> None:
    _seed_job(conn, kind="greenhouse", apply_url="https://g.io/1")
    needs = _seed_job(conn, kind="linkedin", apply_url="https://l.in/1")
    rows = select_pending_jobs(conn, kinds=("linkedin",))
    assert [r[0] for r in rows] == [needs]


def test_select_pending_jobs_handles_empty_kind_tuple(conn: sqlite3.Connection) -> None:
    _seed_job(conn, kind="linkedin", apply_url="https://l.in/1")
    rows = select_pending_jobs(conn, kinds=())
    assert rows == []


def test_select_pending_jobs_treats_empty_string_as_pending(
    conn: sqlite3.Connection,
) -> None:
    """Some sources insert ``''`` instead of NULL — both should backfill."""
    needs = _seed_job(conn, kind="linkedin", apply_url="https://l.in/1", description="")
    rows = select_pending_jobs(conn, kinds=("linkedin",))
    assert [r[0] for r in rows] == [needs]


# ---------------------------------------------------------------------------
# _parse_linkedin_description
# ---------------------------------------------------------------------------


def test_parse_linkedin_description_extracts_text() -> None:
    html = (
        "<html><body>"
        '<div class="description__text"><p>Build cool things.</p>'
        "<p>Remote-friendly.</p></div></body></html>"
    )
    assert _parse_linkedin_description(html) == "Build cool things.Remote-friendly."


def test_parse_linkedin_description_falls_back_to_inner_wrapper() -> None:
    html = '<html><body><div class="show-more-less-html__markup">Markup body</div></body></html>'
    assert _parse_linkedin_description(html) == "Markup body"


def test_parse_linkedin_description_returns_none_on_missing() -> None:
    assert _parse_linkedin_description("<html><body></body></html>") is None


# ---------------------------------------------------------------------------
# backfill_descriptions (end-to-end with fakes)
# ---------------------------------------------------------------------------


async def test_backfill_fills_pending_linkedin_jobs(conn: sqlite3.Connection) -> None:
    job_id = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/42")
    fetcher = _ScriptedFetcher(
        {
            "https://l.in/job/42": _resp(
                200,
                body=(
                    b'<html><body><div class="description__text">'
                    b"Real description body.</div></body></html>"
                ),
            ),
        },
    )

    result = await backfill_descriptions(conn, fetcher)

    assert result == BackfillResult(attempted=1, filled=1, skipped=0)
    row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row[0] == "Real description body."


async def test_backfill_skips_non_2xx(conn: sqlite3.Connection) -> None:
    job_id = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/blocked")
    fetcher = _ScriptedFetcher({"https://l.in/job/blocked": _resp(403)})

    result = await backfill_descriptions(conn, fetcher)

    assert result.attempted == 1
    assert result.filled == 0
    assert result.skipped == 1
    row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row[0] is None


async def test_backfill_continues_after_fetch_exception(conn: sqlite3.Connection) -> None:
    """One job's network failure shouldn't tank the whole batch."""
    bad = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/bad")
    good = _seed_job(conn, kind="linkedin", apply_url="https://l.in/job/good")
    fetcher = _ScriptedFetcher(
        {
            "https://l.in/job/bad": RuntimeError("network down"),
            "https://l.in/job/good": _resp(
                200,
                body=(b'<html><body><div class="description__text">OK</div></body></html>'),
            ),
        },
    )

    result = await backfill_descriptions(conn, fetcher)

    assert result.attempted == 2
    assert result.filled == 1
    assert result.skipped == 1
    bad_row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (bad,)).fetchone()
    good_row = conn.execute("SELECT description_text FROM jobs WHERE id = ?", (good,)).fetchone()
    assert bad_row[0] is None
    assert good_row[0] == "OK"


async def test_backfill_respects_limit(conn: sqlite3.Connection) -> None:
    for i in range(3):
        _seed_job(conn, kind="linkedin", apply_url=f"https://l.in/job/{i}")
    fetcher = _ScriptedFetcher(
        {
            f"https://l.in/job/{i}": _resp(
                200,
                body=(b'<html><body><div class="description__text">D</div></body></html>'),
            )
            for i in range(3)
        },
    )

    result = await backfill_descriptions(conn, fetcher, limit=2)

    assert result.attempted == 2
    assert result.filled == 2
    assert len(fetcher.calls) == 2


async def test_backfill_returns_zero_when_nothing_pending(
    conn: sqlite3.Connection,
) -> None:
    _seed_job(
        conn,
        kind="linkedin",
        apply_url="https://l.in/job/done",
        description="already there",
    )
    fetcher = _ScriptedFetcher({})

    result = await backfill_descriptions(conn, fetcher)

    assert result == BackfillResult(attempted=0, filled=0, skipped=0)


def test_parsers_registry_exports_linkedin() -> None:
    assert "linkedin" in PARSERS
