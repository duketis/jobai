"""Tests for the RecordingFetcher decorator."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobai.db.migrations import apply_pending
from jobai.fetcher.base import Response
from jobai.fetcher.recording import RecordingFetcher
from jobai.sources.repository import upsert_source


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
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
def run_id(conn: sqlite3.Connection, source_id: int) -> int:
    cursor = conn.execute(
        "INSERT INTO scrape_runs (source_id, started_at, status, tier_used) "
        "VALUES (?, ?, 'running', 1)",
        (source_id, datetime.now(tz=UTC).isoformat()),
    )
    conn.commit()
    last_row_id = cursor.lastrowid
    assert last_row_id is not None
    return last_row_id


class _StubFetcher:
    """Minimal Fetcher implementation that returns canned responses."""

    def __init__(self, response: Response) -> None:
        self._response = response
        self.fetch_calls: list[str] = []

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> Response:
        del method, headers, json, timeout
        self.fetch_calls.append(url)
        return self._response

    async def aclose(self) -> None:
        return None


async def test_recording_fetcher_writes_raw_response_row(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    response = Response(
        url="https://example.com/jobs",
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=b'{"jobs": []}',
    )
    inner = _StubFetcher(response)
    recorder = RecordingFetcher(inner, conn=conn, run_id=run_id, source_id=source_id)

    await recorder.fetch("https://example.com/jobs")

    rows = conn.execute("SELECT * FROM raw_responses").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert row["source_id"] == source_id
    assert row["url"] == "https://example.com/jobs"
    assert row["status_code"] == 200


async def test_recording_fetcher_compresses_body(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    body = b'{"large":"' + (b"x" * 5000) + b'"}'
    response = Response(
        url="https://example.com/jobs",
        status_code=200,
        headers={},
        body=body,
    )
    recorder = RecordingFetcher(
        _StubFetcher(response),
        conn=conn,
        run_id=run_id,
        source_id=source_id,
    )

    await recorder.fetch("https://example.com/jobs")

    row = conn.execute("SELECT body_gz FROM raw_responses").fetchone()
    decompressed = gzip.decompress(row["body_gz"])
    assert decompressed == body


async def test_recording_fetcher_records_sha256_of_uncompressed_body(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    body = b'{"jobs": [{"id": 1}]}'
    response = Response(
        url="https://example.com",
        status_code=200,
        headers={},
        body=body,
    )
    recorder = RecordingFetcher(
        _StubFetcher(response),
        conn=conn,
        run_id=run_id,
        source_id=source_id,
    )

    await recorder.fetch("https://example.com")

    expected = hashlib.sha256(body).hexdigest()
    row = conn.execute("SELECT body_sha256 FROM raw_responses").fetchone()
    assert row["body_sha256"] == expected


async def test_recording_fetcher_skips_non_2xx_responses(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    response = Response(
        url="https://example.com",
        status_code=404,
        headers={},
        body=b"not found",
    )
    recorder = RecordingFetcher(
        _StubFetcher(response),
        conn=conn,
        run_id=run_id,
        source_id=source_id,
    )

    await recorder.fetch("https://example.com")

    count = conn.execute("SELECT COUNT(*) FROM raw_responses").fetchone()[0]
    assert count == 0


async def test_recording_fetcher_normalises_header_case(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    response = Response(
        url="https://example.com",
        status_code=200,
        headers={"Content-Type": "application/json", "X-RateLimit": "100"},
        body=b"{}",
    )
    recorder = RecordingFetcher(
        _StubFetcher(response),
        conn=conn,
        run_id=run_id,
        source_id=source_id,
    )

    await recorder.fetch("https://example.com")

    row = conn.execute("SELECT headers_json FROM raw_responses").fetchone()
    headers = json.loads(row["headers_json"])
    assert "content-type" in headers
    assert "x-ratelimit" in headers


async def test_recording_fetcher_sets_expiry_in_future(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    response = Response(
        url="https://example.com",
        status_code=200,
        headers={},
        body=b"{}",
    )
    recorder = RecordingFetcher(
        _StubFetcher(response),
        conn=conn,
        run_id=run_id,
        source_id=source_id,
        retention_days=7,
    )

    await recorder.fetch("https://example.com")

    row = conn.execute("SELECT expires_at, fetched_at FROM raw_responses").fetchone()
    expires = datetime.fromisoformat(row["expires_at"])
    fetched = datetime.fromisoformat(row["fetched_at"])
    assert (expires - fetched).days >= 6  # ~7 days, allowing a small clock skew


async def test_aclose_closes_inner_fetcher(
    conn: sqlite3.Connection,
    source_id: int,
    run_id: int,
) -> None:
    closed = False

    class _ClosingStub:
        async def fetch(self, *args: Any, **kwargs: Any) -> Response:
            del args, kwargs
            return Response(url="x", status_code=200, headers={}, body=b"")

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    recorder = RecordingFetcher(
        _ClosingStub(),
        conn=conn,
        run_id=run_id,
        source_id=source_id,
    )
    await recorder.aclose()
    assert closed is True
