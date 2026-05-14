"""Shared fixtures for tailor-package tests.

The shared pieces:

* :func:`scripted_resume_client` / :func:`scripted_letter_client` — in-memory
  Protocol-conformant fakes that record every call and replay scripted
  poll sequences. The orchestrator + worker tests drive deterministic
  state machines through them.
* :func:`recording_sleeper` — replaces ``asyncio.sleep`` in the
  orchestrator so the chain finishes instantly while still recording
  the requested delays for assertion.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import httpx
import pytest

from jobai.db.migrations import apply_pending
from jobai.tailor.models import (
    CoverletteraiTailorRequest,
    ResumeaiTailorRequest,
    SiblingRunSnapshot,
)

Sleeper = Callable[[float], Awaitable[None]]


@pytest.fixture
def tailor_db_path(tmp_path: Path) -> Path:
    """Fresh, migrated SQLite DB seeded with a single job for chain tests."""
    db = tmp_path / "tailor-test.db"
    conn = sqlite3.connect(db)
    try:
        apply_pending(conn)
        _seed_one_job(conn)
    finally:
        conn.close()
    return db


def _seed_one_job(conn: sqlite3.Connection) -> None:
    """Insert one source + jobs_raw + jobs row so tailor_run FK / apply_url lookups work."""
    conn.execute(
        "INSERT INTO sources (kind, account, display_name, cadence_seconds) "
        "VALUES ('greenhouse', 'acme', 'Acme', 3600)"
    )
    source_id = int(conn.execute("SELECT id FROM sources").fetchone()[0])
    conn.execute(
        "INSERT INTO scrape_runs (source_id, started_at, status, tier_used) "
        "VALUES (?, datetime('now'), 'success', 1)",
        (source_id,),
    )
    conn.execute(
        "INSERT INTO jobs_raw (source_id, source_external_id, raw_json, raw_sha256, "
        "                      first_seen_at, last_seen_at) "
        "VALUES (?, 'ext1', '{}', 'sha', datetime('now'), datetime('now'))",
        (source_id,),
    )
    raw_id = int(conn.execute("SELECT id FROM jobs_raw").fetchone()[0])
    conn.execute(
        "INSERT INTO jobs (dedup_key, title, company, company_norm, apply_url, "
        "                  first_seen_at, last_seen_at) "
        "VALUES ('k', 'Engineer', 'Acme', 'acme', 'https://example.com/jd-1', "
        "        datetime('now'), datetime('now'))"
    )
    conn.execute(
        "INSERT INTO job_sources (job_id, source_id, jobs_raw_id, apply_url) "
        "VALUES (1, ?, ?, 'https://example.com/jd-1')",
        (source_id, raw_id),
    )
    conn.commit()


class ScriptedResumeClient:
    """In-memory :class:`ResumeaiClient` driven by canned scripts.

    ``kick`` always returns ``resume_run_id``. ``poll`` walks
    ``poll_statuses`` left-to-right, returning the last entry forever
    once exhausted. ``stream_pdf`` returns ``stream_response``.
    """

    def __init__(
        self,
        *,
        resume_run_id: str = "rs_1",
        poll_statuses: list[str] | None = None,
        stream_response: httpx.Response | None = None,
        kick_error: Exception | None = None,
        run_record: dict[str, object] | None = None,
    ) -> None:
        self.resume_run_id = resume_run_id
        self._poll_statuses = list(poll_statuses or ["succeeded"])
        self._poll_index = 0
        self._stream_response = stream_response
        self._kick_error = kick_error
        self._run_record = run_record or {
            "id": resume_run_id,
            "status": "succeeded",
            "requirements": {"title": "Engineer"},
            "tailored": {"name": "Jane Doe", "summary": "resume body"},
        }
        self.kick_requests: list[ResumeaiTailorRequest] = []
        self.poll_calls: list[str] = []
        self.stream_calls: list[str] = []
        self.get_run_calls: list[str] = []

    async def kick(self, request: ResumeaiTailorRequest) -> str:
        self.kick_requests.append(request)
        if self._kick_error is not None:
            raise self._kick_error
        return self.resume_run_id

    async def poll(self, run_id: str) -> SiblingRunSnapshot:
        self.poll_calls.append(run_id)
        status = self._poll_statuses[min(self._poll_index, len(self._poll_statuses) - 1)]
        self._poll_index += 1
        return SiblingRunSnapshot(id=run_id, status=status)

    async def get_run(self, run_id: str) -> dict[str, object]:
        self.get_run_calls.append(run_id)
        return dict(self._run_record)

    async def stream_pdf(self, run_id: str) -> httpx.Response:
        self.stream_calls.append(run_id)
        if self._stream_response is None:
            # Default to an empty PDF response so layout-check fetches
            # in the QA stage degrade gracefully (the check itself
            # short-circuits on empty bytes). Tests that exercise the
            # streaming path supply their own ``stream_response``.
            return httpx.Response(200, content=b"")
        return self._stream_response


class ScriptedLetterClient:
    """In-memory :class:`CoverletteraiClient` mirroring :class:`ScriptedResumeClient`."""

    def __init__(
        self,
        *,
        letter_run_id: str = "ls_1",
        poll_statuses: list[str] | None = None,
        stream_response: httpx.Response | None = None,
        kick_error: Exception | None = None,
        run_record: dict[str, object] | None = None,
    ) -> None:
        self.letter_run_id = letter_run_id
        self._poll_statuses = list(poll_statuses or ["succeeded"])
        self._poll_index = 0
        self._stream_response = stream_response
        self._kick_error = kick_error
        self._run_record = run_record or {
            "id": letter_run_id,
            "status": "succeeded",
            "tailored": {"opening": "Dear hiring manager,", "closing": "Thanks."},
        }
        self.kick_requests: list[CoverletteraiTailorRequest] = []
        self.poll_calls: list[str] = []
        self.stream_calls: list[str] = []
        self.get_run_calls: list[str] = []

    async def kick(self, request: CoverletteraiTailorRequest) -> str:
        self.kick_requests.append(request)
        if self._kick_error is not None:
            raise self._kick_error
        return self.letter_run_id

    async def poll(self, run_id: str) -> SiblingRunSnapshot:
        self.poll_calls.append(run_id)
        status = self._poll_statuses[min(self._poll_index, len(self._poll_statuses) - 1)]
        self._poll_index += 1
        return SiblingRunSnapshot(id=run_id, status=status)

    async def get_run(self, run_id: str) -> dict[str, object]:
        self.get_run_calls.append(run_id)
        return dict(self._run_record)

    async def stream_pdf(self, run_id: str) -> httpx.Response:
        self.stream_calls.append(run_id)
        if self._stream_response is None:
            # Default to an empty PDF response so layout-check fetches
            # in the QA stage degrade gracefully (the check itself
            # short-circuits on empty bytes). Tests that exercise the
            # streaming path supply their own ``stream_response``.
            return httpx.Response(200, content=b"")
        return self._stream_response


@pytest.fixture
def scripted_resume_client() -> ScriptedResumeClient:
    """A default-happy-path resume client (kick succeeds, poll returns 'succeeded')."""
    return ScriptedResumeClient()


@pytest.fixture
def scripted_letter_client() -> ScriptedLetterClient:
    """A default-happy-path letter client (kick succeeds, poll returns 'succeeded')."""
    return ScriptedLetterClient()


@pytest.fixture
def recording_sleeper() -> Iterator[tuple[list[float], Sleeper]]:
    """A sleeper that records every requested delay and returns immediately."""
    delays: list[float] = []

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    yield delays, _sleep
