"""SQL CRUD coverage for jobai.tailor.repository."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.db.connection import connect
from jobai.tailor.models import TailorRunStatus
from jobai.tailor.repository import (
    create_tailor_run,
    get_tailor_run,
    list_tailor_runs,
    update_status,
)


@pytest.fixture
def conn(tailor_db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a managed connection against the seeded test DB."""
    with connect(tailor_db_path) as conn:
        yield conn


def test_create_tailor_run_inserts_pending_row(conn: sqlite3.Connection) -> None:
    """A fresh tailor_run starts in ``pending`` with no sibling ids."""
    record = create_tailor_run(conn, job_id=1)
    assert record.id > 0
    assert record.job_id == 1
    assert record.status is TailorRunStatus.PENDING
    assert record.resume_run_id is None
    assert record.letter_run_id is None
    assert record.error is None
    assert record.finished_at is None


def test_get_tailor_run_returns_record_after_create(conn: sqlite3.Connection) -> None:
    """``get_tailor_run`` reads back the same row insert just wrote."""
    created = create_tailor_run(conn, job_id=1)
    fetched = get_tailor_run(conn, created.id)
    assert fetched is not None
    assert fetched.id == created.id


def test_get_tailor_run_returns_none_for_missing_id(conn: sqlite3.Connection) -> None:
    assert get_tailor_run(conn, 99999) is None


def test_update_status_writes_only_supplied_fields(conn: sqlite3.Connection) -> None:
    """Subsequent updates overlay partial fields, leaving others intact."""
    record = create_tailor_run(conn, job_id=1)
    update_status(
        conn,
        record.id,
        status=TailorRunStatus.RESUME_RUNNING,
        resume_run_id="rs_1",
    )
    after_first = get_tailor_run(conn, record.id)
    assert after_first is not None
    assert after_first.status is TailorRunStatus.RESUME_RUNNING
    assert after_first.resume_run_id == "rs_1"
    assert after_first.letter_run_id is None

    update_status(
        conn,
        record.id,
        status=TailorRunStatus.LETTER_RUNNING,
        letter_run_id="ls_1",
        letter_status="tailoring",
    )
    after_second = get_tailor_run(conn, record.id)
    assert after_second is not None
    # The earlier resume fields survive even though we didn't pass them.
    assert after_second.resume_run_id == "rs_1"
    assert after_second.letter_run_id == "ls_1"
    assert after_second.letter_status == "tailoring"


def test_update_status_sets_finished_at_on_terminal(conn: sqlite3.Connection) -> None:
    """Reaching ``succeeded`` or ``failed`` stamps ``finished_at``."""
    record = create_tailor_run(conn, job_id=1)
    update_status(conn, record.id, status=TailorRunStatus.SUCCEEDED)
    succeeded = get_tailor_run(conn, record.id)
    assert succeeded is not None
    assert succeeded.finished_at is not None

    record2 = create_tailor_run(conn, job_id=1)
    update_status(conn, record2.id, status=TailorRunStatus.FAILED, error="boom")
    failed = get_tailor_run(conn, record2.id)
    assert failed is not None
    assert failed.finished_at is not None
    assert failed.error == "boom"


def test_list_tailor_runs_orders_newest_first(conn: sqlite3.Connection) -> None:
    a = create_tailor_run(conn, job_id=1)
    b = create_tailor_run(conn, job_id=1)
    c = create_tailor_run(conn, job_id=1)
    rows = list_tailor_runs(conn)
    assert [r.id for r in rows] == [c.id, b.id, a.id]


def test_list_tailor_runs_respects_limit(conn: sqlite3.Connection) -> None:
    for _ in range(5):
        create_tailor_run(conn, job_id=1)
    rows = list_tailor_runs(conn, limit=2)
    assert len(rows) == 2


def test_list_tailor_runs_filters_by_job(conn: sqlite3.Connection) -> None:
    """Only rows for the requested job come back when ``job_id`` is set."""
    create_tailor_run(conn, job_id=1)
    rows = list_tailor_runs(conn, job_id=1)
    assert len(rows) == 1
    other = list_tailor_runs(conn, job_id=999)
    assert other == []


def test_list_tailor_runs_filters_by_status(conn: sqlite3.Connection) -> None:
    record = create_tailor_run(conn, job_id=1)
    update_status(conn, record.id, status=TailorRunStatus.SUCCEEDED)
    succ = list_tailor_runs(conn, status=TailorRunStatus.SUCCEEDED)
    pend = list_tailor_runs(conn, status=TailorRunStatus.PENDING)
    assert len(succ) == 1
    assert pend == []
