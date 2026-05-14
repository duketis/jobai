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


def test_create_tailor_run_accepts_jd_url_when_no_job_id(
    conn: sqlite3.Connection,
) -> None:
    """One-off URL path: job_id stays null, jd_url carries the URL."""
    record = create_tailor_run(conn, jd_url="https://example.com/jd")
    assert record.job_id is None
    assert record.jd_url == "https://example.com/jd"
    assert record.status is TailorRunStatus.PENDING

    # Reading the row back preserves both fields.
    fetched = get_tailor_run(conn, record.id)
    assert fetched is not None
    assert fetched.job_id is None
    assert fetched.jd_url == "https://example.com/jd"


def test_create_tailor_run_rejects_no_args(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="exactly one of job_id / jd_url"):
        create_tailor_run(conn)


def test_create_tailor_run_rejects_both_args(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="exactly one of job_id / jd_url"):
        create_tailor_run(conn, job_id=1, jd_url="https://example.com/jd")


# ---------------------------------------------------------------------------
# Applied-state (v1.18.0)
# ---------------------------------------------------------------------------


def test_set_applied_stamps_and_clears_applied_at(conn: sqlite3.Connection) -> None:
    """``set_applied(True)`` writes a non-null timestamp;
    ``set_applied(False)`` clears it back to None. Same helper, two
    directions -- maps directly to the PATCH endpoint."""
    from jobai.tailor.repository import set_applied  # noqa: PLC0415

    record = create_tailor_run(conn, job_id=1)
    assert record.applied_at is None

    after_mark = set_applied(conn, record.id, applied=True)
    assert after_mark is not None
    assert after_mark.applied_at is not None

    after_clear = set_applied(conn, record.id, applied=False)
    assert after_clear is not None
    assert after_clear.applied_at is None


def test_set_applied_returns_none_for_unknown_run(conn: sqlite3.Connection) -> None:
    """Unknown run id -> None so the route maps it to a 404 cleanly."""
    from jobai.tailor.repository import set_applied  # noqa: PLC0415

    assert set_applied(conn, 99_999, applied=True) is None


def test_list_tailor_runs_filters_by_applied(conn: sqlite3.Connection) -> None:
    """``applied=True`` -> only rows with applied_at set;
    ``applied=False`` -> only NULL applied_at; ``applied=None`` -> both."""
    from jobai.tailor.repository import set_applied  # noqa: PLC0415

    applied_run = create_tailor_run(conn, job_id=1)
    pending_run = create_tailor_run(conn, job_id=1)
    set_applied(conn, applied_run.id, applied=True)

    applied_ids = {r.id for r in list_tailor_runs(conn, applied=True)}
    pending_ids = {r.id for r in list_tailor_runs(conn, applied=False)}
    all_ids = {r.id for r in list_tailor_runs(conn)}

    assert applied_ids == {applied_run.id}
    assert pending_ids == {pending_run.id}
    assert {applied_run.id, pending_run.id}.issubset(all_ids)
