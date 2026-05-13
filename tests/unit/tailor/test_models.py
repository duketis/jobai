"""Pydantic schema coverage for jobai.tailor.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jobai.tailor.models import (
    TERMINAL_STATUSES,
    KickBatchRequest,
    SiblingRunSnapshot,
    TailorRunStatus,
)


def test_tailor_run_status_values_round_trip() -> None:
    """All five lifecycle states are present and equal their string value."""
    assert TailorRunStatus.PENDING.value == "pending"
    assert TailorRunStatus.RESUME_RUNNING.value == "resume_running"
    assert TailorRunStatus.LETTER_RUNNING.value == "letter_running"
    assert TailorRunStatus.SUCCEEDED.value == "succeeded"
    assert TailorRunStatus.FAILED.value == "failed"


def test_terminal_statuses_match_success_and_failed() -> None:
    """The terminal-state set is exactly {succeeded, failed} — the worker
    relies on this to decide when a row is done."""
    assert frozenset({TailorRunStatus.SUCCEEDED, TailorRunStatus.FAILED}) == TERMINAL_STATUSES


def test_sibling_run_snapshot_ignores_extra_fields() -> None:
    """Both siblings return much larger records than we parse; ``extra=ignore``
    keeps us forward-compatible if they add new fields."""
    snap = SiblingRunSnapshot.model_validate(
        {"id": "rs_1", "status": "tailoring", "tailored": {"x": 1}, "result": None},
    )
    assert snap.id == "rs_1"
    assert snap.status == "tailoring"


def test_kick_batch_request_rejects_empty_list() -> None:
    """A zero-length batch request is a programmer error -- the route
    body would otherwise silently no-op."""
    with pytest.raises(ValidationError):
        KickBatchRequest(job_ids=[])


def test_kick_batch_request_accepts_large_batches() -> None:
    """The endpoint is intentionally generous (10k cap) because the
    TailorPool already serialises concurrency; everything above the
    cap just queues up. The Pydantic ceiling is a safety stop to
    prevent absurdly large payloads, not a quota."""
    request = KickBatchRequest(job_ids=list(range(5_000)))
    assert len(request.job_ids) == 5_000


def test_kick_batch_request_rejects_above_ceiling() -> None:
    """The 10k ceiling exists so a runaway frontend can't post a
    megabyte of integer ids in one request."""
    with pytest.raises(ValidationError):
        KickBatchRequest(job_ids=list(range(10_001)))
