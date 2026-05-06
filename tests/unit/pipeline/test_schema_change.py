"""Tests for the schema-change detection module."""

from __future__ import annotations

import pytest

from jobai.pipeline.schema_change import (
    DEFAULT_THRESHOLD,
    TRACKED_FIELDS,
    FieldChange,
    FieldStats,
    compute_field_stats,
    detect_changes,
    empty_stats,
    update_stats,
)
from jobai.sources.base import NormalizedJob


def _job(**overrides: object) -> NormalizedJob:
    """Build a NormalizedJob with sensible defaults; overrides win."""
    base: dict[str, object] = {
        "source_external_id": "id-1",
        "title": "Senior Engineer",
        "company": "Atlassian",
        "apply_url": "https://example.com/job/1",
        "raw_data": {"id": 1},
        "location_raw": "Sydney NSW",
        "location_country": "Australia",
        "location_city": "Sydney",
        "remote_type": "hybrid",
        "employment_type": "Full Time",
        "posted_at": "2026-05-01T00:00:00Z",
        "salary_min": 120_000,
        "salary_max": 160_000,
        "salary_currency": "AUD",
        "description_text": "Build cool things.",
        "description_html": "<p>Build cool things.</p>",
    }
    base.update(overrides)
    return NormalizedJob(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_field_stats
# ---------------------------------------------------------------------------


def test_compute_field_stats_full_jobs() -> None:
    stats = compute_field_stats([_job(), _job(), _job()])
    assert stats.total == 3
    for field_name in TRACKED_FIELDS:
        assert stats.present[field_name] == 3
        assert stats.null_rate(field_name) == 0.0


def test_compute_field_stats_partial_population() -> None:
    jobs = [
        _job(remote_type=None, salary_min=None),
        _job(remote_type="remote"),
        _job(salary_min=100_000),
    ]
    stats = compute_field_stats(jobs)
    assert stats.total == 3
    assert stats.present["remote_type"] == 2  # 1st had None
    assert stats.present["salary_min"] == 2
    assert stats.null_rate("remote_type") == pytest.approx(1 / 3)


def test_compute_field_stats_empty_string_counts_as_null() -> None:
    stats = compute_field_stats([_job(location_raw=""), _job(location_raw="Sydney")])
    assert stats.present["location_raw"] == 1


def test_compute_field_stats_empty_iterable() -> None:
    stats = compute_field_stats([])
    assert stats.total == 0
    assert stats.null_rate("title") == 0.0


# ---------------------------------------------------------------------------
# update_stats (incremental accumulator)
# ---------------------------------------------------------------------------


def test_update_stats_incremental_matches_compute_field_stats() -> None:
    jobs = [
        _job(remote_type=None),
        _job(salary_max=None),
        _job(),
    ]
    incremental = empty_stats()
    for job in jobs:
        incremental = update_stats(incremental, job)
    bulk = compute_field_stats(jobs)
    assert incremental.total == bulk.total
    assert incremental.present == bulk.present


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_field_stats_json_round_trip() -> None:
    original = compute_field_stats([_job(), _job(remote_type=None)])
    payload = original.to_json()
    restored = FieldStats.from_json(payload)
    assert restored is not None
    assert restored.total == original.total
    assert restored.present == original.present


def test_field_stats_from_json_returns_none_on_garbage() -> None:
    assert FieldStats.from_json(None) is None
    assert FieldStats.from_json("") is None
    assert FieldStats.from_json("{not json") is None
    assert FieldStats.from_json("[]") is None
    assert FieldStats.from_json('{"total": "many"}') is None


# ---------------------------------------------------------------------------
# detect_changes
# ---------------------------------------------------------------------------


def test_detect_changes_returns_empty_when_no_baseline() -> None:
    curr = compute_field_stats([_job() for _ in range(10)])
    assert detect_changes(None, curr) == []


def test_detect_changes_skips_low_volume_runs() -> None:
    prev = compute_field_stats([_job(remote_type="remote")])
    curr = compute_field_stats([_job(remote_type=None)])
    assert detect_changes(prev, curr) == []


def test_detect_changes_flags_field_that_dropped_below_threshold() -> None:
    prev = compute_field_stats([_job() for _ in range(10)])
    curr = compute_field_stats([_job(remote_type=None) for _ in range(10)])
    changes = detect_changes(prev, curr)
    assert len(changes) == 1
    change = changes[0]
    assert change.field == "remote_type"
    assert change.prev_null_rate == 0.0
    assert change.curr_null_rate == 1.0
    assert change.delta == 1.0


def test_detect_changes_ignores_field_that_improved() -> None:
    prev = compute_field_stats([_job(remote_type=None) for _ in range(10)])
    curr = compute_field_stats([_job() for _ in range(10)])
    # null-rate dropped — that's a recovery, not a regression
    assert detect_changes(prev, curr) == []


def test_detect_changes_flags_only_fields_above_threshold() -> None:
    # 30% of jobs lose remote_type, only 10% lose salary_min
    prev = compute_field_stats([_job() for _ in range(10)])
    curr_jobs = (
        [_job(remote_type=None) for _ in range(3)]
        + [_job(salary_min=None) for _ in range(1)]
        + [_job() for _ in range(6)]
    )
    curr = compute_field_stats(curr_jobs)
    changes = detect_changes(prev, curr, threshold=0.25)
    fields = {c.field for c in changes}
    assert "remote_type" in fields
    assert "salary_min" not in fields


def test_default_threshold_is_reasonable() -> None:
    assert 0.1 <= DEFAULT_THRESHOLD <= 0.5


def test_field_change_dataclass_carries_volume_context() -> None:
    prev = compute_field_stats([_job() for _ in range(20)])
    curr = compute_field_stats([_job(salary_min=None) for _ in range(15)])
    changes = detect_changes(prev, curr)
    salary = next(c for c in changes if c.field == "salary_min")
    assert isinstance(salary, FieldChange)
    assert salary.prev_total == 20
    assert salary.curr_total == 15
