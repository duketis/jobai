"""Tests for the agent's tool definitions and executor."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jobai.agent.tools import (
    TOOL_DEFINITIONS,
    ToolExecutor,
    UnknownToolError,
    serialise_result,
)
from jobai.db.migrations import apply_pending
from jobai.dedup.promote import promote_to_canonical_jobs
from jobai.sources.base import NormalizedJob
from jobai.sources.repository import upsert_source


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        apply_pending(connection)
        yield connection
    finally:
        connection.close()


@pytest.fixture
def seeded_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Insert sources, runtime state, and a few canonical jobs."""
    gh = upsert_source(conn, kind="greenhouse", account="atlassian", display_name="Atlassian")
    upsert_source(conn, kind="lever", account="palantir", display_name="Palantir")

    # Mark one source as currently failing within 24h.
    conn.execute(
        "INSERT INTO source_runtime_state "
        "(source_id, current_tier, last_error_at, last_error_class) "
        "VALUES (?, 1, ?, 'network')",
        (gh.id, datetime.now(tz=UTC).isoformat()),
    )

    # Seed two canonical jobs via the production code path so FTS5 stays in sync.
    cursor = conn.execute(
        "INSERT INTO jobs_raw "
        "(source_id, source_external_id, raw_json, raw_sha256, first_seen_at, last_seen_at) "
        "VALUES (?, ?, '{}', 'x', datetime('now'), datetime('now'))",
        (gh.id, "1"),
    )
    raw_id_one = cursor.lastrowid
    assert raw_id_one is not None
    promote_to_canonical_jobs(
        conn,
        source_id=gh.id,
        jobs_raw_id=int(raw_id_one),
        job=NormalizedJob(
            source_external_id="1",
            title="Python Backend Engineer",
            company="Atlassian",
            apply_url="https://example.com/1",
            raw_data={"id": "1"},
            location_country="Australia",
            location_raw="Sydney, Australia",
            remote_type="onsite",
            description_text="Build async Python services on AWS.",
            posted_at="2026-04-15",
        ),
    )

    cursor = conn.execute(
        "INSERT INTO jobs_raw "
        "(source_id, source_external_id, raw_json, raw_sha256, first_seen_at, last_seen_at) "
        "VALUES (?, ?, '{}', 'x', datetime('now'), datetime('now'))",
        (gh.id, "2"),
    )
    raw_id_two = cursor.lastrowid
    assert raw_id_two is not None
    promote_to_canonical_jobs(
        conn,
        source_id=gh.id,
        jobs_raw_id=int(raw_id_two),
        job=NormalizedJob(
            source_external_id="2",
            title="Senior Frontend Engineer",
            company="Atlassian",
            apply_url="https://example.com/2",
            raw_data={"id": "2"},
            location_country="Australia",
            location_raw="Remote, Australia",
            remote_type="remote",
            description_text="React and TypeScript expertise.",
            posted_at="2026-04-20",
        ),
    )

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def test_tool_definitions_have_unique_names() -> None:
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert len(names) == len(set(names))


def test_tool_definitions_cover_required_set() -> None:
    expected = {
        "search_jobs",
        "get_job_detail",
        "mark_job_state",
        "list_sources",
        "get_health",
    }
    assert {t["name"] for t in TOOL_DEFINITIONS} == expected


def test_tool_definitions_have_input_schemas() -> None:
    for tool in TOOL_DEFINITIONS:
        assert tool["description"]
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema


# ---------------------------------------------------------------------------
# Executor dispatch
# ---------------------------------------------------------------------------


def test_unknown_tool_raises(conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(conn)
    with pytest.raises(UnknownToolError) as excinfo:
        executor.execute("not_a_real_tool", {})
    assert excinfo.value.name == "not_a_real_tool"


# ---------------------------------------------------------------------------
# search_jobs
# ---------------------------------------------------------------------------


def test_search_jobs_with_no_filters_returns_all(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("search_jobs", {})
    assert result["total"] == 2
    assert len(result["items"]) == 2


def test_search_jobs_uses_fts(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("search_jobs", {"q": "python"})
    titles = [j["title"] for j in result["items"]]
    assert any("Python" in t for t in titles)


def test_search_jobs_remote_filter(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("search_jobs", {"remote": "remote"})
    assert result["total"] == 1
    assert result["items"][0]["remote_type"] == "remote"


def test_search_jobs_pagination(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    page = executor.execute("search_jobs", {"limit": 1, "offset": 0})
    assert page["limit"] == 1
    assert len(page["items"]) == 1


# ---------------------------------------------------------------------------
# get_job_detail
# ---------------------------------------------------------------------------


def test_get_job_detail_returns_full_record(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    listing = executor.execute("search_jobs", {"q": "python"})
    job_id = listing["items"][0]["id"]

    detail = executor.execute("get_job_detail", {"job_id": job_id})

    assert detail["id"] == job_id
    assert detail["description_text"]
    assert "sources" in detail


def test_get_job_detail_returns_error_dict_for_missing(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("get_job_detail", {"job_id": 999_999})
    assert "error" in result


def test_get_job_detail_raises_when_job_id_missing(conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(conn)
    with pytest.raises(ValueError, match="job_id"):
        executor.execute("get_job_detail", {})


# ---------------------------------------------------------------------------
# mark_job_state
# ---------------------------------------------------------------------------


def test_mark_job_state_persists_value(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    listing = executor.execute("search_jobs", {"q": "python"})
    job_id = listing["items"][0]["id"]

    result = executor.execute(
        "mark_job_state",
        {"job_id": job_id, "state": "saved", "notes": "looks promising"},
    )

    assert result["state"] == "saved"
    assert result["notes"] == "looks promising"
    row = seeded_conn.execute(
        "SELECT state, notes FROM jobs_user_state WHERE job_id = ?", (job_id,)
    ).fetchone()
    assert row["state"] == "saved"
    assert row["notes"] == "looks promising"


def test_mark_job_state_overwrites_previous_value(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    listing = executor.execute("search_jobs", {"q": "python"})
    job_id = listing["items"][0]["id"]

    executor.execute("mark_job_state", {"job_id": job_id, "state": "saved"})
    second = executor.execute("mark_job_state", {"job_id": job_id, "state": "applied"})

    assert second["state"] == "applied"


def test_mark_job_state_returns_error_for_missing_job(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("mark_job_state", {"job_id": 999_999, "state": "saved"})
    assert "error" in result


def test_mark_job_state_raises_for_invalid_state(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    listing = executor.execute("search_jobs", {"q": "python"})
    job_id = listing["items"][0]["id"]

    with pytest.raises(ValueError, match="state must be one of"):
        executor.execute("mark_job_state", {"job_id": job_id, "state": "frobnicated"})


def test_mark_job_state_raises_for_non_string_notes(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    listing = executor.execute("search_jobs", {"q": "python"})
    job_id = listing["items"][0]["id"]

    with pytest.raises(TypeError, match="notes"):
        executor.execute(
            "mark_job_state",
            {"job_id": job_id, "state": "saved", "notes": 42},
        )


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------


def test_list_sources_returns_each_source_with_runtime_state(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("list_sources", {})
    names = [item["name"] for item in result["items"]]
    assert "greenhouse:atlassian" in names
    assert "lever:palantir" in names

    by_name = {item["name"]: item for item in result["items"]}
    # greenhouse:atlassian had a recent failure seeded
    assert by_name["greenhouse:atlassian"]["last_error_class"] == "network"
    # lever:palantir has no runtime_state row
    assert by_name["lever:palantir"]["current_tier"] is None


# ---------------------------------------------------------------------------
# get_health
# ---------------------------------------------------------------------------


def test_get_health_aggregates_counts(seeded_conn: sqlite3.Connection) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("get_health", {})
    assert result["jobs_total"] == 2
    assert result["jobs_added_24h"] == 2
    assert result["sources_total"] == 2
    assert result["sources_enabled"] == 2
    assert result["sources_failing"] == 1
    # No scrape_runs in the seed → last_scrape_at is None.
    assert result["last_scrape_at"] is None


# ---------------------------------------------------------------------------
# serialise_result
# ---------------------------------------------------------------------------


def test_serialise_result_produces_parseable_json() -> None:
    payload: dict[str, Any] = {
        "items": [{"id": 1, "title": "Engineer"}],
        "total": 1,
    }
    rendered = serialise_result(payload)
    assert json.loads(rendered) == payload


def test_serialise_result_handles_non_serialisable_via_default() -> None:
    """The Anthropic tool_result.content is a string — datetimes etc. must
    serialise via the str() fallback rather than blowing up."""
    payload = {"now": datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC)}
    rendered = serialise_result(payload)
    parsed = json.loads(rendered)
    assert "2026-05-05" in parsed["now"]
