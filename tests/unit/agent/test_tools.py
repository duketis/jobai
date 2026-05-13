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
        "kick_tailor",
        "list_tailor_runs",
        "get_tailor_run",
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


def test_mark_job_state_raises_when_job_id_missing(
    seeded_conn: sqlite3.Connection,
) -> None:
    """No ``job_id`` arg means the tool can't act -- surface a clear
    ValueError rather than KeyError."""
    executor = ToolExecutor(seeded_conn)
    with pytest.raises(ValueError, match="job_id"):
        executor.execute("mark_job_state", {"state": "saved"})


def test_mark_job_state_raises_when_job_id_not_int_parseable(
    seeded_conn: sqlite3.Connection,
) -> None:
    """job_id of 'abc' or a list is unusable; raise a clean ValueError."""
    executor = ToolExecutor(seeded_conn)
    with pytest.raises(ValueError, match="job_id"):
        executor.execute("mark_job_state", {"job_id": "not-a-number", "state": "saved"})


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
# Argument coercers (_opt_str / _opt_int / _opt_str_list)
# ---------------------------------------------------------------------------


def test_opt_str_coerces_non_string_input_to_string() -> None:
    """When the agent hands us a non-string value, fall back to str()."""
    from jobai.agent.tools import _opt_str  # noqa: PLC0415

    assert _opt_str(42) == "42"
    assert _opt_str(None) is None
    assert _opt_str("") is None  # empty string -> None
    assert _opt_str("kept") == "kept"


def test_opt_str_list_handles_every_input_shape() -> None:
    """Accepts JSON arrays (preferred) and comma-separated strings
    (defensive). Empty tokens drop out; non-str/non-list inputs map
    to None."""
    from jobai.agent.tools import _opt_str_list  # noqa: PLC0415

    assert _opt_str_list(None) is None
    assert _opt_str_list("senior, ,lead,") == ["senior", "lead"]
    assert _opt_str_list(["senior", "  ", "lead"]) == ["senior", "lead"]
    # Non-string / non-list -> None
    assert _opt_str_list(42) is None
    # All-empty list resolves to None too (cleaned or None).
    assert _opt_str_list(["", "  "]) is None


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


# ---------------------------------------------------------------------------
# kick_tailor / list_tailor_runs / get_tailor_run
# ---------------------------------------------------------------------------


class _FakePool:
    """Captures factories submitted by the kick_tailor handler.

    The real :class:`TailorPool` wraps an asyncio.Semaphore + a task
    group; for unit tests we just need a sink that records the
    submission. The handler doesn't await the factory itself."""

    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def submit(self, factory: Any) -> None:
        self.submitted.append(factory)


def _build_executor(
    conn: sqlite3.Connection,
    *,
    pool: _FakePool,
    db_path: Path,
) -> ToolExecutor:
    """Construct a :class:`ToolExecutor` with tailor deps stubbed.

    Cast through ``Any`` so the structural Protocol mismatch with the
    test fakes doesn't trip mypy -- the real Protocols aren't
    @runtime_checkable and the handler only touches duck-typed methods.
    """
    return ToolExecutor(
        conn,
        tailor_pool=pool,  # type: ignore[arg-type]
        resume_client=_NoOpResumeClient(),
        letter_client=_NoOpResumeClient(),
        db_path=db_path,
    )


class _NoOpResumeClient:
    """Stub Protocol impl -- never actually invoked under unit tests."""

    async def kick(self, request: Any) -> str:
        return "rs_unused"

    async def poll(self, run_id: str) -> Any:
        del run_id
        return None

    async def get_run(self, run_id: str) -> dict[str, Any]:
        del run_id
        return {}

    async def stream_pdf(self, run_id: str) -> Any:
        del run_id
        return None


def test_kick_tailor_creates_row_and_submits_factory(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    result = executor.execute("kick_tailor", {"job_id": 1})
    assert result["job_id"] == 1
    assert result["status"] == "pending"
    assert isinstance(result["tailor_run_id"], int)
    assert len(pool.submitted) == 1


def test_kick_tailor_returns_error_when_tailor_deps_missing(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("kick_tailor", {"job_id": 1})
    assert "error" in result
    assert "not wired" in result["error"]


def test_kick_tailor_returns_error_for_unknown_job(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    result = executor.execute("kick_tailor", {"job_id": 999_999})
    assert result == {"error": "job 999999 not found"}
    assert pool.submitted == []


def test_kick_tailor_raises_when_neither_job_id_nor_jd_url(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    with pytest.raises(ValueError, match="job_id or jd_url"):
        executor.execute("kick_tailor", {})


def test_kick_tailor_raises_when_both_job_id_and_jd_url(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    with pytest.raises(ValueError, match="not both"):
        executor.execute(
            "kick_tailor",
            {"job_id": 1, "jd_url": "https://example.com/jd"},
        )


def test_kick_tailor_raises_when_job_id_not_an_int(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    with pytest.raises(ValueError, match="integer"):
        executor.execute("kick_tailor", {"job_id": "not-an-int"})


def test_kick_tailor_jd_url_matches_catalogue(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """When the URL hits a catalogue row, the run uses the normal
    job_id path and reports matched_job_id."""
    # Seed a job with an apply_url we can target.
    seeded_conn.execute(
        "UPDATE jobs SET apply_url = ? WHERE id = ?",
        ("https://example.com/jd-1", 1),
    )
    seeded_conn.commit()
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    result = executor.execute(
        "kick_tailor",
        {"jd_url": "https://example.com/jd-1?trid=abc"},
    )
    assert result["matched_job_id"] == 1
    assert result["matched_count"] == 1
    assert result["job_id"] == 1
    assert result["jd_url"] is None
    assert len(pool.submitted) == 1


def test_kick_tailor_jd_url_falls_back_when_no_catalogue_match(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """URLs that don't hit the catalogue still kick a chain; the row
    carries the URL directly."""
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    result = executor.execute(
        "kick_tailor",
        {"jd_url": "https://strange.example/never-scraped"},
    )
    assert result["matched_job_id"] is None
    assert result["matched_count"] == 0
    assert result["job_id"] is None
    assert result["jd_url"] == "https://strange.example/never-scraped"
    assert len(pool.submitted) == 1


def test_list_tailor_runs_returns_empty_when_none_kicked(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("list_tailor_runs", {})
    assert result == {"items": []}


def test_list_tailor_runs_returns_records_after_kick(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    executor.execute("kick_tailor", {"job_id": 1})
    executor.execute("kick_tailor", {"job_id": 2})
    listed = executor.execute("list_tailor_runs", {})
    assert len(listed["items"]) == 2
    # Newest-first.
    assert listed["items"][0]["job_id"] == 2
    assert listed["items"][1]["job_id"] == 1


def test_list_tailor_runs_filters_by_job(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    executor.execute("kick_tailor", {"job_id": 1})
    executor.execute("kick_tailor", {"job_id": 2})
    listed = executor.execute("list_tailor_runs", {"job_id": 1})
    assert all(item["job_id"] == 1 for item in listed["items"])


def test_list_tailor_runs_filters_by_status(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    executor.execute("kick_tailor", {"job_id": 1})
    listed = executor.execute("list_tailor_runs", {"status": "pending"})
    assert len(listed["items"]) == 1
    listed_other = executor.execute("list_tailor_runs", {"status": "succeeded"})
    assert listed_other["items"] == []


def test_list_tailor_runs_rejects_bad_status(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    with pytest.raises(ValueError, match="status must be one of"):
        executor.execute("list_tailor_runs", {"status": "bogus"})


def test_list_tailor_runs_rejects_non_int_limit(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    with pytest.raises(ValueError, match="limit must be an integer"):
        executor.execute("list_tailor_runs", {"limit": "not-an-int"})


def test_list_tailor_runs_clamps_oversized_limit(
    seeded_conn: sqlite3.Connection,
) -> None:
    """The handler caps ``limit`` at 100 even when the model asks for more."""
    executor = ToolExecutor(seeded_conn)
    # Won't raise; we just confirm it returns cleanly with the clamp applied.
    result = executor.execute("list_tailor_runs", {"limit": 5_000})
    assert result == {"items": []}


def test_list_tailor_runs_raises_minimum_limit(
    seeded_conn: sqlite3.Connection,
) -> None:
    """``limit=0`` clamps up to 1, which is still a valid value."""
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("list_tailor_runs", {"limit": 0})
    assert result == {"items": []}


def test_get_tailor_run_returns_record(
    seeded_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    pool = _FakePool()
    executor = _build_executor(seeded_conn, pool=pool, db_path=tmp_path / "test.db")
    kicked = executor.execute("kick_tailor", {"job_id": 1})
    got = executor.execute("get_tailor_run", {"tailor_run_id": kicked["tailor_run_id"]})
    assert got["id"] == kicked["tailor_run_id"]
    assert got["job_id"] == 1


def test_get_tailor_run_returns_error_for_missing(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    result = executor.execute("get_tailor_run", {"tailor_run_id": 99_999})
    assert result == {"error": "tailor run 99999 not found"}


def test_get_tailor_run_raises_when_id_missing(
    seeded_conn: sqlite3.Connection,
) -> None:
    executor = ToolExecutor(seeded_conn)
    with pytest.raises(ValueError, match="tailor_run_id"):
        executor.execute("get_tailor_run", {})
