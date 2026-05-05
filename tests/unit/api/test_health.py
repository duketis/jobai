"""Tests for the /api/health endpoint."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from jobai.sources.repository import upsert_source


def test_health_returns_200_on_empty_db(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["jobs_total"] == 0
    assert body["jobs_added_24h"] == 0
    assert body["sources_total"] == 0
    assert body["sources_enabled"] == 0
    assert body["sources_failing"] == 0
    assert "timestamp" in body


def test_health_counts_jobs_and_sources(
    client: TestClient,
    db_path: Path,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        upsert_source(conn, kind="greenhouse", account="atlassian", display_name="A")
        upsert_source(conn, kind="lever", account="palantir", display_name="P")
        # Insert one canonical job (recent)
        conn.execute(
            "INSERT INTO jobs ("
            "  dedup_key, title, company, company_norm, apply_url, "
            "  first_seen_at, last_seen_at, fingerprint_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "deadbeef",
                "Engineer",
                "X",
                "x",
                "https://example.com/1",
                datetime.now(tz=UTC).isoformat(),
                datetime.now(tz=UTC).isoformat(),
                "{}",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    response = client.get("/api/health")
    body = response.json()

    assert body["jobs_total"] == 1
    assert body["jobs_added_24h"] == 1
    assert body["sources_total"] == 2
    assert body["sources_enabled"] == 2


def test_health_marks_degraded_when_source_recently_failed(
    client: TestClient,
    db_path: Path,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        row = upsert_source(
            conn,
            kind="greenhouse",
            account="atlassian",
            display_name="A",
        )
        # Source has failed within 24h with no later success.
        conn.execute(
            "INSERT INTO source_runtime_state "
            "(source_id, current_tier, last_error_at, last_error_class) "
            "VALUES (?, 1, ?, 'network')",
            (row.id, datetime.now(tz=UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    body = client.get("/api/health").json()

    assert body["status"] == "degraded"
    assert body["sources_failing"] == 1


def test_health_status_ok_when_failure_predates_success(
    client: TestClient,
    db_path: Path,
) -> None:
    """A source that recovered (success after error) must NOT count as failing."""
    conn = sqlite3.connect(db_path)
    try:
        row = upsert_source(
            conn,
            kind="greenhouse",
            account="atlassian",
            display_name="A",
        )
        conn.execute(
            "INSERT INTO source_runtime_state "
            "(source_id, current_tier, last_error_at, last_success_at) "
            "VALUES (?, 1, datetime('now', '-2 hours'), datetime('now', '-1 hours'))",
            (row.id,),
        )
        conn.commit()
    finally:
        conn.close()

    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["sources_failing"] == 0


def test_health_excludes_old_failures_from_failing_count(
    client: TestClient,
    db_path: Path,
) -> None:
    """A failure older than 24h with no recovery should not flag the source as
    'currently failing'."""
    conn = sqlite3.connect(db_path)
    try:
        row = upsert_source(
            conn,
            kind="greenhouse",
            account="atlassian",
            display_name="A",
        )
        conn.execute(
            "INSERT INTO source_runtime_state "
            "(source_id, current_tier, last_error_at) "
            "VALUES (?, 1, datetime('now', '-2 days'))",
            (row.id,),
        )
        conn.commit()
    finally:
        conn.close()

    body = client.get("/api/health").json()
    assert body["sources_failing"] == 0
