"""Tests for /api/sources."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from jobai.sources.repository import set_enabled, upsert_source


def test_list_sources_empty(client: TestClient) -> None:
    body = client.get("/api/sources").json()
    assert body == {"items": []}


def test_list_sources_returns_each_configured_source(
    client: TestClient,
    db_path: Path,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        upsert_source(conn, kind="greenhouse", account="atlassian", display_name="A")
        upsert_source(conn, kind="lever", account="palantir", display_name="P")
    finally:
        conn.close()

    body = client.get("/api/sources").json()

    names = [item["name"] for item in body["items"]]
    assert names == ["greenhouse:atlassian", "lever:palantir"]


def test_list_sources_includes_runtime_state_when_present(
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
        conn.execute(
            "INSERT INTO source_runtime_state "
            "(source_id, current_tier, last_success_at, consecutive_failures) "
            "VALUES (?, 1, ?, 0)",
            (row.id, datetime.now(tz=UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

    item = client.get("/api/sources").json()["items"][0]
    assert item["current_tier"] == 1
    assert item["last_success_at"] is not None
    assert item["consecutive_failures"] == 0


def test_list_sources_returns_nulls_for_sources_with_no_runtime_state(
    client: TestClient,
    db_path: Path,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        upsert_source(conn, kind="greenhouse", account="atlassian", display_name="A")
    finally:
        conn.close()

    item = client.get("/api/sources").json()["items"][0]
    assert item["current_tier"] is None
    assert item["last_success_at"] is None
    assert item["consecutive_failures"] == 0


def test_list_sources_enabled_only_filter(
    client: TestClient,
    db_path: Path,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        upsert_source(conn, kind="greenhouse", account="a", display_name="A")
        upsert_source(conn, kind="greenhouse", account="b", display_name="B")
        set_enabled(conn, kind="greenhouse", account="b", enabled=False)
    finally:
        conn.close()

    body = client.get("/api/sources", params={"enabled_only": "true"}).json()
    names = [item["name"] for item in body["items"]]
    assert names == ["greenhouse:a"]
