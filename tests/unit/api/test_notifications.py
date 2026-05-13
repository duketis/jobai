"""Tests for /api/notifications."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient


def _seed_notification(
    db_path: Path,
    *,
    kind: str = "source_failing",
    severity: str = "warn",
    title: str = "Greenhouse failing",
    body: str | None = None,
) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO notifications (kind, severity, title, body) VALUES (?, ?, ?, ?)",
            (kind, severity, title, body),
        )
        last_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()
    assert last_id is not None
    return int(last_id)


def test_list_notifications_empty(client: TestClient) -> None:
    body = client.get("/api/notifications").json()
    assert body == {"total": 0, "unread_count": 0, "items": []}


def test_list_notifications_returns_seeded_items(
    client: TestClient,
    db_path: Path,
) -> None:
    _seed_notification(db_path, title="Source A failing")
    _seed_notification(db_path, title="Source B schema change", kind="schema_change")

    body = client.get("/api/notifications").json()
    titles = [item["title"] for item in body["items"]]
    assert body["total"] == 2
    assert body["unread_count"] == 2
    assert "Source A failing" in titles
    assert "Source B schema change" in titles


def test_unread_only_filter(client: TestClient, db_path: Path) -> None:
    a = _seed_notification(db_path, title="Already read")
    _seed_notification(db_path, title="Still unread")
    client.post(f"/api/notifications/{a}/read")

    body = client.get("/api/notifications", params={"unread_only": "true"}).json()
    titles = [item["title"] for item in body["items"]]
    assert titles == ["Still unread"]
    assert body["total"] == 1
    assert body["unread_count"] == 1


def test_mark_read_sets_read_at_timestamp(
    client: TestClient,
    db_path: Path,
) -> None:
    nid = _seed_notification(db_path, title="x")
    response = client.post(f"/api/notifications/{nid}/read")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == nid
    assert body["read_at"]


def test_mark_read_is_idempotent(client: TestClient, db_path: Path) -> None:
    nid = _seed_notification(db_path, title="x")
    first = client.post(f"/api/notifications/{nid}/read").json()
    second = client.post(f"/api/notifications/{nid}/read").json()
    # The first read_at value sticks; mark-read does not bump the timestamp.
    assert first["read_at"] == second["read_at"]


def test_mark_read_404_when_missing(client: TestClient) -> None:
    response = client.post("/api/notifications/9999/read")
    assert response.status_code == 404


def test_pagination(client: TestClient, db_path: Path) -> None:
    for i in range(5):
        _seed_notification(db_path, title=f"n{i}")

    body = client.get("/api/notifications", params={"limit": 2, "offset": 0}).json()
    assert len(body["items"]) == 2
    assert body["total"] == 5

    body = client.get("/api/notifications", params={"limit": 2, "offset": 4}).json()
    assert len(body["items"]) == 1


def test_list_surfaces_body_when_present(client: TestClient, db_path: Path) -> None:
    """Notifications with a non-null ``body`` should round-trip the value
    through the response. Exercises the non-null branch of _optional_str."""
    _seed_notification(db_path, title="with-body", body="extra context here")
    body = client.get("/api/notifications").json()
    items = {item["title"]: item for item in body["items"]}
    assert items["with-body"]["body"] == "extra context here"
