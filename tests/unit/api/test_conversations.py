"""Tests for the ``/api/conversations`` endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from jobai.agent.conversations import append_message, create_conversation


def _seed(db_path: Path, *, title: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conv = create_conversation(conn, title=title)
        append_message(
            conn,
            conversation_id=conv.id,
            role="user",
            content="hello",
        )
        append_message(
            conn,
            conversation_id=conv.id,
            role="assistant",
            content=[{"type": "text", "text": "hi back"}],
        )
        return conv.id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_returns_empty_when_no_conversations(client: TestClient) -> None:
    response = client.get("/api/conversations")
    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_list_orders_by_updated_at_desc(client: TestClient, db_path: Path) -> None:
    older = _seed(db_path, title="older thread")
    newer = _seed(db_path, title="newer thread")

    response = client.get("/api/conversations")
    assert response.status_code == 200
    items = response.json()["items"]
    assert [i["id"] for i in items] == [newer, older]
    assert items[0]["title"] == "newer thread"


def test_list_pagination(client: TestClient, db_path: Path) -> None:
    ids = [_seed(db_path, title=f"thread {i}") for i in range(3)]

    page = client.get("/api/conversations", params={"limit": 2, "offset": 0}).json()
    assert len(page["items"]) == 2
    # Newest first → ids[2], ids[1]
    assert [i["id"] for i in page["items"]] == [ids[2], ids[1]]

    page2 = client.get("/api/conversations", params={"limit": 2, "offset": 2}).json()
    assert len(page2["items"]) == 1
    assert page2["items"][0]["id"] == ids[0]


def test_list_rejects_invalid_limit(client: TestClient) -> None:
    assert client.get("/api/conversations", params={"limit": 0}).status_code == 422
    assert client.get("/api/conversations", params={"limit": 1000}).status_code == 422


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_detail_returns_messages_oldest_first(client: TestClient, db_path: Path) -> None:
    conv_id = _seed(db_path, title="t")

    response = client.get(f"/api/conversations/{conv_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == conv_id
    assert body["title"] == "t"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "hello"
    assert body["messages"][1]["content"] == [{"type": "text", "text": "hi back"}]


def test_detail_404_when_missing(client: TestClient) -> None:
    response = client.get("/api/conversations/9999")
    assert response.status_code == 404
    assert "9999" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_conversation_and_messages(
    client: TestClient,
    db_path: Path,
) -> None:
    conv_id = _seed(db_path, title="to delete")

    response = client.delete(f"/api/conversations/{conv_id}")
    assert response.status_code == 204

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        convs = conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchall()
        msgs = conn.execute(
            "SELECT id FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchall()
    finally:
        conn.close()
    assert convs == []
    assert msgs == []


def test_delete_404_when_missing(client: TestClient) -> None:
    response = client.delete("/api/conversations/9999")
    assert response.status_code == 404


def test_patch_rename_updates_title(client: TestClient, db_path: Path) -> None:
    """PATCH /api/conversations/{id} with a new title round-trips."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO conversations (title) VALUES (?)",
            ("orig",),
        )
        conv_id = int(cursor.lastrowid or 0)
        conn.commit()
    finally:
        conn.close()

    response = client.patch(
        f"/api/conversations/{conv_id}",
        json={"title": "renamed"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == conv_id
    assert body["title"] == "renamed"


def test_patch_rename_404_when_missing(client: TestClient) -> None:
    response = client.patch("/api/conversations/9999", json={"title": "new"})
    assert response.status_code == 404


def test_patch_rename_422_when_blank_title(client: TestClient, db_path: Path) -> None:
    """Whitespace-only titles are rejected by the persistence layer's
    validator; the route maps that to a 422."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO conversations (title) VALUES (?)",
            ("orig",),
        )
        conv_id = int(cursor.lastrowid or 0)
        conn.commit()
    finally:
        conn.close()
    response = client.patch(f"/api/conversations/{conv_id}", json={"title": "   "})
    assert response.status_code == 422
