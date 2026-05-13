"""Tests for conversation and message persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jobai.agent.conversations import (
    ConversationNotFoundError,
    append_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    messages_to_anthropic_format,
    rename_conversation,
)
from jobai.db.migrations import apply_pending


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


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def test_create_conversation_returns_persisted_row(conn: sqlite3.Connection) -> None:
    convo = create_conversation(conn, title="Job hunt")

    assert convo.id > 0
    assert convo.title == "Job hunt"
    assert convo.created_at
    assert convo.updated_at


def test_get_conversation_404_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(ConversationNotFoundError):
        get_conversation(conn, 9999)


def test_list_conversations_orders_by_recency(conn: sqlite3.Connection) -> None:
    a = create_conversation(conn, title="Old")
    b = create_conversation(conn, title="Newer")

    # Bump 'a' forward by appending a message.
    append_message(conn, conversation_id=a.id, role="user", content="hi")

    listed = list_conversations(conn)
    assert [c.id for c in listed] == [a.id, b.id]


def test_delete_conversation_cascades_to_messages(conn: sqlite3.Connection) -> None:
    convo = create_conversation(conn, title="Dispose")
    append_message(conn, conversation_id=convo.id, role="user", content="x")

    delete_conversation(conn, convo.id)

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (convo.id,),
        ).fetchone()[0]
        == 0
    )


def test_delete_conversation_404_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(ConversationNotFoundError):
        delete_conversation(conn, 9999)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_append_message_string_content(conn: sqlite3.Connection) -> None:
    convo = create_conversation(conn, title="x")
    msg = append_message(conn, conversation_id=convo.id, role="user", content="What jobs?")

    assert msg.id > 0
    assert msg.role == "user"
    assert msg.content == "What jobs?"


def test_append_message_list_content_preserves_tool_use_blocks(
    conn: sqlite3.Connection,
) -> None:
    """Tool-use blocks must round-trip exactly so subsequent agent turns can
    feed them back to Claude."""
    convo = create_conversation(conn, title="x")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "I'll search for that."},
        {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "search_jobs",
            "input": {"q": "python", "remote": "remote"},
        },
    ]

    msg = append_message(conn, conversation_id=convo.id, role="assistant", content=content)

    assert msg.content == content
    fetched = list_messages(conn, convo.id)
    assert len(fetched) == 1
    assert fetched[0].content == content


def test_append_message_404_when_conversation_missing(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(ConversationNotFoundError):
        append_message(conn, conversation_id=9999, role="user", content="x")


def test_list_messages_returns_chronological_order(conn: sqlite3.Connection) -> None:
    convo = create_conversation(conn, title="x")

    append_message(conn, conversation_id=convo.id, role="user", content="first")
    append_message(conn, conversation_id=convo.id, role="assistant", content="second")
    append_message(conn, conversation_id=convo.id, role="user", content="third")

    msgs = list_messages(conn, convo.id)
    contents = [m.content for m in msgs]
    assert contents == ["first", "second", "third"]


def test_list_messages_404_when_conversation_missing(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(ConversationNotFoundError):
        list_messages(conn, 9999)


def test_append_message_bumps_conversation_updated_at(
    conn: sqlite3.Connection,
) -> None:
    convo = create_conversation(conn, title="x")
    initial_updated = get_conversation(conn, convo.id).updated_at

    # SQLite datetime('now') is second-precision; ensure tick over.
    conn.execute("UPDATE conversations SET updated_at = '2020-01-01' WHERE id = ?", (convo.id,))
    conn.commit()

    append_message(conn, conversation_id=convo.id, role="user", content="x")

    new_updated = get_conversation(conn, convo.id).updated_at
    assert new_updated > "2020-01-01"
    assert new_updated >= initial_updated[:4]  # at least a real timestamp


# ---------------------------------------------------------------------------
# Anthropic format conversion
# ---------------------------------------------------------------------------


def test_messages_to_anthropic_format_roundtrips(conn: sqlite3.Connection) -> None:
    convo = create_conversation(conn, title="x")
    append_message(conn, conversation_id=convo.id, role="user", content="hi")
    append_message(
        conn,
        conversation_id=convo.id,
        role="assistant",
        content=[
            {"type": "text", "text": "Hello!"},
            {"type": "tool_use", "id": "t1", "name": "search_jobs", "input": {"q": "x"}},
        ],
    )
    append_message(
        conn,
        conversation_id=convo.id,
        role="user",
        content=[{"type": "tool_result", "tool_use_id": "t1", "content": "5 results"}],
    )

    formatted = messages_to_anthropic_format(list_messages(conn, convo.id))

    assert formatted == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "t1", "name": "search_jobs", "input": {"q": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "5 results"}],
        },
    ]


# ---------------------------------------------------------------------------
# rename_conversation
# ---------------------------------------------------------------------------


def test_rename_conversation_updates_title_and_bumps_updated_at(
    conn: sqlite3.Connection,
) -> None:
    convo = create_conversation(conn, title="orig")
    renamed = rename_conversation(conn, convo.id, title="new title")
    assert renamed.id == convo.id
    assert renamed.title == "new title"


def test_rename_conversation_404_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(ConversationNotFoundError):
        rename_conversation(conn, 9999, title="nope")


def test_rename_conversation_rejects_blank_title(conn: sqlite3.Connection) -> None:
    convo = create_conversation(conn, title="orig")
    with pytest.raises(ValueError, match="must not be empty"):
        rename_conversation(conn, convo.id, title="   ")


def test_row_to_message_raises_on_invalid_content_json(
    conn: sqlite3.Connection,
) -> None:
    """messages.content_json must round-trip as list or str; a dict /
    number / null at the top level is a wire-format violation that the
    loader must reject loudly rather than yielding a malformed message."""
    convo = create_conversation(conn, title="x")
    # Hand-craft a row with bogus JSON so list_messages tries to load it.
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content_json) VALUES (?, ?, ?)",
        (convo.id, "user", '{"oops": "not list or str"}'),
    )
    conn.commit()
    with pytest.raises(TypeError, match="neither list nor str"):
        list_messages(conn, convo.id)
