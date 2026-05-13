"""Conversation + message persistence for the agent.

The Anthropic API is stateless. Every chat turn we send the entire
prior conversation back. These helpers keep that history in SQLite so
the same conversation can resume across process restarts.

Messages store the full Anthropic content array (text, tool_use,
tool_result blocks) as JSON. Round-tripping through JSON preserves the
exact shape Claude needs for the next turn — title strings would lose
the tool-use blocks that drive multi-turn agentic loops.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Conversation:
    """One chat thread."""

    id: int
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """One message in a conversation, content already deserialised."""

    id: int
    conversation_id: int
    role: str
    content: list[dict[str, Any]] | str
    created_at: str


class ConversationNotFoundError(LookupError):
    """Raised when a conversation lookup misses."""

    def __init__(self, conversation_id: int) -> None:
        super().__init__(f"conversation {conversation_id} not found")
        self.conversation_id = conversation_id


def create_conversation(conn: sqlite3.Connection, *, title: str) -> Conversation:
    """Insert a new conversation row and return it."""
    cursor = conn.execute(
        "INSERT INTO conversations (title) VALUES (?)",
        (title,),
    )
    conn.commit()
    last_id = cursor.lastrowid
    # SQLite always returns lastrowid on a successful INSERT; defensive only.
    if last_id is None:  # pragma: no cover
        raise RuntimeError("INSERT INTO conversations returned no lastrowid")
    return _row_to_conversation(_must_fetch(conn, int(last_id)))


def get_conversation(conn: sqlite3.Connection, conversation_id: int) -> Conversation:
    return _row_to_conversation(_must_fetch(conn, conversation_id))


def list_conversations(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[Conversation]:
    # Secondary `id DESC` breaks ties when two conversations share the
    # same millisecond `updated_at` — without it ordering on ties is
    # undefined and the recent-conversations list flickers.
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_to_conversation(r) for r in rows]


def delete_conversation(conn: sqlite3.Connection, conversation_id: int) -> None:
    cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    if cursor.rowcount == 0:
        raise ConversationNotFoundError(conversation_id)
    conn.commit()


def rename_conversation(
    conn: sqlite3.Connection,
    conversation_id: int,
    *,
    title: str,
) -> Conversation:
    """Update the conversation's title and bump ``updated_at``.

    Raises :class:`ConversationNotFoundError` if the row doesn't
    exist; that surfaces as a 404 in the HTTP layer.
    """
    cleaned = title.strip()
    if not cleaned:
        msg = "title must not be empty"
        raise ValueError(msg)
    cursor = conn.execute(
        "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
        (cleaned, conversation_id),
    )
    if cursor.rowcount == 0:
        raise ConversationNotFoundError(conversation_id)
    conn.commit()
    return get_conversation(conn, conversation_id)


def append_message(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
    role: str,
    content: list[dict[str, Any]] | str,
) -> StoredMessage:
    """Append a new message and bump the conversation's ``updated_at``."""
    if (
        conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        is None
    ):
        raise ConversationNotFoundError(conversation_id)

    content_json = json.dumps(content)
    cursor = conn.execute(
        "INSERT INTO messages (conversation_id, role, content_json) VALUES (?, ?, ?)",
        (conversation_id, role, content_json),
    )
    # Millisecond precision (`%f`) avoids ties when two messages land in
    # the same second — the list endpoint sorts by ``updated_at DESC``
    # and we want a stable ordering for the recent-conversations view.
    conn.execute(
        "UPDATE conversations SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') WHERE id = ?",
        (conversation_id,),
    )
    conn.commit()

    last_id = cursor.lastrowid
    # SQLite always returns lastrowid on a successful INSERT; defensive only.
    if last_id is None:  # pragma: no cover
        raise RuntimeError("INSERT INTO messages returned no lastrowid")
    row = conn.execute(
        "SELECT id, conversation_id, role, content_json, created_at FROM messages WHERE id = ?",
        (int(last_id),),
    ).fetchone()
    return _row_to_message(row)


def list_messages(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> list[StoredMessage]:
    """Return every message in a conversation, oldest first."""
    if (
        conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        is None
    ):
        raise ConversationNotFoundError(conversation_id)

    rows = conn.execute(
        "SELECT id, conversation_id, role, content_json, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    return [_row_to_message(r) for r in rows]


def messages_to_anthropic_format(
    messages: Sequence[StoredMessage],
) -> list[dict[str, Any]]:
    """Translate stored messages into the shape ``messages.create`` expects."""
    return [{"role": m.role, "content": m.content} for m in messages]


def _must_fetch(conn: sqlite3.Connection, conversation_id: int) -> sqlite3.Row | tuple[Any, ...]:
    row: sqlite3.Row | tuple[Any, ...] | None = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        raise ConversationNotFoundError(conversation_id)
    return row


def _row_to_conversation(row: sqlite3.Row | tuple[Any, ...]) -> Conversation:
    return Conversation(
        id=int(row[0]),
        title=str(row[1]),
        created_at=str(row[2]),
        updated_at=str(row[3]),
    )


def _row_to_message(row: sqlite3.Row | tuple[Any, ...]) -> StoredMessage:
    raw_content = json.loads(row[3])
    if not isinstance(raw_content, (list, str)):
        msg = (
            f"messages.content_json for id={row[0]} is neither list nor str: "
            f"{type(raw_content).__name__}"
        )
        raise TypeError(msg)
    return StoredMessage(
        id=int(row[0]),
        conversation_id=int(row[1]),
        role=str(row[2]),
        content=raw_content,
        created_at=str(row[4]),
    )
