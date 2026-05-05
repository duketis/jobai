"""Tests for the ``POST /api/agent/chat`` SSE endpoint.

We override the Anthropic client dependency with a fake that yields
canned stream events — same pattern as ``tests/unit/agent/test_loop.py``
but driven through the FastAPI app so we also assert persistence and
SSE wire format.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jobai.api.dependencies import get_anthropic_client, get_anthropic_model

# ---------------------------------------------------------------------------
# Fakes — mirror the shape the real SDK exposes to the loop
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async-context-manager + async iterator returning canned events."""

    def __init__(self, events: list[Any], final: Any) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    async def get_final_message(self) -> Any:
        return self._final


class _FakeMessagesAPI:
    def __init__(self, streams: list[_FakeStream]) -> None:
        # Hold a reference (not a copy) so tests can append after the
        # fixture has constructed the fake.
        self._streams = streams
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        return self._streams.pop(0)


class _FakeClient:
    def __init__(self, streams: list[_FakeStream]) -> None:
        self.messages = _FakeMessagesAPI(streams)


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _tool_use_start(*, tool_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type="tool_use", id=tool_id, name=name),
    )


def _final_message(
    *,
    content: list[dict[str, Any]],
    stop_reason: str,
) -> SimpleNamespace:
    blocks = [
        SimpleNamespace(type=b["type"], **{k: v for k, v in b.items() if k != "type"})
        for b in content
    ]
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_streams() -> list[_FakeStream]:
    """Per-test list of canned streams. Tests append to this before
    issuing the request."""
    return []


@pytest.fixture
def app_with_fake_client(app: FastAPI, fake_streams: list[_FakeStream]) -> FastAPI:
    """Wire the fake Anthropic client into the app."""
    fake = _FakeClient(fake_streams)
    app.dependency_overrides[get_anthropic_client] = lambda: fake
    app.dependency_overrides[get_anthropic_model] = lambda: "claude-opus-4-7-test"
    # Stash on app so tests can introspect calls.
    app.state.fake_client = fake
    return app


@pytest.fixture
def chat_client(app_with_fake_client: FastAPI) -> Iterator[TestClient]:
    with TestClient(app_with_fake_client) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Helpers — parse SSE response body into a list of (event, data) tuples
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE response body into ``(event_type, data)`` tuples.

    SSE separates events by a blank line. ``sse-starlette`` writes
    ``\\r\\n`` line endings, so we normalise to ``\\n`` first.
    """
    normalised = body.replace("\r\n", "\n").replace("\r", "\n")
    events: list[tuple[str, dict[str, Any]]] = []
    for raw_chunk in normalised.split("\n\n"):
        chunk = raw_chunk.strip("\n")
        if not chunk:
            continue
        event_type = ""
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if not event_type:
            continue
        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        events.append((event_type, payload))
    return events


# ---------------------------------------------------------------------------
# Happy path: text-only response creates conversation + persists messages
# ---------------------------------------------------------------------------


def test_chat_creates_conversation_and_streams_text(
    chat_client: TestClient,
    fake_streams: list[_FakeStream],
    db_path: Path,
) -> None:
    fake_streams.append(
        _FakeStream(
            events=[_text_delta("Hello "), _text_delta("there.")],
            final=_final_message(
                content=[{"type": "text", "text": "Hello there."}],
                stop_reason="end_turn",
            ),
        )
    )

    response = chat_client.post(
        "/api/agent/chat",
        json={"message": "hi"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    types = [t for t, _ in events]
    assert types == ["conversation", "text_delta", "text_delta", "done"]

    conv_event = events[0]
    conversation_id = conv_event[1]["conversation_id"]
    assert isinstance(conversation_id, int)
    assert conversation_id > 0

    # User + assistant messages persisted.
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT role, content_json FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert rows[0][0] == "user"
    assert json.loads(rows[0][1]) == "hi"
    assert rows[1][0] == "assistant"
    assert json.loads(rows[1][1]) == [{"type": "text", "text": "Hello there."}]


def test_chat_continues_existing_conversation(
    chat_client: TestClient,
    fake_streams: list[_FakeStream],
    db_path: Path,
) -> None:
    """A second call with the same conversation_id appends rather than creates."""
    fake_streams.extend(
        [
            _FakeStream(
                events=[_text_delta("First.")],
                final=_final_message(
                    content=[{"type": "text", "text": "First."}],
                    stop_reason="end_turn",
                ),
            ),
            _FakeStream(
                events=[_text_delta("Second.")],
                final=_final_message(
                    content=[{"type": "text", "text": "Second."}],
                    stop_reason="end_turn",
                ),
            ),
        ]
    )

    first = chat_client.post("/api/agent/chat", json={"message": "turn 1"})
    first_events = _parse_sse(first.text)
    conversation_id = first_events[0][1]["conversation_id"]

    second = chat_client.post(
        "/api/agent/chat",
        json={"conversation_id": conversation_id, "message": "turn 2"},
    )
    second_events = _parse_sse(second.text)
    assert second_events[0][1]["conversation_id"] == conversation_id

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM conversations",
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT role FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()
    assert count == 1
    assert [r[0] for r in rows] == ["user", "assistant", "user", "assistant"]


def test_chat_history_is_threaded_into_request(
    chat_client: TestClient,
    fake_streams: list[_FakeStream],
    app_with_fake_client: FastAPI,
) -> None:
    """The second call must include the first turn's messages in the request."""
    fake_streams.extend(
        [
            _FakeStream(
                events=[],
                final=_final_message(
                    content=[{"type": "text", "text": "first reply"}],
                    stop_reason="end_turn",
                ),
            ),
            _FakeStream(
                events=[],
                final=_final_message(
                    content=[{"type": "text", "text": "second reply"}],
                    stop_reason="end_turn",
                ),
            ),
        ]
    )

    first = chat_client.post("/api/agent/chat", json={"message": "first"})
    conversation_id = _parse_sse(first.text)[0][1]["conversation_id"]
    chat_client.post(
        "/api/agent/chat",
        json={"conversation_id": conversation_id, "message": "second"},
    )

    fake = app_with_fake_client.state.fake_client
    second_call = fake.messages.calls[1]
    sent_messages = second_call["messages"]
    # Expect: prior user turn, prior assistant turn, current user turn
    assert sent_messages[0]["role"] == "user"
    assert sent_messages[0]["content"] == "first"
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[1]["content"] == [{"type": "text", "text": "first reply"}]
    assert sent_messages[2] == {"role": "user", "content": "second"}


# ---------------------------------------------------------------------------
# Tool round trip persists tool_result blocks too
# ---------------------------------------------------------------------------


def test_chat_tool_round_trip_persists_tool_result(
    chat_client: TestClient,
    fake_streams: list[_FakeStream],
    db_path: Path,
) -> None:
    fake_streams.extend(
        [
            _FakeStream(
                events=[_tool_use_start(tool_id="toolu_1", name="list_sources")],
                final=_final_message(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "list_sources",
                            "input": {},
                        }
                    ],
                    stop_reason="tool_use",
                ),
            ),
            _FakeStream(
                events=[_text_delta("No sources configured.")],
                final=_final_message(
                    content=[{"type": "text", "text": "No sources configured."}],
                    stop_reason="end_turn",
                ),
            ),
        ]
    )

    response = chat_client.post(
        "/api/agent/chat",
        json={"message": "what sources?"},
    )
    events = _parse_sse(response.text)
    types = [t for t, _ in events]

    assert "tool_use_start" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert "done" in types

    conv_id = events[0][1]["conversation_id"]
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT role, content_json FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    finally:
        conn.close()
    # user / assistant(tool_use) / user(tool_result) / assistant(text)
    roles = [r[0] for r in rows]
    assert roles == ["user", "assistant", "user", "assistant"]

    tool_use_payload = json.loads(rows[1][1])
    assert tool_use_payload[0]["type"] == "tool_use"
    assert tool_use_payload[0]["name"] == "list_sources"

    tool_result_payload = json.loads(rows[2][1])
    assert tool_result_payload[0]["type"] == "tool_result"
    assert tool_result_payload[0]["tool_use_id"] == "toolu_1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_chat_rejects_empty_message(chat_client: TestClient) -> None:
    response = chat_client.post("/api/agent/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_with_unknown_conversation_id_starts_fresh(
    chat_client: TestClient,
    fake_streams: list[_FakeStream],
    db_path: Path,
) -> None:
    """Posting with a stale id falls back to a new conversation rather than 404.

    The agent UI may hold a conversation id from a deleted thread; it
    should still get a working stream rather than an error reply.
    """
    fake_streams.append(
        _FakeStream(
            events=[],
            final=_final_message(
                content=[{"type": "text", "text": "ok"}],
                stop_reason="end_turn",
            ),
        )
    )

    response = chat_client.post(
        "/api/agent/chat",
        json={"conversation_id": 9999, "message": "hi"},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    new_id = events[0][1]["conversation_id"]
    assert new_id != 9999

    conn = sqlite3.connect(db_path)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM conversations").fetchall()]
    finally:
        conn.close()
    assert new_id in ids


# ---------------------------------------------------------------------------
# Title derivation truncates long messages
# ---------------------------------------------------------------------------


def test_chat_long_message_title_is_truncated(
    chat_client: TestClient,
    fake_streams: list[_FakeStream],
    db_path: Path,
) -> None:
    fake_streams.append(
        _FakeStream(
            events=[],
            final=_final_message(
                content=[{"type": "text", "text": "ok"}],
                stop_reason="end_turn",
            ),
        )
    )

    long_message = "x" * 500
    response = chat_client.post("/api/agent/chat", json={"message": long_message})
    conv_id = _parse_sse(response.text)[0][1]["conversation_id"]

    conn = sqlite3.connect(db_path)
    try:
        title = conn.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert len(title) <= 80
    assert title.endswith("…")
