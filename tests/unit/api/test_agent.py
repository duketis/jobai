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


def test_derive_title_returns_fallback_for_whitespace_message() -> None:
    """Pydantic min_length=1 prevents an all-empty post, but a message
    that's just whitespace (already validated as len>0) lands at the
    title helper with nothing to slice. The fallback fires."""
    from jobai.api.routes.agent import _derive_title  # noqa: PLC0415

    assert _derive_title("   \n  ") == "Untitled conversation"


def test_load_history_returns_empty_when_no_prior_messages(
    app_with_fake_client: FastAPI,
    db_path: Path,
) -> None:
    """``_load_history`` returns [] for a brand-new conversation with no
    persisted messages (defensive: the route appends the user turn
    before calling this helper, but an empty list is still handled)."""
    del app_with_fake_client
    from jobai.agent.conversations import create_conversation  # noqa: PLC0415
    from jobai.api.routes.agent import _load_history  # noqa: PLC0415

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        convo = create_conversation(conn, title="t")
        history = _load_history(conn, convo.id)
    finally:
        conn.close()
    assert history == []


def test_chat_subscription_backend_routes_through_subscription_loop(
    app_with_fake_client: FastAPI,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When agent_backend='subscription', the SSE stream comes from
    run_subscription_chat_turn (not the API loop). We swap that
    function for a fake so the test exercises the route's branch
    without spawning a real claude CLI."""
    from jobai.agent.loop import StreamEvent  # noqa: PLC0415
    from jobai.api import routes  # noqa: PLC0415

    async def fake_subscription_turn(**kwargs: Any) -> AsyncIterator[StreamEvent]:
        del kwargs
        yield StreamEvent(type="text_delta", data={"text": "sub-mode"})
        yield StreamEvent(
            type="done",
            data={"stop_reason": "end_turn", "usage": {}},
        )

    monkeypatch.setattr(
        routes.agent,
        "run_subscription_chat_turn",
        fake_subscription_turn,
    )
    # Set agent_backend=subscription via runtime_settings.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO app_settings (key, value, updated_at) "
            "VALUES ('agent_backend', 'subscription', datetime('now'))",
        )
        conn.commit()
    finally:
        conn.close()

    with TestClient(app_with_fake_client) as client:
        response = client.post("/api/agent/chat", json={"message": "hi"})
    events = _parse_sse(response.text)
    types = [ev for ev, _ in events]
    assert "text_delta" in types
    assert any(payload.get("text") == "sub-mode" for _, payload in events)


def test_chat_surfaces_runtime_failure_as_sse_error_event(
    app_with_fake_client: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected exception inside the loop must surface as an SSE
    ``error`` event rather than tearing down the HTTP response with a 500."""
    from jobai.api import routes  # noqa: PLC0415

    async def boom(**kwargs: Any) -> AsyncIterator[Any]:
        del kwargs
        msg = "loop went bang"
        raise RuntimeError(msg)
        # Unreachable yield keeps this an async generator at the type level.
        yield None  # type: ignore[unreachable]  # pragma: no cover

    monkeypatch.setattr(routes.agent, "run_chat_turn", boom)

    with TestClient(app_with_fake_client) as client:
        response = client.post("/api/agent/chat", json={"message": "hi"})
    events = _parse_sse(response.text)
    error_events = [(t, d) for t, d in events if t == "error"]
    assert error_events
    assert error_events[0][1]["error"] == "loop went bang"


def test_persist_turn_handles_tool_result_without_paired_assistant(
    db_path: Path,
) -> None:
    """``_persist_turn`` zips assistant + tool_result messages. If the
    tool_result list is longer (eg a tool ran but the assistant turn
    that triggered it didn't make it into the list), the loop must
    still persist the orphan tool_result rather than crashing."""
    from jobai.agent.conversations import create_conversation, list_messages  # noqa: PLC0415
    from jobai.agent.loop import TurnResult  # noqa: PLC0415
    from jobai.api.routes.agent import _persist_turn  # noqa: PLC0415

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        convo = create_conversation(conn, title="t")
        result = TurnResult()
        result.tool_result_messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}],
            },
        )
        _persist_turn(conn, convo.id, result)
        messages = list_messages(conn, convo.id)
    finally:
        conn.close()
    # The orphan tool_result landed as a 'user' role row.
    assert any(m.role == "user" and "tool_result" in str(m.content) for m in messages)
