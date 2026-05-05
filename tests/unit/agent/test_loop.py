"""Tests for the streaming agent loop.

The loop is tested with a fake ``AsyncAnthropic`` client whose
``messages.stream`` yields canned events and final messages. This
exercises every stop-reason path (end_turn, tool_use, pause_turn,
unknown), the tool round-trip, error propagation from the SDK, and
the max-iterations safety net — without ever calling the real API.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jobai.agent.loop import (
    DEFAULT_MAX_ITERATIONS,
    StreamEvent,
    TurnResult,
    run_chat_turn,
)
from jobai.agent.tools import ToolExecutor
from jobai.db.migrations import apply_pending

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async-context-manager + async iterator returning canned stream events."""

    def __init__(
        self,
        events: list[Any],
        final: Any,
        *,
        raise_on_enter: BaseException | None = None,
    ) -> None:
        self._events = events
        self._final = final
        self._raise = raise_on_enter

    async def __aenter__(self) -> _FakeStream:
        if self._raise is not None:
            raise self._raise
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
        self._streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        return self._streams.pop(0)


class _FakeClient:
    def __init__(self, streams: list[_FakeStream]) -> None:
        self.messages = _FakeMessagesAPI(streams)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def executor(conn: sqlite3.Connection) -> ToolExecutor:
    return ToolExecutor(conn)


# ---------------------------------------------------------------------------
# Helpers to build canned events
# ---------------------------------------------------------------------------


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _thinking_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=text),
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
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> SimpleNamespace:
    """Build a fake `Message` with the fields the loop reads.

    Content blocks are passed as dicts; the loop's `_block_to_dict`
    handles dicts directly.
    """
    blocks = [
        SimpleNamespace(type=b["type"], **{k: v for k, v in b.items() if k != "type"})
        for b in content
    ]
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Happy path: text-only response
# ---------------------------------------------------------------------------


async def test_text_only_response_yields_deltas_and_done(
    executor: ToolExecutor,
) -> None:
    final = _final_message(
        content=[{"type": "text", "text": "Hello there."}],
        stop_reason="end_turn",
    )
    fake_stream = _FakeStream(
        events=[_text_delta("Hello "), _text_delta("there.")],
        final=final,
    )
    client = _FakeClient([fake_stream])
    result = TurnResult()

    events = await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="hi",
            history=[],
            tool_executor=executor,
            result=result,
        )
    )

    types = [e.type for e in events]
    assert types == ["text_delta", "text_delta", "done"]
    assert events[0].data["text"] == "Hello "
    assert events[1].data["text"] == "there."
    assert events[2].data["stop_reason"] == "end_turn"
    assert result.stop_reason == "end_turn"
    assert result.usage["input_tokens"] == 100
    assert result.usage["output_tokens"] == 50


# ---------------------------------------------------------------------------
# Tool use round trip
# ---------------------------------------------------------------------------


async def test_tool_use_round_trip(executor: ToolExecutor) -> None:
    """Iter 1: assistant calls list_sources. Iter 2: assistant returns text."""
    iter1_final = _final_message(
        content=[
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "list_sources",
                "input": {},
            },
        ],
        stop_reason="tool_use",
    )
    iter2_final = _final_message(
        content=[{"type": "text", "text": "You have 0 sources configured."}],
        stop_reason="end_turn",
    )
    streams = [
        _FakeStream(
            events=[_tool_use_start(tool_id="toolu_1", name="list_sources")],
            final=iter1_final,
        ),
        _FakeStream(
            events=[_text_delta("You have 0 sources configured.")],
            final=iter2_final,
        ),
    ]
    client = _FakeClient(streams)
    result = TurnResult()

    events = await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="what sources are configured?",
            history=[],
            tool_executor=executor,
            result=result,
        )
    )

    types = [e.type for e in events]
    # tool_use_start (from stream) -> tool_call -> tool_result -> text_delta -> done
    assert types == ["tool_use_start", "tool_call", "tool_result", "text_delta", "done"]

    tool_call_event = events[1]
    assert tool_call_event.data["name"] == "list_sources"
    assert tool_call_event.data["id"] == "toolu_1"

    tool_result_event = events[2]
    assert tool_result_event.data["id"] == "toolu_1"
    assert tool_result_event.data["result"] == {"items": []}

    assert len(result.assistant_messages) == 2
    assert len(result.tool_result_messages) == 1
    assert result.usage["input_tokens"] == 200  # 100 per iteration


# ---------------------------------------------------------------------------
# Tool error
# ---------------------------------------------------------------------------


async def test_tool_error_emits_event_and_continues_loop(
    executor: ToolExecutor,
) -> None:
    """If a tool raises, the loop emits a tool_error event AND feeds an
    is_error=true tool_result back so the model can recover."""
    iter1_final = _final_message(
        content=[
            {
                "type": "tool_use",
                "id": "toolu_bad",
                "name": "get_job_detail",
                "input": {},  # missing required job_id -> ValueError
            },
        ],
        stop_reason="tool_use",
    )
    iter2_final = _final_message(
        content=[{"type": "text", "text": "I need a job id."}],
        stop_reason="end_turn",
    )
    client = _FakeClient(
        [
            _FakeStream(
                events=[_tool_use_start(tool_id="toolu_bad", name="get_job_detail")],
                final=iter1_final,
            ),
            _FakeStream(events=[], final=iter2_final),
        ]
    )
    result = TurnResult()

    events = await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="show me details",
            history=[],
            tool_executor=executor,
            result=result,
        )
    )

    error_event = next(e for e in events if e.type == "tool_error")
    assert error_event.data["error_class"] == "ValueError"
    assert "job_id" in error_event.data["error"]
    # The tool_result block fed back must carry is_error=true
    tool_result_block = result.tool_result_messages[0]["content"][0]
    assert tool_result_block["is_error"] is True


# ---------------------------------------------------------------------------
# Pause turn
# ---------------------------------------------------------------------------


async def test_pause_turn_loops_back(executor: ToolExecutor) -> None:
    iter1_final = _final_message(
        content=[{"type": "text", "text": "Working..."}],
        stop_reason="pause_turn",
    )
    iter2_final = _final_message(
        content=[{"type": "text", "text": "Done."}],
        stop_reason="end_turn",
    )
    client = _FakeClient(
        [
            _FakeStream(events=[], final=iter1_final),
            _FakeStream(events=[], final=iter2_final),
        ]
    )

    events = await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="search the web for X",
            history=[],
            tool_executor=executor,
        )
    )
    types = [e.type for e in events]
    assert "pause_turn" in types
    assert types[-1] == "done"
    assert events[-1].data["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# Max iterations safety
# ---------------------------------------------------------------------------


async def test_max_iterations_terminates_runaway_loop(
    executor: ToolExecutor,
) -> None:
    """If the model keeps calling tools forever, we cap at max_iterations."""

    def _looping_stream() -> _FakeStream:
        return _FakeStream(
            events=[],
            final=_final_message(
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_loop",
                        "name": "list_sources",
                        "input": {},
                    }
                ],
                stop_reason="tool_use",
            ),
        )

    streams = [_looping_stream() for _ in range(3)]
    client = _FakeClient(streams)

    events = await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="loop forever",
            history=[],
            tool_executor=executor,
            max_iterations=3,
        )
    )

    last = events[-1]
    assert last.type == "done"
    assert last.data["stop_reason"] == "max_iterations"
    assert last.data["iterations"] == 3


# ---------------------------------------------------------------------------
# SDK error propagation
# ---------------------------------------------------------------------------


async def test_sdk_error_surfaces_as_event(executor: ToolExecutor) -> None:
    boom = RuntimeError("network down")
    client = _FakeClient([_FakeStream(events=[], final=None, raise_on_enter=boom)])
    result = TurnResult()

    events = await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="hi",
            history=[],
            tool_executor=executor,
            result=result,
        )
    )

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].data["error_class"] == "RuntimeError"
    assert "network down" in events[0].data["error"]
    assert result.stop_reason == "error"


# ---------------------------------------------------------------------------
# History is forwarded
# ---------------------------------------------------------------------------


async def test_history_is_threaded_into_request(executor: ToolExecutor) -> None:
    final = _final_message(
        content=[{"type": "text", "text": "ok"}],
        stop_reason="end_turn",
    )
    client = _FakeClient([_FakeStream(events=[], final=final)])

    history: list[dict[str, Any]] = [
        {"role": "user", "content": "earlier message"},
        {"role": "assistant", "content": [{"type": "text", "text": "earlier reply"}]},
    ]

    await _collect(
        run_chat_turn(
            client=client,  # type: ignore[arg-type]
            model="claude-opus-4-7",
            user_message="follow-up",
            history=history,
            tool_executor=executor,
        )
    )

    sent = client.messages.calls[0]["messages"]
    assert sent[0] == history[0]
    assert sent[1] == history[1]
    assert sent[2] == {"role": "user", "content": "follow-up"}


# ---------------------------------------------------------------------------
# Constants / sanity
# ---------------------------------------------------------------------------


def test_default_max_iterations_is_reasonable() -> None:
    assert 5 <= DEFAULT_MAX_ITERATIONS <= 20
