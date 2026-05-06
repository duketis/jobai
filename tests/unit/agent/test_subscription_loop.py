"""Tests for the subscription-mode agent loop.

These exercise the SDK-message → StreamEvent translation, the
history-stream adapter, and the MCP tool wrapping. The Claude
Agent SDK's actual transport is replaced with a fake ``query`` that
yields canned messages — no ``claude`` CLI subprocess is spawned.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from jobai.agent import subscription_loop
from jobai.agent.loop import TurnResult
from jobai.agent.tools import ToolExecutor
from jobai.db.migrations import apply_pending


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "agent.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    apply_pending(connection)
    return connection


@pytest.fixture
def executor(conn: sqlite3.Connection) -> ToolExecutor:
    return ToolExecutor(conn)


def _patch_query(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> list[dict[str, Any]]:
    """Replace ``query`` with a fake that yields ``messages`` in order.

    Returns a list that captures the serialised prompt-stream the
    fake received, so tests can assert what we sent.
    """
    captured_prompt: list[dict[str, Any]] = []

    def fake_query(
        *, prompt: Any, options: Any = None, transport: Any = None
    ) -> AsyncIterator[Any]:
        del options, transport  # unused in tests

        async def _gen() -> AsyncIterator[Any]:
            # Drain the streaming-input prompt the loop built so we
            # can assert prior messages were forwarded.
            async for entry in prompt:
                captured_prompt.append(entry)
            for msg in messages:
                yield msg

        return _gen()

    monkeypatch.setattr(subscription_loop, "query", fake_query)
    return captured_prompt


def _result_message(
    stop_reason: str = "end_turn", usage: dict[str, int] | None = None
) -> ResultMessage:
    """Build a minimal ``ResultMessage`` — most fields are mandatory but
    the loop only reads ``stop_reason`` and ``usage``."""
    return ResultMessage(
        subtype="success",
        duration_ms=0,
        duration_api_ms=0,
        is_error=False,
        num_turns=1,
        session_id="test",
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Translation: AssistantMessage with text → text_delta + assistant message
# ---------------------------------------------------------------------------


async def test_text_block_emits_text_delta_and_records_assistant(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    _patch_query(
        monkeypatch,
        [
            AssistantMessage(
                content=[TextBlock(text="Hi there!")],
                model="claude-opus-4-7",
            ),
            _result_message(),
        ],
    )
    result = TurnResult()
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model="claude-opus-4-7",
            user_message="hi",
            history=[],
            tool_executor=executor,
            result=result,
        )
    ]
    types = [e.type for e in events]
    assert types == ["text_delta", "done"]
    assert events[0].data == {"text": "Hi there!"}
    assert result.assistant_messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
    ]
    assert result.stop_reason == "end_turn"


async def test_thinking_block_emits_thinking_delta(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    _patch_query(
        monkeypatch,
        [
            AssistantMessage(
                content=[ThinkingBlock(thinking="reasoning…", signature="sig")],
                model="claude-opus-4-7",
            ),
            _result_message(),
        ],
    )
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="hi",
            history=[],
            tool_executor=executor,
        )
    ]
    assert events[0].type == "thinking_delta"
    assert events[0].data == {"text": "reasoning…"}


# ---------------------------------------------------------------------------
# Translation: tool_use + tool_result round trip
# ---------------------------------------------------------------------------


async def test_tool_use_block_strips_mcp_prefix_and_emits_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """The SDK exposes our tools as ``mcp__jobai__search_jobs`` —
    the chat UI keys off the bare name, so the prefix is stripped
    before emitting the tool_call event."""
    _patch_query(
        monkeypatch,
        [
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="call-1",
                        name="mcp__jobai__search_jobs",
                        input={"q": "python"},
                    ),
                ],
                model="claude-opus-4-7",
            ),
            _result_message(),
        ],
    )
    result = TurnResult()
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="find me python jobs",
            history=[],
            tool_executor=executor,
            result=result,
        )
    ]
    tool_calls = [e for e in events if e.type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].data == {
        "id": "call-1",
        "name": "search_jobs",
        "input": {"q": "python"},
    }
    # The persisted assistant block also uses the bare name.
    assert result.assistant_messages[-1]["content"][-1]["name"] == "search_jobs"


async def test_tool_result_block_in_user_echo_emits_tool_result(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """After our MCP tool returns, the SDK loops back a UserMessage
    carrying the tool_result block. We surface it as a tool_result
    event so the chat UI shows the round-trip."""
    _patch_query(
        monkeypatch,
        [
            AssistantMessage(
                content=[
                    ToolUseBlock(id="call-1", name="mcp__jobai__get_health", input={}),
                ],
                model="claude-opus-4-7",
            ),
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="call-1",
                        content=[{"type": "text", "text": '{"status":"ok"}'}],
                    ),
                ],
            ),
            AssistantMessage(
                content=[TextBlock(text="System is healthy.")],
                model="claude-opus-4-7",
            ),
            _result_message(),
        ],
    )
    result = TurnResult()
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="how's the system?",
            history=[],
            tool_executor=executor,
            result=result,
        )
    ]
    tool_results = [e for e in events if e.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].data["id"] == "call-1"
    assert tool_results[0].data["result"] == '{"status":"ok"}'

    # The tool_result block is also stashed for persistence so the
    # next turn's history preserves the round-trip.
    assert len(result.tool_result_messages) == 1
    persisted = result.tool_result_messages[0]["content"][0]
    assert persisted["type"] == "tool_result"
    assert persisted["tool_use_id"] == "call-1"


# ---------------------------------------------------------------------------
# History adapter
# ---------------------------------------------------------------------------


async def test_history_is_forwarded_then_user_message_last(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    captured = _patch_query(monkeypatch, [_result_message()])
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": [{"type": "text", "text": "earlier reply"}]},
    ]
    _ = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="next question",
            history=history,
            tool_executor=executor,
        )
    ]
    # The fake ``query`` recorded each prompt-stream entry. We expect
    # the prior history first, then the new user turn at the end.
    assert len(captured) == 3
    assert captured[0]["type"] == "user"
    assert captured[0]["message"]["content"] == "earlier question"
    assert captured[1]["type"] == "assistant"
    assert captured[2]["type"] == "user"
    assert captured[2]["message"] == {"role": "user", "content": "next question"}


# ---------------------------------------------------------------------------
# ResultMessage closes the turn
# ---------------------------------------------------------------------------


async def test_result_message_emits_done_with_usage(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    _patch_query(
        monkeypatch,
        [
            _result_message(stop_reason="end_turn", usage={"input_tokens": 12, "output_tokens": 4}),
        ],
    )
    result = TurnResult()
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
            result=result,
        )
    ]
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].data["stop_reason"] == "end_turn"
    assert done[0].data["usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert result.usage == {"input_tokens": 12, "output_tokens": 4}


async def test_system_message_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """Init / sidecar SystemMessages aren't surfaced to the chat UI."""
    _patch_query(
        monkeypatch,
        [
            SystemMessage(subtype="init", data={"session_id": "x"}),
            _result_message(),
        ],
    )
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
        )
    ]
    assert [e.type for e in events] == ["done"]


# ---------------------------------------------------------------------------
# SDK exception → error event
# ---------------------------------------------------------------------------


async def test_sdk_exception_surfaces_as_error_event(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    sentinel = object()

    async def _gen() -> AsyncIterator[Any]:
        # Yield the sentinel so the body is unambiguously an async
        # generator (mypy needs the yield), then raise on the next
        # iteration — same shape as the SDK throwing transport errors
        # mid-stream.
        yield sentinel
        msg = "fake CLI not found"
        raise RuntimeError(msg)

    def boom_query(**_kwargs: Any) -> AsyncIterator[Any]:
        return _gen()

    monkeypatch.setattr(subscription_loop, "query", boom_query)
    result = TurnResult()
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
            result=result,
        )
    ]
    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].data["error_class"] == "RuntimeError"
    assert "fake CLI not found" in events[0].data["error"]
    assert result.stop_reason == "error"


# ---------------------------------------------------------------------------
# MCP-tool wrappers actually call back into the executor
# ---------------------------------------------------------------------------


async def test_wrapped_tool_routes_through_executor(
    executor: ToolExecutor,
) -> None:
    """The @tool wrappers we register with the SDK must dispatch
    via :meth:`ToolExecutor.execute`, otherwise the two backends
    diverge on what tools actually do."""
    tools = subscription_loop._build_mcp_tools(executor)
    by_name = {t.name: t for t in tools}
    assert "get_health" in by_name

    # MCP tools are SdkMcpTool; the underlying coroutine lives on the
    # ``handler`` attribute (or equivalent) — invoke it.
    health_tool = by_name["get_health"]
    output = await health_tool.handler({})

    assert "content" in output
    text = output["content"][0]["text"]
    parsed = json.loads(text)
    # ``get_health`` returns the same payload as /api/health.
    assert "jobs_total" in parsed
    assert "sources_total" in parsed


def test_strip_mcp_prefix_removes_only_jobai_namespace() -> None:
    assert subscription_loop._strip_mcp_prefix("mcp__jobai__search_jobs") == "search_jobs"
    # Other namespaces (built-in tools, other MCP servers) pass through.
    assert subscription_loop._strip_mcp_prefix("Bash") == "Bash"
    assert subscription_loop._strip_mcp_prefix("mcp__other__foo") == "mcp__other__foo"
