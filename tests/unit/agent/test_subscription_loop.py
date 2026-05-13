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
from claude_agent_sdk import StreamEvent as SdkStreamEvent

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


async def test_partial_event_emits_text_delta(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """Per-token deltas come through the SDK's partial-event channel
    (``include_partial_messages=True``); the assembled AssistantMessage
    that lands later only persists content + emits tool_call."""

    _patch_query(
        monkeypatch,
        [
            SdkStreamEvent(
                uuid="u1",
                session_id="s",
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hi "},
                },
            ),
            SdkStreamEvent(
                uuid="u2",
                session_id="s",
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "there!"},
                },
            ),
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
    # Two partial deltas + done. AssistantMessage doesn't re-emit text.
    assert types == ["text_delta", "text_delta", "done"]
    assert events[0].data == {"text": "Hi "}
    assert events[1].data == {"text": "there!"}
    # The assembled assistant turn is still persisted for next-turn history.
    assert result.assistant_messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
    ]


async def test_partial_event_emits_thinking_delta(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:

    _patch_query(
        monkeypatch,
        [
            SdkStreamEvent(
                uuid="u1",
                session_id="s",
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "reasoning…"},
                },
            ),
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


async def test_oauth_token_is_forwarded_to_options(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """When an OAuth token is supplied, the loop forwards it to the SDK
    via ``options.env`` rather than leaking it into the parent process
    environment. Exercises the ``if oauth_token:`` True branch."""
    captured_env: dict[str, dict[str, str]] = {}

    def fake_query(
        *, prompt: Any, options: Any = None, transport: Any = None
    ) -> AsyncIterator[Any]:
        del prompt, transport
        captured_env["env"] = dict(options.env)

        async def _gen() -> AsyncIterator[Any]:
            yield _result_message()

        return _gen()

    monkeypatch.setattr(subscription_loop, "query", fake_query)
    async for _ in subscription_loop.run_subscription_chat_turn(
        model=None,
        user_message="ping",
        history=[],
        tool_executor=executor,
        oauth_token="sk-ant-oat-abc",
    ):
        pass
    assert captured_env["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-abc"


async def test_mcp_tool_handler_returns_error_payload_on_exception(
    executor: ToolExecutor,
) -> None:
    """The MCP tool wrapper catches executor failures and returns them
    as ``is_error: True`` payloads so the model sees a clear failure
    rather than crashing the SDK transport."""
    tools = subscription_loop._build_mcp_tools(executor)
    by_name = {t.name: t for t in tools}
    # ``mark_job_state`` raises ValueError when 'state' is missing.
    output = await by_name["mark_job_state"].handler({"job_id": 1})
    assert output["is_error"] is True
    assert "ValueError" in output["content"][0]["text"]


async def test_mcp_tool_handler_treats_non_dict_args_as_empty(
    executor: ToolExecutor,
) -> None:
    """If the SDK hands the handler something that isn't a dict (eg a
    list / string), normalise to an empty dict before calling the
    executor. Exercises the ``isinstance(args, dict)`` False branch."""
    tools = subscription_loop._build_mcp_tools(executor)
    by_name = {t.name: t for t in tools}
    # ``get_health`` doesn't need any args, so the empty-dict fallback
    # produces a successful payload.
    output = await by_name["get_health"].handler(["unexpected"])
    assert "content" in output
    assert output.get("is_error") is None


async def test_result_message_with_non_int_usage_values_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """``usage`` may carry non-int values (None, str). The accumulator
    must skip those rather than crashing -- exercises the
    ``isinstance(value, int)`` False branch in _translate_message."""
    _patch_query(
        monkeypatch,
        [
            _result_message(
                stop_reason="end_turn",
                usage={"input_tokens": 5, "cache_read_input_tokens": None},
            ),
        ],
    )
    result = TurnResult()
    async for _ in subscription_loop.run_subscription_chat_turn(
        model=None,
        user_message="ping",
        history=[],
        tool_executor=executor,
        result=result,
    ):
        pass
    # ``input_tokens`` was accepted; ``cache_read_input_tokens`` was filtered out.
    assert result.usage == {"input_tokens": 5}


async def test_partial_event_with_unknown_delta_type_yields_nothing(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """A content_block_delta with an unknown delta_type produces no
    StreamEvent. Exercises the elif-fall-through branch in
    _translate_partial_event."""
    delta_event = SdkStreamEvent(
        uuid="evt-1",
        session_id="sess-1",
        event={"type": "content_block_delta", "delta": {"type": "signature_delta"}},
    )
    _patch_query(monkeypatch, [delta_event, _result_message()])
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
        )
    ]
    types = [e.type for e in events]
    # The signature_delta produced no event; only the closing 'done' survives.
    assert "text_delta" not in types
    assert "thinking_delta" not in types
    assert "done" in types


async def test_partial_event_with_non_dict_event_payload_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """SDK partial events should normally carry a dict ``event`` payload;
    if a future SDK version hands us a non-dict, we treat it as an
    empty record rather than crash. Exercises the
    ``isinstance(message.event, dict)`` False branch."""
    weird = SdkStreamEvent(uuid="evt-w", session_id="sess-w", event="not-a-dict")  # type: ignore[arg-type]
    _patch_query(monkeypatch, [weird, _result_message()])
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
        )
    ]
    # No partial deltas surface; the run still completes cleanly.
    assert events[-1].type == "done"


def test_flatten_tool_result_content_uniform_text_blocks_joined() -> None:
    """A list of dict text blocks should flatten into a single string."""
    out = subscription_loop._flatten_tool_result_content(
        [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
    )
    assert out == "hello world"


def test_flatten_tool_result_content_handles_objects_with_text_attr() -> None:
    """Block objects with a ``.text`` attribute (the SDK's own block
    classes) also flatten cleanly."""
    from types import SimpleNamespace  # noqa: PLC0415

    out = subscription_loop._flatten_tool_result_content(
        [SimpleNamespace(text="from-attr")],
    )
    assert out == "from-attr"


def test_flatten_tool_result_content_passes_through_when_not_a_list() -> None:
    """A string content is returned verbatim -- the early ``isinstance(content, list)``
    False branch."""
    assert subscription_loop._flatten_tool_result_content("plain") == "plain"


def test_flatten_tool_result_content_returns_input_when_no_text_chunks() -> None:
    """A list whose blocks have neither ``type=text`` nor a ``.text``
    attribute returns the original list (no chunks accumulated)."""
    payload = [{"type": "image", "data": "..."}, 42]
    assert subscription_loop._flatten_tool_result_content(payload) is payload


def test_iter_user_blocks_returns_empty_for_string_content() -> None:
    """``UserMessage.content`` may be a plain string for synthetic user
    inputs the loop generates -- normalise to an empty block list."""
    msg = UserMessage(content="hi there", parent_tool_use_id=None)
    assert subscription_loop._iter_user_blocks(msg) == []


async def test_user_echo_without_tool_result_blocks_yields_nothing(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """A UserMessage whose blocks are all text (not ToolResultBlock)
    produces no tool_result events. Exercises the ToolResultBlock False
    branch in _translate_user_echo."""
    msg = UserMessage(
        content=[TextBlock(text="hi")],
        parent_tool_use_id=None,
    )
    _patch_query(monkeypatch, [msg, _result_message()])
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
        )
    ]
    assert all(e.type != "tool_result" for e in events)


async def test_partial_event_empty_text_delta_yields_no_event(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """An empty-text content_block_delta is a no-op. Exercises the
    ``if text:`` False branch on the text/thinking delta paths."""
    empty_text = SdkStreamEvent(
        uuid="evt-et",
        session_id="sess-et",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": ""}},
    )
    empty_thinking = SdkStreamEvent(
        uuid="evt-eth",
        session_id="sess-eth",
        event={
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": ""},
        },
    )
    _patch_query(monkeypatch, [empty_text, empty_thinking, _result_message()])
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
        )
    ]
    types = [e.type for e in events]
    assert "text_delta" not in types
    assert "thinking_delta" not in types


async def test_assistant_message_with_only_empty_blocks_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """An AssistantMessage carrying only empty-text + empty-thinking
    blocks produces no tool_call event and stores no persisted blocks.
    Covers the ``if blocks: append`` False branch in _translate_assistant."""
    msg = AssistantMessage(
        content=[TextBlock(text=""), ThinkingBlock(thinking="", signature="")],
        model="claude-sonnet-4-6",
        parent_tool_use_id=None,
    )
    _patch_query(monkeypatch, [msg, _result_message()])
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
    # No tool_call events, no persisted assistant turn.
    assert all(e.type != "tool_call" for e in events)
    assert result.assistant_messages == []


async def test_assistant_message_with_unhandled_block_type_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """A block that isn't TextBlock / ThinkingBlock / ToolUseBlock skips
    every elif and loops to the next iteration. Exercises the
    ``elif ToolUseBlock`` False branch in _translate_assistant."""

    class _UnknownBlock:
        """SDK adding a new block type would land here until we extend
        the translator -- the runtime must not crash on it."""

    msg = AssistantMessage(
        content=[_UnknownBlock(), TextBlock(text="visible")],
        model="claude-sonnet-4-6",
        parent_tool_use_id=None,
    )
    _patch_query(monkeypatch, [msg, _result_message()])
    result = TurnResult()
    async for _ in subscription_loop.run_subscription_chat_turn(
        model=None,
        user_message="ping",
        history=[],
        tool_executor=executor,
        result=result,
    ):
        pass
    # The TextBlock survived; the unknown block was silently dropped.
    assert result.assistant_messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "visible"}]},
    ]


async def test_unhandled_sdk_message_type_is_logged_and_dropped(
    monkeypatch: pytest.MonkeyPatch,
    executor: ToolExecutor,
) -> None:
    """A message type the translator doesn't recognise (eg a future
    SDK addition) is logged at debug level and produces no events."""

    class _MysteryMessage:
        pass

    _patch_query(monkeypatch, [_MysteryMessage(), _result_message()])
    events = [
        ev
        async for ev in subscription_loop.run_subscription_chat_turn(
            model=None,
            user_message="ping",
            history=[],
            tool_executor=executor,
        )
    ]
    # Only the closing 'done' from _result_message survives.
    assert [e.type for e in events] == ["done"]
