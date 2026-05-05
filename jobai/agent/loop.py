"""Streaming agent loop.

Drives the manual tool-use loop using ``AsyncAnthropic.messages.stream``
and yields :class:`StreamEvent` objects the chat endpoint forwards as
Server-Sent Events.

Why manual: the tool runner returns whole messages after the agentic
loop completes, which loses the per-token / per-tool-call visibility
the UI needs to render progress as the agent works. Manual gives us
text deltas, thinking deltas, tool-call announcements, tool results,
and the final usage rollup as discrete events.

Stop-reason handling:

* ``end_turn``: the model finished. Emit a ``done`` event and return.
* ``tool_use``: execute every tool the model requested, append the
  ``tool_result`` blocks as the next user turn, and loop.
* ``pause_turn``: a server-side tool (web search, code execution) hit
  its iteration cap. Re-send the conversation as-is so the API
  resumes; the SDK detects the trailing ``server_tool_use`` block and
  continues from there.
* anything else (``max_tokens``, ``refusal``): emit ``done`` with the
  reason and let the caller surface it.

The loop persists nothing — the caller (the API endpoint) is
responsible for writing user / assistant messages to the
``conversations`` store. Keeping the loop pure makes it trivially
testable against a mocked SDK.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam

from jobai.agent.prompts import SYSTEM_PROMPT
from jobai.agent.tools import TOOL_DEFINITIONS, ToolExecutor, serialise_result

_log = logging.getLogger(__name__)

#: Default ceiling on the number of tool-use round trips per user turn.
#: Stops a misbehaving model from looping indefinitely.
DEFAULT_MAX_ITERATIONS = 10

#: Hard ceiling on output tokens per iteration. Keeps non-streaming SDK
#: paths under their HTTP timeout while leaving room for thoughtful
#: tool-driven responses.
DEFAULT_MAX_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One event yielded by the agent loop.

    The chat endpoint maps these to Server-Sent Events on the wire.
    Type values are stable; data shape varies by type and is documented
    in the constructors below.
    """

    type: str
    data: dict[str, Any]


@dataclass(slots=True)
class TurnResult:
    """Aggregate state of one user turn after the loop completes.

    Exposed alongside the streamed events so the caller (the API
    endpoint) can persist the assistant turns to the conversation
    store without re-deriving them.
    """

    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_result_messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


async def run_chat_turn(
    *,
    client: AsyncAnthropic,
    model: str,
    user_message: str,
    history: list[dict[str, Any]],
    tool_executor: ToolExecutor,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    result: TurnResult | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run one user turn through the agent loop, streaming events.

    Args:
        client: an ``AsyncAnthropic`` instance.
        model: model id (e.g. ``claude-opus-4-7``).
        user_message: the user's text for this turn.
        history: prior conversation messages in Anthropic format
            (``[{"role": ..., "content": ...}, ...]``).
        tool_executor: dispatches tool calls to local handlers.
        max_iterations: hard ceiling on tool-use round trips.
        max_tokens: per-iteration output cap.
        result: optional :class:`TurnResult` the caller passes in to
            collect assistant + tool-result messages for persistence.
            The loop populates the dataclass in place.
    """
    if result is None:
        result = TurnResult()

    messages: list[dict[str, Any]] = [*history, {"role": "user", "content": user_message}]

    for _iteration in range(max_iterations):
        try:
            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=cast("list[ToolParam]", TOOL_DEFINITIONS),
                messages=cast("list[MessageParam]", messages),
            ) as stream:
                async for event in stream:
                    forwarded = _translate_stream_event(event)
                    if forwarded is not None:
                        yield forwarded
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001  - surface any SDK failure as an event
            _log.warning(
                "agent_stream_failed",
                extra={"error_class": type(exc).__name__, "error": str(exc)},
            )
            yield StreamEvent(
                type="error",
                data={"error_class": type(exc).__name__, "error": str(exc)},
            )
            result.stop_reason = "error"
            return

        assistant_content = [_block_to_dict(b) for b in final.content]
        assistant_message = {"role": "assistant", "content": assistant_content}
        result.assistant_messages.append(assistant_message)
        messages.append(assistant_message)
        _accumulate_usage(result.usage, final.usage)

        stop_reason = getattr(final, "stop_reason", None)

        if stop_reason == "tool_use":
            tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
            tool_results = await _run_tools(tool_uses, tool_executor)
            for tool_event in tool_results.events:
                yield tool_event
            user_turn = {"role": "user", "content": tool_results.blocks}
            messages.append(user_turn)
            result.tool_result_messages.append(user_turn)
            continue

        if stop_reason == "pause_turn":
            yield StreamEvent(type="pause_turn", data={})
            continue

        result.stop_reason = stop_reason or "unknown"
        yield StreamEvent(
            type="done",
            data={"stop_reason": result.stop_reason, "usage": dict(result.usage)},
        )
        return

    result.stop_reason = "max_iterations"
    yield StreamEvent(
        type="done",
        data={"stop_reason": "max_iterations", "iterations": max_iterations},
    )


@dataclass(slots=True)
class _ToolBatchResult:
    blocks: list[dict[str, Any]]
    events: list[StreamEvent]


async def _run_tools(
    tool_uses: list[Any],
    tool_executor: ToolExecutor,
) -> _ToolBatchResult:
    """Execute every tool call in order, returning the result blocks +
    matching SSE events."""
    blocks: list[dict[str, Any]] = []
    events: list[StreamEvent] = []
    for tool_use in tool_uses:
        tu_id = str(getattr(tool_use, "id", ""))
        tu_name = str(getattr(tool_use, "name", ""))
        tu_input = dict(getattr(tool_use, "input", {}) or {})
        events.append(
            StreamEvent(
                type="tool_call",
                data={"id": tu_id, "name": tu_name, "input": tu_input},
            )
        )
        try:
            result_value = tool_executor.execute(tu_name, tu_input)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": serialise_result(result_value),
                }
            )
            events.append(
                StreamEvent(
                    type="tool_result",
                    data={"id": tu_id, "name": tu_name, "result": result_value},
                )
            )
        except Exception as exc:  # noqa: BLE001  - report any tool failure to the model
            error_msg = f"{type(exc).__name__}: {exc}"
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": error_msg,
                    "is_error": True,
                }
            )
            events.append(
                StreamEvent(
                    type="tool_error",
                    data={
                        "id": tu_id,
                        "name": tu_name,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            )
    return _ToolBatchResult(blocks=blocks, events=events)


def _translate_stream_event(event: Any) -> StreamEvent | None:
    """Map Anthropic's stream events to our SSE-shaped events.

    We surface text deltas, thinking deltas, and tool-use block starts
    (with the tool name + id) as soon as they arrive. Other internal
    events (block stops, message_start, etc.) are filtered out; the
    final message is read after the stream completes.
    """
    event_type = getattr(event, "type", None)
    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        delta_type = getattr(delta, "type", None)
        if delta_type == "text_delta":
            return StreamEvent(type="text_delta", data={"text": getattr(delta, "text", "")})
        if delta_type == "thinking_delta":
            return StreamEvent(
                type="thinking_delta",
                data={"text": getattr(delta, "thinking", "")},
            )
    elif event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if getattr(block, "type", None) == "tool_use":
            return StreamEvent(
                type="tool_use_start",
                data={
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                },
            )
    return None


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Render an Anthropic content block as a plain dict for storage and
    re-feeding."""
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "model_dump"):
        dumped = block.model_dump()
        if isinstance(dumped, dict):
            return dumped
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": dict(getattr(block, "input", {}) or {}),
        }
    if block_type == "thinking":
        return {
            "type": "thinking",
            "thinking": getattr(block, "thinking", ""),
            "signature": getattr(block, "signature", ""),
        }
    return {"type": block_type or "unknown"}


def _accumulate_usage(running: dict[str, int], usage: Any) -> None:
    """Sum per-iteration usage into a single running total dict."""
    if usage is None:
        return
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            running[key] = running.get(key, 0) + value
