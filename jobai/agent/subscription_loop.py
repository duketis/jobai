"""Subscription-mode agent loop driven by the Claude Agent SDK.

The default agent loop in :mod:`jobai.agent.loop` drives the
Anthropic SDK directly with an API key — pay-per-token billing.
This loop is the alternative for users who'd rather have their
Claude Pro/Max subscription quota cover the agent's calls.

It works by running the logged-in ``claude`` CLI as a subprocess
via the ``claude-agent-sdk`` package. The CLI auths with whichever
account it's logged in as on the host (typically the user's
``~/.claude/`` credentials), so token usage bills against that
subscription rather than an API key.

The five jobai tools (search_jobs, get_job_detail, mark_job_state,
list_sources, get_health) register as an in-process MCP server so
the SDK can call them without any extra subprocess. Stream events
are translated to the same :class:`StreamEvent` shape the API
loop produces, so the SSE endpoint is backend-agnostic.

Why a separate file rather than abstract behind a Protocol: the
two SDKs disagree on most of their surface area (tool registration,
message-stream events, conversation state). Forcing them through
a single interface would mean the LCD of both. Two parallel loops
let each backend speak its own SDK's idioms.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SdkMcpTool,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from claude_agent_sdk import StreamEvent as SdkStreamEvent

from jobai.agent.loop import StreamEvent, TurnResult
from jobai.agent.prompts import SYSTEM_PROMPT
from jobai.agent.tools import TOOL_DEFINITIONS, ToolExecutor, serialise_result

_log = logging.getLogger(__name__)

#: Cap on agentic round-trips per user turn — same default as the
#: API-mode loop. The SDK enforces this server-side via ``max_turns``.
DEFAULT_MAX_ITERATIONS = 10

#: MCP server name that namespaces our tools. The SDK exposes them
#: to the model as ``mcp__jobai__<tool_name>``.
_MCP_SERVER_NAME = "jobai"


async def run_subscription_chat_turn(
    *,
    model: str | None,
    user_message: str,
    history: list[dict[str, Any]],
    tool_executor: ToolExecutor,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    result: TurnResult | None = None,
    oauth_token: str | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run one user turn through the Claude Agent SDK, streaming events.

    Args:
        model: model id (e.g. ``claude-opus-4-7``) or ``None`` to let
            the local ``claude`` CLI pick its default.
        user_message: this turn's user text.
        history: prior conversation messages in the canonical
            ``[{"role": ..., "content": ...}, ...]`` shape. Forwarded
            to the SDK via streaming-input mode so the model sees
            the full thread, not just the latest message.
        tool_executor: the existing :class:`ToolExecutor` — we wrap
            each of its handlers as an SDK tool so subscription
            mode and API mode share one execution path.
        max_iterations: surfaces as ``ClaudeAgentOptions.max_turns``.
        result: optional turn-aggregate the caller passes in; the
            loop populates it in place for persistence.
    """
    if result is None:
        result = TurnResult()

    server = create_sdk_mcp_server(
        name=_MCP_SERVER_NAME,
        tools=_build_mcp_tools(tool_executor),
    )
    allowed = [f"mcp__{_MCP_SERVER_NAME}__{td['name']}" for td in TOOL_DEFINITIONS]

    # The ``claude`` CLI reads its long-lived auth from
    # ``CLAUDE_CODE_OAUTH_TOKEN``; forwarding it via ``options.env``
    # keeps the secret off the parent process's environment so the
    # rest of the API server doesn't see it.
    cli_env: dict[str, str] = {}
    if oauth_token:
        cli_env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        # Disable the SDK's built-in coding tools (Bash, Read, Edit,
        # WebFetch, …) — jobai's agent has nothing to do with files.
        # Only our MCP tools should be reachable.
        tools=[],
        mcp_servers={_MCP_SERVER_NAME: server},
        allowed_tools=allowed,
        model=model,
        max_turns=max_iterations,
        # ``bypassPermissions`` is the only preset that doesn't gate
        # MCP tool calls behind a prompt the chat UI can't answer —
        # our tools are read-mostly DB queries so nothing dangerous
        # ever asks regardless of which preset is in effect.
        permission_mode="bypassPermissions",
        include_partial_messages=True,
        env=cli_env,
    )

    try:
        async for message in query(prompt=_history_stream(history, user_message), options=options):
            async for event in _translate_message(message, result):
                yield event
                if event.type in {"done", "error"}:
                    return
        # The SDK normally ends every turn with a ResultMessage (triggering
        # the early ``return`` above). Falling out of the loop here means
        # the transport closed without one -- defensive-only, surfaced as
        # a clean turn end so the chat UI doesn't hang forever.
        return  # pragma: no cover  # noqa: TRY300
    except Exception as exc:  # noqa: BLE001 - report any SDK failure as an event
        _log.warning(
            "agent_subscription_failed",
            extra={"error_class": type(exc).__name__, "error": str(exc)},
        )
        yield StreamEvent(
            type="error",
            data={"error_class": type(exc).__name__, "error": str(exc)},
        )
        result.stop_reason = "error"
        return


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _build_mcp_tools(executor: ToolExecutor) -> list[SdkMcpTool[Any]]:
    """Wrap every TOOL_DEFINITIONS entry as an in-process MCP tool.

    The SDK's :func:`tool` decorator expects an async function whose
    body returns ``{"content": [{"type": "text", "text": "..."}]}``.
    We thread through the existing :class:`ToolExecutor` so both
    modes (API-key and subscription) share one execution path —
    differences live entirely in *how* the model calls the tool,
    not in *what the tool does*.
    """
    tools: list[SdkMcpTool[Any]] = []
    for definition in TOOL_DEFINITIONS:
        name = str(definition["name"])
        tools.append(
            _wrap_tool(executor, name, str(definition["description"]), definition["input_schema"])
        )
    return tools


def _wrap_tool(
    executor: ToolExecutor,
    name: str,
    description: str,
    input_schema: Any,
) -> SdkMcpTool[Any]:
    """Build one ``SdkMcpTool`` that delegates to the shared executor."""

    async def handler(args: Any) -> dict[str, Any]:
        try:
            value = executor.execute(name, args if isinstance(args, dict) else {})
        except Exception as exc:  # noqa: BLE001 - surface the message to the model
            return {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "is_error": True,
            }
        return {"content": [{"type": "text", "text": serialise_result(value)}]}

    decorator = tool(name, description, input_schema)
    return decorator(handler)


# ---------------------------------------------------------------------------
# History → SDK streaming-input adapter
# ---------------------------------------------------------------------------


async def _history_stream(
    history: list[dict[str, Any]],
    user_message: str,
) -> AsyncIterable[dict[str, Any]]:
    """Yield prior messages followed by the new user turn.

    The SDK's streaming-input mode expects each message wrapped as
    ``{"type": "user", "message": {"role": ..., "content": ...}}``.
    Assistant turns ride through with ``"type": "assistant"`` so
    the model sees the full conversation context.
    """
    for entry in history:
        wrapper_type = "assistant" if entry.get("role") == "assistant" else "user"
        yield {"type": wrapper_type, "message": entry}
    yield {
        "type": "user",
        "message": {"role": "user", "content": user_message},
    }


# ---------------------------------------------------------------------------
# Message → StreamEvent translation
# ---------------------------------------------------------------------------


async def _translate_message(
    message: Any,
    result: TurnResult,
) -> AsyncIterator[StreamEvent]:
    """Map one SDK message to zero or more :class:`StreamEvent`s.

    With ``include_partial_messages=True``, the SDK emits per-token
    ``SdkStreamEvent`` messages first, then the assembled
    :class:`AssistantMessage` once a block completes. To avoid
    double-emitting text, we surface text/thinking deltas ONLY from
    the partial stream events; the full AssistantMessage is used for
    persistence and to emit ``tool_call`` events (tool-use blocks
    land whole, not as deltas, so they need the assembled view).
    """
    if isinstance(message, SdkStreamEvent):
        async for event in _translate_partial_event(message):
            yield event
        return
    if isinstance(message, AssistantMessage):
        async for event in _translate_assistant(message, result):
            yield event
        return
    if isinstance(message, UserMessage):
        # User echoes contain tool_result blocks the SDK assembled
        # after our MCP server returned. We surface them so the
        # chat UI's streaming view shows the round-trip, and we
        # stash them on the turn result for persistence.
        async for event in _translate_user_echo(message, result):
            yield event
        return
    if isinstance(message, ResultMessage):
        result.stop_reason = "end_turn"
        usage = getattr(message, "usage", None)
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, int):
                    result.usage[key] = result.usage.get(key, 0) + value
        yield StreamEvent(
            type="done",
            data={
                "stop_reason": result.stop_reason,
                "usage": dict(result.usage),
            },
        )
        # Caller's outer loop short-circuits on the 'done' event above and
        # never re-enters this generator -- defensive only.
        return  # pragma: no cover
    if isinstance(message, SystemMessage):
        # Init / sidecar metadata; the chat UI doesn't need it.
        return
    # Rate-limit notices, etc — no UI surface yet, but log so they're
    # not silently lost.
    _log.debug("agent_subscription_unhandled_message", extra={"type": type(message).__name__})


async def _translate_partial_event(message: SdkStreamEvent) -> AsyncIterator[StreamEvent]:
    """Emit text/thinking deltas as they arrive on the SDK stream.

    The wrapped ``event`` mirrors the Anthropic API's partial-message
    shape — same as what :func:`jobai.agent.loop._translate_stream_event`
    handles for the API-key path. We only forward the deltas users
    can read incrementally; tool_use block opens come through later
    as part of the assembled AssistantMessage so we skip them here.
    """
    raw = message.event if isinstance(message.event, dict) else {}
    event_type = raw.get("type")
    if event_type == "content_block_delta":
        delta = raw.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text", "")
            if text:
                yield StreamEvent(type="text_delta", data={"text": text})
        elif delta_type == "thinking_delta":
            text = delta.get("thinking", "")
            if text:
                yield StreamEvent(type="thinking_delta", data={"text": text})


async def _translate_assistant(
    message: AssistantMessage,
    result: TurnResult,
) -> AsyncIterator[StreamEvent]:
    """Emit ``tool_call`` events + persist the assembled assistant turn.

    Text and thinking content was already streamed live via
    :func:`_translate_partial_event`; we don't re-emit it here or
    the chat UI would show every reply twice. We do still persist
    the full assembled blocks so the next turn's history reads back
    as a valid Anthropic-format message list.
    """
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock) and block.text:
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock) and block.thinking:
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": getattr(block, "signature", ""),
                },
            )
        elif isinstance(block, ToolUseBlock):
            tool_input = block.input if isinstance(block.input, dict) else {}
            tool_name = _strip_mcp_prefix(block.name)
            yield StreamEvent(
                type="tool_call",
                data={"id": block.id, "name": tool_name, "input": tool_input},
            )
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": tool_name,
                    "input": tool_input,
                },
            )
    if blocks:
        result.assistant_messages.append({"role": "assistant", "content": blocks})


async def _translate_user_echo(
    message: UserMessage,
    result: TurnResult,
) -> AsyncIterator[StreamEvent]:
    blocks: list[dict[str, Any]] = []
    for block in _iter_user_blocks(message):
        if isinstance(block, ToolResultBlock):
            content = _flatten_tool_result_content(block.content)
            yield StreamEvent(
                type="tool_result",
                data={
                    "id": block.tool_use_id,
                    "name": "result",
                    "result": content,
                },
            )
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": content,
                    "is_error": bool(getattr(block, "is_error", False)),
                },
            )
    if blocks:
        result.tool_result_messages.append({"role": "user", "content": blocks})


def _iter_user_blocks(message: UserMessage) -> list[Any]:
    """``UserMessage.content`` is a string or block list; normalise."""
    content = message.content
    if isinstance(content, list):
        return list(content)
    return []


def _flatten_tool_result_content(content: Any) -> Any:
    """The SDK delivers tool_result content as either a string or a
    list of content blocks. The chat UI accepts either; flatten
    to a string when the blocks are uniformly text so the JSON
    payload stays readable in the SSE stream."""
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                chunks.append(str(block.text))
        if chunks:
            return "".join(chunks)
    return content


def _strip_mcp_prefix(name: str) -> str:
    """Drop the ``mcp__jobai__`` prefix the SDK adds to MCP tools.

    The chat UI keys off the bare tool name (``search_jobs``); the
    namespacing is an SDK implementation detail.
    """
    prefix = f"mcp__{_MCP_SERVER_NAME}__"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


__all__ = ["DEFAULT_MAX_ITERATIONS", "run_subscription_chat_turn"]
