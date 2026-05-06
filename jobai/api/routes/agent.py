"""Agent chat endpoint — Server-Sent Events stream of one turn.

``POST /api/agent/chat`` runs one user turn through
:func:`jobai.agent.loop.run_chat_turn` and streams the loop's events
to the client as SSE. The endpoint also owns conversation
persistence: it creates new conversations on first turn, stores the
user's message before kickoff, and stores every assistant /
tool_result message after the loop completes.

Why SSE not WebSocket: SSE is one-way (server → client), survives
proxy buffering with ``Cache-Control: no-cache``, and is dirt simple
to consume from a browser via ``EventSource``. WebSockets would let
the client cancel mid-turn but add reconnect / framing complexity we
don't need yet.

Why persist after the loop completes (not incrementally): keeps
persistence atomic per turn — partial state from a crashed turn
won't poison the next call's history.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from itertools import zip_longest
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from jobai.agent.conversations import (
    Conversation,
    ConversationNotFoundError,
    append_message,
    create_conversation,
    get_conversation,
    list_messages,
    messages_to_anthropic_format,
)
from jobai.agent.loop import StreamEvent, TurnResult, run_chat_turn
from jobai.agent.subscription_loop import run_subscription_chat_turn
from jobai.agent.tools import ToolExecutor
from jobai.api.dependencies import AnthropicDep, ConnDep, ModelDep
from jobai.api.models import AgentChatRequest
from jobai.api.runtime_settings import get_effective_agent_config

router = APIRouter()

_log = logging.getLogger(__name__)

#: Cap conversation titles to keep the sidebar tidy. Picked to match a
#: common single-line tweet length so titles don't wrap on narrow UIs.
_TITLE_MAX_CHARS = 80


@router.post(
    "/chat",
    summary="Stream one chat turn as Server-Sent Events",
    response_class=EventSourceResponse,
)
async def chat(
    body: AgentChatRequest,
    conn: ConnDep,
    fallback_client: AnthropicDep,
    fallback_model: ModelDep,
) -> EventSourceResponse:
    """Stream one turn through the agent loop as SSE.

    The first event is always ``conversation`` carrying the
    conversation id (the client needs this to resume on the next
    turn). Subsequent events come straight from
    :func:`run_chat_turn`: ``text_delta``, ``thinking_delta``,
    ``tool_use_start``, ``tool_call``, ``tool_result``, ``tool_error``,
    ``pause_turn``, ``done``, or ``error``.

    Backend selection (API key vs subscription) and credentials are
    resolved per-request from :mod:`jobai.api.runtime_settings`, so a
    Settings UI update takes effect on the next chat turn without
    restarting the process.
    """
    conversation = _resolve_conversation(conn, body.conversation_id, body.message)
    append_message(
        conn,
        conversation_id=conversation.id,
        role="user",
        content=body.message,
    )

    return EventSourceResponse(
        _stream_turn(
            conn=conn,
            conversation_id=conversation.id,
            user_message=body.message,
            fallback_client=fallback_client,
            fallback_model=fallback_model,
        ),
    )


def _resolve_conversation(
    conn: sqlite3.Connection,
    conversation_id: int | None,
    user_message: str,
) -> Conversation:
    """Fetch the existing conversation or create a new one.

    The new-conversation title is the first ``_TITLE_MAX_CHARS``
    characters of the user's message. We avoid a ``LLM_summary`` round
    trip just for a title — the user can always rename later.
    """
    if conversation_id is not None:
        try:
            return get_conversation(conn, conversation_id)
        except ConversationNotFoundError:
            # Surface as a stream-level error so the client sees a
            # consistent SSE shape rather than mixed JSON / SSE replies.
            return create_conversation(
                conn,
                title=_derive_title(user_message),
            )
    return create_conversation(conn, title=_derive_title(user_message))


def _derive_title(message: str) -> str:
    cleaned = message.strip().replace("\n", " ")
    if not cleaned:
        return "Untitled conversation"
    if len(cleaned) <= _TITLE_MAX_CHARS:
        return cleaned
    return cleaned[: _TITLE_MAX_CHARS - 1] + "…"


async def _stream_turn(
    *,
    conn: sqlite3.Connection,
    conversation_id: int,
    user_message: str,
    fallback_client: AsyncAnthropic,
    fallback_model: str,
) -> AsyncIterator[dict[str, str]]:
    """Drive the agent loop and yield SSE-shaped dicts.

    Each yielded dict has ``event`` (the type) and ``data`` (a JSON
    string). ``EventSourceResponse`` formats these as
    ``event: <type>\\n data: <json>\\n\\n`` on the wire.

    The backend selection (API key vs Claude Pro/Max subscription)
    happens here, not in the loops themselves: both
    :func:`run_chat_turn` and :func:`run_subscription_chat_turn`
    yield the same :class:`StreamEvent` shape, so the SSE wire
    format is identical regardless of which auth path is in use.
    """
    yield _sse("conversation", {"conversation_id": conversation_id})

    history = _load_history(conn, conversation_id)
    executor = ToolExecutor(conn)
    result = TurnResult()
    cfg = get_effective_agent_config(conn)

    async def _events() -> AsyncIterator[StreamEvent]:
        if cfg.agent_backend == "subscription":
            async for ev in run_subscription_chat_turn(
                model=cfg.anthropic_model or None,
                user_message=user_message,
                history=history,
                tool_executor=executor,
                result=result,
                oauth_token=cfg.claude_code_oauth_token,
            ):
                yield ev
        else:
            # API mode. If the user set an explicit api_key via the
            # Settings UI, build a fresh client with that override;
            # otherwise reuse the dependency-injected client (tests
            # override this dep with a fake; production uses
            # AsyncAnthropic() with env-default credentials).
            client = (
                AsyncAnthropic(api_key=cfg.anthropic_api_key)
                if cfg.anthropic_api_key
                else fallback_client
            )
            model = cfg.anthropic_model or fallback_model
            async for ev in run_chat_turn(
                client=client,
                model=model,
                user_message=user_message,
                history=history,
                tool_executor=executor,
                result=result,
            ):
                yield ev

    try:
        async for stream_event in _events():
            yield _sse(stream_event.type, stream_event.data)
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure as an SSE error
        _log.exception("agent_chat_stream_failed", extra={"backend": cfg.agent_backend})
        yield _sse(
            "error",
            {"error_class": type(exc).__name__, "error": str(exc)},
        )

    _persist_turn(conn, conversation_id, result)


def _load_history(
    conn: sqlite3.Connection,
    conversation_id: int,
) -> list[dict[str, Any]]:
    """Return all messages prior to the just-stored user turn.

    The user message we just saved is excluded — the loop appends it
    to ``messages`` itself and would otherwise duplicate it.
    """
    stored = list_messages(conn, conversation_id)
    if not stored:
        return []
    return messages_to_anthropic_format(stored[:-1])


def _persist_turn(
    conn: sqlite3.Connection,
    conversation_id: int,
    result: TurnResult,
) -> None:
    """Write the assistant + tool_result messages back to SQLite.

    Interleaves them in the same order the loop produced them so the
    next turn's history reads back as a valid Anthropic message list:
    ``A0, T0, A1, T1, ..., An``.
    """
    for assistant, tool in zip_longest(
        result.assistant_messages,
        result.tool_result_messages,
        fillvalue=None,
    ):
        if assistant is not None:
            append_message(
                conn,
                conversation_id=conversation_id,
                role="assistant",
                content=assistant["content"],
            )
        if tool is not None:
            append_message(
                conn,
                conversation_id=conversation_id,
                role="user",
                content=tool["content"],
            )


def _sse(event_type: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event_type, "data": json.dumps(data)}
