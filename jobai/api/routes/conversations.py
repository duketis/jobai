"""Conversations CRUD endpoints.

Reads + deletions only — conversations are *created* implicitly by
:mod:`jobai.api.routes.agent` on the first chat turn, so there is no
``POST /api/conversations``. Letting the agent route own creation
keeps title derivation and the first-turn message persistence in one
place; a separate create endpoint would invite drift between the two
code paths.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from jobai.agent.conversations import (
    ConversationNotFoundError,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
)
from jobai.api.dependencies import ConnDep
from jobai.api.models import (
    ConversationDetailResponse,
    ConversationItem,
    ConversationMessageItem,
    ConversationsListResponse,
)

router = APIRouter()

#: Same defaults as the jobs list endpoint — keeps the API consistent.
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


@router.get(
    "",
    response_model=ConversationsListResponse,
    summary="List conversations, newest activity first",
)
def list_(
    conn: ConnDep,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_LIMIT, description=f"Max items per page (1-{MAX_LIMIT})."),
    ] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0, description="Page offset.")] = 0,
) -> ConversationsListResponse:
    rows = list_conversations(conn, limit=limit, offset=offset)
    return ConversationsListResponse(
        items=[
            ConversationItem(
                id=r.id,
                title=r.title,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in rows
        ],
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Fetch one conversation with its full message history",
)
def detail(conn: ConnDep, conversation_id: int) -> ConversationDetailResponse:
    """Return the conversation row plus every message, oldest first."""
    try:
        conversation = get_conversation(conn, conversation_id)
        messages = list_messages(conn, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"conversation {conversation_id} not found",
        ) from exc
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            ConversationMessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.delete(
    "/{conversation_id}",
    status_code=204,
    summary="Delete a conversation and all its messages",
)
def delete(conn: ConnDep, conversation_id: int) -> None:
    """Delete the conversation row; messages cascade via foreign key."""
    try:
        delete_conversation(conn, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"conversation {conversation_id} not found",
        ) from exc
