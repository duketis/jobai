-- Conversations and messages for the AI agent layer.
--
-- The Anthropic API is stateless — we send the full conversation history
-- on every request. These tables persist that history per conversation so
-- the agent can resume across sessions and the UI can render past chats.
--
-- `messages.content_json` stores the full Anthropic content array (text +
-- tool_use + tool_result blocks) as serialised JSON, so reconstructing a
-- prior turn for a follow-up request is a single deserialise + assign.

CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_conversations_updated_at ON conversations (updated_at DESC);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content_json    TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_messages_conversation ON messages (conversation_id, id);
