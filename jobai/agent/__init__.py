"""Conversational AI agent layer.

Wraps the data layer's HTTP API in an agent that takes natural-language
input from a user, calls Claude with tool definitions for search /
detail / state / health, executes tool calls, and streams the final
response back as Server-Sent Events.

The agent is stateless at the API level (every request sends full
history); ``conversations`` persists history per chat so the UI can
resume across sessions.
"""

from __future__ import annotations
