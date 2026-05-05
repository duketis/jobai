"""Tests for the agent's system prompt.

The prompt is frozen and tested for shape invariants only — no dynamic
interpolation, no per-request rendering. Behavioural validation lives
with the agent loop tests where we can mock the model.
"""

from __future__ import annotations

from jobai.agent.prompts import SYSTEM_PROMPT


def test_system_prompt_is_non_empty_string() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()


def test_system_prompt_mentions_every_tool() -> None:
    """Defensive: if a tool is added or renamed, the prompt should be updated
    to teach the model about it. This test catches drift."""
    for tool_name in (
        "search_jobs",
        "get_job_detail",
        "mark_job_state",
        "list_sources",
        "get_health",
    ):
        assert tool_name in SYSTEM_PROMPT, f"prompt missing reference to {tool_name}"


def test_system_prompt_has_no_dynamic_interpolation_markers() -> None:
    """A `{...}` placeholder, an f-string artifact, or a stale jinja2 tag
    means a future render path could mutate the prefix bytes and silently
    invalidate prompt caching."""
    suspicious = ["{{", "}}", "{%", "%}"]
    for marker in suspicious:
        assert marker not in SYSTEM_PROMPT, f"prompt contains template marker {marker!r}"
