"""Async Anthropic client factory for the agent.

Centralises API-key resolution and model defaults so the rest of the
agent layer doesn't depend on :mod:`jobai.config` directly. Tests can
override the key by passing ``api_key=`` explicitly; production reads
from settings (which falls back to the SDK's default
``ANTHROPIC_API_KEY`` env-var resolution if unset).
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from jobai.config import get_settings


def build_client(*, api_key: str | None = None) -> AsyncAnthropic:
    """Construct an :class:`AsyncAnthropic` client.

    Resolution order for the API key:
    1. The explicit ``api_key=`` argument (used in tests).
    2. ``settings.anthropic_api_key`` (loaded from
       ``JOBAI_ANTHROPIC_API_KEY``).
    3. The SDK's default ``ANTHROPIC_API_KEY`` env-var lookup.
    """
    if api_key is None:
        api_key = get_settings().anthropic_api_key
    if api_key is not None:
        return AsyncAnthropic(api_key=api_key)
    return AsyncAnthropic()


def get_model() -> str:
    """Return the configured model id (defaults to ``claude-opus-4-7``)."""
    return get_settings().anthropic_model
