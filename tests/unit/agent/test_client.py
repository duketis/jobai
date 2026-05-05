"""Tests for the Anthropic client factory."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from anthropic import AsyncAnthropic

from jobai.agent.client import build_client, get_model
from jobai.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_build_client_returns_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # SDK requires *some* key — env or arg — even to construct.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = build_client()
    assert isinstance(client, AsyncAnthropic)


def test_build_client_honors_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit kwarg takes precedence over env / settings."""
    monkeypatch.setenv("JOBAI_ANTHROPIC_API_KEY", "from-settings")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    client = build_client(api_key="from-arg")
    assert client.api_key == "from-arg"


def test_build_client_uses_settings_when_no_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings-provided key wins over the SDK's env-var fallback."""
    monkeypatch.setenv("JOBAI_ANTHROPIC_API_KEY", "from-settings")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    client = build_client()
    assert client.api_key == "from-settings"


def test_build_client_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When neither arg nor settings provide a key, the SDK's default
    ANTHROPIC_API_KEY env-var resolution kicks in."""
    monkeypatch.delenv("JOBAI_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    client = build_client()
    assert client.api_key == "from-env"


def test_get_model_returns_settings_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBAI_ANTHROPIC_MODEL", "claude-haiku-4-5")
    assert get_model() == "claude-haiku-4-5"


def test_get_model_default_is_opus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOBAI_ANTHROPIC_MODEL", raising=False)
    assert get_model() == "claude-opus-4-7"
