"""Tests for the configuration loader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from jobai.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Run each test from an empty temp directory and clear the cache.

    The empty workdir means no stray ``.env`` from the developer's
    environment can leak into a test. The cache_clear ensures
    :func:`get_settings` reflects per-test environment changes.
    """
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_set() -> None:
    settings = Settings()

    assert settings.db_path == Path("jobai.db")
    assert settings.log_level == "INFO"
    assert settings.log_dir == Path("logs")
    assert settings.log_max_bytes == 10_000_000
    assert settings.log_backup_count == 5
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8421
    assert settings.default_cadence_seconds == 1800
    assert "jobai" in settings.user_agent.lower()
    assert settings.anthropic_api_key is None
    assert settings.anthropic_model == "claude-opus-4-7"


def test_anthropic_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBAI_ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("JOBAI_ANTHROPIC_MODEL", "claude-haiku-4-5")

    settings = Settings()

    assert settings.anthropic_api_key == "sk-ant-test-key"
    assert settings.anthropic_model == "claude-haiku-4-5"


def test_env_var_overrides_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_db = tmp_path / "custom.db"
    monkeypatch.setenv("JOBAI_DB_PATH", str(custom_db))
    monkeypatch.setenv("JOBAI_API_PORT", "9999")
    monkeypatch.setenv("JOBAI_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.db_path == custom_db
    assert settings.api_port == 9999
    assert settings.log_level == "DEBUG"


def test_env_file_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "JOBAI_LOG_LEVEL=DEBUG\nJOBAI_API_PORT=7000\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.log_level == "DEBUG"
    assert settings.api_port == 7000


def test_env_var_takes_precedence_over_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("JOBAI_API_PORT=7000\n", encoding="utf-8")
    monkeypatch.setenv("JOBAI_API_PORT", "8000")

    settings = Settings()

    assert settings.api_port == 8000


def test_invalid_port_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBAI_API_PORT", "not-a-number")
    with pytest.raises(ValidationError):
        Settings()


def test_port_out_of_range_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBAI_API_PORT", "70000")
    with pytest.raises(ValidationError):
        Settings()


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown ``JOBAI_*`` variables must not raise.

    Failing on unknown names would make the loader brittle against env
    churn (e.g. an old variable left behind by a removed feature).
    """
    monkeypatch.setenv("JOBAI_SOMETHING_UNKNOWN", "value")
    settings = Settings()
    assert settings.api_port == 8421


def test_get_settings_returns_cached_instance() -> None:
    """Two calls to :func:`get_settings` return the same instance."""
    a = get_settings()
    b = get_settings()
    assert a is b


def test_get_settings_picks_up_environment_after_cache_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cache_clear`` lets tests rebuild settings against a new env."""
    monkeypatch.setenv("JOBAI_API_PORT", "1234")
    first = get_settings()
    assert first.api_port == 1234

    monkeypatch.setenv("JOBAI_API_PORT", "5678")
    get_settings.cache_clear()
    second = get_settings()
    assert second.api_port == 5678
