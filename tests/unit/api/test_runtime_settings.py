"""Tests for the runtime-settings repository + effective-config resolver."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from jobai.api.runtime_settings import (
    ALLOWED_KEYS,
    SECRET_KEYS,
    get_effective_agent_config,
    read_all,
    redacted_view,
    write_many,
)
from jobai.config import get_settings
from jobai.db.migrations import apply_pending


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sqlite3.Connection]:
    # Strip the env so the tests measure only the DB-override behaviour.
    for var in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "JOBAI_AGENT_BACKEND",
        "JOBAI_ANTHROPIC_API_KEY",
        "JOBAI_ANTHROPIC_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    # Force a fresh Settings load with the cleaned env.
    get_settings.cache_clear()

    db_path = tmp_path / "settings.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    apply_pending(connection)
    try:
        yield connection
    finally:
        connection.close()
        get_settings.cache_clear()


def test_allow_list_is_what_the_ui_expects() -> None:
    """If you add a setting key, update this list — and remember
    whether it's a secret. The Settings UI's submit handler keys
    off the same list."""
    assert {
        "agent_backend",
        "anthropic_api_key",
        "claude_code_oauth_token",
        "anthropic_model",
        # Apply-profile fields v1.17.0+ — drive the per-job
        # CHECKLIST.md the snapshot module writes for each tailored
        # application. Plain strings, none secret.
        "apply_profile_full_name",
        "apply_profile_email",
        "apply_profile_phone",
        "apply_profile_location",
        "apply_profile_linkedin_url",
        "apply_profile_github_url",
        "apply_profile_right_to_work",
        "apply_profile_notice_period",
        "apply_profile_salary_expectation",
    } == ALLOWED_KEYS
    assert {"anthropic_api_key", "claude_code_oauth_token"} == SECRET_KEYS


def test_effective_config_falls_back_to_defaults_when_table_empty(
    conn: sqlite3.Connection,
) -> None:
    cfg = get_effective_agent_config(conn)
    assert cfg.agent_backend == "api"
    assert cfg.anthropic_api_key is None
    assert cfg.claude_code_oauth_token is None
    assert cfg.anthropic_model == "claude-opus-4-7"


def test_write_then_read_round_trip(conn: sqlite3.Connection) -> None:
    write_many(
        conn,
        [
            ("agent_backend", "subscription"),
            ("claude_code_oauth_token", "sk-ant-oat-test"),
        ],
    )
    cfg = get_effective_agent_config(conn)
    assert cfg.agent_backend == "subscription"
    assert cfg.claude_code_oauth_token == "sk-ant-oat-test"  # noqa: S105 - test fixture


def test_blank_value_clears_override(conn: sqlite3.Connection) -> None:
    """An empty string in PUT means 'use the env default again'."""
    write_many(conn, [("anthropic_api_key", "sk-ant-test")])
    assert read_all(conn) == {"anthropic_api_key": "sk-ant-test"}

    write_many(conn, [("anthropic_api_key", "")])
    assert read_all(conn) == {}


def test_none_value_also_clears_override(conn: sqlite3.Connection) -> None:
    write_many(conn, [("anthropic_api_key", "sk-ant-test")])
    write_many(conn, [("anthropic_api_key", None)])
    assert read_all(conn) == {}


def test_write_rejects_unknown_keys(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="unknown setting key"):
        write_many(conn, [("not_a_real_setting", "x")])


def test_redacted_view_collapses_secrets_to_booleans(conn: sqlite3.Connection) -> None:
    """The UI never sees the raw API key / OAuth token; only ``has_*`` flags."""
    write_many(
        conn,
        [
            ("anthropic_api_key", "sk-ant-secret"),
            ("claude_code_oauth_token", "sk-ant-oat-secret"),
            ("agent_backend", "subscription"),
        ],
    )
    snapshot = redacted_view(conn)
    assert snapshot["agent_backend"] == "subscription"
    assert snapshot["has_anthropic_api_key"] is True
    assert snapshot["has_claude_code_oauth_token"] is True
    # No raw value leaks.
    assert "sk-ant-secret" not in str(snapshot)
    assert "sk-ant-oat-secret" not in str(snapshot)


def test_redacted_view_reports_missing_secrets_as_false(
    conn: sqlite3.Connection,
) -> None:
    snapshot = redacted_view(conn)
    assert snapshot["has_anthropic_api_key"] is False
    assert snapshot["has_claude_code_oauth_token"] is False


def test_env_value_surfaces_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A value set in the process env (e.g. inherited from .env)
    should still drive the effective config when the DB has no
    override — the UI is *additive*, never required."""
    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")

    db_path = tmp_path / "envtest.db"
    conn = sqlite3.connect(db_path)
    apply_pending(conn)
    try:
        cfg = get_effective_agent_config(conn)
        assert cfg.anthropic_api_key == "sk-ant-from-env"
    finally:
        conn.close()
        get_settings.cache_clear()


def test_db_override_beats_env(
    monkeypatch: pytest.MonkeyPatch,
    conn: sqlite3.Connection,
) -> None:
    """When both a DB override and an env value exist, the DB wins —
    the UI is the source of truth once the user has used it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    write_many(conn, [("anthropic_api_key", "sk-ant-from-ui")])
    cfg = get_effective_agent_config(conn)
    assert cfg.anthropic_api_key == "sk-ant-from-ui"


# ---------------------------------------------------------------------------
# Apply profile (v1.17.0+)
# ---------------------------------------------------------------------------


def test_get_apply_profile_returns_empty_when_no_overrides(
    conn: sqlite3.Connection,
) -> None:
    """A fresh DB with no Settings-UI submissions returns an empty
    profile -- the snapshot module then writes a checklist with no
    profile lines (still works, just less convenient)."""
    from jobai.api.runtime_settings import get_apply_profile  # noqa: PLC0415

    assert get_apply_profile(conn) == {}


def test_get_apply_profile_strips_prefix_for_short_keys(
    conn: sqlite3.Connection,
) -> None:
    """Persisted keys are ``apply_profile_*`` but the snapshot module
    consumes them as ``full_name``, ``email``, etc -- the helper
    strips the prefix so the contract stays clean at the seam."""
    from jobai.api.runtime_settings import get_apply_profile  # noqa: PLC0415

    write_many(
        conn,
        [
            ("apply_profile_full_name", "Jane Doe"),
            ("apply_profile_email", "jane@example.com"),
            ("apply_profile_phone", "+61 400 000 000"),
        ],
    )
    profile = get_apply_profile(conn)
    assert profile == {
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+61 400 000 000",
    }


def test_redacted_view_emits_blank_strings_for_unset_apply_fields(
    conn: sqlite3.Connection,
) -> None:
    """The Settings UI needs stable field names to bind to. Every
    apply-profile key surfaces in redacted_view -- set keys carry
    their value, unset ones come back as blank strings rather than
    being omitted (which would force the UI to handle 'undefined')."""
    write_many(conn, [("apply_profile_email", "jane@example.com")])
    view = redacted_view(conn)
    # Set field surfaces with its value...
    assert view["apply_profile_email"] == "jane@example.com"
    # ...unset fields come back as empty strings for stable binding.
    assert view["apply_profile_full_name"] == ""
    assert view["apply_profile_phone"] == ""
    assert view["apply_profile_linkedin_url"] == ""
