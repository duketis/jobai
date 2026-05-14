"""Runtime-tunable settings backed by the ``app_settings`` table.

The :mod:`jobai.config` ``Settings`` object captures the boot-time
configuration loaded from environment variables / ``.env``. That's
fine for ops-style knobs (DB path, log dir, log level) but a public
app shouldn't make users edit a dotfile to switch agent backends or
rotate API keys. This module layers a SQLite-backed override on top:
boot defaults stay where they are, but anything the user changes via
the Settings UI lives in ``app_settings`` and wins at request time.

Currently surfaced:

* ``agent_backend``           — ``"api"`` / ``"subscription"``
* ``anthropic_api_key``       — used in API mode
* ``claude_code_oauth_token`` — used in subscription mode
* ``anthropic_model``         — model id

The two secrets are stored verbatim. The DB file lives on the user's
own machine (single-tenant, local-first) so there's no cross-user
leakage; storing in plaintext is the same trust model as a ``.env``
file in the project directory.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from jobai.config import get_settings

#: Keys that make up the user's "apply profile" — the contact details
#: the snapshot module pastes into each job's ``CHECKLIST.md`` so the
#: user can copy-paste them straight into application forms. Stored
#: as plain strings (no secrets in here -- the most sensitive thing
#: is a phone number, and that's on the user's resume anyway).
APPLY_PROFILE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "apply_profile_full_name",
        "apply_profile_email",
        "apply_profile_phone",
        "apply_profile_location",
        "apply_profile_linkedin_url",
        "apply_profile_github_url",
        "apply_profile_right_to_work",
        "apply_profile_notice_period",
        "apply_profile_salary_expectation",
    },
)

#: All keys the UI is allowed to read/write. Anything outside this set
#: is rejected so a future change to this module is the only way to
#: extend the surface — random other env values stay out of reach.
ALLOWED_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "agent_backend",
            "anthropic_api_key",
            "claude_code_oauth_token",
            "anthropic_model",
        },
    )
    | APPLY_PROFILE_KEYS
)

#: Keys whose values are secrets and should be redacted in GET
#: responses. The UI only ever needs to know whether a value is set,
#: not what it is.
SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {"anthropic_api_key", "claude_code_oauth_token"},
)


@dataclass(frozen=True, slots=True)
class EffectiveAgentConfig:
    """Resolved runtime config for the agent layer.

    Built fresh on each request so a settings UPDATE takes effect on
    the next chat turn without restarting the process.
    """

    agent_backend: str
    anthropic_api_key: str | None
    claude_code_oauth_token: str | None
    anthropic_model: str


def get_effective_agent_config(conn: sqlite3.Connection) -> EffectiveAgentConfig:
    """Return the live agent config, merging DB overrides over env.

    Resolution order for each field:
    1. ``app_settings.value`` if non-empty (set via the UI / API).
    2. ``Settings`` from ``jobai.config`` (env / .env at boot).
    3. The relevant ``ANTHROPIC_*`` / ``CLAUDE_CODE_*`` env-var if
       neither of the above set it (matches the SDK fallbacks).
    """
    overrides = _read_overrides(conn)
    boot = get_settings()

    agent_backend = (
        (overrides.get("agent_backend") or (boot.agent_backend if boot.agent_backend else "api"))
        .strip()
        .lower()
    )

    api_key = (
        overrides.get("anthropic_api_key")
        or boot.anthropic_api_key
        or os.environ.get(
            "ANTHROPIC_API_KEY",
        )
    )
    oauth_token = overrides.get("claude_code_oauth_token") or os.environ.get(
        "CLAUDE_CODE_OAUTH_TOKEN",
    )
    model = overrides.get("anthropic_model") or boot.anthropic_model

    return EffectiveAgentConfig(
        agent_backend=agent_backend,
        anthropic_api_key=_blank_to_none(api_key),
        claude_code_oauth_token=_blank_to_none(oauth_token),
        anthropic_model=model,
    )


def read_all(conn: sqlite3.Connection) -> dict[str, str]:
    """Return every override row keyed by setting name."""
    return _read_overrides(conn)


def get_apply_profile(conn: sqlite3.Connection) -> dict[str, str]:
    """Return the user's apply profile as ``{short_key: value}``.

    Short keys drop the ``apply_profile_`` prefix so the snapshot
    module can use them directly as field labels (``full_name``,
    ``email``, ...). Missing fields are omitted; the snapshot module
    only emits checklist lines for keys that have a value.
    """
    overrides = _read_overrides(conn)
    profile: dict[str, str] = {}
    for key in APPLY_PROFILE_KEYS:
        value = overrides.get(key)
        if value:
            short = key.removeprefix("apply_profile_")
            profile[short] = value
    return profile


def write_many(conn: sqlite3.Connection, items: Iterable[tuple[str, str | None]]) -> None:
    """Persist each ``(key, value)`` pair, validating against the allow-list.

    A ``None`` value deletes the override (so the field falls back to
    the env-default again). An empty string is treated the same way —
    the UI's "blank input" should always mean "use the default".
    """
    pairs = list(items)
    for key, _value in pairs:
        if key not in ALLOWED_KEYS:
            msg = f"unknown setting key: {key!r}"
            raise ValueError(msg)
    for key, value in pairs:
        if value is None or value == "":
            conn.execute("DELETE FROM app_settings WHERE key = ?", (key,))
            continue
        conn.execute(
            "INSERT INTO app_settings (key, value, updated_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET "
            "  value = excluded.value, updated_at = excluded.updated_at",
            (key, value),
        )
    conn.commit()


def redacted_view(conn: sqlite3.Connection) -> dict[str, str | bool]:
    """Return a UI-safe snapshot of the effective config.

    Secret keys collapse to a boolean ``has_*`` flag so the UI can
    show "set / not set" without ever round-tripping the value back
    through the browser. Non-secret keys are returned verbatim,
    including every apply-profile field (none of those are secret --
    a user opening the Settings modal needs to see what's currently
    there).
    """
    cfg = get_effective_agent_config(conn)
    view: dict[str, str | bool] = {
        "agent_backend": cfg.agent_backend,
        "anthropic_model": cfg.anthropic_model,
        "has_anthropic_api_key": cfg.anthropic_api_key is not None,
        "has_claude_code_oauth_token": cfg.claude_code_oauth_token is not None,
    }
    profile = get_apply_profile(conn)
    for short_key, value in profile.items():
        view[f"apply_profile_{short_key}"] = value
    # Always emit every apply_profile_* key so the UI has stable fields
    # to bind to -- blanks come back as empty strings.
    for key in APPLY_PROFILE_KEYS:
        view.setdefault(key, "")
    return view


def _read_overrides(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    return {str(row[0]): str(row[1]) for row in rows if row[1] is not None}


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None
