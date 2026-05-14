"""Settings endpoints — let the UI tune runtime config without .env edits.

* ``GET /api/settings``  — return the effective config with secrets
  redacted (collapsed to ``has_*`` booleans). The UI uses this to
  populate the Settings modal on open.
* ``PUT /api/settings``  — write any subset of the user-tunable
  fields. Empty / null values clear the override so the field falls
  back to the env default again.

Why redact in GET: the values are stored on the user's local SQLite
file, but round-tripping a long-lived OAuth token through the
browser tab and (potentially) browser dev-tools history is an
unnecessary leak. The UI only needs to know "is it set?" — the
checkbox-or-text widget shows the value while the user types it,
then forgets it on submit.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jobai.api.dependencies import ConnDep
from jobai.api.runtime_settings import (
    ALLOWED_KEYS,
    SECRET_KEYS,
    redacted_view,
    write_many,
)

router = APIRouter()


class SettingsView(BaseModel):
    """Effective settings as the Settings modal sees them."""

    agent_backend: str = Field(description="One of 'api' or 'subscription'.")
    anthropic_model: str = Field(description="Configured model id.")
    has_anthropic_api_key: bool = Field(
        description="True if an API key is configured (value redacted).",
    )
    has_claude_code_oauth_token: bool = Field(
        description="True if a Claude Code OAuth token is configured (value redacted).",
    )
    apply_profile_full_name: str = Field(
        default="",
        description="Apply profile: full name.",
    )
    apply_profile_email: str = Field(
        default="",
        description="Apply profile: email address.",
    )
    apply_profile_phone: str = Field(
        default="",
        description="Apply profile: phone (with country code).",
    )
    apply_profile_location: str = Field(
        default="",
        description="Apply profile: city/region.",
    )
    apply_profile_linkedin_url: str = Field(
        default="",
        description="Apply profile: LinkedIn URL.",
    )
    apply_profile_github_url: str = Field(
        default="",
        description="Apply profile: GitHub URL.",
    )
    apply_profile_right_to_work: str = Field(
        default="",
        description=(
            "Apply profile: free-text right-to-work statement (e.g. 'Yes -- Australian citizen')."
        ),
    )
    apply_profile_notice_period: str = Field(
        default="",
        description="Apply profile: notice period (e.g. 'Immediate', '4 weeks').",
    )
    apply_profile_salary_expectation: str = Field(
        default="",
        description="Apply profile: salary expectation (e.g. '120k AUD + super').",
    )


class SettingsUpdateRequest(BaseModel):
    """Partial update — only fields that are set in the body change.

    ``extra='forbid'`` rejects unknown keys with a 422 so a typo in
    a future client doesn't silently no-op.
    """

    model_config = ConfigDict(extra="forbid")

    agent_backend: str | None = None
    anthropic_api_key: str | None = None
    claude_code_oauth_token: str | None = None
    anthropic_model: str | None = None
    apply_profile_full_name: str | None = None
    apply_profile_email: str | None = None
    apply_profile_phone: str | None = None
    apply_profile_location: str | None = None
    apply_profile_linkedin_url: str | None = None
    apply_profile_github_url: str | None = None
    apply_profile_right_to_work: str | None = None
    apply_profile_notice_period: str | None = None
    apply_profile_salary_expectation: str | None = None

    @field_validator("agent_backend")
    @classmethod
    def _validate_backend(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return value
        normalised = value.strip().lower()
        if normalised not in {"api", "subscription"}:
            msg = "agent_backend must be 'api' or 'subscription'"
            raise ValueError(msg)
        return normalised


@router.get(
    "",
    response_model=SettingsView,
    summary="Read the effective settings (secrets redacted).",
)
def get_settings_view(conn: ConnDep) -> SettingsView:
    snapshot = redacted_view(conn)
    return SettingsView.model_validate(snapshot)


@router.put(
    "",
    response_model=SettingsView,
    summary="Update one or more settings; returns the new effective view.",
)
def update_settings(body: SettingsUpdateRequest, conn: ConnDep) -> SettingsView:
    # Only forward fields the client actually included. Fields left
    # unset in the body don't change; explicit None / "" clears them.
    updates: list[tuple[str, str | None]] = []
    payload = body.model_dump(exclude_unset=True)
    for key, value in payload.items():
        if key not in ALLOWED_KEYS:
            # Pydantic already prevents unknown fields, but defence in
            # depth — this also enforces the same allow-list the
            # repository uses.
            raise HTTPException(status_code=400, detail=f"unknown setting key: {key}")
        if isinstance(value, str):
            updates.append((key, value))
        # All allow-listed fields are ``str | None`` in the Pydantic model so
        # ``value`` is always one of those two; the value-is-not-None
        # continuation branch is genuinely unreachable via this route.
        elif value is None:  # pragma: no branch
            updates.append((key, None))
    try:
        write_many(conn, updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SettingsView.model_validate(redacted_view(conn))


# Re-export the constants for tests that want to assert the surface.
__all__: list[str] = [
    "ALLOWED_KEYS",
    "SECRET_KEYS",
    "SettingsUpdateRequest",
    "SettingsView",
    "router",
]
