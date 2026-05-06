"""HTTP-level tests for the settings endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_returns_default_view_when_table_empty(client: TestClient) -> None:
    response = client.get("/api/settings")
    assert response.status_code == 200
    body = response.json()
    assert body["agent_backend"] == "api"
    assert body["has_anthropic_api_key"] is False
    assert body["has_claude_code_oauth_token"] is False


def test_put_persists_overrides_and_returns_redacted_view(
    client: TestClient,
) -> None:
    response = client.put(
        "/api/settings",
        json={
            "agent_backend": "subscription",
            "claude_code_oauth_token": "sk-ant-oat-test",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["agent_backend"] == "subscription"
    assert body["has_claude_code_oauth_token"] is True
    # No raw token in the response.
    assert "sk-ant-oat-test" not in response.text


def test_put_with_blank_string_clears_a_secret(client: TestClient) -> None:
    """The UI sends `""` to mean 'forget the saved value'."""
    client.put("/api/settings", json={"anthropic_api_key": "sk-ant-test"})
    snapshot = client.get("/api/settings").json()
    assert snapshot["has_anthropic_api_key"] is True

    client.put("/api/settings", json={"anthropic_api_key": ""})
    snapshot = client.get("/api/settings").json()
    assert snapshot["has_anthropic_api_key"] is False


def test_put_rejects_invalid_backend_value(client: TestClient) -> None:
    response = client.put("/api/settings", json={"agent_backend": "lol"})
    assert response.status_code == 422
    assert "agent_backend" in response.text


def test_put_with_partial_body_only_updates_listed_fields(
    client: TestClient,
) -> None:
    """Fields absent from the PUT body must not change — the modal
    only sends the fields the user touched."""
    client.put(
        "/api/settings",
        json={
            "agent_backend": "subscription",
            "claude_code_oauth_token": "tok-1",
        },
    )
    # Now update only the model. Backend + token should survive.
    client.put("/api/settings", json={"anthropic_model": "claude-haiku-4-5"})
    snapshot = client.get("/api/settings").json()
    assert snapshot["agent_backend"] == "subscription"
    assert snapshot["has_claude_code_oauth_token"] is True
    assert snapshot["anthropic_model"] == "claude-haiku-4-5"


def test_put_unknown_field_is_rejected(client: TestClient) -> None:
    response = client.put(
        "/api/settings",
        json={"not_a_real_setting": "value"},
    )
    # Pydantic strict-by-default would reject unknown fields with 422;
    # accept either 400 or 422 since we have a defence-in-depth check.
    assert response.status_code in {400, 422}
