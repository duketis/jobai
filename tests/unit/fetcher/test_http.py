"""Tests for :class:`jobai.fetcher.http.HttpFetcher`."""

from __future__ import annotations

import httpx
import pytest
import respx

from jobai.fetcher.http import HttpFetcher


async def test_fetch_returns_response_with_status_and_body() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.example.com/jobs").mock(
            return_value=httpx.Response(200, json={"jobs": [{"id": 1}]}),
        )

        async with HttpFetcher() as fetcher:
            response = await fetcher.fetch("https://api.example.com/jobs")

        assert response.status_code == 200
        assert response.is_ok is True
        assert b'"jobs"' in response.body


async def test_fetch_returns_response_for_non_2xx_without_raising() -> None:
    """A 404 must come back as a Response with status 404, not as an
    exception."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.example.com/missing").mock(
            return_value=httpx.Response(404, text="not found"),
        )

        async with HttpFetcher() as fetcher:
            response = await fetcher.fetch("https://api.example.com/missing")

        assert response.status_code == 404
        assert response.is_ok is False
        assert "not found" in response.text


async def test_fetch_sends_default_user_agent() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://api.example.com/").mock(
            return_value=httpx.Response(200, text="ok"),
        )

        async with HttpFetcher(user_agent="jobai-test/1.0") as fetcher:
            await fetcher.fetch("https://api.example.com/")

        assert route.called
        request = route.calls.last.request
        assert request.headers["user-agent"] == "jobai-test/1.0"


async def test_fetch_passes_through_custom_headers() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://api.example.com/").mock(
            return_value=httpx.Response(200, text="ok"),
        )

        async with HttpFetcher() as fetcher:
            await fetcher.fetch(
                "https://api.example.com/",
                headers={"X-Custom": "value"},
            )

        assert route.called
        request = route.calls.last.request
        assert request.headers["x-custom"] == "value"


async def test_fetch_supports_post_with_json_body() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.example.com/q").mock(
            return_value=httpx.Response(200, json={"ok": True}),
        )

        async with HttpFetcher() as fetcher:
            response = await fetcher.fetch(
                "https://api.example.com/q",
                method="POST",
                json={"includeCompensation": True},
            )

        assert response.is_ok
        assert route.called
        request = route.calls.last.request
        assert b"includeCompensation" in request.content


async def test_fetch_supports_post_with_form_encoded_data() -> None:
    """``data=`` URL-encodes a mapping for ``application/x-www-form-urlencoded``
    bodies — Salesforce Aura, OAuth token exchanges, classic HTML
    forms.
    """
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.example.com/aura").mock(
            return_value=httpx.Response(200, text="ok"),
        )

        async with HttpFetcher() as fetcher:
            response = await fetcher.fetch(
                "https://api.example.com/aura",
                method="POST",
                data={"message": '{"a":1}', "aura.token": "null"},
            )

        assert response.is_ok
        assert route.called
        request = route.calls.last.request
        assert request.headers["content-type"].startswith(
            "application/x-www-form-urlencoded",
        )
        # httpx URL-encodes the JSON values (curly braces and colons).
        assert b"message=%7B%22a%22%3A1%7D" in request.content
        assert b"aura.token=null" in request.content


async def test_fetch_records_fetched_at_timestamp_in_utc() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.example.com/").mock(
            return_value=httpx.Response(200, text="ok"),
        )

        async with HttpFetcher() as fetcher:
            response = await fetcher.fetch("https://api.example.com/")

        assert response.fetched_at.tzinfo is not None


async def test_fetch_raises_on_network_failure() -> None:
    """Genuine connection failures must surface as exceptions; only
    HTTP-level non-2xx responses are returned as Response objects."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.example.com/").mock(
            side_effect=httpx.ConnectError("connection refused"),
        )

        async with HttpFetcher() as fetcher:
            with pytest.raises(httpx.ConnectError):
                await fetcher.fetch("https://api.example.com/")


async def test_async_context_manager_closes_underlying_client() -> None:
    """Exiting the ``async with`` must call aclose on the wrapped client."""
    fetcher = HttpFetcher()
    async with fetcher:
        pass
    assert fetcher._client.is_closed is True
