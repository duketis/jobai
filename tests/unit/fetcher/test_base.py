"""Tests for the Fetcher Protocol and Response dataclass."""

from __future__ import annotations

from datetime import UTC, datetime

from jobai.fetcher.base import Fetcher, Response
from jobai.fetcher.http import HttpFetcher


def test_response_is_ok_returns_true_for_2xx() -> None:
    for status in (200, 201, 204, 299):
        response = Response(
            url="https://example.com",
            status_code=status,
            headers={},
            body=b"",
        )
        assert response.is_ok is True


def test_response_is_ok_returns_false_for_non_2xx() -> None:
    for status in (199, 300, 404, 500, 503):
        response = Response(
            url="https://example.com",
            status_code=status,
            headers={},
            body=b"",
        )
        assert response.is_ok is False


def test_response_text_decodes_utf8_body() -> None:
    response = Response(
        url="https://example.com",
        status_code=200,
        headers={},
        body=b"hello world",
    )
    assert response.text == "hello world"


def test_response_text_replaces_invalid_bytes() -> None:
    response = Response(
        url="https://example.com",
        status_code=200,
        headers={},
        body=b"valid \xff\xfe invalid",
    )
    assert "valid" in response.text
    assert "invalid" in response.text  # should not raise


def test_response_fetched_at_defaults_to_now() -> None:
    before = datetime.now(tz=UTC)
    response = Response(
        url="https://example.com",
        status_code=200,
        headers={},
        body=b"",
    )
    after = datetime.now(tz=UTC)
    assert before <= response.fetched_at <= after


def test_http_fetcher_satisfies_fetcher_protocol() -> None:
    """The runtime-checkable Protocol must accept the concrete tier-1 fetcher."""
    fetcher = HttpFetcher()
    assert isinstance(fetcher, Fetcher)
